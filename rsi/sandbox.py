"""Non-container scientific execution: UID + Landlock + seccomp, fail closed.

Requires Linux x86_64, Landlock ABI >= 3, libseccomp and a root supervisor.
Only explicitly granted Agent working copies are writable. Credentials, other
tasks and evaluation labels are not granted.
"""
from __future__ import annotations

import ctypes
import errno
from functools import lru_cache
import json
import os
import re
import resource
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path


@lru_cache(maxsize=32)
def _gpu_uuid(process_id, gpu):
    """Resolve an assigned GPU once per host process, before confinement.

    UUIDs, rather than mutable device ordinals, are passed to CUDA. Including
    the PID prevents a forked worker from inheriting its parent's cached mapping.
    Only successful, single-device responses are cached; transient probe errors
    get bounded retries and persistent errors still fail closed.
    """
    command = ['nvidia-smi', '-i', gpu, '--query-gpu=uuid', '--format=csv,noheader']
    for attempt in range(3):
        try:
            probe = subprocess.run(command, capture_output=True, text=True,
                                   timeout=20, check=True)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            if attempt == 2:
                raise
            time.sleep(1)
            continue
        value = probe.stdout.strip()
        if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value):
            raise RuntimeError('Cannot resolve assigned GPU UUID')
        return value


def _landlock(read_paths, write_paths):
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 3:
        raise RuntimeError(f'Landlock ABI >= 3 required, got {abi}')
    # All filesystem rights through ABI 3; never permit creating device nodes.
    handled = (1 << 15) - 1
    read = (1 << 0) | (1 << 2) | (1 << 3)
    file_rights = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)

    class Ruleset(ctypes.Structure):
        _fields_ = [('handled_access_fs', ctypes.c_uint64)]

    class Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int)]

    attr = Ruleset(handled)
    fd = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'landlock_create_ruleset')
    try:
        for path, rights in [(p, read) for p in read_paths] + [(p, handled & ~((1 << 6) | (1 << 11))) for p in write_paths]:
            path = Path(path)
            if not path.exists():
                continue
            pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                if not stat.S_ISDIR(os.fstat(pfd).st_mode):
                    rights &= file_rights
                rule = Beneath(rights, pfd)
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0) < 0:
                    raise OSError(ctypes.get_errno(), f'landlock_add_rule: {path}')
            finally:
                os.close(pfd)
        if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.syscall(446, fd, 0) < 0:
            raise OSError(ctypes.get_errno(), 'landlock_restrict_self')
    finally:
        os.close(fd)


def _seccomp(restrict_affinity=False):
    lib = ctypes.CDLL('libseccomp.so.2')
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not ctx:
        raise RuntimeError('seccomp_init failed')
    denied = ('connect', 'sendto', 'sendmsg', 'sendmmsg',
              'ptrace', 'process_vm_readv', 'process_vm_writev', 'pidfd_getfd',
              'mount', 'umount2', 'pivot_root', 'chroot', 'unshare', 'setns',
              'bpf', 'perf_event_open', 'open_by_handle_at', 'name_to_handle_at',
              'io_uring_setup', 'keyctl', 'add_key', 'request_key', 'reboot')
    if restrict_affinity:
        denied += ('sched_setaffinity',)
    try:
        class Compare(ctypes.Structure):
            _fields_ = [('arg', ctypes.c_uint), ('op', ctypes.c_int),
                        ('datum_a', ctypes.c_uint64), ('datum_b', ctypes.c_uint64)]
        lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                               ctypes.c_int, ctypes.c_uint, ctypes.POINTER(Compare)]
        # CUDA 590 initializes a local Unix listener for UVM file descriptors.
        # Internet sockets and outgoing connections/messages remain forbidden.
        comparison = Compare(0, 1, 1, 0)  # arg0 != AF_UNIX, SCMP_CMP_NE=1
        if lib.seccomp_rule_add_array(ctx, 0x50000 | errno.EPERM,
                                     lib.seccomp_syscall_resolve_name(b'socket'),
                                     1, ctypes.byref(comparison)) != 0:
            raise RuntimeError('seccomp socket rule failed')
        for name in denied:
            num = lib.seccomp_syscall_resolve_name(name.encode())
            if num >= 0 and lib.seccomp_rule_add(ctx, 0x50000 | errno.EPERM, num, 0) != 0:
                raise RuntimeError(f'seccomp rule failed: {name}')
        if lib.seccomp_load(ctx) != 0:
            raise RuntimeError('seccomp_load failed')
    finally:
        lib.seccomp_release(ctx)


def assert_plain_tree(root, ignore=()):
    """Reject user-controlled links/devices before any privileged snapshot/read."""
    root = Path(root)
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError('Workspace root must be a real directory')
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if name not in ignore]
        for name in dirs + files:
            if name in ignore:
                continue
            path = Path(base) / name
            mode = path.lstat().st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError(f'Workspace contains a link or special file: {path.relative_to(root)}')
            if stat.S_ISREG(mode) and path.stat().st_nlink != 1:
                raise ValueError(f'Workspace contains a hard link: {path.relative_to(root)}')


def prepare_workspace(root, uid):
    """Grant ordinary files to the sandbox without following links or devices.

    A confined command must remain able to unlink an invalid artifact. Privileged
    snapshots still require a plain tree; execution relies on UID and Landlock
    restrictions, including for paths reached through symlinks.
    """
    root = Path(root)
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError('Workspace root must be a real directory')
    for base, dirs, files in os.walk(root, followlinks=False):
        os.chown(base, uid, uid, follow_symlinks=False)
        os.chmod(base, 0o700, follow_symlinks=False)
        dirs[:] = [name for name in dirs
                   if stat.S_ISDIR((Path(base) / name).lstat().st_mode)]
        for name in files:
            path = Path(base) / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                continue
            os.chown(path, uid, uid, follow_symlinks=False)
            os.chmod(path, (info.st_mode & 0o111) | 0o600, follow_symlinks=False)


def prepare_frozen_workspace(root, uid):
    """Inference may write only predictions and temporary files, never its model."""
    root = Path(root)
    assert_plain_tree(root)
    for path in [root, *root.rglob('*')]:
        os.chown(path, os.geteuid(), os.getegid(), follow_symlinks=False)
        path.chmod(0o555 if path.is_dir() else (0o555 if path.stat().st_mode & 0o111 else 0o444))
    for name in ('output', '.tmp'):
        path = root/name
        path.mkdir(exist_ok=True)
        prepare_workspace(path, uid)


def _kill_uid(uid):
    # Covers daemonized children that escaped the initial process group.
    killed = set()
    for _ in range(4):
        found = False
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():
                continue
            try:
                if path.stat().st_uid == uid:
                    if (path/'stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                        continue
                    os.kill(int(path.name), signal.SIGKILL)
                    killed.add(int(path.name))
                    found = True
            except (FileNotFoundError, ProcessLookupError):
                pass
        if not found:
            break
        time.sleep(0.03)
    return sorted(killed)


def uid_resources(uid):
    """Conservative summed RSS (shared pages counted per process), not cgroup accounting."""
    import psutil
    rss, pids = 0, []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid == uid:
                process = psutil.Process(int(path.name))
                rss += process.memory_info().rss
                pids.append(process.pid)
        except (OSError, psutil.Error):
            pass
    return {'rss_bytes': rss, 'pids': pids}


def validate_resources(value):
    value = dict(value or {})
    if set(value) - {'cpu_affinity', 'threads', 'memory_limit_bytes', 'max_processes'}:
        raise ValueError('Unknown sandbox resource setting')
    cpus = value.get('cpu_affinity')
    if cpus is not None and (not isinstance(cpus, list) or not cpus or
            any(type(cpu) is not int or cpu not in os.sched_getaffinity(0) for cpu in cpus) or
            len(cpus) != len(set(cpus))):
        raise ValueError('cpu_affinity must be a nonempty unique subset of allowed CPUs')
    for key in ('threads', 'memory_limit_bytes', 'max_processes'):
        if key in value and (type(value[key]) is not int or value[key] < 1):
            raise ValueError('Invalid sandbox resource setting: ' + key)
    return value


class Sandbox:
    def __init__(self, python, problem, workspace, logs, gpu, uid=None, read_extra=(), env_extra=None,
                 limits=None, frozen_workspace=False, audit_sources=False, write_extra=()):
        if os.geteuid() != 0:
            raise RuntimeError('Local sandbox requires root to drop to a dedicated UID')
        # Keep the virtualenv entry path: resolving a venv's python symlink
        # silently switches to its base environment and loses task dependencies.
        self.python = str(Path(python).absolute())
        self.problem, self.workspace, self.logs = map(lambda p: Path(p).resolve(), (problem, workspace, logs))
        self.gpu = gpu
        self.frozen_workspace, self.audit_sources = frozen_workspace, audit_sources
        self.limits = validate_resources(limits)
        self.gpu_uuid = ''
        if gpu is not None:
            self.gpu_uuid = _gpu_uuid(os.getpid(), str(gpu))
        self.read_extra, self.env_extra = list(map(str, read_extra)), env_extra or {}
        # These paths come only from the host, never from an Agent request.
        self.write_extra = [str(Path(path).resolve()) for path in write_extra]
        if frozen_workspace and self.write_extra:
            raise ValueError('Frozen inference cannot grant additional writable directories')
        self.uid = uid or (200000 + os.getpid())
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.index = 0

    def run(self, code, timeout):
        if timeout is not None and timeout <= 0:
            raise TimeoutError('No solve budget left')
        if not isinstance(code, str) or len(code.encode()) > 100000:
            raise ValueError('Python action too large')
        # The caller supplies its remaining budget. Preparing a large inherited
        # workspace must not grant the scientific process extra time.
        started = time.monotonic()
        deadline = float('inf') if timeout is None else started + timeout
        from rsi.host_io import run_io
        run_io('prepare_frozen_workspace' if self.frozen_workspace else 'prepare_workspace',
               deadline, workspace=str(self.workspace), uid=self.uid)
        for path in self.write_extra:
            run_io('prepare_workspace', deadline, workspace=path, uid=self.uid)
        self.index += 1
        def observe_sources(phase):
            try:
                return run_io('source_snapshot', deadline, workspace=str(self.workspace),
                              directory=str(self.logs/'source-audit'), name=f'exec-{self.index:03d}-{phase}')
            except Exception as exc:
                # Missing observation must be explicit; it must not prevent
                # commands that repair a malformed scientific workspace.
                return dict(complete=False, error=f'{type(exc).__name__}: {exc}')
        source_before = observe_sources('before') if self.audit_sources else None
        config = self.logs / f'exec-{self.index:03d}.json'
        log = self.logs / f'exec-{self.index:03d}.log'
        config.write_text(json.dumps(dict(python=self.python, problem=str(self.problem),
                                         workspace=str(self.workspace), gpu=self.gpu,
                                         gpu_uuid=self.gpu_uuid,
                                         uid=self.uid, code=code, read_extra=self.read_extra,
                                         write_extra=self.write_extra,
                                         env_extra=self.env_extra, limits=self.limits,
                                         frozen_workspace=self.frozen_workspace)))
        config.chmod(0o600)
        reason = None
        peak_rss, resource_samples, next_resource_check = 0, 0, 0
        env = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'}
        with log.open('wb') as output:
            if time.monotonic() >= deadline:
                raise TimeoutError('Sandbox preparation exhausted action deadline')
            proc = subprocess.Popen([self.python, '-I', str(Path(__file__).resolve()), str(config)],
                                    env=env, stdout=output, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
            try:
                while proc.poll() is None:
                    if time.monotonic() >= next_resource_check:
                        measured = uid_resources(self.uid)
                        if measured['pids']:
                            peak_rss = max(peak_rss, measured['rss_bytes'])
                            resource_samples += 1
                        next_resource_check = time.monotonic() + .5
                        if (self.limits.get('memory_limit_bytes')
                                and measured['rss_bytes'] > self.limits['memory_limit_bytes']):
                            reason = 'memory_limit'
                            break
                    if time.monotonic() >= deadline:
                        reason = 'timeout'
                        break
                    if log.stat().st_size > 16 * 1024 * 1024:
                        reason = 'output_limit'
                        break
                    time.sleep(0.1)
            finally:
                killed = _kill_uid(self.uid)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 16000))
            tail = stream.read().decode(errors='replace')
        result = {'returncode': proc.returncode, 'reason': reason,
                'time_limit_seconds': timeout,
                'seconds': time.monotonic() - started, 'stdout': tail, 'log': str(log),
                'peak_sampled_rss_bytes': peak_rss if resource_samples else None,
                'resource_samples': resource_samples, 'rss_sample_interval_seconds': .5,
                'uid': self.uid,
                'resource_limits': self.limits}
        if reason is None and killed:
            result.update(reason='background_processes_terminated',
                error='The command returned with child processes still running. They were stopped; '
                      'background jobs do not survive a tool call. Run the program in the foreground '
                      'or wait for all children inside the same command. Do not sleep/poll for this job.',
                background_processes_terminated=len(killed))
        if self.audit_sources:
            from rsi.common import write_json
            import hashlib
            report = dict(code=code, before=source_before, after=observe_sources('after'),
                          process={key: result[key] for key in ('returncode', 'reason', 'seconds', 'log')},
                          scope='Host-recorded command and source observations; actual fitting membership is not proven')
            audit = self.logs/f'exec-{self.index:03d}.audit.json'
            write_json(audit, report); audit.chmod(0o444)
            result['execution_audit'] = dict(path=str(audit), sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
                complete=bool(report['before'].get('complete') and report['after'].get('complete')))
        return result


def _child(config_path):
    config = json.loads(Path(config_path).read_text())
    limits = config.get('limits', {})
    if limits.get('cpu_affinity'):
        os.sched_setaffinity(0, limits['cpu_affinity'])
    workspace = Path(config['workspace'])
    tmp = workspace / '.tmp'
    # Only these inherited variables reach scientific code; never API credentials.
    env = {'PATH': str(Path(config['python']).parent) + ':/usr/bin:/bin',
           'HOME': str(tmp), 'TMPDIR': str(tmp), 'XDG_CACHE_HOME': str(tmp),
           'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1', 'LANG': 'C.UTF-8',
           'OPENBLAS_NUM_THREADS': str(limits.get('threads', 4)),
           'OMP_NUM_THREADS': str(limits.get('threads', 4)),
           'MKL_NUM_THREADS': str(limits.get('threads', 4)),
           'CUDA_VISIBLE_DEVICES': config['gpu_uuid'], 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
           'DATA_DIR': str(Path(config['problem']) / 'data'),
           'PROBLEM_DIR': config['problem'], 'OUTPUT_DIR': str(workspace / 'output'),
           'WORKSPACE': str(workspace)}
    env.update(config.get('env_extra', {}))
    read_paths = ['/usr', '/bin', '/lib', '/lib64', '/etc/ld.so.cache', '/etc/alternatives',
                  '/sys', '/proc/cpuinfo', '/proc/meminfo', '/proc/stat', '/proc/self', '/proc/devices',
                  '/proc/driver/nvidia', str(Path(config['python']).parents[1]),
                  sys.base_prefix, config['problem']]
    read_paths.extend(config.get('read_extra', []))
    if config.get('frozen_workspace'):
        read_paths.append(str(workspace))
    devices = ['/dev/null', '/dev/zero', '/dev/random', '/dev/urandom']
    if config['gpu'] is not None:
        devices += ['/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools', f'/dev/nvidia{config["gpu"]}']
    # CUDA also initializes in subprocesses (e.g. python run.py). A rule on
    # /proc/self resolves only the launcher PID, so descendants need /proc.
    # Dedicated UIDs enforce proc ownership: other processes' environ/mem/fds
    # and all root-owned kernel controls remain inaccessible. ptrace and
    # process_vm_* are additionally denied by seccomp.
    writable = [str(workspace/'output'), str(workspace/'.tmp')] if config.get('frozen_workspace') else [str(workspace)]
    if not config.get('frozen_workspace'):
        writable.extend(config.get('write_extra', []))
    _landlock(read_paths, [*writable, '/proc', *devices])
    _seccomp(restrict_affinity=bool(limits.get('cpu_affinity')))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 2048))
    max_processes = limits.get('max_processes', 512)
    resource.setrlimit(resource.RLIMIT_NPROC, (max_processes, max_processes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**3, 16 * 1024**3))
    os.setgroups([])
    os.setresgid(config['uid'], config['uid'], config['uid'])
    os.setresuid(config['uid'], config['uid'], config['uid'])
    os.chdir(workspace)
    os.umask(0o077)
    # Do this only after confinement and dropping privileges. Even a malformed
    # .tmp entry must leave the Agent able to run a command that removes it.
    if not os.path.lexists(tmp):
        tmp.mkdir(mode=0o700)
    os.execve(config['python'], [config['python'], '-c', config['code']], env)


if __name__ == '__main__':
    _child(sys.argv[1])
