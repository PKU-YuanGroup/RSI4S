"""Host-owned source observations around scientific commands, not fitting proof."""
import hashlib
import json
import os
from pathlib import Path
import stat

from rsi.common import write_json


def snapshot(workspace, directory, name):
    """Copy bounded source evidence without following user-controlled links."""
    workspace, directory = Path(workspace), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    blobs = directory/'sources'; blobs.mkdir(exist_ok=True)
    files, omitted = {}, []
    remaining = 8 * 1024 * 1024
    extensions = {'.py', '.sh', '.bash', '.r', '.jl', '.m', '.c', '.cc', '.cpp', '.h', '.hpp', '.cu', '.ipynb'}
    skip = {'.tmp', '__pycache__', '.git', '.venv', 'venv', 'node_modules'}
    for base, dirs, names in os.walk(workspace, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in skip and not (Path(base)/d).is_symlink())
        for filename in sorted(names):
            path = Path(base)/filename
            if path.suffix.lower() not in extensions and filename != 'Makefile':
                continue
            relative = path.relative_to(workspace).as_posix()
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                omitted.append(dict(path=relative, reason='Not a plain source file'))
                continue
            limit = min(1024 * 1024, remaining)
            if info.st_size > limit:
                omitted.append(dict(path=relative, reason='Source evidence size limit', bytes=info.st_size))
                continue
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, 'rb') as stream:
                data = stream.read(limit+1)
                after = os.fstat(stream.fileno())
            if (len(data) > limit or (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
                omitted.append(dict(path=relative, reason='Source changed during observation'))
                continue
            remaining -= len(data)
            digest = hashlib.sha256(data).hexdigest()
            target = blobs/(digest+'.txt')
            if not target.exists():
                # A deadline may kill this worker during a write. Publish only
                # complete blobs so later snapshots never reuse a partial file.
                temporary = target.with_suffix('.partial')
                temporary.write_bytes(data); temporary.chmod(0o444)
                temporary.replace(target)
            files[relative] = dict(sha256=digest, bytes=len(data), source=str(target))
    report = dict(files=files, omitted=omitted, complete=not omitted,
                  scope='Source files present at a command boundary; not proof that every file was executed')
    path = directory/(name+'.json')
    if path.exists():
        raise ValueError('Cannot overwrite scientific execution evidence')
    write_json(path, report); path.chmod(0o444)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                files=len(files), omitted=len(omitted), complete=not omitted)


def review_evidence(reference, run_directory, source_budget=60000):
    """Verify host receipts and attach bounded historical source evidence for review."""
    from judge.sources import clip_text_bytes
    run_directory = Path(run_directory).resolve()

    def read_checked(item, key='path'):
        path = Path(item[key]).resolve()
        if not path.is_relative_to(run_directory):
            raise ValueError('Execution evidence escaped the experiment directory')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('Host execution evidence changed: ' + str(path))
        return data

    report = json.loads(read_checked(reference))
    sources, seen = [], set()
    remaining = source_budget
    for run in report['runs']:
        for action in run['actions']:
            receipt = action['outcome'].get('execution_audit')
            if not receipt:
                continue
            observation = json.loads(read_checked(receipt))
            for phase in ('before', 'after'):
                manifest = observation[phase]
                if not manifest.get('path'):
                    continue
                files = json.loads(read_checked(manifest))['files']
                for name, item in files.items():
                    if item['sha256'] in seen or remaining <= 0:
                        continue
                    data = read_checked(item, key='source')
                    text = clip_text_bytes(data.decode(errors='replace'), remaining)
                    sources.append(dict(run_id=run['run_id'], phase=phase, file=name,
                                        sha256=item['sha256'], source=text))
                    remaining -= min(len(data), remaining)
                    seen.add(item['sha256'])
    return dict(record=clip_text_bytes(json.dumps(report, ensure_ascii=False), 60000),
                source_observations=sources, source_display_budget_bytes=source_budget,
                actual_fitting_membership_verified=False, in_memory_fitting_excluded=False,
                interpretation='Commands and source observations are host-recorded. '
                    'Source presence or data authorization does not prove actual fitting; '
                    'use the code and execution evidence to assess the declared method.')
