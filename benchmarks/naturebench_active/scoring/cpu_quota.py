"""Per-terminal CPU quota in an operator-delegated cgroup v2 directory."""
from contextlib import contextmanager
import os
from pathlib import Path
import uuid


def physical_id(cpu):
    root=Path('/sys/devices/system/cpu')/f'cpu{cpu}'/'topology'
    return tuple((root/name).read_text().strip() for name in ('physical_package_id','core_id'))


def check_physical(cpus):
    if len({physical_id(cpu) for cpu in cpus})!=len(cpus):
        raise ValueError('Terminal allocation must use distinct physical CPU cores')


def effective_quota(directory, mount=Path('/sys/fs/cgroup')):
    directory,mount=Path(directory).resolve(),Path(mount).resolve()
    if not directory.is_relative_to(mount): raise ValueError('CPU cgroup must be within cgroup v2 mount')
    limit=float('inf')
    while True:
        path=directory/'cpu.max'
        if path.exists():
            quota,period=path.read_text().split()
            if quota!='max': limit=min(limit,int(quota)/int(period))
        if directory==mount:break
        directory=directory.parent
    return limit


@contextmanager
def cpu_quota(root, cores):
    if root is None and cores is None:
        yield None
        return
    if root is None or type(cores) is not int or cores<1:
        raise ValueError('CPU quota needs positive cores and an operator-delegated cgroup directory')
    root=Path(root).resolve();mount=Path('/sys/fs/cgroup')
    if effective_quota(root,mount)<cores:
        raise ValueError('Ancestor CPU quota is smaller than the requested terminal allocation')
    if 'cpu' not in (root/'cgroup.subtree_control').read_text().split() or not os.access(root,os.W_OK):
        raise ValueError('Provide a writable cgroup with delegated CPU controller before launching terminal jobs')
    old=next(line.split('::',1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    original=mount/old.lstrip('/')
    group=root/('rsi-terminal-'+uuid.uuid4().hex);group.mkdir()
    moved=False
    try:
        value=f'{cores*100000} 100000'
        (group/'cpu.max').write_text(value)
        if (group/'cpu.max').read_text().strip()!=value:raise ValueError('CPU quota did not take effect')
        (group/'cgroup.procs').write_text(str(os.getpid()));moved=True
        yield dict(path=str(group),cpu_max=value,cores=cores,ancestor_quota=effective_quota(root,mount))
    finally:
        if moved:(original/'cgroup.procs').write_text(str(os.getpid()))
        group.rmdir()
