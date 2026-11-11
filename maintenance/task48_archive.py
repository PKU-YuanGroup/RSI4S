"""Lossless content archives, verified before publication or payload deletion.

Code and history originals remain readable. Known large worker inputs can be
read through open_source or restored at their original path. Successful checks
are reused only in this process, bound to an unchanged inode. Old corrupt final
archives require the explicit repair command below.
"""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import zlib

from rsi.common import write_json, record_path

BLOCK = 4 * 1024 * 1024
COMPRESSION_LEVEL = 1
INPUT_MIN_BYTES = 16 * 1024 * 1024


def archived_input(path, run):
    """Only host-generated worker inputs, never similarly named Agent files."""
    parts=path.relative_to(run).parts
    replay=(len(parts)==7 and parts[0]=='sessions' and parts[2]=='attempts' and
            parts[4]=='replay' and parts[5] in ('train_input','validation_input'))
    final=(len(parts)==3 and parts[:2]==('final','final_input'))
    return ((replay or final) and path.name in ('records.jsonl','validation_records.jsonl') and
            path.stat().st_size>=INPUT_MIN_BYTES)


def identity(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Archive/source must be a regular file: '+str(path))
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_json(path, value):
    """Persist the path/content mapping before allowing removal of originals."""
    write_json(path,value)
    with Path(path).open('rb') as stream:os.fsync(stream.fileno())
    sync_directory(Path(path).parent)


class Archives:
    def __init__(self, run, *, progress=True):
        self.progress_enabled = progress
        self.run = Path(run).resolve()
        self.root = self.run/'retained-source'
        if self.root.is_symlink():
            raise ValueError('Archive directory must not be a symlink')
        self.root.mkdir(exist_ok=True)
        self.hashes, self.verified = {}, {}
        self.last_progress = 0.
        self.progress = dict(status='preparing', started_at=time.time(), pid=os.getpid(),
                             compression_level=COMPRESSION_LEVEL, completed_paths=0,
                             archives_created=0, archives_reused=0)

    def publish(self, operation, path, count=0, *, force=False):
        if not self.progress_enabled:return
        now = time.monotonic()
        if force or now-self.last_progress >= 1:
            self.progress.update(operation=operation, path=str(path), bytes_processed=count,
                                 updated_at=time.time())
            write_json(self.run/'retention-preparation.json', self.progress)
            self.last_progress = now

    def hash(self, path):
        path = Path(path)
        key = identity(path)
        if key in self.hashes:
            return self.hashes[key]
        h, count = hashlib.sha256(), 0
        self.publish('hash', path)
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(BLOCK), b''):
                h.update(block); count += len(block)
                self.publish('hash', path, count)
        if identity(path) != key:
            raise ValueError('Source changed while hashing: '+str(path))
        self.hashes[key] = h.hexdigest()
        return h.hexdigest()

    def verify(self, path, expected):
        key = identity(path)
        if self.verified.get(key) == expected:
            return
        h, count = hashlib.sha256(), 0
        self.publish('verify_archive', path)
        try:
            with gzip.open(path, 'rb') as stream:
                for block in iter(lambda: stream.read(BLOCK), b''):
                    h.update(block); count += len(block)
                    self.publish('verify_archive', path, count)
            if h.hexdigest() != expected or identity(path) != key:
                raise ValueError('Decoded content hash or archive identity differs')
        except (OSError, EOFError, ValueError, zlib.error) as exc:
            raise ValueError('Invalid source archive; preserve the original and use '
                             'python -m maintenance.task48_archive --run ... --repair-source ...: '+str(path)) from exc
        self.verified[key] = expected

    def prepare(self, source, digest):
        """Keep an interrupted temporary file as evidence; never publish it."""
        source = Path(source)
        before = identity(source)
        temporary = self.root/('.'+digest+'.'+uuid.uuid4().hex+'.partial')
        count, h = 0, hashlib.sha256()
        self.publish('compress', source)
        with source.open('rb') as src, temporary.open('xb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw,
                               compresslevel=COMPRESSION_LEVEL, mtime=0) as out:
                for block in iter(lambda: src.read(BLOCK), b''):
                    h.update(block); out.write(block); count += len(block)
                    self.publish('compress', source, count)
            raw.flush(); os.fsync(raw.fileno())
        if identity(source) != before or h.hexdigest() != digest:
            raise ValueError('Source changed while archiving: '+str(source))
        self.verify(temporary, digest)
        return temporary

    def add(self, source):
        digest = self.hash(source)
        saved = self.root/(digest+'.gz')
        if saved.exists() or saved.is_symlink():
            self.verify(saved, digest)
            self.progress['archives_reused'] += 1
        else:
            temporary = self.prepare(source, digest)
            temporary.replace(saved)
            sync_directory(self.root)
            # rename changes ctime; the verified bytes were published by us.
            self.verified[identity(saved)] = digest
            self.progress['archives_created'] += 1
        self.progress['completed_paths'] += 1
        return dict(sha256=digest, archive=str(saved.relative_to(self.run)))

    def verify_index(self, index):
        for info in index.values():
            expected = info['sha256']
            if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
                raise ValueError('Invalid archive content hash')
            if info['archive'] != 'retained-source/'+expected+'.gz':
                raise ValueError('Unexpected content archive path')
            self.verify(self.run/info['archive'], expected)


def locate(path):
    path=Path(path).absolute()
    for root in path.parents:
        index=root/'retained-source/index.json'
        if index.is_file():
            plan=root/'retention/plan.json'
            if plan.exists():
                expected=json.loads(plan.read_text())['source_index_sha256']
                if hashlib.sha256(index.read_bytes()).hexdigest()!=expected:
                    raise ValueError('Retention source index changed')
            info=json.loads(index.read_text()).get(str(path.relative_to(root)))
            if info is not None:
                return root,info
    raise FileNotFoundError(path)


@contextmanager
def open_source(path):
    """Read an original or fully verified archive, preserving existing audits."""
    path=Path(path)
    if path.is_symlink():raise ValueError('Original source must not be a symlink')
    if path.exists():
        with path.open() as stream:yield stream
    else:
        root,info=locate(path)
        Archives(root,progress=False).verify_index({'source':info})
        with gzip.open(root/info['archive'],'rt') as stream:yield stream


def restore(run, relative):
    from maintenance.task48_retention_journal import locked, safe_path
    run=Path(run).resolve()
    with locked(run):
        target=safe_path(run,relative)
        index=json.loads((run/'retained-source/index.json').read_text())
        info=index[relative];archives=Archives(run)
        archives.verify_index({relative:info})
        if target.exists():
            if archives.hash(target)!=info['sha256']:raise ValueError('Existing original differs from archive')
            return dict(status='already_present',source=relative)
        temporary=target.with_name(target.name+'.restore-'+uuid.uuid4().hex)
        target.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(run/info['archive'],'rb') as src, temporary.open('xb') as out:
            for block in iter(lambda:src.read(BLOCK),b''):out.write(block)
            out.flush();os.fsync(out.fileno())
        if archives.hash(temporary)!=info['sha256']:raise ValueError('Restored bytes differ from archive')
        temporary.replace(target);sync_directory(target.parent)
        result=dict(status='restored',source=relative,sha256=info['sha256'],time=time.time())
        write_json(archives.root/'restorations'/(uuid.uuid4().hex+'.json'),result)
        return result


def repair(run, relative):
    """Explicit recovery only; never resumes retention or deletes originals."""
    from maintenance.task48_retention import validate_terminal
    from maintenance.task48_retention_journal import locked, safe_path
    run = Path(run).resolve()
    with locked(run):
        if validate_terminal(run, verify_artifact=not (run/'retention/plan.json').exists()) is not None:
            raise ValueError('Incomplete run cannot repair retention')
        source = safe_path(run, relative)
        archives = Archives(run)
        digest = archives.hash(source)
        saved = archives.root/(digest+'.gz')
        # Missing final files are simply re-created by the next retention retry.
        if not saved.exists():
            raise ValueError('No existing archive to repair')
        try:
            archives.verify(saved, digest)
            return dict(status='already_valid', source=relative, sha256=digest)
        except ValueError:
            pass
        before = identity(saved)
        damaged_hash = archives.hash(saved)
        temporary = archives.prepare(source, digest)
        if identity(saved) != before:
            raise ValueError('Damaged archive changed during repair')
        evidence = archives.root/'recovery'/uuid.uuid4().hex
        evidence.mkdir(parents=True)
        # Hard-link the damaged bytes before atomically replacing the final name.
        os.link(saved, evidence/'damaged.gz')
        report = dict(status='prepared', source=relative, sha256=digest,
                      original_archive_sha256=damaged_hash,
                      damaged_archive=str((evidence/'damaged.gz').relative_to(run)),
                      temporary=str(temporary.relative_to(run)))
        durable_json(evidence/'repair.json', report)
        sync_directory(evidence); sync_directory(evidence.parent)
        temporary.replace(saved); sync_directory(archives.root)
        report.update(status='repaired', finished_at=time.time(), original_preserved=True)
        durable_json(evidence/'repair.json', report)
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=record_path)
    operation=parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--repair-source', help='Original path relative to run')
    operation.add_argument('--restore-source', help='Restore archived bytes at their original path')
    args = parser.parse_args()
    print(json.dumps(repair(args.run,args.repair_source) if args.repair_source else restore(args.run,args.restore_source)))
