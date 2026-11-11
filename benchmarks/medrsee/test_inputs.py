"""Host-private, prepared ENV_02 Test inputs. Never cache or export Test labels."""
from contextlib import closing, contextmanager
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from .membership import connect, native_row
from .source import COUNTS, NativeReader


def resolve_cache_root(data_root, configured=None):
    """Relative cache paths follow the dataset when it is moved or downloaded."""
    path = Path(configured or 'datasets/external/EE-02/prepared-test-inputs')
    return path if path.is_absolute() else Path(data_root) / path


def sha(path, deadline=float('inf')):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('Prepared Test input deadline exhausted')
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                return value.hexdigest()
            value.update(chunk)


class PreparedTestInputs:
    """Preparation is explicit and offline; consuming a missing cache fails closed."""
    def __init__(self, root, task, index, data_root):
        if task != 'ENV_02':
            raise ValueError('Prepared tabular Test inputs currently support ENV_02 only')
        if Path(root).is_symlink():
            raise ValueError('Test input cache must be a plain directory')
        self.root, self.task = Path(root).resolve(), task
        self.index, self.data_root = Path(index).resolve(), Path(data_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        here = Path(__file__).parent
        # Bind portable relative paths and contents, not machine-local timestamps.
        # No label array is read when computing or checking this descriptor.
        self.index_state = self.source_state(self.index)
        self.input_states = {}
        with closing(connect(self.index)) as db, NativeReader(self.data_root) as reader:
            sources = {}
            members = db.execute("SELECT * FROM members WHERE split='T' GROUP BY source_id")
            for member in members:
                row = native_row(task, dict(member))
                for name in row['input_path']:
                    path = reader.path(name)
                    relative = path.relative_to(self.data_root).as_posix()
                    if relative not in sources:
                        state = self.source_state(path)
                        self.input_states[str(path)] = state
                        sources[relative] = dict(bytes=state[0], sha256=sha(path))
            count = db.execute("SELECT count(*) FROM members WHERE split='T'").fetchone()[0]
        if count != COUNTS[task][4]:
            raise ValueError('Unexpected Test membership count')
        self.identity = dict(format=2, task=task, count=count, index_sha256=sha(self.index),
            input_sources=sources, exporter={name: sha(here / name) for name in
            ('evidence.py', 'tabular.py', 'source.py', 'membership.py',
             'native_reader.py', 'source_templates.json', 'test_inputs.py')})
        self.check_sources()
        self.key = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.entry = self.root / self.key

    @staticmethod
    def source_state(path):
        st = Path(path).stat()
        return [st.st_size, st.st_mtime_ns, st.st_ctime_ns]

    def check_sources(self):
        if self.source_state(self.index) != self.index_state or any(
                self.source_state(path) != state for path, state in self.input_states.items()):
            raise ValueError('Native input sources changed after cache selection')

    @contextmanager
    def lock(self):
        with (self.root / (self.key + '.lock')).open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def verify(self, deadline=float('inf')):
        self.check_sources()
        if self.entry.is_symlink() or (self.entry/'data').is_symlink():
            raise ValueError('Prepared Test inputs must be a plain directory')
        manifest = self.entry / 'manifest.json'
        if not manifest.is_file():
            raise FileNotFoundError('ENV_02 Test inputs are not prepared for these data/code. '
                'Run python -m maintenance.medrsee.prepare_test_inputs before launching research.')
        value = json.loads(manifest.read_text())
        if value['identity'] != self.identity or value['labels_included'] is not False:
            raise ValueError('Prepared Test input identity mismatch')
        if sum(b['count'] for b in value['batches']) != self.identity['count']:
            raise ValueError('Incomplete prepared Test inputs')
        if [b['directory'] for b in value['batches']] != [f'{i:06d}' for i in range(len(value['batches']))]:
            raise ValueError('Invalid prepared batch order')
        for relative, expected in value['files'].items():
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts or len(path.parts) != 2:
                raise ValueError('Invalid prepared input path')
            source = self.entry / 'data' / path
            if (source.is_symlink() or source.parent.is_symlink() or not source.is_file()
                    or source.stat().st_size != expected['bytes']
                    or sha(source, deadline) != expected['sha256']):
                raise ValueError('Prepared Test input integrity failure: ' + relative)
        return value

    def prepare(self, progress=None):
        """Read exact T input selectors once; publish only a complete checked cache."""
        from .evidence import export_batch
        with self.lock():
            if self.entry.exists():
                return dict(cache_hit=True, key=self.key, **self.verify())
            temporary = Path(tempfile.mkdtemp(prefix='.building-', dir=self.root))
            try:
                batches, files, chunk, completed = [], {}, [], 0
                with closing(connect(self.index)) as db, NativeReader(self.data_root) as reader:
                    def flush():
                        nonlocal completed
                        name = f'{len(batches):06d}'
                        target = temporary / 'data' / name
                        result = export_batch(self.task, chunk, reader, target, labels=False)
                        if result['labeled'] != 0:
                            raise ValueError('Test inputs must never contain labels')
                        batches.append(dict(directory=name, count=len(chunk)))
                        for path in sorted(target.iterdir()):
                            path.chmod(0o444)
                            files[name + '/' + path.name] = dict(bytes=path.stat().st_size, sha256=sha(path))
                        completed += len(chunk)
                        if progress:
                            progress(dict(completed=completed, total=self.identity['count'], batches=len(batches)))
                    for member in db.execute("SELECT * FROM members WHERE split='T' ORDER BY ordinal"):
                        chunk.append(dict(member))
                        if len(chunk) == 512:
                            flush()
                            chunk.clear()
                    if chunk:
                        flush()
                if completed != self.identity['count']:
                    raise ValueError('Incomplete Test input preparation')
                self.check_sources()
                value = dict(identity=self.identity, labels_included=False, batches=batches, files=files)
                (temporary / 'manifest.json').write_text(json.dumps(value, sort_keys=True) + '\n')
                (temporary / 'manifest.json').chmod(0o400)
                temporary.rename(self.entry)
                return dict(cache_hit=False, key=self.key, **self.verify())
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)

    def materialize(self, destination, deadline=float('inf')):
        """Grant a fresh final-only view, never the private cache directory itself."""
        value = self.verify(deadline)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        batches = []
        for batch in value['batches']:
            path = destination / batch['directory']
            path.mkdir()
            batches.append(dict(path=str(path), count=batch['count'], order='fixed ordinal', labeled=0))
        for relative in value['files']:
            if time.monotonic() >= deadline:
                raise TimeoutError('Prepared Test input deadline exhausted')
            source, target = self.entry / 'data' / relative, destination / relative
            try:
                os.link(source, target)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                shutil.copyfile(source, target)
            target.chmod(0o444)
        return batches
