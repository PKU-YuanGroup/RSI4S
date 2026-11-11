"""Private, lossless native-volume cache; only authorized arrays are exported.

Inputs and labels use separate entries. Cache paths are never granted to Agent
sandboxes. Published batches contain hard links to individual permitted NPY
arrays, which can be memory-mapped without decompressing an entire cohort.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from .native_reader import original_id

VOLUME_TASKS = {'MED_01', 'MED_03', 'MED_04', 'MED_07', 'MED_06'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class VolumeCache:
    def __init__(self, root, reader):
        self.root, self.reader = Path(root), reader
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        here = Path(__file__).parent
        self.schema = digest({name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                              for name in ('cache.py', 'images.py', 'native_reader.py')})

    def _key(self, row, labels):
        kind = 'label' if labels else 'input'
        paths = row[kind + '_path']
        selectors = row[kind + '_selector']
        signatures = []
        for name, selector in zip(paths, selectors):
            path = self.reader.path(name)
            if selector['kind'] == 'directory':
                path = path / selector['member']
            st = path.stat()
            signatures.append((str(path), st.st_size, st.st_mtime_ns))
        return digest(dict(schema=self.schema, task=original_id(row['task_id']), sample_id=row['sample_id'],
                           paths=signatures, selectors=selectors, labels=labels))

    @contextmanager
    def _entry(self, row, labels, deadline):
        key = self._key(row, labels)
        parent = self.root / self.schema / ('labels' if labels else 'inputs') / key[:2]
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        entry = parent / key
        with (parent / (key + '.lock')).open('a') as lock:
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Evidence cache deadline reached')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(.05)
            try:
                yield entry
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def get(self, row, labels=False, deadline=float('inf')):
        from .images import decode, nifti
        if row['task_id'] not in VOLUME_TASKS:
            raise ValueError('Volume cache requires a native 3-D image task')
        # Labels may only be requested by the host after authorization. Geometry
        # validation uses the input cache and never reveals a hidden label.
        geometry = self.get(row, deadline=deadline)[0] if labels else None
        with self._entry(row, labels, deadline) as entry:
            manifest = entry / 'manifest.json'
            if manifest.is_file():
                return json.loads(manifest.read_text()), entry, True
            temporary = Path(tempfile.mkdtemp(prefix='.building-', dir=entry.parent))
            try:
                arrays, metadata = {}, {}
                if labels:
                    native = [self.reader.load(p, s) for p, s in
                              zip(row['label_path'], row['label_selector'])]
                    if row['task_id'] == 'MED_04':
                        metadata['annotations'] = native
                    else:
                        if len(native) != len(geometry['parts']):
                            raise ValueError('Native image/label part count differs')
                        for i, raw in enumerate(native):
                            y, affine = nifti(raw)
                            part = geometry['parts'][i]
                            if list(y.shape) != part['shape'] or not np.allclose(affine, part['affine'], atol=1e-4):
                                raise ValueError('Native image/label geometry differs')
                            arrays[f'label_{i}'] = y.astype(np.int16)
                else:
                    parts = decode(row, self.reader, labels=False)
                    metadata['parts'] = []
                    for i, part in enumerate(parts):
                        arrays[f'input_{i}'] = part['x']
                        arrays[f'affine_{i}'] = part['affine']
                        metadata['parts'].append(dict(shape=list(part['x'].shape),
                                                      affine=part['affine'].tolist()))
                metadata['arrays'] = {}
                for name, array in arrays.items():
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Evidence cache deadline reached')
                    filename = name + '.npy'
                    np.save(temporary / filename, array, allow_pickle=False)
                    (temporary / filename).chmod(0o444)
                    metadata['arrays'][name] = filename
                (temporary / 'manifest.json').write_text(json.dumps(metadata))
                (temporary / 'manifest.json').chmod(0o444)
                temporary.rename(entry)
                return metadata, entry, False
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)

    def export(self, row, destination, number, labels=False, deadline=float('inf')):
        destination = Path(destination)
        output, annotations, hits = {}, None, 0
        for permitted_labels in ([False, True] if labels else [False]):
            meta, entry, hit = self.get(row, labels=permitted_labels, deadline=deadline)
            hits += int(hit)
            for name, filename in meta['arrays'].items():
                relative = f'{number:06d}-{name}.npy'
                source, target = entry / filename, destination / relative
                try:
                    os.link(source, target)
                except OSError as exc:
                    import errno
                    if exc.errno != errno.EXDEV:
                        raise
                    shutil.copyfile(source, target)
                output[name] = relative
            if 'annotations' in meta:
                annotations = meta['annotations']
        return output, annotations, hits
