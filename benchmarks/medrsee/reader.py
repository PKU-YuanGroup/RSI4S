"""Public batch reader. This file never accesses the organizer's raw data."""
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
import numpy as np


def volume_resize_plan(shape, dtype, target_shape, intent='downsample'):
    """Validate an explicit 3-D grid before allocation; bytes exclude model activations."""
    import math
    import operator
    def dimensions(value):
        if len(value) != 3 or any(isinstance(n, (bool, np.bool_)) for n in value):
            raise ValueError('Expected three positive integer dimensions')
        result = tuple(operator.index(n) for n in value)
        if any(n < 1 for n in result):
            raise ValueError('Expected three positive integer dimensions')
        return result
    shape, target = dimensions(shape), dimensions(target_shape)
    if intent not in ('downsample', 'resize'):
        raise ValueError('intent must be downsample or resize')
    ratio = math.prod(target) / math.prod(shape)
    if intent == 'downsample' and any(out > old for old, out in zip(shape, target)):
        raise ValueError(f'Downsampling would enlarge {shape} to {target} ({ratio:g}x voxels). '
                         'Divide dimensions by downsampling factors; scipy zoom takes output/input ratios.')
    dtype = np.dtype(dtype)
    if dtype.kind not in 'biufc':
        raise ValueError('Volume arrays must have a numeric dtype')
    return dict(input_shape=list(shape), target_shape=list(target), intent=intent,
                zoom=[out/old for old, out in zip(shape, target)], voxel_ratio=ratio,
                input_bytes=math.prod(shape)*dtype.itemsize,
                output_bytes=math.prod(target)*dtype.itemsize)


def resize_volume(array, target_shape, order=1, intent='downsample', max_output_bytes=None):
    """Resize to an explicit grid; order=0 for masks, order=1 for linear images.

    Exported native arrays/affines stay unchanged. Restore predictions with
    intent='resize'. No default memory cap is imposed.
    """
    plan = volume_resize_plan(array.shape, array.dtype, target_shape, intent)
    if type(order) is not int or order not in (0, 1):
        raise ValueError('Use order=0 for nearest-neighbor masks or order=1 for linear interpolation')
    if max_output_bytes is not None:
        if type(max_output_bytes) is not int or max_output_bytes < 1:
            raise ValueError('max_output_bytes must be a positive integer')
        if plan['output_bytes'] > max_output_bytes:
            raise ValueError('Planned output exceeds the caller-provided max_output_bytes')
    from scipy.ndimage import zoom
    result = zoom(array, plan['zoom'], order=order, prefilter=False)
    if tuple(result.shape) != tuple(plan['target_shape']):
        raise RuntimeError('Resampled grid differs from the explicit target shape')
    return result


def downsample_volume(array, factors, order=1):
    """Factors are divisors: (4,4,2) gives roughly 1/32 as many voxels."""
    if len(factors) != 3 or any(isinstance(f, (bool, np.bool_)) or not np.isfinite(f) or f < 1 for f in factors):
        raise ValueError('Use three finite downsampling divisors >= 1')
    target = tuple(max(1, round(n/float(f))) for n, f in zip(array.shape, factors))
    return resize_volume(array, target, order=order, intent='downsample')


class ImageInputs(Sequence):
    """Load one image at a time; NPY exports are read-only memory maps.

    No sample arrays are retained by this sequence. Copy an array explicitly
    before in-place preprocessing. Legacy NPZ batches remain supported.
    """
    def __init__(self, path, samples):
        self.path, self.samples = Path(path), samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        sample = self.samples[index]
        if 'arrays' in sample:
            return {name: np.load(self.path / filename, mmap_mode='r', allow_pickle=False)
                    for name, filename in sample['arrays'].items()}
        with np.load(self.path / sample['file'], allow_pickle=False) as data:
            return {key: data[key] for key in data.files}


def _samples(path):
    """Refresh membership roles without changing an export's label authorization."""
    path = Path(path)
    samples = json.loads((path/'samples.json').read_text())
    marker = path/'membership.json'
    if marker.exists():
        pool = path/json.loads(marker.read_text())['pool_database']
    else:
        # Compatibility with historical batches in their original data directory.
        pool = path.parent.parent/'pool.sqlite'
        if path.parent.name != 'batches' or not pool.exists():
            return samples  # Standalone exports, including frozen final inputs.
    if not pool.is_file():
        raise ValueError('Research batch membership database is missing; use the original DATA_DIR')
    ids = [sample['sample_id'] for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate sample IDs within a batch')
    db = sqlite3.connect(pool.resolve().as_uri()+'?mode=ro', uri=True)
    try:
        db.execute('BEGIN')
        roles = {}
        for start in range(0, len(ids), 900):
            chunk = ids[start:start+900]
            roles.update(db.execute('SELECT sample_id,role FROM samples WHERE sample_id IN ('
                                    + ','.join('?' for _ in chunk) + ')', chunk))
        if len(roles) != len(ids):
            raise ValueError('Research batch contains samples missing from its membership database')
        for sample in samples:
            sample['role'] = roles[sample['sample_id']]
    finally:
        db.close()
    return samples


def load_batch(path):
    """Return (samples, inputs, labels).

    Images: inputs is a lazy sequence of dicts with input_0/affine_0/etc; authorized masks
    are label_0/etc in those dicts. LUNA annotations are in samples[i].
    EE: inputs[i] is a list of named-column dictionaries (one per native table).
    labels is N×1 or N×24, with NaN for unacquired samples. Check samples[i].labeled.
    Roles reflect the current public pool, even for batches fetched before a
    validation declaration. Labels remain exactly those authorized at export.
    Reload metadata after declaring validation; raw samples.json is a snapshot.
    """
    path = Path(path)
    samples = _samples(path)
    if (path/'labels.npy').exists():
        inputs = [[] for _ in samples]
        for table_path in sorted(path.glob('input_*.npz')):
            with np.load(table_path,allow_pickle=False) as table:
                offsets = table['offsets']
                columns = table['columns']
                arrays = [table[f'column_{j}'] for j in range(len(columns))]
                for i in range(len(samples)):
                    inputs[i].append({str(name):array[offsets[i]:offsets[i+1]]
                                      for name,array in zip(columns,arrays)})
        return samples, inputs, np.load(path/'labels.npy',allow_pickle=False)
    return samples, ImageInputs(path, samples), None


def iter_batch(path, role=None, sample_ids=None):
    """Yield (sample, inputs, label) after filtering metadata, before image reads.

    For images, authorized masks are in inputs and label is None. For tables,
    label is the corresponding row from labels.npy. Record actual fitting membership.
    """
    samples, inputs, labels = load_batch(path)
    wanted = None if sample_ids is None else set(sample_ids)
    for i, sample in enumerate(samples):
        if role is not None and sample['role'] != role:
            continue
        if wanted is not None and sample['sample_id'] not in wanted:
            continue
        yield sample, inputs[i], None if labels is None else labels[i]


def iter_dataset(data_dir, role=None, sample_ids=None, labeled_only=False):
    """Read fetched samples once each, in native ordinal order, using current roles.

    Prefer a labeled export when available, then the latest export. This does
    not fetch data or reveal labels absent from an already authorized export.
    It avoids counting overlapping fetches as additional training evidence.
    """
    data_dir = Path(data_dir)
    wanted = None if sample_ids is None else set(sample_ids)
    selected = {}
    for batch in json.loads((data_dir/'batches.json').read_text()):
        path = Path(batch['path'])
        if not path.is_absolute():
            path = data_dir/path
        for index, sample in enumerate(_samples(path)):
            sid = sample['sample_id']
            if (role is not None and sample['role'] != role
                    or wanted is not None and sid not in wanted
                    or labeled_only and not sample['labeled']):
                continue
            previous = selected.get(sid)
            if previous is not None:
                if previous[0]['ordinal'] != sample['ordinal']:
                    raise ValueError('A sample ID has inconsistent ordinals across batches')
                if previous[0]['labeled'] and not sample['labeled']:
                    continue
            selected[sid] = (sample, path, index)
    loaded_path = None
    for sample, path, index in sorted(selected.values(), key=lambda row: row[0]['ordinal']):
        if path != loaded_path:
            samples, inputs, labels = load_batch(path)
            loaded_path = path
        current = samples[index]
        if role is not None and current['role'] != role:
            continue
        yield current, inputs[index], None if labels is None else labels[index]
