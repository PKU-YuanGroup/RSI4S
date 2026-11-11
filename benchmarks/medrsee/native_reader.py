"""Read the compact fixed split CSV against native data. Python 3.10+, numpy/pandas/h5py.

This is an organizer data reader, not an agent access-control service.
Images/NIfTI/MHD are returned as original bytes; arrays and tables are decoded.
"""
import csv
import io
import json
import shutil
import tarfile
import zipfile
from collections import OrderedDict
from pathlib import Path
import sys

try:
    from ..task_names import original_id, public_name
except ImportError:  # Isolated scorer and direct CLI use have no benchmarks package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from task_names import original_id, public_name


PROFILES = {
    'gloria_rrs': [f'Rrs_{nm}' for nm in range(350, 901)],
    'camels_weather': ['date', 'precipitation', 'pet', 'temperature', 'peti',
                       'humidity', 'shortwave_rad', 'longwave_rad', 'windspeed'],
    'camels_static': ['gauge_lat', 'gauge_lon', 'gauge_elev', 'area', 'dpsbar', 'elev_mean'],
    'gpp_weather': ['Unnamed: 0', 'Unnamed: 1', 'SW_IN_F_MDS', 'TA_F_MDS',
                    'VPD_F_MDS', 'swvl1', 'swvl2', 'swvl3', 'swvl4'],
    'gpp_static': ['towerlat', 'towerlon', 'alt', 'IGBP_Code'],
}


# This fragment is incorporated into the delivered data_reader.py.
SOURCE_TEMPLATES = json.loads(Path(__file__).with_name('source_templates.json').read_text())


# MED_02: fixed custom1000_251_v1 members, original archive byte offsets.
SOURCE_TEMPLATES.append({'origin_split': 'BraTS2021_official_training', 'input_path': ['datasets/external/MED-04/raw/BraTS2021_Training_Data.tar', 'datasets/external/MED-04/raw/BraTS2021_Training_Data.tar', 'datasets/external/MED-04/raw/BraTS2021_Training_Data.tar', 'datasets/external/MED-04/raw/BraTS2021_Training_Data.tar'], 'input_selector': [{'kind': 'tar_offset', 'span': {'$': 0}}, {'kind': 'tar_offset', 'span': {'$': 1}}, {'kind': 'tar_offset', 'span': {'$': 2}}, {'kind': 'tar_offset', 'span': {'$': 3}}], 'label_path': ['datasets/external/MED-04/raw/BraTS2021_Training_Data.tar'], 'label_selector': [{'kind': 'tar_offset', 'span': {'$': 4}}]})

def _fill(value, slots):
    if isinstance(value, dict):
        if set(value) == {'$'}:
            return slots[value['$']]
        return {k: _fill(v, slots) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, slots) for v in value]
    return value


def iter_samples(csv_path=None, task=None, split=None):
    """Expand fixed membership lazily; no seed, data scan or label access.

    Ordinary records locate one sample. Daily records use run lengths over native
    row numbers: D=D0, Q=Q, V=V, T=T, X=not included. Counts default to one.
    Example Q3DX2T2 means Q,Q,Q,D0,X,X,T,T at successive native row numbers.
    """
    import re
    from datetime import date, timedelta
    csv.field_size_limit(16 * 1024 * 1024)
    csv_path = Path(csv_path) if csv_path else Path(__file__).with_name('splits.csv')
    wanted = {split} if isinstance(split, str) else set(split or [])
    with csv_path.open(encoding='utf-8-sig', newline='') as f:
        for record in csv.DictReader(f):
            # splits.csv keeps the original IDs; everything above it uses public names.
            if task and public_name(record['task_id']) != task:
                continue
            template = SOURCE_TEMPLATES[int(record['source_id'])]
            if record['split'] != '@':
                if wanted and record['split'] not in wanted:
                    continue
                refs = _fill(template, json.loads(record['locator']))
                yield {**refs, 'task_id': public_name(record['task_id']), **{k: record[k] for k in
                       ['sample_id', 'group_id', 'split']}}
                continue
            start, runs, dates = json.loads(record['locator'])
            anchors = [(int(i), date.fromisoformat(d)) for i, d in dates]
            anchor_index = 0
            cursor = start
            for match in re.finditer(r'([DQVTX])(\d*)', runs):
                code, length = match.groups()
                count = int(length) if length else 1
                role = 'D0' if code == 'D' else code
                if code == 'X' or (wanted and role not in wanted):
                    cursor += count
                    continue
                for i in range(cursor, cursor + count):
                    while anchor_index + 1 < len(anchors) and anchors[anchor_index + 1][0] <= i:
                        anchor_index += 1
                    origin_row, origin_date = anchors[anchor_index]
                    day = (origin_date + timedelta(days=i-origin_row)).isoformat()
                    x = [dict(s) for s in template['input_selector']]
                    y = [dict(s) for s in template['label_selector']]
                    if public_name(record['task_id']) == 'ENV_02':
                        x[0]['slice'] = [i-365, i+1]
                    else:
                        x[0]['row'] = i
                    y[0]['row'] = i
                    yield dict(task_id=public_name(record['task_id']),
                               sample_id=record['sample_id']+':'+day,
                               group_id=record['group_id'], split=role,
                               origin_split=template['origin_split'],
                               input_path=template['input_path'], input_selector=x,
                               label_path=template['label_path'], label_selector=y)
                cursor += count


class NativeReader:
    def __init__(self, root='/mnt/workspace/RSI4Science'):
        self.root = Path(root).resolve()
        self.handles = OrderedDict()
        self.tables = OrderedDict()

    def path(self, relative):
        p = (self.root / relative).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError('Path is outside data root')
        return p

    def table(self, path, options):
        import pandas as pd
        key = (path, json.dumps(options, sort_keys=True))
        if key not in self.tables:
            self.tables[key] = pd.read_csv(self.path(path), **options)
            if len(self.tables) > 6:
                self.tables.popitem(last=False)
        self.tables.move_to_end(key)
        return self.tables[key]

    def handle(self, path, kind):
        import numpy as np
        key = (path, kind)
        if key not in self.handles:
            p = self.path(path)
            if kind == 'zip':
                h = zipfile.ZipFile(p)
            elif kind == 'tar':
                h = tarfile.open(p, 'r:')
            elif kind == 'npy':
                h = np.load(p, mmap_mode='r', allow_pickle=False)
            elif kind == 'h5':
                import h5py
                h = h5py.File(p, 'r')
            elif kind == 'csv':
                with p.open(encoding='utf-8-sig', newline='') as f:
                    h = list(csv.DictReader(f))
            else:
                raise ValueError(kind)
            self.handles[key] = h
            if len(self.handles) > 12:
                _, old = self.handles.popitem(last=False)
                if hasattr(old, 'close'):
                    old.close()
        self.handles.move_to_end(key)
        return self.handles[key]

    def load(self, path, selector):
        kind = selector['kind']
        if kind == 'tar_offset':
            offset, size = selector['span']
            if type(offset) is not int or type(size) is not int or offset < 0 or size <= 0:
                raise ValueError('Invalid native archive span')
            with self.path(path).open('rb') as stream:
                stream.seek(offset)
                value = stream.read(size)
            if len(value) != size:
                raise ValueError('Incomplete native archive member')
            return value
        if kind == 'table':
            import pandas as pd
            df = self.table(path, selector.get('read_csv', {}))
            cols = (PROFILES[selector['profile']] if 'profile' in selector
                    else selector['columns'])
            if 'row' in selector:
                part = df.iloc[[selector['row']]]
            elif 'rows' in selector:
                part = df.iloc[selector['rows']]
            else:
                part = df.iloc[slice(*selector['slice'])]
            value = part.loc[:, cols].reset_index(drop=True).copy()
            if selector.get('nonnegative_only'):
                value = value.where(value >= 0)
            if selector.get('reduce') == 'max':
                value = pd.DataFrame([value.max(axis=0)])
            if 'scale' in selector:
                value *= float(selector['scale'])
            if selector.get('relative_time'):
                value['time/s'] -= value['time/s'].iloc[0]
            for name, constant in selector.get('constant_columns', {}).items():
                value[name] = constant
            return value
        if kind == 'directory':
            p = self.path(str(Path(path) / selector['member']))
            if 'key' in selector:
                import h5py
                with h5py.File(p, 'r') as f:
                    value = f[selector['key']][:]
            else:
                value = p.read_bytes()
        else:
            h = self.handle(path, kind)
            if kind == 'zip':
                value = h.read(selector['member'])
            elif kind == 'tar':
                f = h.extractfile(selector['member'])
                if f is None:
                    raise ValueError('Not a regular archive member')
                with f:
                    value = f.read()
            elif kind == 'npy':
                import numpy as np
                value = np.asarray(h[selector['row']])
            elif kind == 'h5':
                value = ({k: h[k][selector['row']] for k in selector['keys']}
                         if 'keys' in selector else h[selector['key']][selector['row']])
            elif kind == 'csv':
                value = h
        if 'where' in selector:
            if isinstance(value, bytes):
                value = list(csv.DictReader(io.StringIO(value.decode('utf-8-sig'))))
            value = [r for r in value if all(r[k] == v for k, v in selector['where'].items())]
        return value

    def read(self, row, labels=False):
        """Return {inputs: [...], labels: [...]} in selector order.

        labels=False never reads label selectors. labels=True is trusted use only.
        Group IDs constrain split assignment; they never grant extra labels.
        """
        result = {}
        for prefix, name in [('input', 'inputs'), ('label', 'labels')]:
            if prefix == 'label' and not labels:
                continue
            paths = row[prefix + '_path']
            selectors = row[prefix + '_selector']
            if len(paths) != len(selectors):
                raise ValueError('Path/selector count mismatch')
            result[name] = [self.load(p, s) for p, s in zip(paths, selectors)]
        return result

    def close(self):
        for h in self.handles.values():
            if hasattr(h, 'close'):
                h.close()
        self.handles.clear()
        self.tables.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def prepare(root='/mnt/workspace/RSI4Science', tasks=None):
    """Extract the few native files required from downloaded original ZIPs.

    No network requests, split regeneration or preprocessing of numeric values.
    Existing output files are retained. Other tasks read their archives directly.
    """
    root = Path(root).resolve()
    tasks = set(tasks or ['ENV_01', 'ENV_03', 'ENV_05', 'MED_05'])

    def extract(archive, dest, members=None):
        dest = dest.resolve()
        if not dest.is_relative_to(root):
            raise ValueError('Destination outside data root')
        with zipfile.ZipFile(archive) as z:
            for name, output in (members or [(n, n) for n in z.namelist()]):
                target = (dest / output).resolve()
                if not target.is_relative_to(dest):
                    raise ValueError('Archive member outside destination')
                info = z.getinfo(name)
                if info.is_dir() or target.exists():
                    continue
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Archive symlink is not supported')
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst)

    for task in sorted(tasks):
        d = root / 'datasets/external' / original_id(task)  # Data on disk keeps original names.
        if task == 'ENV_01':
            extract(d / 'raw/GLORIA-2022.zip', d / 'native')
        elif task == 'ENV_03':
            extract(d / 'raw/GEFCom2014.zip', d / 'native')
            w = d / 'native/GEFCom2014 Data'
            extract(w / 'GEFCom2014-W_V2.zip', w)
            for name in ['Task15_W_Zone1_10.zip', 'TaskExpVars15_W_Zone1_10.zip']:
                extract(w / 'Wind/Task 15' / name, w / 'Wind/Task 15')
        elif task == 'ENV_05':
            for name in ['Dataset_1_NCA_battery.zip', 'Dataset_2_NCM_battery.zip', 'Dataset_3_NCM_NCA_battery.zip']:
                extract(d / 'raw' / name, d / 'native')
        elif task == 'MED_05':
            extract(d / 'Kaggle_public_mirror/CoNIC.zip',
                    root / 'data_aligned/task73_v1/rebuilds_v5/MED-11.source',
                    [('data/images.npy', 'images.npy'), ('data/labels.npy', 'labels.npy')])
        print(task, 'native files ready', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/mnt/workspace/RSI4Science')
    parser.add_argument('--csv', default=str(Path(__file__).with_name('splits.csv')))
    parser.add_argument('--task')
    parser.add_argument('--split', choices=['D0', 'Q', 'V', 'T'])
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root, [args.task] if args.task else None)
    else:
        row = next(iter_samples(args.csv, args.task, args.split))
        with NativeReader(args.root) as reader:
            x = reader.read(row)
        print(row['task_id'], row['sample_id'], row['split'])
        print('inputs:', [(type(v).__name__, getattr(v, 'shape', None)) for v in x['inputs']])
