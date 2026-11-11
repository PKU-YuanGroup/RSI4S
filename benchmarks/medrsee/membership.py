"""Compact organizer index. No labels are loaded while indexing membership."""
from __future__ import annotations

import csv
from contextlib import closing
from datetime import date, timedelta
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .source import WORK, COUNTS, SOURCE_TEMPLATES, NativeReader
from .native_reader import _fill, public_name


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    return db


def fingerprint():
    h = hashlib.sha256()
    for path in [Path(__file__), WORK/'splits.csv', WORK/'native_reader.py',
                 WORK/'source_templates.json']:
        h.update(path.read_bytes())
    return h.hexdigest()


def build_index(task, destination):
    """Serialize builders of one index and reuse its atomic publication."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock file: unlinking it could let waiters lock different inodes.
    with destination.with_suffix(destination.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # A previous builder may have finished while this process was waiting.
        return _locked_index(task, destination)


def _locked_index(task, destination):
    source_sha256 = fingerprint()
    if destination.exists():
        with closing(connect(destination)) as db:
            meta = dict(db.execute('SELECT key,value FROM meta'))
        if meta != {'task': task, 'source_sha256': source_sha256}:
            raise ValueError('Stale membership index; use a new destination')
        return destination
    temporary = destination.with_suffix('.building.sqlite')
    if temporary.exists():
        raise ValueError(f'Incomplete index exists: {temporary}; inspect it before rebuilding')
    # An abandoned partial is evidence of a real failure. Do not erase it or
    # mistake it for an active builder now that we hold the exclusive lock.
    with closing(connect(temporary)) as db:
        _write_index(task, db, source_sha256)
    temporary.chmod(0o644)
    temporary.rename(destination)
    return destination


def _write_index(task, db, source_sha256):
    """Preserve native CSV expansion order, including daily run-length records."""
    db.executescript('''
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE members (ordinal INTEGER PRIMARY KEY, sample_id TEXT UNIQUE NOT NULL,
          group_id TEXT NOT NULL, split TEXT NOT NULL, source_id INTEGER NOT NULL, locator TEXT NOT NULL);
    ''')
    csv.field_size_limit(16 * 1024 * 1024)
    counts = dict(D0=0, Q=0, V=0, T=0)
    batch = []
    ordinal = 0
    with (WORK/'splits.csv').open(encoding='utf-8-sig', newline='') as stream:
        for record in csv.DictReader(stream):
            if public_name(record['task_id']) != task:
                continue
            if record['split'] != '@':
                expanded = [(record['sample_id'], record['split'], record['locator'])]
            else:
                def daily():
                    cursor, runs, anchors = json.loads(record['locator'])
                    anchors = [(int(i), date.fromisoformat(d)) for i, d in anchors]
                    anchor = 0
                    for code, size in re.findall(r'([DQVTX])(\d*)', runs):
                        length = int(size or 1)
                        if code != 'X':
                            for row in range(cursor, cursor + length):
                                while anchor + 1 < len(anchors) and anchors[anchor+1][0] <= row:
                                    anchor += 1
                                offset, day = anchors[anchor]
                                sid = record['sample_id'] + ':' + (day + timedelta(days=row-offset)).isoformat()
                                yield sid, 'D0' if code == 'D' else code, json.dumps(row)
                        cursor += length
                expanded = daily()
            for sid, split, locator in expanded:
                batch.append((ordinal, sid, record['group_id'], split, int(record['source_id']), locator))
                counts[split] += 1
                ordinal += 1
                if len(batch) >= 10000:
                    db.executemany('INSERT INTO members VALUES (?,?,?,?,?,?)', batch)
                    batch.clear()
    db.executemany('INSERT INTO members VALUES (?,?,?,?,?,?)', batch)
    n, d0, _, v, t = COUNTS[task]
    if counts != dict(D0=d0, Q=n-d0, V=v, T=t):
        raise ValueError(f'Unexpected split counts: {counts}')
    db.execute('CREATE INDEX member_split ON members(split, ordinal)')
    db.executemany('INSERT INTO meta VALUES (?,?)', [('task',task), ('source_sha256',source_sha256)])
    db.commit()


def native_row(task, member):
    template = SOURCE_TEMPLATES[member['source_id']]
    locator = json.loads(member['locator'])
    if isinstance(locator, int):
        refs = {**template, 'input_selector': [dict(s) for s in template['input_selector']],
                'label_selector': [dict(s) for s in template['label_selector']]}
        if task == 'ENV_02':
            refs['input_selector'][0]['slice'] = [locator-365, locator+1]
        else:
            refs['input_selector'][0]['row'] = locator
        refs['label_selector'][0]['row'] = locator
    else:
        refs = _fill(template, locator)
    return {**refs, 'task_id':task, **{k: member[k] for k in ('sample_id','group_id','split')}}


def spatial_groups(index, data_root):
    """GPP station connected components at 10 km, not just station IDs."""
    import numpy as np
    db = connect(index)
    members = db.execute('SELECT * FROM members GROUP BY group_id').fetchall()
    db.close()
    positions = []
    with NativeReader(data_root) as reader:
        for member in members:
            row = native_row('ENV_04', member)
            table = reader.load(row['input_path'][1], row['input_selector'][1])
            positions.append(table[['towerlat','towerlon']].to_numpy(float)[0])
    coords = np.radians(positions)
    parent = list(range(len(members)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, (lat, lon) in enumerate(coords):
        distance = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(
            np.sin((coords[:,0]-lat)/2)**2 + np.cos(lat)*np.cos(coords[:,0])*np.sin((coords[:,1]-lon)/2)**2,0,1)))
        for j in np.flatnonzero(distance <= 10):
            parent[root(int(j))] = root(i)
    groups = {}
    for i, member in enumerate(members):
        groups.setdefault(root(i), []).append(member['group_id'])
    return {station: min(stations) for stations in groups.values() for station in stations}
