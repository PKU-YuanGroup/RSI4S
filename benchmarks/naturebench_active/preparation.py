"""Incremental trusted input I/O under the existing research deadline.

Checks run between SQLite instructions, records and file blocks. JSON bytes and
the historical scientific digest encoding stay unchanged; no phase gets a quota.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time


def check(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('Input preparation exceeded the shared deadline')


@contextmanager
def deadline_connection(db, deadline):
    check(deadline)
    db.set_progress_handler(lambda: int(deadline is not None and time.monotonic()>=deadline),1000)
    try:
        yield db
    except sqlite3.OperationalError:
        check(deadline)
        raise
    finally:
        db.set_progress_handler(None,0)
    check(deadline)


@contextmanager
def database(path, deadline=None):
    check(deadline)
    remaining = 5 if deadline is None else max(0, min(5, deadline-time.monotonic()))
    db = sqlite3.connect(path, timeout=remaining)
    db.row_factory = sqlite3.Row
    try:
        with deadline_connection(db,deadline):yield db
    finally:
        db.close()


def file_hash(path, deadline=None):
    h = hashlib.sha256()
    check(deadline)
    with Path(path).open('rb') as stream:
        while True:
            check(deadline)
            block = stream.read(4*1024*1024)
            if not block: break
            h.update(block)
    check(deadline)
    return h.hexdigest()


def finite(value, deadline, memo):
    check(deadline)
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, (dict, tuple, list)):
        if id(value) in memo: return memo[id(value)]
        result = ({k: finite(v, deadline, memo) for k, v in value.items()} if isinstance(value, dict)
                  else [finite(v, deadline, memo) for v in value])
        memo[id(value)] = result
        return result
    return value


def digest(value, deadline=None):
    """Byte-identical to rsi.common.digest without one giant encoded string.

    Canonical dictionaries use string keys in scientific signatures. Each row
    uses the C encoder; shared descriptors are normalized only once.
    """
    value = finite(value, deadline, {})
    h = hashlib.sha256()
    def emit(item, depth=0):
        check(deadline)
        if depth >= 2 or not isinstance(item, (dict, list)):
            data = json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
            for start in range(0, len(data), 4*1024*1024):
                check(deadline); h.update(data[start:start+4*1024*1024])
        elif isinstance(item, dict):
            h.update(b'{')
            for i, key in enumerate(sorted(item)):
                if i: h.update(b', ')
                h.update(json.dumps(key, ensure_ascii=False).encode()); h.update(b': ')
                emit(item[key], depth+1)
            h.update(b'}')
        else:
            h.update(b'[')
            for i, row in enumerate(item):
                if i: h.update(b', ')
                emit(row, depth+1)
            h.update(b']')
    emit(value)
    check(deadline)
    return h.hexdigest()


def copy_file(source, target, deadline=None):
    check(deadline)
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name+'.partial')
    try:
        with Path(source).open('rb') as src, partial.open('xb') as out:
            while True:
                check(deadline)
                block = src.read(4*1024*1024)
                if not block: break
                out.write(block)
        check(deadline)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def jsonl(path, rows, deadline=None):
    check(deadline)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.tmp')
    try:
        with partial.open('w') as stream:
            for row in rows:
                check(deadline)
                line = json.dumps(row, allow_nan=False)+'\n'
                for start in range(0, len(line), 1024*1024):
                    check(deadline); stream.write(line[start:start+1024*1024])
        check(deadline)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
