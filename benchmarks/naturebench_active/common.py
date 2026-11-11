"""Stable data identities and small I/O primitives; no preparation or model calls."""
import hashlib
import json
from pathlib import Path
import sqlite3

def stable(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(stable(value).encode()).hexdigest()

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def connect(path):
    db=sqlite3.connect(path);db.row_factory=sqlite3.Row;return db

def jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    tmp.replace(path)
