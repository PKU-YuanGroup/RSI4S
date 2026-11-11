from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def record_path(value, *, area=None):
    """Resolve persistent records inside this checkout, rejecting escaped links."""
    if area not in (None, 'runs', 'provenance'):
        raise ValueError('Unknown record area')
    root = ROOT.resolve()
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    for name in ((area,) if area else ('runs', 'provenance')):
        base = root / name
        if base.resolve() == base and path != base and path.is_relative_to(base):
            return path
    expected = f'{root}/{area}/' if area else f'{root}/runs/ or {root}/provenance/'
    raise ValueError(f'Records must stay below {expected}: {value}')


def git_commit():
    """HEAD of this checkout, or None for a source archive or a repository without commits."""
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', '--verify', '--quiet', 'HEAD'], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def encoded(value):
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None  # Official missing/nonfinite score semantics are unchanged.
        if isinstance(item, dict):
            return {k: finite(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [finite(v) for v in item]
        return item
    return json.dumps(finite(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(encoded(value) + '\n')
    tmp.replace(path)


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, kind, **fields):
        record = {'time': time.time(), 'event': kind, **fields}
        with self.path.open('a') as out:
            out.write(encoded(record) + '\n')
            out.flush()
            os.fsync(out.fileno())
        return record


def parse_json(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('Model response must be one JSON object')
    # This endpoint sometimes echoes response_format.type into the content.
    # It is a transport marker, not an agent field; all other unknown fields
    # still reach the consumer's policy or verdict schema validation.
    if value.get('type') == 'json_object':
        value.pop('type')
    return value


def parse_action(text):
    """Some compatible APIs ignore JSON mode. Execute only the FIRST object.

    Never execute prose, queue additional actions, or treat narrated results as
    observations. Meta policies still use strict whole-response JSON parsing.
    """
    decoder = json.JSONDecoder()
    for start in re.finditer(r'\{', text):
        try:
            value, _ = decoder.raw_decode(text[start.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and 'tool' in value:
            return value
    raise ValueError('No JSON action found; return one object with tool and its arguments')
