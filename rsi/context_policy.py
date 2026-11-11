"""Versioned Agent display preferences; host evidence and capabilities are separate."""
import json
from pathlib import Path


AGENT_TRANSPORT_BYTES = 8 * 1024 * 1024

DEFAULT_POLICY = json.loads(
    (Path(__file__).resolve().parents[1] / 'agent/shared/context_policy.json').read_text())


def validate_policy(value=None):
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - DEFAULT_POLICY.keys():
        raise ValueError('context_policy.json must be an object containing known context policy keys')
    policy = {**DEFAULT_POLICY, **value}
    for key, default in DEFAULT_POLICY.items():
        item = policy[key]
        if type(default) is bool:
            if type(item) is not bool:
                raise ValueError(f'context_policy.{key} must be boolean')
        else:
            nullable = key in ('tool_display_bytes', 'stdout_preview_chars')
            minimum = 1024 if key == 'tool_display_bytes' else 2 if key == 'stdout_preview_chars' else 1
            if nullable and item is None:
                continue
            if type(item) is not int or item < minimum:
                raise ValueError(f'context_policy.{key} must be an integer >= {minimum}'
                                 + (' or null' if nullable else ''))
    if policy['archive_recent_chars'] > policy['archive_trigger_chars']:
        raise ValueError('context_policy.archive_recent_chars must not exceed archive_trigger_chars')
    return policy


def load_policy(source):
    path = Path(source) / 'context_policy.json'
    if not path.exists():
        return validate_policy()  # Older bundles retain their original defaults.
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('context_policy.json must be a JSON object')
    return validate_policy(value)
