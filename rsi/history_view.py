"""Deterministic, bounded views of immutable evidence (never candidate ranking)."""
from rsi.common import encoded

DISPLAY_BYTES = 32768
PAGE_BYTES = 24576
FACT_BYTES = 8192


def json_bytes(value):
    return len(encoded(value).encode('utf-8'))


def text_prefix(text, budget=PAGE_BYTES):
    """Bound the JSON-escaped UTF-8 representation; offsets still count characters."""
    if json_bytes(text) <= budget:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if json_bytes(text[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def text_page(text, offset=0, limit=8000):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 50000:
        raise ValueError('Use a nonnegative character offset and limit in 1..50000')
    shown = text_prefix(text[offset:offset + limit])
    end = offset + len(shown)
    return dict(text=shown, offset=offset, total_chars=len(text),
                next_offset=end if end < len(text) else None,
                offset_unit='characters', page_byte_limit=PAGE_BYTES)


def object_view(value, budget, depth=2):
    """Keep exact small values/scalars in source order; flag every incomplete view."""
    if json_bytes(value) <= budget:
        return value, False
    if isinstance(value, str):
        return text_prefix(value, budget), True
    result = {}
    if isinstance(value, dict) and depth:
        for key, item in value.items():
            if isinstance(item, dict) and depth > 1:
                item, _ = object_view(item, min(budget // 2, 2048), depth - 1)
            elif isinstance(item, (dict, list)):
                continue
            if json_bytes({**result, key: item}) <= budget:
                result[key] = item
    return ([] if isinstance(value, list) else result), True


def evaluation_view(evaluation):
    """Exact headline scores/cohort and audit flags; large tables stay in records."""
    result, omitted = {}, list(evaluation.get('details_omitted', []))
    for key in ('attempt', 'artifact', 'snapshot', 'call_id', 'aggregate_improvement',
                'selection_metric', 'aggregate_scope', 'metric_comparison', 'generation_validation',
                'best_aggregate_improvement', 'best_attempt', 'valid_primary_score',
                'task_version', 'meta_version', 'validation_cohort', 'validation_coverage', 'raw_scores',
                'error', 'per_instance_improvement', 'inference_verification', 'training_audit'):
        if key not in evaluation:
            continue
        value = evaluation[key]
        budget = 2048 if key == 'raw_scores' else 1024
        if key == 'training_audit' and isinstance(value, dict):
            value = {k: v for k, v in value.items() if k in (
                'status', 'validation_overlap', 'verification_level',
                'actual_fitting_membership_verified', 'in_memory_fitting_excluded')}
        if key == 'inference_verification' and isinstance(value, dict):
            value = {k: v for k, v in value.items() if k in (
                'status', 'filesystem_policy', 'in_memory_fitting_excluded', 'source_files_sha256')}
        if key == 'generation_validation' and isinstance(value, dict):
            value = dict(status=value.get('status'), probe_records=len(value.get('records', [])),
                         warnings=[text_prefix(v, 240) for v in value.get('warnings', [])[:3]],
                         observed_exception_types=sorted({v['exception'] for v in value.get('observed_model_exceptions', [])}),
                         test_inputs_read=value.get('test_inputs_read'), test_truth_read=value.get('test_truth_read'))
        value, clipped = object_view(value, budget)
        if clipped or value != evaluation[key]:
            omitted.append(key)
        if json_bytes({**result, key: value}) <= 6000:
            result[key] = value
        else:
            omitted.append(key)
    if omitted:
        result['details_omitted'] = sorted(set(omitted))
    return result


def row_page(rows, offset, limit, budget=PAGE_BYTES):
    """A prefix, never a ranked/filtered subset. Caller provides bounded rows."""
    result, used = [], 2
    for row in rows:
        cost = json_bytes(row) + 2
        if used + cost > budget or len(result) >= limit:
            break
        result.append(row)
        used += cost
    return result, offset + len(result)


def manifest_view(info, offset=0, limit=20):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 50000:
        raise ValueError('Invalid manifest pagination; limit must be 1..50000 entries')
    names = sorted(info.get('files', {}))
    entries, end = row_page(({name: info['files'][name]} for name in names[offset:]),
                            offset, limit, budget=PAGE_BYTES - 4096)
    metadata, clipped = object_view({k: v for k, v in info.items() if k != 'files'}, 3072)
    return {**metadata, 'files': {k: v for row in entries for k, v in row.items()},
            'metadata_clipped': clipped, 'total_files': len(names), 'offset': offset,
            'next_offset': end if end < len(names) else None, 'offset_unit': 'file entries',
            'page_byte_limit': PAGE_BYTES,
            'details': 'Use section=manifest for exact paginated manifest JSON; path selects a file.'}
