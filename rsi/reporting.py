"""Provider-reported token accounting by Agent role and immutable version."""
import json
from pathlib import Path

from rsi.common import write_json
from rsi.workflow import phase_after


def token_report(directory):
    directory = Path(directory)
    calls = {}
    phase = 'unattributed'
    for line in (directory / 'events.jsonl').read_text().splitlines():
        event = json.loads(line)
        phase = phase_after(event, phase)
        if 'call' not in event:
            continue
        row = calls.setdefault(event['call'], {'call': event['call'], 'role': 'unknown', 'version': None,
                                             'prompt_tokens': None, 'completion_tokens': None,
                                             'reported_total_tokens': None, 'usage_known': False})
        if event['event'] == 'api_reserved':
            row.update(reserved_tokens=event['reserved_tokens'], charged_tokens=event['reserved_tokens'])
        elif event['event'] == 'api_start':
            row.update(role=event['role'], version=event['version'], model=event['model'], start_time=event['time'])
            row['phase'] = 'historical_judge' if event['role'] == 'judge' else phase
            row.update(input_characters=event.get('input_characters'),
                       input_characters_by_category=event.get('input_characters_by_category'))
        elif event['event'] == 'api_settled':
            usage = event.get('usage')
            if usage is not None:
                total = usage['prompt_tokens'] + usage['completion_tokens']
                row.update(prompt_tokens=usage['prompt_tokens'], completion_tokens=usage['completion_tokens'],
                           reported_total_tokens=total, charged_tokens=total, usage_known=True)
                reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
                row.update(reasoning_tokens=reasoning,
                           non_reasoning_output_tokens=(usage['completion_tokens'] - reasoning if reasoning is not None else None),
                           cache_hit_input_tokens=usage.get('prompt_cache_hit_tokens',
                               (usage.get('prompt_tokens_details') or {}).get('cached_tokens')),
                           cache_miss_input_tokens=usage.get('prompt_cache_miss_tokens'))
                hit = row.get('cache_hit_input_tokens')
                if (row['cache_miss_input_tokens'] is None and type(hit) is int
                        and 0 <= hit <= usage['prompt_tokens']):
                    row['cache_miss_input_tokens'] = usage['prompt_tokens'] - hit
        elif event['event'] == 'api_error':
            row.update(error_type=event['error_type'], http_status=event['http_status'])
        elif event['event'] == 'api_end':
            row.update(seconds=event['seconds'], finish_reason=event.get('finish_reason'))

    def total(rows):
        result = {'calls': len(rows), 'prompt_tokens': sum(r['prompt_tokens'] or 0 for r in rows),
                'completion_tokens': sum(r['completion_tokens'] or 0 for r in rows),
                'reported_total_tokens': sum(r['reported_total_tokens'] or 0 for r in rows),
                'charged_tokens': sum(r.get('charged_tokens', 0) for r in rows),
                'unknown_usage_calls': sum(not r['usage_known'] for r in rows)}
        for field in ('reasoning_tokens', 'non_reasoning_output_tokens', 'cache_hit_input_tokens', 'cache_miss_input_tokens'):
            result[field] = sum(r.get(field) or 0 for r in rows)
            result[field + '_unknown_calls'] = sum(r.get(field) is None for r in rows)
        result['input_characters'] = sum(r.get('input_characters') or 0 for r in rows)
        result['input_characters_unknown_calls'] = sum(r.get('input_characters') is None for r in rows)
        categories = {}
        for row in rows:
            for key, value in (row.get('input_characters_by_category') or {}).items():
                categories[key] = categories.get(key, 0) + value
        result['input_characters_by_category'] = categories
        return result

    rows = sorted(calls.values(), key=lambda r: r['call'])
    roles = {role: total([r for r in rows if r['role'] == role]) for role in sorted({'meta', 'task', 'judge'} | {r['role'] for r in rows})}
    versions = [{'role': role, 'version': version,
                 **total([r for r in rows if r['role'] == role and r['version'] == version])}
                for role, version in sorted({(r['role'], r['version']) for r in rows}, key=str)]
    phases = {stage: total([r for r in rows if r.get('phase', 'unattributed') == stage])
              for stage in sorted({r.get('phase', 'unattributed') for r in rows})}
    report = {'by_role': roles, 'by_version': versions, 'by_phase': phases,
              'api_cost': None, 'api_cost_reason': 'No verified billing receipt or price schedule supplied',
              'total': total(rows), 'calls': rows,
              'note': 'Reported tokens come from provider usage. Unknown usage is not zero; charged_tokens retains its conservative reservation. Historical Judge calls remain separate from Meta and Task; current runs do not invoke Judge.'}
    write_json(directory / 'token_report.json', report)
    lines = ['# Token 使用报告', '', '| 角色 | 调用数 | 输入 token | 输出 token | 已报告合计 | 用量未知调用 |',
             '|---|---:|---:|---:|---:|---:|']
    for role, value in roles.items():
        lines.append(f"| {role} | {value['calls']} | {value['prompt_tokens']} | {value['completion_tokens']} | {value['reported_total_tokens']} | {value['unknown_usage_calls']} |")
    lines += ['', '## 输出与缓存明细', '', '| 角色 | Reasoning | 其他输出 | 缓存命中输入 | 未命中输入 |',
              '|---|---:|---:|---:|---:|']
    for role, value in roles.items():
        lines.append(f"| {role} | {value['reasoning_tokens']} | {value['non_reasoning_output_tokens']} | {value['cache_hit_input_tokens']} | {value['cache_miss_input_tokens']} |")
    lines += ['', '上述明细仅累加 API 已报告值，缺失字段不代表零用量；各字段未知调用数见 JSON。Reasoning 包含在输出合计内。',
              '逐 HTTP 请求（含重试）的输入类别字符量另存于 JSON；字符量不是 token 或费用。']
    lines += ['', '## 各 Agent 版本', '', '| 角色 | 版本 | 调用数 | 输入 token | 输出 token | 已报告合计 |', '|---|---|---:|---:|---:|---:|']
    for value in versions:
        lines.append(f"| {value['role']} | `{value['version']}` | {value['calls']} | {value['prompt_tokens']} | {value['completion_tokens']} | {value['reported_total_tokens']} |")
    lines += ['', '来源：events.jsonl 与 calls/ 中的原始 API 交互。未知用量单列，不作为零消耗；judge 独立统计。逐调用明细及保守记账见 token_report.json。', '']
    (directory / 'token_report.md').write_text('\n'.join(lines))
    return report
