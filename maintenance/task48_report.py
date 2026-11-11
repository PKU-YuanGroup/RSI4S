"""Provenance-linked, loss-aware tables for the paired 48-task experiment."""
import csv
from collections import Counter
import json
import math
from pathlib import Path

from rsi.common import write_json
from rsi.freeze import sha
from maintenance.task48_results import session_counts, executions, events_at, terminal, leaves, escape, annotation_counts, scientific_completion_status
from maintenance.task48_costs import CostAccumulator
from maintenance.task48_audit import audit_chains, audit_run, audit_pairs
from rsi.chain import SCHEDULING_POLICY as CHAIN_POLICY


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def numeric_leaves(value, prefix=''):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from numeric_leaves(item, prefix + '/' + str(key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from numeric_leaves(item, prefix + '/' + str(i))
    elif value is None or type(value) in (int, float):
        yield prefix.lstrip('/'), value if value is not None and math.isfinite(value) else None


def write_csv(path, rows, fields):
    fields = list(dict.fromkeys([*fields, *(k for row in rows for k in row)]))
    with path.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict,list)) else v
                             for k, v in row.items()})


def compare(rows):
    pairs = {}
    for row in rows:
        if row['split'] not in ('test', 'episode'):
            continue
        key = tuple(row[k] for k in ('task_id','instance','seed','split','metric'))
        pair = pairs.setdefault(key, {})
        if row['mode'] in pair:
            raise ValueError('Ambiguous final comparison key: ' + str(key))
        pair[row['mode']] = row
    results = []
    for key, pair in sorted(pairs.items(), key=str):
        fixed, coevolve = pair.get('fixed', {}), pair.get('coevolve', {})
        direction = fixed.get('direction') or coevolve.get('direction')
        status = 'paired'
        if not fixed or not coevolve or any(r.get('value') is None for r in (fixed, coevolve)):
            status = 'missing'
        elif any(r.get('status') != 'ok' for r in (fixed, coevolve)):
            status = 'invalid_or_incomplete'
        elif direction not in ('higher_is_better', 'lower_is_better') or fixed.get('direction') != coevolve.get('direction'):
            status = 'direction_unknown'
        difference = None
        if status == 'paired':
            difference = coevolve['value'] - fixed['value']
            if direction == 'lower_is_better':
                difference = -difference
        results.append(dict(zip(('task_id','instance','seed','split','metric'),key),
            experiment_id=(fixed or coevolve)['experiment_id'], direction=direction,
            fixed_value=fixed.get('value'), coevolve_value=coevolve.get('value'),
            oriented_difference=difference, pair_status=status,
            source_fixed=fixed.get('source'), source_coevolve=coevolve.get('source')))
    return results


def report(directory, *, deep_audit=False):
    directory = Path(directory).resolve()
    study = read(directory/'study.json')
    state = read(directory/'supervisor.json', {'jobs':read(directory/'jobs.json'), 'status':'prepared'})
    output = directory/'report'; output.mkdir(exist_ok=True)
    sources = {}
    missing_sources = set()

    def source(path, pointer=''):
        relative = str(path.relative_to(directory))
        if path.is_file():
            observed = sha(path)
            if relative in sources and sources[relative] != observed:
                raise RuntimeError('Report source changed during export: ' + relative)
            sources[relative] = observed
        else:
            missing_sources.add(relative)
        return relative + ('#' + pointer if pointer else '')

    for name in ('study.json', 'supervisor.json', 'jobs.json', 'environment.json'):
        if (directory/name).is_file():
            source(directory/name)
    telemetry = directory/'metrics.jsonl'
    accumulators = {}
    if telemetry.is_file():
        source(telemetry)
    runs, metrics, sessions, stages, audits = [], [], [], [], []
    for job in state['jobs']:
        run = Path(job['directory'])
        summary = read(run/'summary.json', {})
        completion = read(run/'completion.json', {})
        workflow = read(run/'workflow.json', {})
        retention = read(run/'retention.json', {})
        records = [read(p) for p in sorted((run/'records').glob('*.json'))]
        for path in (run/'records').glob('*.json'):
            source(path)
        events = events_at(run/'events.jsonl')
        if events is not None:
            source(run/'events.jsonl')
        inventory = executions(records, events)
        token_data = read(run/'token_report.json', {})
        tokens = token_data.get('total', {})
        accumulators[job['id']] = CostAccumulator(job, events, read(run/'workflow.json', {}), token_data)
        sessions.extend(dict(run_id=job['id'], task_id=job['task'], mode=job['mode'], **r) for r in inventory)
        base = dict(experiment_id=directory.name, run_id=job['id'], task_id=job['task'],
                    mode=job['mode'], seed=study.get('seeds', {}).get('model_training'),
                    status='failed' if completion.get('status')=='failed' else summary.get('status', job['status']))
        if workflow.get('status') == 'failed' and scientific_completion_status(completion) == 'completed':
            base['status'] = 'maintenance_failed'
        if study.get('interruption_reason') and ((job.get('returncode') or 0) < 0 or
                str(summary.get('error','')).startswith('KeyboardInterrupt: Terminated;')):
            base['status'] = 'interrupted'
        row = dict(base, scientific_status=scientific_completion_status(completion),
                   workflow_status=workflow.get('status'), retention_status=retention.get('status',workflow.get('retention_status')),
                   maintenance_error=retention.get('error') or workflow.get('maintenance_error') or completion.get('retention_error'),
                   error=summary.get('error') or completion.get('research_error') or completion.get('error') or (job.get('result') or {}).get('error'),
                   research_error=summary.get('error', completion.get('research_error')), terminal_error=completion.get('error'),
                   worker_returncode=job.get('returncode'), recorded_research_status=summary.get('status'),
                   supervisor_source=source(directory/'supervisor.json'),
                   completion_source=source(run/'completion.json'),
                   worker_log_source=source(directory/'worker-logs'/(job['id'].replace('/','-')+'.log')),
                   research_status=summary.get('research_status',summary.get('status')),
                   **session_counts(records, events), selected_artifact=(summary.get('final_test') or {}).get('artifact'),
                   input_tokens=tokens.get('prompt_tokens'), output_tokens=tokens.get('completion_tokens'),
                   cache_hit_input_tokens=tokens.get('cache_hit_input_tokens'),
                   cache_miss_input_tokens=tokens.get('cache_miss_input_tokens'),
                   unknown_usage_calls=tokens.get('unknown_usage_calls'),
                   cache_hit_unknown_calls=tokens.get('cache_hit_input_tokens_unknown_calls'),
                   cache_miss_unknown_calls=tokens.get('cache_miss_input_tokens_unknown_calls'),
                   annotation_queries=None, annotations_acquired=None, research_seconds=summary.get('research_wall_seconds'),
                   # Absent (None) for runs before 2026-11-07, which counted retry time as research.
                   infrastructure_excluded_seconds=summary.get('infrastructure_excluded_seconds'),
                   terminal_seconds=completion.get('terminal_seconds', (summary.get('final_test') or {}).get('elapsed_seconds')),
                   gpu=job['gpu'], cpu_cores=job['cpu_affinity'], terminal_status=completion.get('terminal_status'),
                   judge_valid=(summary.get('judge') or {}).get('is_valid'),
                   judge_status=(summary.get('judge') or {}).get('status',
                       'historical_review' if summary.get('judge') else 'not_recorded'),
                   result_policy='native_metrics_without_judge_v1',
                   source=source(run/'summary.json'), token_source=source(run/'token_report.json'))
        # Annotation cost is recorded at research closure, before terminal work.
        annotation = summary.get('annotation_status') or {}
        row['annotations_acquired'] = annotation.get('annotations_used')
        row['annotation_budget'] = annotation.get('annotation_budget')
        ledger = annotation_counts(run)
        row['annotation_queries'] = ledger['queries'] if ledger else None
        for name in ('evidence/organizer/ledger.sqlite', 'evidence/problem/data/pool.sqlite'):
            if (run/name).is_file():
                source(run/name)
        runs.append(row)
        expected = study.get('primary_metrics', {}).get(job['task'], [])
        specifications = {(e['instance'],e['metric']):e for e in expected}

        def emit(instance, name, value, path, pointer, split, artifact=None, status='ok', coverage=None, evidence_issues=None):
            spec = specifications.get((instance,name), {})
            metrics.append(dict(base, instance=instance, split=split, artifact=artifact,
                metric=name, value=value, direction=spec.get('direction'), unit=spec.get('unit'),
                is_primary=spec.get('is_primary',False), coverage=coverage, status=status if value is not None else 'missing',
                missing_reason=None if value is not None else 'No finite native metric',
                evidence_issues=evidence_issues or [], source=source(path,pointer)))

        # Every research evaluation, including auxiliary native metrics, remains visible.
        for record in records:
            for i, evaluation in enumerate(record.get('evaluations', [])):
                for instance, values in evaluation.get('raw_scores', {}).items():
                    for name, value, pointer in leaves(values):
                        emit(instance,name,value,run/'records'/f"{record['run_id']}.json",
                             f'/evaluations/{i}/raw_scores/{escape(instance)}'+pointer, 'validation',
                             evaluation.get('artifact'), 'ok' if evaluation.get('valid_primary_score') else 'invalid')
        native_rows, receipt = terminal(run, job['task'], summary, completion)
        available = {(r['instance'],r['metric']) for r in native_rows if r['value'] is not None}
        missing_primary = [key for key,spec in specifications.items() if spec.get('is_primary') and key not in available]
        if missing_primary and receipt['status'] == 'ok':
            receipt['status'] = 'invalid_or_incomplete'
            receipt['evidence_issues'].append('Required primary metric missing or nonfinite: '+str(missing_primary))
            for native in native_rows:
                if native['status'] == 'ok':
                    native['status'] = 'invalid_or_incomplete'
        seen = set()
        for native in native_rows:
            emit(native['instance'], native['metric'], native['value'], native['path'], native['pointer'],
                 native['split'], row['selected_artifact'], native['status'], native['coverage'], native['evidence_issues'])
            seen.add((native['instance'], native['metric']))
        for (instance,name), spec in specifications.items():
            if spec.get('is_primary') and (instance,name) not in seen:
                emit(instance,name,None,receipt['source'],'',receipt['split'],row['selected_artifact'],'missing')
        row.update(native_result_status=receipt['status'], native_result_issues=receipt['evidence_issues'],
                   test_samples=receipt['test_samples'], scored_samples=receipt['samples'],
                   membership=receipt['membership'], episode_protocol=receipt['protocol'],
                   trajectory_sha256=receipt['trajectory'])
        if 'selected_episode' in receipt:
            row.update(episode_selection=receipt['selection'], selected_episode=receipt['selected_episode'],
                       last_completed_episode=receipt['last_completed_episode'],
                       completed_episodes=receipt['completed_episodes'], valid_episodes=receipt['valid_episodes'],
                       episode_scores=receipt['episodes'])
        audits.append(audit_run(job, study, summary, completion, events, inventory, receipt, source, deep=deep_audit))
    if telemetry.is_file():
        with telemetry.open() as stream:
            for line in stream:
                sample = json.loads(line)
                for measured in sample.get('jobs', []):
                    if measured['id'] in accumulators:
                        accumulators[measured['id']].add(sample, measured)
    for row in runs:
        stage_rows, totals = accumulators[row['run_id']].finish()
        row.update(totals)
        stages.extend(dict(run_id=row['run_id'], task_id=row['task_id'], mode=row['mode'], **r) for r in stage_rows)
    comparisons = compare(metrics)
    chained = study.get('resources', {}).get('scheduling_policy') == CHAIN_POLICY
    # Chain steps start from different inherited Agents by design; audit lineage instead of pairs.
    audit = dict(schema_version=1, scope='Recorded evidence, not scientific method validity',
                 deep_archive_check=deep_audit, runs=audits,
                 pairs=[] if chained else audit_pairs(state['jobs']),
                 chains=audit_chains(state['jobs']) if chained else [])
    audit['status_counts'] = dict(Counter(r['status'] for r in audits))
    lines = [f'# {directory.name}', '',
             ('问题：任务链中，后续任务继承前序任务蒸馏的 Task/Meta Agent 与通用科研经验后，原生指标、成本和实际演化轮数如何变化？'
              if chained else '问题：相同任务与预算下，固定 Meta 与共同演化的原生指标、成本和实际演化轮数有何差异？'), '',
             f"状态：{state['status']}；{len({j['task'] for j in state['jobs']})} 个任务、{len(state['jobs'])} 次计划运行；调度：{study.get('resources', {}).get('scheduling_policy', '未记录')}。", '',
             '模型与各组预算以 study.json 和逐运行配置为准；阶段记录不新增时间限制。', '',
             'Task 在 fixed/coevolve 均可自改。*_versions 按实际执行的不同源码摘要计数；'
             '*_published_versions 统计执行过的已发布起始版本，*_configured_versions 仅统计配置。'
             '步骤入口源码与行覆盖是执行证据，不表示包内每个文件都被执行。', '',
             '汇总协议：native_metrics_without_judge_v1。Judge 不作为成绩纳入条件；'
             '历史判定与费用保留，停用审查不表示方法已通过科学有效性审核。', '',
             ('链上第一个任务从共享初始 Agent 开始，后续任务的初始 Agent 来自前序封存交接，核验见下方任务链；科学模型独立生成，训练种子未由宿主强制。'
              if chained else '计划要求固定/共同演化使用同一数据契约和初始 Agent，实际配对核验见 audit.json；科学模型独立生成，训练种子未由宿主强制。'),
             '仅一次配对，无独立重复；不报告统计显著性或置信区间，不把差值解释为 Meta 自演化的因果效应。', '',
             '演化口径：执行次数由源码执行事件确认；启动尝试、未启动、失败及配置版本另列，完成数须有执行证据且 finished；M-final 单列。', '',
             '| 模式 | 运行状态 | 已报告缓存命中输入 | 已报告未命中输入 | 已报告输出 | 用量未知调用 | 完成 Task / Meta 会话 |',
             '|---|---|---:|---:|---:|---:|---|']
    if study.get('interruption_reason'):
        lines[2:2] = ['本次尝试中断原因：'+study['interruption_reason'], '',
                      '中断记录保留；未完成任务及未执行的排队任务不构成方法优劣证据。', '']
    for mode in ('fixed','coevolve'):
        subset = [r for r in runs if r['mode']==mode]
        totals = []
        for key in ('cache_hit_input_tokens','cache_miss_input_tokens','output_tokens','unknown_usage_calls',
                    'task_sessions_completed','meta_sessions_completed'):
            known = [r[key] for r in subset if r.get(key) is not None]
            missing = len(subset)-len(known)
            totals.append(str(sum(known)) + (f'（{missing} 次记录缺失）' if missing else '') if known else '未知')
        lines.append(f"| {mode} | {dict(Counter(r['status'] for r in subset))} | " +
                     ' | '.join(map(str,totals[:4])) + f' | {totals[4]} / {totals[5]} |')
    lines += ['', '用量是服务端已报告值的小计，未知调用和缺失缓存字段不是零消耗；费用为 null（未使用未经核实的单价）。', '',
              f"终评比较状态：{dict(Counter(r['pair_status'] for r in comparisons))}。",
              '高者优差值为 coevolve−fixed；低者优为 fixed−coevolve。不同任务/单位不直接平均。', '',
              '全部原生指标及缺失项见 [metrics.csv](metrics.csv)，逐项配对见 [comparisons.csv](comparisons.csv)。',
              '每次运行状态和成本见 [runs.csv](runs.csv)，执行事件与会话对应关系见 [sessions.csv](sessions.csv)。',
              '分阶段时间、采样缺口、跨阶段分摊及 API 用量见 [stages.csv](stages.csv)。',
              '证据审计见 [audit.json](audit.json)，缺失证据不视作通过；它不评价方法科学有效性。',
              '来源路径、JSON 字段和文件 SHA256 见各表 source 列与 [sources.json](sources.json)。', '',
              'CPU、进程内存、GPU 利用率/显存的原始时间序列位于 ../metrics.jsonl；共享 GPU 利用率不能当作单进程独占使用率。',
              'runs.csv 的 CPU 核秒与 GPU 驻留秒为采样估计：短命进程可能遗漏；GPU 驻留秒不是独占计算时长。',
              '超过 15 秒的监控间隔不外推资源消耗；跨阶段区间按时间比例分摊。短命进程和采样缺口使资源统计不完整。',
              '新实验单独目录、预算和报告；独立维护及 API 探针费用不混入任务。固定 split 0 不代表官方十划分平均，部分 Vina 不代表完整终评。',
              '运行环境与冻结信息见 ../study.json 及每次运行的 release.json/config.json/allocated_resources.json。', '',
              '失败、无效、终评不完整和覆盖不足均保留在表中；MAT_01/CHEM_01 为原生 episode，没有独立 Test。', '']
    probe=read(directory/'api-probe/token_report.json',{}).get('total')
    if probe is not None:
        source(directory/'api-probe/token_report.json')
        lines += ['独立 API 连通性探针用量（不归入科学任务）：'+json.dumps(probe,ensure_ascii=False), '']

    if chained:
        lines += ['## 任务链', '',
                  '每条链按顺序运行；后续任务只继承前序任务封存的 Task/Meta Agent 与 research_experience.md，'
                  '不继承科学产物、数据、标签、历史或 Test 结果。蒸馏在最终选模之后、冻结 Test 推理之前进行。'
                  '蒸馏到时停止操作，经检查后自动交接已落盘的 Agent 和笔记，继续该链；'
                  '其他蒸馏或交接失败时停止该链，后续任务记为 blocked，不沿用旧 Agent 或旧笔记回退。'
                  '同一任务与独立运行的差异受链上位置和前序任务影响，只作观察，不作因果结论。', '',
                  '| 链 | 模式 | 位置 | 任务 | 作业状态 | 蒸馏 | 交接 | 经验字符数 | 继承核验 |',
                  '|---|---|---:|---|---|---|---|---:|---|']
        for row in audit['chains']:
            lines.append(f"| {row['chain']} | {row['mode']} | {row['position']} | {row['task_id']} | "
                         f"{row['job_status']} | {row['distillation_status'] or '—'} | "
                         f"{row['handoff_status'] or row['handoff_error'] or '—'} | "
                         f"{row['experience_characters'] if row['experience_characters'] is not None else '—'} | "
                         f"{row['status']}{'：'+'；'.join(row['issues']) if row['issues'] else ''} |")
        lines += ['', '完整链审计见 audit.json 的 chains；运行内记录见各运行 chain.json 与 chain_handoff/manifest.json。', '']
    lines += [f"审计状态：{audit['status_counts']}；完整归档检查：{deep_audit}。", '']
    # Validate the complete source set before replacing any previous report.
    for relative, expected_hash in sources.items():
        if sha(directory/relative) != expected_hash:
            raise RuntimeError('Report source changed during export: ' + relative)
    write_csv(output/'runs.csv', runs, ['experiment_id','run_id','task_id','mode','seed','status'])
    write_csv(output/'metrics.csv', metrics, ['experiment_id','run_id','task_id','instance','mode','seed','split','metric','value'])
    write_csv(output/'comparisons.csv', comparisons, ['experiment_id','task_id','instance','seed','split','metric'])
    write_csv(output/'sessions.csv', sessions, ['run_id','session_id','kind','phase','status','started','executed'])
    write_csv(output/'stages.csv', stages, ['run_id','phase','wall_seconds','observed_seconds','unsampled_seconds'])
    write_json(output/'audit.json', audit)
    write_json(output/'sources.json', sources)
    write_json(output/'missing-sources.json', sorted(missing_sources))
    (output/'summary.md').write_text('\n'.join(lines))
    for relative, expected_hash in sources.items():
        if sha(directory/relative) != expected_hash:
            raise RuntimeError('Report source changed during publication: ' + relative)
    outputs = ('summary.md', 'runs.csv', 'metrics.csv', 'comparisons.csv', 'sessions.csv',
               'stages.csv', 'audit.json', 'sources.json', 'missing-sources.json')
    write_json(output/'manifest.json', dict(schema_version=1,
        files={name:sha(output/name) for name in outputs}, source_count=len(sources)))
    return dict(runs=len(runs), metrics=len(metrics), comparisons=len(comparisons), audit_statuses=audit['status_counts'])
