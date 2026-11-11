"""Entry point for executable Meta/Task source evolution."""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import importlib.util
import json
import sys
from pathlib import Path

from rsi.common import ROOT, write_json, record_path
from rsi.freeze import freeze, verify_freeze, code_hashes
from rsi.harness import run_task
from rsi.processes import task_executor
from rsi.providers import API_REQUEST_TIMEOUT_SECONDS
from rsi.schedule import (evolution_round_limit, retire_stage_limits, meta_selection_seconds,
                          final_inference_seconds, terminal_scoring_seconds)
from rsi.sandbox import validate_resources
from eval_service import _get_per_instance_primaries
from rsi.tasks import task_directory, selected_tasks, MEDRSEE_TASKS, SIMULATOR_TASKS


def preflight(data_root, tasks=None, simulator_environments=None):
    result = []
    for task in tasks or selected_tasks():
        root = task_directory(data_root,task)
        table, issues = _get_per_instance_primaries(root / 'metadata.json', task)
        tree = ast.parse((root / 'evaluation/evaluator.py').read_text())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                imports.add(node.module.split('.')[0])
        missing = sorted(name for name in imports if importlib.util.find_spec(name) is None
                         and not (root / 'evaluation' / (name + '.py')).exists()
                         and not (task in MEDRSEE_TASKS and name=='medrsee'))
        result.append({'task': task, 'primary_instances': len(table), 'metadata_issues': issues,
                       'missing_evaluator_imports': missing,
                       'data_directory': (root / 'problem/data').is_dir(),
                       'archived_data': (root / 'problem/data_archives').is_dir(),
                       'evaluator_executed': False})
        if task in MEDRSEE_TASKS:
            from benchmarks.medrsee.source import DATA
            result[-1].update(benchmark='medrsee',input_access='host fetch_data',
                raw_data_root=str(DATA),raw_data_present=DATA.is_dir(),
                missing_metric_imports=[m for m in ('monai','medpy','nibabel','rasterio','scipy','sklearn')
                                        if importlib.util.find_spec(m) is None])
        if task in SIMULATOR_TASKS:
            from benchmarks.simulator.source import verify_source
            try:
                simulator_root, simulator_python = verify_source(task, **(simulator_environments or {}).get(task, {}))
                result[-1].update(benchmark='simulator', source_verified=True,
                    raw_data_root=str(simulator_root), simulator_python=simulator_python,
                    protocol='repeated_fixed_episode_v1', independent_test=False)
            except Exception as exc:
                result[-1].update(benchmark='simulator', source_verified=False, error=str(exc))
    return {'tasks': result, 'task_count': len(result),
            'primary_instances': sum(x['primary_instances'] for x in result),
            'scope': 'Static inventory only; does not prove all selected scientific environments are ready.'}


def successful_result(task, result):
    return (result.get('status') in ('completed', 'budget_exhausted')
            and bool(result.get('valid_score_observed'))
            and (task not in MEDRSEE_TASKS or (result.get('final_test') or {}).get('status') == 'completed')
            and (result.get('benchmark_protocol')!='naturebench-active' or (result.get('final_test') or {}).get('status')=='completed'))


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    cfg.pop('judge', None)  # Retired; old configurations cannot enable model review.
    if cfg.get('freeze_manifest'):
        cfg['freeze_manifest'] = str(record_path(cfg['freeze_manifest'], area='provenance'))
    required = {'data_root', 'python', 'credentials', 'model', 'mode', 'rounds', 'gpus', 'budget'}
    if not required <= cfg.keys():
        raise ValueError(f'Missing config fields: {required - cfg.keys()}')
    if cfg['mode'] not in ('fixed', 'coevolve'):
        raise ValueError('Invalid mode')
    for key, default in [('cpu_cores_per_slot', 4), ('gpu_wait_seconds', 600)]:
        value = cfg.setdefault(key, default)
        if type(value) is not int or value < 1:
            raise ValueError(key + ' must be a positive integer')
    meta_selection_seconds(cfg)
    if cfg.get('naturebench_active'):
        cfg['final_inference_seconds'] = final_inference_seconds(cfg)
        cfg['terminal_scoring_seconds'] = terminal_scoring_seconds(cfg)
    cfg['rounds'] = evolution_round_limit(cfg)
    retire_stage_limits(cfg)
    cfg.pop('task0_handoff_after_valid_evaluation', None)  # Retired; evaluate never ends Task0.
    for name in ('api_stop_after_action', 'reuse_artifacts'):
        if name in cfg and type(cfg[name]) is not bool:
            raise ValueError(name + ' must be boolean')
    from rsi.providers import validate_model
    validate_model(cfg['model'])
    maximum = cfg.get('max_output_tokens', 'model_max')
    if maximum is None:
        maximum = 'model_max'
    if maximum != 'model_max' and (type(maximum) is not int or maximum < 1):
        raise ValueError('max_output_tokens must be model_max, null or a positive integer')
    cfg['max_output_tokens'] = maximum
    if 'max_actions_per_round' in cfg:
        raise ValueError('Use task_calls/meta_calls; max_actions_per_round belongs to the retired policy runner')
    for name in ('task_calls', 'meta_calls'):
        cfg.setdefault(name, None)
        if cfg[name] is None:
            continue
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError(f'Invalid {name}')
    if not cfg['gpus'] or any(type(g) is not int or not 0 <= g < 8 for g in cfg['gpus']):
        raise ValueError('gpus must be a nonempty subset of GPU IDs 0..7')
    if len(set(cfg['gpus'])) != len(cfg['gpus']):
        raise ValueError('Duplicate GPU IDs')
    slots = cfg.setdefault('gpu_slots_per_gpu', 2)
    if type(slots) is not int or not 1 <= slots <= 8:
        raise ValueError('gpu_slots_per_gpu must be an integer in 1..8')
    cfg['sandbox_resources'] = validate_resources(cfg.get('sandbox_resources'))
    if cfg.get('naturebench_active'):
        if not cfg.get('naturebench_source_root'):raise ValueError('Active NatureBench requires native metric source_root')
        if cfg['budget'].get('solve_seconds') != cfg['budget'].get('wall_seconds'):
            raise ValueError('Active NatureBench requires equal solve_seconds and wall_seconds for one shared research budget')
    for key in ('solve_seconds', 'api_calls', 'api_tokens'):
        if key in ('api_calls', 'api_tokens') and key in cfg['budget'] and cfg['budget'][key] is None:
            continue  # Explicit null removes a cumulative limit; accounting continues.
        if type(cfg['budget'].get(key)) is not int or cfg['budget'][key] < 1:
            raise ValueError(f'Invalid budget {key}')
    if 'wall_seconds' in cfg['budget']:
        if type(cfg['budget']['wall_seconds']) is not int or cfg['budget']['wall_seconds'] < 1:
            raise ValueError('Invalid budget wall_seconds')
    if cfg.get('finalize_seconds', 0) != 0:
        raise ValueError('finalize_seconds was removed; final inference follows research closure')
    cfg.pop('finalize_seconds', None)
    for key, default, minimum in (('api_request_timeout_seconds', API_REQUEST_TIMEOUT_SECONDS, 1), ('api_max_retries', 3, 0)):
        cfg.setdefault(key, default)
        if type(cfg[key]) is not int or cfg[key] < minimum:
            raise ValueError(f'Invalid {key}')
    if 'phase_budget' in cfg:
        raise ValueError('phase_budget was removed; stages share the global time budget')
    cfg.pop('max_tool_calls', None)  # Retired setting; old configs cannot restore the limit.
    if 'medrsee_final_seconds' in cfg and (type(cfg['medrsee_final_seconds']) is not int or cfg['medrsee_final_seconds'] < 1):
        raise ValueError('medrsee_final_seconds must be a positive integer')
    if Path(cfg['python']).resolve() != Path(sys.executable).resolve():
        raise ValueError('Launch with configured science Python; official evaluator uses sys.executable')
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'freeze', 'verify'):
        p = sub.add_parser(name)
        p.add_argument('--data-root', help='Task data root (a group config\'s naturebench_source_root or data_root)')
        p.add_argument('--manifest',help='New freeze identity path; never overwrite an existing experiment freeze')
        if name == 'preflight':
            p.add_argument('--tasks',nargs='+')
            p.add_argument('--config', help='Group config: supplies the data root and simulator environments')
        if name == 'freeze':
            p.add_argument('--evidence', help='Optional developer offline-test report for the current code')
    run = sub.add_parser('run')
    run.add_argument('--config', required=True)
    run.add_argument('--tasks', nargs='+', help='Task IDs from the config\'s group (e.g. LIFE_13); omit to use the config tasks')
    run.add_argument('--out', required=True, help='New experiment directory; existing directories refused')
    run.add_argument('--workers', type=int, help='Concurrent tasks; defaults to all configured GPU slots')
    run.add_argument('--mode', choices=['fixed', 'coevolve'], help='Override the configuration mode')
    run.add_argument('--integration', action='store_true', help='Bounded pre-freeze acceptance; requires passing offline tests')
    index = sub.add_parser('medrsee-index',help='Build immutable membership indices; does not read any labels')
    index.add_argument('--tasks',nargs='+',choices=MEDRSEE_TASKS,default=list(MEDRSEE_TASKS))
    score = sub.add_parser('medrsee-score',help='Organizer-only one-shot final prediction submission scoring')
    score.add_argument('--task',required=True,choices=MEDRSEE_TASKS)
    score.add_argument('--run',required=True,type=record_path)
    score.add_argument('--raw-data',help='MedRSEE raw data root (default: $MEDRSEE_HOME/data/medrsee)')
    args = parser.parse_args()
    try:
        if args.command == 'run':
            args.out = str(record_path(args.out, area='runs'))
        if args.command == 'freeze':
            if args.evidence:
                args.evidence = str(record_path(args.evidence, area='provenance'))
            if args.manifest:
                args.manifest = str(record_path(args.manifest, area='provenance'))
    except ValueError as exc:
        parser.error(str(exc))
    if args.command in ('preflight', 'freeze', 'verify') and not args.data_root and not getattr(args, 'config', None):
        parser.error('--data-root is required' + (' (or --config)' if args.command == 'preflight' else ''))
    if args.command == 'preflight':
        group = json.loads(Path(args.config).read_text()) if args.config else {}
        data_root = args.data_root or group.get('naturebench_source_root', group.get('data_root'))
        report = preflight(data_root, args.tasks or group.get('tasks'), group.get('simulator_environments'))
        write_json(ROOT / 'provenance/preflight.json', report)
        print(json.dumps(report, indent=2))
    elif args.command == 'freeze':
        print(freeze(args.data_root, args.evidence,args.manifest))
    elif args.command == 'verify':
        print(verify_freeze(args.data_root,manifest=args.manifest))
    elif args.command in ('medrsee-index','medrsee-score'):
        from benchmarks.medrsee.membership import build_index,fingerprint
        for task in args.tasks if args.command=='medrsee-index' else [args.task]:
            path = build_index(task,ROOT/'runs/medrsee-indices'/fingerprint()/f'{task}.sqlite')
            if args.command=='medrsee-index':
                print(task,path,flush=True)
            else:
                from benchmarks.medrsee.scoring import final_submission
                from benchmarks.medrsee.source import DATA
                print(json.dumps(final_submission(task,args.run,path,args.raw_data or DATA),ensure_ascii=False,indent=2))
    else:
        cfg = load_config(args.config)
        if args.mode:
            cfg['mode'] = args.mode
        if args.integration:
            report = json.loads((ROOT / 'provenance/offline_tests.json').read_text())
            if not report.get('success') or report['code_hashes'] != code_hashes():
                parser.error('Integration requires successful tests of the current code')
            if not args.tasks or len(args.tasks) != 1 or cfg['rounds'] > 3 or cfg['budget']['solve_seconds'] > 3600:
                parser.error('Integration is limited to one task, <=3 rounds and <=3600 seconds')
            identity = {'phase': 'integration_not_yet_frozen', 'code_hashes': code_hashes()}
        elif cfg.get('naturebench_active'):
            from benchmarks.naturebench_active.release import release
            targets=args.tasks or cfg.get('tasks')
            if not targets:parser.error('Specify active NatureBench tasks explicitly')
            identities={t:release(cfg,t)['identity'] for t in targets}
            identity=dict(protocol='naturebench-active',tasks=identities)
        else:
            identity = verify_freeze(cfg['data_root'],manifest=cfg.get('freeze_manifest'))
        manifest = selected_tasks()
        tasks = args.tasks or cfg.get('tasks') or (ROOT/'configs/tasks_44.txt').read_text().split()
        if len(tasks) != len(set(tasks)) or set(tasks) - set(manifest):
            parser.error('Tasks must be a unique subset of the registered task manifests')
        capacity = len(cfg['gpus']) * cfg['gpu_slots_per_gpu']
        workers = args.workers if args.workers is not None else capacity
        if not 1 <= workers <= capacity:
            parser.error('workers must be between 1 and total configured GPU slots')
        out = record_path(args.out, area='runs')
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / 'experiment.json', {'schema': 'rsi4s.experiment.v1',
                                            'experiment_id': out.name,
                                            'freeze': identity, 'tasks': tasks,
                                            'budget_per_task': cfg['budget'], 'mode': cfg['mode'],
                                            'resources': {'gpus': cfg['gpus'], 'slots_per_gpu': cfg['gpu_slots_per_gpu'],
                                                          'workers': min(workers, len(tasks)), 'capacity': capacity}})
        results = {}
        # spawn prevents evaluator/service state from leaking across tasks.
        with task_executor(min(workers, len(tasks))) as executor:
            futures = {executor.submit(run_task, cfg, t, out / t, None, not args.integration): t for t in tasks}
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {'status': 'failed', 'error': str(exc)}
                results[task] = result
                write_json(out / 'summary.json', results)
                print(task, result['status'], 'valid_score=', result.get('valid_score_observed'), flush=True)
        if any(not successful_result(t, r) for t, r in results.items()):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
