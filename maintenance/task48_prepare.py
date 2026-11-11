"""Prepare a unique paired study; validate all environments before live calls."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

from rsi.chain import HANDOFF, check_handoff_status
from benchmarks.task_names import public_name
from rsi.tasks import MEDRSEE_TASKS, SIMULATOR_TASKS
from rsi.common import ROOT, git_commit, record_path, write_json
from rsi.freeze import code_hashes, sha
from rsi.schedule import STAGE_POLICY
from rsi.providers import API_REQUEST_TIMEOUT_SECONDS
from maintenance.task48 import (MODE_ORDER, SCHEDULING_POLICY, CONCURRENT_POLICY, CHAIN_POLICY,
                                COEVOLVE_FIRST_POLICY, plan, plan_chains, plan_selected, task_configs,
                                scheduling_policy)


def continuation_plan(study):
    """What a stopped sequential study left: failed runs to rerun from their parent, never-started jobs.

    Finished runs stay results of the stopped study. Nothing is copied or resumed from it."""
    study = record_path(study, area='runs')
    state = json.loads((study/'supervisor.json').read_text())
    record = json.loads((study/'study.json').read_text())
    if state.get('status') not in ('interrupted', 'finished'):
        raise ValueError('Continue only from a stopped study')
    if state.get('scheduling_policy') not in (SCHEDULING_POLICY, COEVOLVE_FIRST_POLICY):
        raise ValueError('Continue only a sequential fixed/coevolve study')
    kept, reruns, continued = [], {}, []
    for job in state['jobs']:
        name, run = f"{job['mode']}/{job['task']}", Path(job['directory'])
        if job['status'] == 'finished':
            kept.append(name)
        elif job['status'] == 'failed':
            if not (run/'completion.json').is_file() or run.parts[-2:] != (job['mode'], job['task']):
                raise ValueError('A failed job needs its completed original run: '+name)
            reruns[name] = str(record_path(run, area='runs'))
        elif job['status'] in ('cancelled', 'queued'):
            if run.exists():
                raise ValueError('A never-started job already has a run directory: '+name)
            continued.append(name)
        else:
            raise ValueError(f'Unexpected job status {job["status"]!r}: {name}')
    if not reruns and not continued:
        raise ValueError('The stopped study has nothing left to run')
    return dict(study=str(study), model=record['model']['id'], scheduling_policy=state['scheduling_policy'],
                mode_order=list(state['mode_order']), kept=kept, reruns=reruns, continued=continued)


def chain_specs(values, inherits=None):
    """Parse NAME=TASK,TASK,... and optional NAME=PREDECESSOR_RUN continuation points."""
    chains, inherits = [], dict(inherits or {})
    for value in values:
        name, separator, tasks = value.partition('=')
        tasks = [task for task in tasks.split(',') if task]
        if not separator or not name or not tasks:
            raise ValueError('Use --chain NAME=TASK1,TASK2,...')
        chains.append(dict(name=name, tasks=tasks, start=1, inherit=None))
    if set(inherits) - {chain['name'] for chain in chains}:
        raise ValueError('--chain-inherit names an unknown chain')
    for chain in chains:
        if chain['name'] in inherits:
            # The worker verifies the sealed handoff in full before research.
            run = record_path(inherits[chain['name']], area='runs')
            manifest = json.loads((run/HANDOFF/'manifest.json').read_text())
            check_handoff_status(manifest)
            if (manifest['chain']['name'] != chain['name'] or manifest['chain']['tasks'] != chain['tasks']
                    or manifest['chain']['position'] >= len(chain['tasks'])):
                raise ValueError('Inherited handoff belongs to a different chain or ends it: '+chain['name'])
            chain.update(start=manifest['chain']['position']+1, inherit=str(run))
    return chains


def protocol_check(config_path, tasks, evidence, check_directory):
    """Run under the group's configured interpreter: check readiness, then create or verify the freeze.

    `evidence` (optional) is a developer offline-test report; when given it must match the current code
    and the group's environment test modules also run. Without it the check is data/environment readiness.
    """
    from run_rsi import load_config, preflight
    from rsi.harness import release
    config = load_config(config_path)
    before = code_hashes()
    if evidence is not None:
        baseline = json.loads(Path(evidence).read_text())
        if baseline.get('success') is not True or baseline.get('code_hashes') != before:
            raise ValueError('Full offline test evidence must match the current infrastructure')
    readiness = preflight(config.get('naturebench_source_root', config['data_root']), tasks,
                          simulator_environments=config.get('simulator_environments'))
    for row in readiness['tasks']:
        if (row.get('metadata_issues') or row.get('missing_evaluator_imports') or
            row.get('missing_metric_imports') or row.get('source_verified') is False):
            raise ValueError('Task readiness failed: ' + json.dumps(row))
    # Group-specific real data/environment checks, with no model API (developer test suite only).
    modules = []
    if evidence is not None:
        modules = (['tests.test_naturebench_active','tests.test_naturebench_final_outputs',
                    'tests.test_active_release_protocol','tests.test_validation_design']
                   if config.get('naturebench_active') else
                   ['tests.test_simulator'] if tasks[0] in SIMULATOR_TASKS else
                   ['tests.test_prepared_test_inputs','tests.test_medrsee'])
        if 'pending' in str(config_path):
            modules += ['tests.test_naturebench_pending']
        if 'plm_interact' in str(config_path):
            modules += ['tests.test_plm_interact']
        if tasks == ['LIFE_01']:
            modules += ['tests.test_resp']
    check_directory = Path(check_directory);check_directory.mkdir(parents=True,exist_ok=False)
    returncode = 0
    if modules:
        with (check_directory/'environment-tests.log').open('w') as out:
            returncode = subprocess.run([sys.executable,'-m','unittest',*modules,'-v'],cwd=ROOT,
                                        stdout=out,stderr=subprocess.STDOUT).returncode
    report = dict(success=returncode==0 and code_hashes()==before, code_hashes=before,
                  full_offline_evidence=None if evidence is None else str(Path(evidence).resolve()),
                  full_offline_sha256=None if evidence is None else sha(evidence),
                  python=sys.executable,python_version=platform.python_version(),
                  test_modules=modules, readiness=readiness, model_api_called=False)
    if config.get('naturebench_active'):
        from benchmarks.naturebench_active.release import protocol_settings
        report['protocol_settings'] = protocol_settings(config)
    checked = check_directory/'evidence.json'
    write_json(checked,report)
    if not report['success']:
        raise ValueError('Configured environment tests failed: '+str(check_directory))
    if Path(config['freeze_manifest']).exists():
        for task in tasks:
            release(config,task)
        return report
    if config.get('naturebench_active'):
        from benchmarks.naturebench_active.release import freeze
        freeze(config, checked, config['freeze_manifest'], tasks)
    else:
        from rsi.freeze import freeze
        freeze(config['data_root'], checked, config['freeze_manifest'], tasks=tasks)
    for task in tasks:
        release(config,task)
    return report


def prepare(directory, evidence=None, *, tasks=None, solve_seconds=None, concurrent_modes=False,
            fresh_releases=False, cold_indices=False, mode_order=MODE_ORDER,
            model=None, credentials=None, jobs=None, slot_offset=0, parents=None, rerun_reason=None,
            chains=None, chain_modes=('coevolve',), chain_distillation_seconds=None, continue_from=None,
            api_max_retries=None):
    """`jobs` selects explicit mode/task reruns; each names its parent run and reason.

    `chains` (from chain_specs) runs each chain's tasks in order, inheriting distilled Agents and notes.
    `continue_from` (a stopped sequential study) runs what it left, see continuation_plan.
    `api_max_retries` overrides the group configs' retries per provider call (attempts = retries + 1).
    `evidence` (optional) is a developer offline-test report bound to the current code.
    """
    from rsi.providers import validate_model, generation_settings
    if model is not None:
        validate_model(model)
    if credentials is not None and not Path(credentials).is_file():
        raise ValueError('Credential source must be an existing file')
    if api_max_retries is not None and (type(api_max_retries) is not int or not 0 <= api_max_retries <= 20):
        raise ValueError('api_max_retries must be an integer in 0..20')
    directory = record_path(directory,area='runs')
    before = code_hashes()
    if evidence is not None:
        evidence = record_path(evidence,area='provenance')
        offline = json.loads(evidence.read_text())
        if offline.get('success') is not True or offline.get('code_hashes') != before:
            raise ValueError('Require successful full offline tests of current code')
    pairs=task_configs()
    if jobs is not None:
        if tasks is not None or concurrent_modes or tuple(mode_order)!=MODE_ORDER:
            raise ValueError('Selected reruns define their own tasks and concurrent schedule')
        if not rerun_reason or not isinstance(parents,dict) or set(parents)!=set(jobs):
            raise ValueError('Each selected rerun requires a parent run and a rerun reason')
        for job,parent in parents.items():
            parent=record_path(parent,area='runs')
            if not (parent/'completion.json').is_file() or parent.parts[-2] != job.split('/',1)[0] or parent.parts[-1] not in (job.split('/',1)[1],public_name(job.split('/',1)[1])):
                raise ValueError('Parent must be the completed original run of the same mode/task: '+job)
        tasks=list(dict.fromkeys(job.split('/',1)[1] for job in jobs))
    continuation=None
    if continue_from is not None:
        if tasks is not None or jobs is not None or chains is not None or concurrent_modes:
            raise ValueError('A continuation derives its jobs from the stopped study')
        continuation=continuation_plan(continue_from)
        if model is None or model!=continuation['model']:
            raise ValueError('A continuation must name the stopped study model: '+continuation['model'])
        if list(mode_order)!=continuation['mode_order']:
            raise ValueError('A continuation keeps the stopped study mode order')
        if continuation['reruns'] and not rerun_reason:
            raise ValueError('Failed runs of the stopped study are reruns; give a rerun reason')
    if chains is not None:
        if tasks is not None or jobs is not None or concurrent_modes or tuple(mode_order)!=MODE_ORDER:
            raise ValueError('Chains define their own tasks and sequential schedule')
        if chain_distillation_seconds is not None and (type(chain_distillation_seconds) is not int
                                                       or chain_distillation_seconds<1):
            raise ValueError('Chain distillation seconds must be a positive integer')
        tasks=[task for chain in chains for task in chain['tasks'][chain['start']-1:]]
    if tasks is not None:
        available=dict(pairs)
        if not tasks or len(tasks)!=len(set(tasks)) or not set(tasks)<=set(available):
            raise ValueError('Select distinct tasks from task_48')
        pairs=[(task,available[task]) for task in tasks]
    if solve_seconds is not None and (type(solve_seconds) is not int or solve_seconds<=0):
        raise ValueError('Research seconds must be a positive integer')
    mode_order=tuple(mode_order)
    policy=scheduling_policy(mode_order,concurrent_modes)
    if concurrent_modes and mode_order != MODE_ORDER:
        raise ValueError('Mode order applies only to sequential studies')
    rerun_jobs=jobs
    if rerun_jobs is not None:
        policy=CONCURRENT_POLICY
    if chains is not None:
        policy=CHAIN_POLICY
    directory.mkdir(parents=True,exist_ok=False)
    jobs=(plan_selected(directory,rerun_jobs,slot_offset=slot_offset) if rerun_jobs is not None else
          plan_chains(directory,chains,chain_modes,slot_offset=slot_offset) if chains is not None else
          plan(directory,pairs,concurrent_modes=concurrent_modes,mode_order=mode_order,
               include=None if continuation is None else [*continuation['reruns'],*continuation['continued']]))
    if chain_distillation_seconds is not None:
        for job in jobs:
            job['settings']['chain_distillation_seconds']=chain_distillation_seconds
    configs={}
    for task, name in pairs:
        configs.setdefault(name,[]).append(task)
    prepared_configs={}
    for name, selected in configs.items():
        cfg=json.loads((ROOT/name).read_text())
        if model is not None:
            cfg['model']=model
        if credentials is not None:
            cfg['credentials']=str(Path(credentials).resolve())
        if api_max_retries is not None:
            cfg['api_max_retries']=api_max_retries
        if solve_seconds is not None:
            cfg['budget']={**cfg['budget'],'solve_seconds':solve_seconds,'wall_seconds':solve_seconds}
        if fresh_releases:
            cfg['freeze_manifest']=str(ROOT/'provenance/releases'/directory.name/(Path(name).stem+'.json'))
        if cold_indices and any(task in MEDRSEE_TASKS for task in selected):
            cfg['medrsee_index_root']=str(directory/'shared-membership')
        if (solve_seconds is not None or fresh_releases or cold_indices or model is not None or credentials is not None
                or api_max_retries is not None):
            cfg['tasks']=selected
            path=directory/'group-configs'/Path(name).name
            write_json(path,cfg)
        else:
            path=ROOT/name
        prepared_configs[name]=(path,cfg)
        for job in jobs:
            if job['source_config']==name:
                for key in ('budget','freeze_manifest','medrsee_index_root','model','credentials','api_max_retries'):
                    if key in cfg:job['settings'][key]=cfg[key]
    write_json(directory/'jobs.json',[{k:v for k,v in j.items() if k!='settings'} for j in jobs])
    study=dict(schema_version=1,reporting_schema='new_run_evidence_v3',experiment_id=directory.name,
        question='Fixed Meta versus coevolving Meta under matched task, data, initial Agent and resource budgets',
        tasks=[task for task,_ in pairs],instances={},
        modes=['fixed','coevolve'] if rerun_jobs is None else sorted({j['mode'] for j in jobs}),
        seeds=dict(data_split=None,initial_labels=None,model_training=None,
                   note='Use existing immutable dataset membership and initial labels; model training is Agent-selected, no host seed override'),
        model=dict(id=next(iter(prepared_configs.values()))[1]['model'],version=None),
        result_policy='native_metrics_without_judge_v1', scientific_review='disabled',
        stage_policy=STAGE_POLICY, per_stage_time_limits=False,
        generation_settings=dict(max_output_tokens='model_max',
                                 **generation_settings(next(iter(prepared_configs.values()))[1]['model'])['sampling_parameters'],
                                 **generation_settings(next(iter(prepared_configs.values()))[1]['model']),
                                 note='Output maximum resolved from provider metadata or explicitly sourced official model specification'),
        initial_agents={k:v for k,v in before.items() if k.startswith('agent/')},
        data_contract={},budgets={name:cfg['budget'] for name,(_,cfg) in prepared_configs.items()},
        resources=dict(wave_tasks=16 if rerun_jobs is None else len(jobs),
                       worker_processes=16 if rerun_jobs is None else len(jobs),
                       gpus=list(range(8)) if rerun_jobs is None else sorted({j['gpu'] for j in jobs}),gpu_slots=2,
                       logical_cpu_cores_per_worker=4,physical_cpu_cores_per_worker=4,
                       hardware_threads_per_core=1,library_threads_per_worker=4,
                       fixed_gpus=list(range(8)),coevolve_gpus=list(range(8)),
                       scheduling_policy=policy,
                       mode_order=list(mode_order),
                       mode_barrier=None if concurrent_modes or rerun_jobs is not None else f'All {mode_order[0]} workers including terminal scoring and retention',
                       planned_slots={j['id']:j['slot'] for j in jobs},
                       wave_barrier='All workers including native terminal scoring and retention'),
        primary_metrics={},analysis_plan=dict(repetitions=1,paired_by=['task','instance','split','metric'],
            missing='Explicit missing/failed/invalid/partial coverage; never zero-fill',
            statistics='Native-unit oriented differences only; no cross-task mean or significance claims',
            causal_limit='T00 science models independently generated; differences are not a causal effect of Meta evolution'),
        code_revision=dict(git=git_commit(),code_hashes=before),
        freeze={},environment={},
        parent_experiment=(continuation['reruns'] or None) if continuation is not None else
                          None if rerun_jobs is None else {job:str(record_path(path,area='runs')) for job,path in parents.items()},
        continued_from=None if continuation is None else {k:continuation[k] for k in ('study','kept','continued')},
        rerun_reason=rerun_reason,preflight_status='checking',
        api_max_retries={name:cfg.get('api_max_retries') for name,(_,cfg) in prepared_configs.items()},
        validation_run=dict(solve_seconds=solve_seconds,concurrent_modes=concurrent_modes,cold_indices=cold_indices),
        full_offline_evidence=None if evidence is None else str(evidence),
        created_at=datetime.now(timezone.utc).isoformat())
    if chains is not None:
        from rsi.chain import DISTILLATION_SECONDS
        study.update(question=('Task chains: does a later task benefit from the Task/Meta Agents and general '
                               'research notes distilled after earlier tasks?'),
            modes=list(chain_modes),
            chains=[{k:chain[k] for k in ('name','tasks','start','inherit')} for chain in chains],
            chain_protocol=dict(carried='Sealed Task and Meta Agent sources plus research_experience.md only',
                distillation='M-chain session after final selection and before frozen Test inference',
                distillation_seconds=chain_distillation_seconds or DISTILLATION_SECONDS,
                deadline_policy='Stop model/tool work at the time limit; validate and transfer saved Agents and notes',
                failure_policy='Timeout alone continues the chain; other failures stop it and block later steps',
                test_feedback=False))
        study['resources'].update(wave_tasks=len(chains)*len(chain_modes),
            worker_processes=len(chains)*len(chain_modes), gpus=sorted({j['gpu'] for j in jobs}),
            mode_barrier=None, wave_barrier=('A chain step starts after its predecessor worker ends, including '
                                             'terminal scoring and retention; chains run concurrently'))
        study['analysis_plan'].update(paired_by=['chain','position','task','instance','split','metric'],
            causal_limit=('Inherited Agents and notes depend on the earlier tasks and their order; comparisons '
                          'with independent runs of the same task are observations, not causal transfer effects'))
    write_json(directory/'study.json',study)
    hardware={}
    for key,command in [('cpu',['lscpu']),('gpu',['nvidia-smi','-q']),('kernel',['uname','-a']),
                        ('disk',['df','-h',str(directory)])]:
        observed=subprocess.run(command,capture_output=True,text=True,timeout=30)
        hardware[key]=dict(returncode=observed.returncode,stdout=observed.stdout,stderr=observed.stderr)
    write_json(directory/'environment.json',hardware)
    try:
        for name,tasks in configs.items():
            config_path,cfg=prepared_configs[name]
            tag=Path(name).stem
            checks=ROOT/'provenance/checks'/directory.name/tag
            # Separate process is essential: evaluator Python is part of release identity.
            result=subprocess.run([cfg['python'],'-m','maintenance.task48_prepare','check',
                '--config',str(config_path),'--tasks',*tasks,'--out',str(checks)]
                +([] if evidence is None else ['--evidence',str(evidence)]),cwd=ROOT)
            if result.returncode:
                raise RuntimeError('Environment/freeze check failed: '+name)
            frozen=json.loads(Path(cfg['freeze_manifest']).read_text())
            study['freeze'][name]=dict(path=cfg['freeze_manifest'],sha256=sha(cfg['freeze_manifest']),identity=frozen['identity'])
            study['environment'][name]={k:frozen[k] for k in ('python','python_version','packages')}
            study['data_contract'][name]=frozen['tasks']
            from rsi.tasks import task_directory
            for task in tasks:
                metadata=task_directory(cfg.get('naturebench_source_root',cfg['data_root']),task)/'metadata.json'
                native=json.loads(metadata.read_text())
                specs=[]
                for entry in native['performance_entries']:
                    for metric in entry['metrics']:
                        specs.append(dict(instance=entry['dataset_name'],metric=metric['name'],
                            direction=metric.get('metric_direction'),unit=metric.get('unit'),
                            is_primary=metric.get('is_primary',False)))
                study['primary_metrics'][task]=specs
                study['instances'][task]=list(dict.fromkeys(s['instance'] for s in specs))
            write_json(directory/'study.json',study)
        for job in jobs:
            write_json(job['config'],job['settings'])
        study['job_config_sha256'] = {job['id']:sha(job['config']) for job in jobs}
        # Use final job configurations; cache data belongs to the task dataset.
        from maintenance.task48_inputs import check_inputs
        study['prepared_inputs'] = check_inputs(jobs, directory/'prepared-inputs', prepare=True)
        write_json(directory/'study.json',study)
        # One neutral connectivity/usage check, recorded independently of science.
        from rsi.api import APIClient, Budget
        from rsi.common import Journal
        from rsi.reporting import token_report
        cfg=jobs[0]['settings']
        probe=directory/'api-probe';probe.mkdir()
        journal=Journal(probe/'events.jsonl')
        budget=Budget(dict(api_calls=1,api_tokens=None),journal)
        client=APIClient(cfg['credentials'],cfg['model'],budget,probe/'calls',journal,
                         max_output='model_max',request_timeout=API_REQUEST_TIMEOUT_SECONDS,max_retries=0)
        try:
            reply=client.complete([dict(role='user',content='Reply with the JSON object {"ok":true}.')],
                                  'probe','connectivity-v1',API_REQUEST_TIMEOUT_SECONDS)
            if json.loads(reply) != {'ok':True}:
                raise ValueError('Unexpected connectivity probe response')
            study['model']['provider_metadata']=client.model_limits
        finally:
            token_report(probe)
        study['preflight_status']='ready'
        write_json(directory/'study.json',study)
    except BaseException as exc:
        study.update(preflight_status='failed',preflight_error=repr(exc))
        write_json(directory/'study.json',study)
        raise
    return study


if __name__=='__main__':
    from rsi.providers import SUPPORTED_MODELS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','check'])
    p.add_argument('--out',required=True)
    p.add_argument('--evidence',help='Optional developer offline-test report (tests/run_offline.py) for the current code')
    p.add_argument('--config')
    p.add_argument('--tasks',nargs='+')
    p.add_argument('--solve-seconds',type=int)
    p.add_argument('--concurrent-modes',action='store_true')
    p.add_argument('--mode-order',nargs=2,choices=MODE_ORDER,default=MODE_ORDER)
    p.add_argument('--fresh-releases',action='store_true')
    p.add_argument('--cold-indices',action='store_true')
    p.add_argument('--model',choices=SUPPORTED_MODELS)
    p.add_argument('--credentials')
    p.add_argument('--jobs',nargs='+',help='Selected mode/task reruns')
    p.add_argument('--slot-offset',type=int,default=0)
    p.add_argument('--parent',action='append',default=[],help='mode/task=original run directory')
    p.add_argument('--rerun-reason')
    p.add_argument('--continue-from',help='Stopped sequential study: rerun its failed jobs, run its never-started ones')
    p.add_argument('--chain',action='append',help='NAME=TASK1,TASK2,...: run in order, inheriting distilled Agents and notes')
    p.add_argument('--chain-modes',nargs='+',choices=MODE_ORDER,default=['coevolve'])
    p.add_argument('--chain-inherit',action='append',default=[],
                   help='NAME=RUN: continue that chain after the sealed handoff of an existing run')
    p.add_argument('--chain-distillation-seconds',type=int)
    p.add_argument('--api-max-retries',type=int,help='Retries per provider call (attempts = N + 1); default: group configs')
    args=p.parse_args()
    if args.command=='check':
        if not args.config or not args.tasks:
            p.error('check requires --config and --tasks')
        protocol_check(args.config,args.tasks,args.evidence,record_path(args.out,area='provenance'))
    else:
        prepare(args.out,args.evidence,tasks=args.tasks,solve_seconds=args.solve_seconds,
                concurrent_modes=args.concurrent_modes,fresh_releases=args.fresh_releases,
                cold_indices=args.cold_indices,mode_order=args.mode_order,
                model=args.model,credentials=args.credentials,jobs=args.jobs,slot_offset=args.slot_offset,
                parents=dict(item.split('=',1) for item in args.parent) if args.jobs else None,
                rerun_reason=args.rerun_reason,
                chains=chain_specs(args.chain,dict(item.split('=',1) for item in args.chain_inherit))
                       if args.chain else None,
                chain_modes=args.chain_modes,chain_distillation_seconds=args.chain_distillation_seconds,
                continue_from=args.continue_from,api_max_retries=args.api_max_retries)
