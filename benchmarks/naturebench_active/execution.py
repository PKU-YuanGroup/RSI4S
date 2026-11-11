"""Isolated fit/predict execution shared by RSI and optional diagnostics."""
import json
import os
import shutil

from rsi.common import write_json
from rsi.sandbox import assert_plain_tree
from benchmarks.naturebench_active.common import connect, digest, sha256
from benchmarks.naturebench_active.adapters.pending import FEATURE_TASKS, LOSS_TASKS, TASKS as PENDING_TASKS

WORKER = '''import importlib.util,json,pickle,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('scientific_model',Path('model.py'))
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
records=[json.loads(line) for line in Path(__RECORDS__).open()]
if __MODE__ == 'fit':
    assert all(r.get('label') is not None for r in records)
    if __TASK__ in __LOSS_TASKS__:
        for name in ('validation_loss', 'predict'):
            if not callable(getattr(module, name, None)):
                raise ValueError('Loss-validation submissions must define callable '+name+'(model, records) before fitting')
        validation=[json.loads(line) for line in Path(__VALIDATION__).open()]
        assert all(r.get('label') is not None for r in validation)
        assert not {r['sample_id'] for r in validation}&{r['sample_id'] for r in records}
        model=module.fit(__TASK__,records,validation)
    else:
        model=module.fit(__TASK__,records)
    if __TASK__ == 'MAT_05':
        import math
        probes=[json.loads(line) for line in Path(__PROBES__).open()]
        assert all('label' not in r for r in probes)
        try:
            predictions=module.predict(model,probes)
            assert len(predictions)==len(probes), 'Prediction length mismatch'
            for value in predictions:
                assert set(value)=={'target'}, 'Output must contain target only'
                assert math.isfinite(float(value['target'])), 'Target must be finite'
        except Exception as exc:
            raise ValueError('Public-schema prediction check failed for 1–5 regions (core + 0–4 shells); keep feature width fixed and handle the fifth region: '+str(exc)) from exc
        print(json.dumps(dict(public_schema_probe=True,regions=[1,2,3,4,5],test_inputs_read=False,test_truth_read=False)))
    if __TASK__ == 'CHEM_09':
        import copy,numpy as np
        probe={k:v for k,v in records[0].items() if k!='label'}
        prediction=module.predict(model,[probe])[0]
        training_energies=np.array([r['label']['energy'] for r in records],float)
        assert abs(float(prediction['energy'])-training_energies.mean()) <= max(50.,100*training_energies.std()), 'Unlabeled inference lost the fitted energy offset'
        forces=np.asarray(prediction['forces'],float); errors=[]; epsilon=1e-4
        for atom,axis in [(0,0),(0,2),(1,1),(3,0),(5,2),(8,1)]:
            plus=copy.deepcopy(probe);minus=copy.deepcopy(probe)
            plus['inputs']['positions'][atom][axis]+=epsilon
            minus['inputs']['positions'][atom][axis]-=epsilon
            outputs=module.predict(model,[plus,minus])
            force=-(outputs[0]['energy']-outputs[1]['energy'])/(2*epsilon)
            errors.append(abs(force-forces[atom,axis]))
        assert max(errors)<1e-3, 'Learned forces disagree with negative energy gradient: '+str(errors)
        assert np.max(np.abs(forces.sum(axis=0)))<1e-5, 'Forces fail translation invariance'
        print(json.dumps(dict(training_only_scientific_probe=True,max_force_gradient_error=float(max(errors)),energy_offset_preserved=True)))
    with Path('model.pkl').open('wb') as f:pickle.dump(model,f)
elif __MODE__ == 'loss':
    import math
    assert all(r.get('label') is not None for r in records)
    with Path('model.pkl').open('rb') as f:model=pickle.load(f)
    losses=module.validation_loss(model,records)
    assert len(losses)==len(records), 'Return one validation loss per record'
    with Path('output/predictions.jsonl').open('w') as f:
        for row,value in zip(records,losses):
            assert not isinstance(value,bool) and math.isfinite(float(value)), 'Loss must be finite'
            f.write(json.dumps(dict(sample_id=row['sample_id'],prediction={'loss':float(value)}),allow_nan=False)+'\\n')
else:
    assert all('label' not in r for r in records)
    with Path('model.pkl').open('rb') as f:model=pickle.load(f)
    observed_exceptions={}
    scientific_filename=str(Path('model.py').resolve())
    def trace_exceptions(frame,event,arg):
        if frame.f_code.co_filename != scientific_filename:return None
        frame.f_trace_lines=False
        if event=='exception':
            kind,value,_=arg
            key=str(frame.f_lineno)+':'+kind.__name__
            if key not in observed_exceptions and len(observed_exceptions)<20:
                observed_exceptions[key]=dict(line=frame.f_lineno,exception=kind.__name__,message=str(value)[:240],count=0)
            if key in observed_exceptions:observed_exceptions[key]['count']+=1
        return trace_exceptions
    if __MODE__ == 'generation_probe':sys.settrace(trace_exceptions)
    try:
        result=module.predict(model,records)
    finally:
        if __MODE__ == 'generation_probe':
            sys.settrace(None)
            Path('output/generation_diagnostics.json').write_text(json.dumps(dict(
                observed_model_exceptions=list(observed_exceptions.values()),
                interpretation='Observed exceptions can be intentionally handled; this is an engineering diagnostic, not a method-validity verdict.')))
    assert len(result)==len(records), 'Prediction length mismatch'
    with Path('output/predictions.jsonl').open('w') as f:
        for row,value in zip(records,result):
            f.write(json.dumps(dict(sample_id=row['sample_id'],prediction=value),allow_nan=False)+'\\n')
print(json.dumps(dict(mode=__MODE__,records=len(records))))
'''


def development_rows(evidence, role, *, deadline=None, enrich=True):
    # Only the curated public DB is read here. Q stays hidden, including Q in
    # an internal validation group; no label is acquired for evaluation.
    from .preparation import database, check
    records = []
    with database(evidence.pool, deadline) as db:
        for row in db.execute('SELECT sample_id,instance,group_id,inputs,label FROM samples WHERE acquired=1 AND role=? ORDER BY instance,ordinal', (role,)):
            check(deadline)
            r = dict(row)
            r['inputs'] = json.loads(r['inputs']); r['label'] = json.loads(r['label'])
            records.append(r)
    if enrich:
        enrich_public_inputs(records,evidence.pool,evidence.manifest['task_id'],deadline=deadline)
    check(deadline)
    return records


def enrich_public_inputs(records,pool,task,*,deadline=None):
    """Resolve explicitly public descriptor references, never organizer labels."""
    from .preparation import database, check
    check(deadline)
    if task=='LIFE_04':
        from benchmarks.naturebench_active.adapters.graph import enrich, network_features
        enrich(records,network_features(pool,deadline=deadline),deadline=deadline)
        return
    if task!='MAT_03':return
    assets = {}
    with database(pool,deadline) as db:
        for r in db.execute('SELECT asset_id,payload FROM assets'):
            check(deadline)
            assets[r['asset_id']] = json.loads(r['payload'])
    decoded={}
    for r in records:
        check(deadline)
        for kind,field in [('molecule','molecule_features'),('framework','framework_features')]:
            reference=r['inputs'][field]
            if reference not in assets:raise ValueError('Missing public descriptor '+reference)
            columns=assets[kind+'/columns'];values=assets[reference]
            if len(columns)!=len(values):raise ValueError('Descriptor column mismatch')
            if reference not in decoded:decoded[reference]=dict(zip(columns,values))
            r['inputs'][kind+'_descriptors']=decoded[reference]


def attach_feature_paths(evidence,records,role,destination,*,deadline=None):
    from .preparation import check, file_hash, copy_file
    check(deadline)
    if evidence.manifest['task_id'] in PENDING_TASKS:
        from benchmarks.naturebench_active.adapters.pending import copy_resources
        paths=copy_resources(evidence.package,destination,evidence.manifest,deadline=deadline)
        check(deadline)
        if paths:
            for row in records:row['inputs']['resource_paths']=paths
    if evidence.manifest['task_id'] not in ('MAT_02','LIFE_07','LIFE_16',*FEATURE_TASKS):return
    from benchmarks.naturebench_active.common import digest
    manifest_path=destination/'input_files_manifest.json'
    files=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for relative,expected in files.items():
        existing=(destination/relative).resolve()
        if not existing.is_relative_to(destination.resolve()) or file_hash(existing,deadline)!=expected:
            raise ValueError('Existing input attachment manifest changed')
    by_id={r['sample_id']:r for r in records}
    for start in range(0,len(records),128):
        check(deadline)
        ids=[r['sample_id'] for r in records[start:start+128]]
        for row in evidence.fetch(role,ids,limit=128,deadline=deadline if deadline is not None else float('inf'))['records']:
            paths=[]
            for relative in row.get('files',[]):
                source=evidence.problem/relative
                target=destination/'files'/digest(row['sample_id'])/source.name
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    if file_hash(target,deadline)!=file_hash(source,deadline):raise ValueError('Sanitized input attachment changed')
                else:copy_file(source,target,deadline);target.chmod(0o444)
                paths.append(str(target));files[str(target.relative_to(destination))]=file_hash(target,deadline)
            by_id[row['sample_id']]['inputs']['feature_paths']=paths
    write_json(manifest_path,files)


def scientific_worker(python, task, mode, problem, workspace, logs, seconds, resource_limits=None, *, gpu):
    """Confine scientific code; only frozen prediction may omit its deadline."""
    from rsi.sandbox import Sandbox
    if mode not in ('fit', 'predict', 'loss', 'generation_probe') or (mode in ('loss', 'generation_probe') and task not in LOSS_TASKS):
        raise ValueError('Unknown scientific worker mode')
    if seconds is None and mode != 'predict':
        raise ValueError('Only frozen prediction may run without a time limit')
    if seconds is not None and seconds <= 0:
        raise TimeoutError('Original research deadline has expired')
    if mode == 'fit' and task == 'MAT_05':
        from benchmarks.naturebench_active.adapters.nanoparticle import schema_probes
        from benchmarks.naturebench_active.common import jsonl
        jsonl(problem / 'schema_probes.jsonl', schema_probes())
    logs = logs / ('attempt-%03d' % (len(list(logs.glob('attempt-*'))) + 1))
    allowed = sorted(os.sched_getaffinity(0))
    limits = dict(resource_limits or {})
    cpus = sorted(limits.get('cpu_affinity') or allowed)
    if not set(cpus) <= set(allowed):
        raise ValueError('Assigned CPUs outside host affinity')
    threads = limits.get('threads', len(cpus))
    if type(threads) is not int or not 1 <= threads <= len(cpus):
        raise ValueError('Worker threads must fit the assigned CPU allocation')
    # Library thread pools also consume RLIMIT_NPROC; match local Sandbox headroom.
    limits.update(threads=threads,
                  max_processes=min(512, limits.get('max_processes', 512)), cpu_affinity=cpus)
    code = WORKER.replace('__RECORDS__', repr(str(problem / 'records.jsonl')))
    code = code.replace('__PROBES__', repr(str(problem / 'schema_probes.jsonl')))
    code = code.replace('__VALIDATION__', repr(str(problem / 'validation_records.jsonl')))
    code = code.replace('__LOSS_TASKS__', repr(LOSS_TASKS))
    code = code.replace('__MODE__', repr(mode)).replace('__TASK__', repr(task))
    sandbox = Sandbox(python, problem, workspace, logs, gpu=gpu,
                      uid=3000000 + os.getpid(), limits=limits, frozen_workspace=mode != 'fit')
    result = sandbox.run(code, seconds)
    result.update(mode=mode, uid=sandbox.uid, gpu=gpu, gpu_uuid=sandbox.gpu_uuid,
                  confinement='RSI UID + Landlock + seccomp; '+('assigned GPU visible' if gpu is not None else 'CPU only'))
    write_json(logs / 'result.json', result)
    if result['returncode']:
        # Replay logs are private to the host. Return enough of this worker's
        # own diagnostics for Task/Meta repair without opening that directory.
        # Put the authoritative termination reason before the output: a warning
        # in a timed-out process is not evidence that the warning caused a crash.
        failure = dict(mode=mode, returncode=result['returncode'],
                       reason=result.get('reason') or 'process_exit',
                       seconds=result['seconds'], time_limit_seconds=seconds)
        output = result.get('stdout', '').encode()
        clipped = len(output) > 2400
        tail = output[-2400:].decode(errors='replace').strip()
        message = 'Scientific worker failed: ' + json.dumps(failure)
        if tail:
            message += '\nWorker output tail' + (' (truncated)' if clipped else '') + ':\n' + tail
        message += '\nOrganizer log: ' + result['log']
        raise RuntimeError(message)
    assert_plain_tree(workspace)
    return result


def cpu_worker(python, task, mode, problem, workspace, logs, seconds, resource_limits=None):
    """Explicit CPU diagnostic/fixture entry; formal RSI uses scientific_worker."""
    return scientific_worker(python, task, mode, problem, workspace, logs, seconds,
                             resource_limits, gpu=None)
