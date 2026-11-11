"""Organizer-only RNA cost profiling; sampled timings are never benchmark scores.

Use complete frozen predictions. Select across CM size and sequence-length
strata, including their tails. Censored probes retain a lower bound, never an
invented completion time. A resource plan must budget research, final selection,
frozen inference, queueing, native scoring and review separately.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from benchmarks.naturebench_active.common import digest, sha256
from benchmarks.naturebench_active.scoring.process import NativeProcesses
from benchmarks.naturebench_active.scoring.resources import terminal_resources
from benchmarks.naturebench_active.scoring.rna_terminal import allocation, families, score_family
from benchmarks.naturebench_active.scoring.terminal import frozen_result
from rsi.common import write_json, digest as cache_digest


def cm_headers(path):
    result, current = {}, {}
    with Path(path).open() as stream:
        for line in stream:
            words = line.split(maxsplit=1)
            if not words:
                continue
            key = words[0]
            if key == '//':
                # Infernal bundles also contain HMMER records with the same
                # accession and their own // terminator. They are not CMs.
                if all(name in current for name in ('ACC','CLEN','STATES','W')):
                    result[current['ACC']] = current
                current = {}
            elif len(words) == 2 and key in ('ACC', 'CLEN', 'STATES', 'W'):
                current[key] = words[1].strip() if key == 'ACC' else int(words[1])
    return result


def size_bin(number):
    return 'short' if number <= 100 else 'medium' if number <= 500 else 'long'


def inventory(native):
    jobs = families(native)
    headers = {path:cm_headers(path) for path in {j['cm_bundle'] for j in jobs}}
    unique = {}
    for job in jobs:
        header = headers[job['cm_bundle']][job['family']]
        job = dict(job, cm_length=header['CLEN'], cm_states=header['STATES'], cm_window=header['W'])
        job['stratum'] = size_bin(job['cm_length'])+'/'+size_bin(job['length_max'])
        # This is a sampling priority only, not a runtime complexity formula.
        job['tail_priority'] = job['cm_states']*job['length_max']
        key = digest([job['family'],job['fasta_sha256'],job['cm_bundle_sha256']])
        unique.setdefault(key, job)
    groups = {}
    for key, job in unique.items():
        groups.setdefault(job['stratum'], []).append(key)
    selected = set()
    for keys in groups.values():
        ordered = sorted(keys, key=lambda k:(unique[k]['tail_priority'],k))
        selected.update([ordered[len(ordered)//2], ordered[-1]])
    selected.add(max(unique, key=lambda k:unique[k]['cm_length']))
    selected.add(max(unique, key=lambda k:unique[k]['length_max']))
    return unique, sorted(selected), {k:len(v) for k,v in groups.items()}


def sample_sequences(sequences, count):
    ordered = sorted(sequences, key=lambda row:(len(row[1]),row[0]))
    if count < 2:
        raise ValueError('At least two sequences are required to cover length endpoints')
    indices = sorted({round(i*(len(ordered)-1)/(count-1)) for i in range(count)})
    return [ordered[index] for index in indices]


def cost_scenarios(jobs, measurements, workers):
    """Empirical scenarios, not confidence intervals or promises about the tail."""
    rates, censored = {}, []
    for key, row in measurements.items():
        if row['status'] != 'completed':
            censored.append(key); continue
        # A few selected endpoints exaggerate rare long sequences and may not
        # occupy every Infernal thread. Startup overhead also is not linear in N.
        # Use them to expose expensive tails, never multiply them into an ETA.
        if not row.get('full_family'):
            continue
        job = jobs[key]
        rates.setdefault(job['stratum'], []).append(row['seconds']/row['sequences'])
    missing = sorted({j['stratum'] for j in jobs.values()}-rates.keys())
    estimate = None
    if not missing and not censored:
        durations = [[], []]
        for key, job in jobs.items():
            row = measurements.get(key)
            if row and row.get('full_family'):
                low = high = row['seconds']
            else:
                values = rates[job['stratum']]
                low, high = min(values)*len(job['sequence_ids']), max(values)*len(job['sequence_ids'])
            durations[0].append(low);durations[1].append(high)
        def schedule(values):
            heap = [0.]*workers
            for seconds in sorted(values,reverse=True):
                current = heapq.heappop(heap);heapq.heappush(heap,current+seconds)
            return max(heap)
        estimate = [schedule(values) for values in durations]
    return dict(empirical_wall_scenarios_seconds=estimate,unknown_strata=missing,censored_or_failed_probes=censored,
                guaranteed_completion_time=None,
                interpretation='Conditional timing scenarios under the same threads/workers and input strata; '
                    'not a statistical interval or upper bound. Sequence difficulty and contention can dominate. '
                    'Only complete-family observations inform these scenarios. Bounded subset probes are '
                    'not extrapolated. Any missing full-family stratum or censored probe suppresses the estimate.')


def load_native(run, source):
    frozen = frozen_result(run)
    if frozen['task'] != 's41592-023-02148-8':
        raise ValueError('RNA capacity profiling requires frozen RNA generation predictions')
    evaluator = Path(source)/frozen['task']/'evaluation/evaluator.py'
    release = json.loads((Path(run)/'release.json').read_text())
    if release['task_contract'].get(str(evaluator.resolve())) != sha256(evaluator):
        raise ValueError('Native evaluator differs from the frozen release')
    os.environ['OUTPUT_DIR'] = str(Path(run)/'final/final_output')
    os.environ['PATH'] = str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    spec=importlib.util.spec_from_file_location('capacity_native',evaluator)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    return native, frozen


def profile(run, source, directory, *, observed=None, sequences=4, seconds=180,
            threads=6, workers=2, cpu_affinity=None, memory_budget_mib=None, sample_cache_root=None,
            pool_path=None):
    if type(sequences) is not int or sequences < 2 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Require at least two probe sequences and a positive finite timeout')
    run, directory = Path(run).resolve(),Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=False)
    native, frozen = load_native(run, source)
    jobs, selected, strata = inventory(native)
    sources = [Path(__file__),Path(score_family.__code__.co_filename),Path(NativeProcesses.run.__code__.co_filename)]
    code = {str(p):sha256(p) for p in sources}
    executables = {name:shutil.which(name) for name in ('cmalign','cmfetch')}
    if not all(executables.values()):raise ValueError('Missing Infernal executables')
    identity = dict(adapter_sha256=code[str(Path(score_family.__code__.co_filename))],
        evaluator_sha256=sha256(Path(native.__file__)),
        **{name+'_sha256':sha256(Path(path)) for name,path in executables.items()})
    measurements = {}
    if observed is not None:
        observation = json.loads(Path(observed).read_text())
        for key, job in jobs.items():
            for row in observation.get('completed',{}).values():
                binding = row.get('binding',{})
                if (all(binding.get(k)==v for k,v in identity.items()) and binding.get('threads')==threads
                        and binding.get('family')==job['family'] and binding.get('fasta_sha256')==job['fasta_sha256']
                        and row.get('n_sequences')==len(job['sequence_ids'])):
                    entry=Path(row['cache_entry'])
                    model=entry.parents[1]/'models'/cache_digest(dict(bundle=job['cm_bundle_sha256'],family=job['family'],
                        cmfetch=identity['cmfetch_sha256']))/'family.cm'
                    if not model.is_file() or sha256(model)!=binding.get('cm_sha256'):
                        continue
                    saved=json.loads((entry/'result.json').read_text())
                    if saved['binding']!=binding or saved['execution']!=row['execution']:
                        continue
                    measurements[key] = dict(status='completed',seconds=row['execution']['seconds'],
                        sequences=row['n_sequences'],full_family=True,source=str(observed),
                        observed_receipt_sha256=digest(row))
                    break
    public_jobs = {k:{name:value for name,value in j.items() if name!='sequence_ids'} for k,j in jobs.items()}
    report = dict(status='profiling',run=str(run),frozen_identity=digest(frozen),identity=identity,
                  settings=dict(threads=threads,workers=workers,cpu_cores=threads*workers),
                  strata=strata,selected=selected,jobs=public_jobs,measurements=measurements,
                  probe_sequences=sequences,probe_timeout_seconds=seconds,started_at=time.time(),
                  source_sha256=code,feedback_to_research=False,benchmark_score_computed=False)
    write_json(directory/'profile.json',report)
    def probe(key, processes):
        job = jobs[key]; path=directory/'samples'/key;path.mkdir(parents=True)
        records=sample_sequences(native.validate_fasta(job['fasta'],job['family']),sequences)
        fasta=path/(job['family']+'.fa');fasta.write_text(''.join('>'+name+'\n'+sequence+'\n' for name,sequence in records))
        sample=dict(job,fasta=str(fasta),fasta_sha256=sha256(fasta),sequence_ids=[r[0] for r in records])
        started=time.monotonic()
        try:
            result=score_family(native,sample,Path(sample_cache_root) if sample_cache_root else directory/'cache',executables,identity,threads,seconds,processes,
                lambda job,receipt:write_json(path/'started.json',receipt))
            return dict(status='completed',seconds=result['execution']['seconds'],sequences=len(records),
                        full_family=False,cache_entry=result['cache_entry'],cache_hit=result['cache_hit'],
                        length_min=min(len(s) for _,s in records),length_max=max(len(s) for _,s in records))
        except subprocess.TimeoutExpired as exc:
            return dict(status='censored',seconds=time.monotonic()-started,sequences=len(records),
                        duration_lower_bound_seconds=seconds,error=str(exc),full_family=False)
        except Exception as exc:
            return dict(status='failed',seconds=time.monotonic()-started,sequences=len(records),
                        error=f'{type(exc).__name__}: {exc}',full_family=False)
    memory=workers*(threads*8192+1024)
    with terminal_resources(run,device='cpu',cpu_cores=threads*workers,cpu_affinity=cpu_affinity,
        memory_mib=memory,memory_budget_mib=memory_budget_mib,queue_seconds=600,pool_path=pool_path) as resources:
        report['resources']=resources
        allocation(resources['cpu_threads'],threads,workers,memory)
        with NativeProcesses().managed() as processes:
            executor=ThreadPoolExecutor(max_workers=workers)
            try:
                pending={executor.submit(probe,key,processes):key for key in selected if key not in measurements}
                for future in as_completed(pending):
                    measurements[pending[future]]=future.result();write_json(directory/'profile.json',report)
            finally:
                processes.stop()
                executor.shutdown(wait=True,cancel_futures=True)
    report.update(status='completed',finished_at=time.time(),scenarios=cost_scenarios(jobs,measurements,workers),
        representative_coverage_complete=all(key in measurements for key in selected),
        probe_failures=[k for k,v in measurements.items() if v['status']=='failed'])
    if frozen_result(run)!=frozen or any(sha256(Path(p))!=value for p,value in code.items()):
        raise ValueError('Frozen inputs or capacity profiler code changed during profiling')
    report['frozen_inputs_unchanged']=True
    write_json(directory/'profile.json',report)
    return report


def verify_profile(path, frozen, source, resources, scoring):
    """A new queued RNA score requires a representative profile of its own inputs."""
    profile_path=Path(path)
    report=json.loads(profile_path.read_text())
    if (report.get('status')!='completed' or not report.get('representative_coverage_complete')
            or report.get('probe_failures') or not report.get('frozen_inputs_unchanged')):
        raise ValueError('RNA capacity profile is incomplete or has failed probes')
    if report.get('frozen_identity')!=digest(frozen):
        raise ValueError('RNA capacity profile belongs to different frozen predictions')
    if report['identity']['evaluator_sha256']!=sha256(Path(source)):
        raise ValueError('RNA capacity profile uses a different native evaluator')
    for source_path, expected in report['source_sha256'].items():
        if sha256(Path(source_path))!=expected:raise ValueError('Capacity profiling code changed; refresh the profile using bound sample checkpoints')
    wanted=dict(threads=scoring.get('rna_threads',6),workers=scoring.get('rna_workers',4),cpu_cores=resources.get('cpu_cores'))
    if report.get('settings')!=wanted:
        raise ValueError('RNA scoring allocation differs from the capacity profile; regenerate its scheduling scenarios')
    return dict(profile_sha256=sha256(profile_path),
                scenarios=report['scenarios'],settings=report['settings'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','source','directory'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--observed')
    parser.add_argument('--sample-cache-root')
    parser.add_argument('--pool-path',help='Shared CPU/GPU reservation registry; set explicitly when running from an isolated code snapshot')
    parser.add_argument('--cpus',type=lambda s:list(map(int,s.split(','))))
    parser.add_argument('--threads',type=int,default=6);parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--sequences',type=int,default=4);parser.add_argument('--seconds',type=float,default=180)
    parser.add_argument('--memory-budget-mib',type=int)
    a=parser.parse_args()
    result=profile(a.run,a.source,a.directory,observed=a.observed,sequences=a.sequences,seconds=a.seconds,
                   threads=a.threads,workers=a.workers,cpu_affinity=a.cpus,memory_budget_mib=a.memory_budget_mib,
                   sample_cache_root=a.sample_cache_root,pool_path=a.pool_path)
    print(json.dumps({k:result[k] for k in ('status','strata','scenarios','representative_coverage_complete','probe_failures')}))
