"""Parallel, resumable orchestration of the unchanged Infernal RNA score.

This is host-only Test scoring. Each cache entry binds the exact FASTA, extracted
CM, evaluator, executable bytes and Infernal arguments. Aggregation follows the
upstream evaluator's family order, numpy mean and GA normalization.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import math
import os
from pathlib import Path
import shutil
import time
import uuid
import threading

from benchmarks.naturebench_active.common import sha256
from benchmarks.naturebench_active.scoring.process import NativeProcesses
from rsi.common import digest, write_json
import json


@contextmanager
def entry_lock(path, processes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        while True:
            if processes.stopped:
                raise InterruptedError('RNA scoring was interrupted while waiting for a checkpoint')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(.1)
        yield


def available_memory_mib():
    """Use both Linux host availability and the current cgroup's remaining RAM."""
    available = next(int(line.split()[1])//1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            group = Path('/sys/fs/cgroup')/line[3:].lstrip('/')
            while group.is_relative_to('/sys/fs/cgroup'):
                try:
                    limit = (group/'memory.max').read_text().strip()
                    if limit != 'max':
                        current = int((group/'memory.current').read_text())
                        stat = dict(line.split() for line in (group/'memory.stat').read_text().splitlines())
                        # Inactive file pages can be reclaimed by this cgroup.
                        # Treating the entire page cache as unreclaimable made a
                        # mostly cached 800-GiB group appear to have only 80 GiB.
                        working = max(0, current-int(stat.get('inactive_file', 0)))
                        available = min(available, max(0, int(limit)-working)//(1024**2))
                except FileNotFoundError:
                    pass
                if group == Path('/sys/fs/cgroup'):
                    break
                group = group.parent
    return available


def allocation(cpu_budget, threads, workers, memory_mib):
    for name, value in [('cpu_budget', cpu_budget), ('threads', threads), ('workers', workers), ('memory_mib', memory_mib)]:
        if type(value) is not int or value < 1:
            raise ValueError(name+' must be a positive integer')
    if cpu_budget > len(os.sched_getaffinity(0)):
        raise ValueError('RNA CPU budget exceeds the allowed affinity')
    # --mxsize is per worker thread. Reserve its full matrix allowance plus
    # process overhead, instead of admitting many large CMs from thread count alone.
    per_family_mib = 8192*threads+1024
    actual = min(workers, cpu_budget//threads, memory_mib//per_family_mib)
    if actual < 1:
        raise ValueError(f'RNA needs at least {threads} CPUs and {per_family_mib} MiB per family; '
                         'allocate more resources or lower threads per family')
    return dict(workers=actual, requested_workers=workers, threads_per_family=threads,
                cpu_budget=cpu_budget, memory_budget_mib=memory_mib,
                reserved_mib_per_family=per_family_mib)


def families(native):
    jobs = []
    headers = {}
    for instance in native.INSTANCES:
        directory = Path(native.OUTPUT_DIR)/instance
        paths = sorted(p for p in directory.iterdir() if p.suffix in ('.fa', '.fasta'))
        if not paths:
            raise ValueError('No FASTA files: '+str(directory))
        indexed = {p.stem: p for p in paths}
        if len(indexed) != len(paths):
            raise ValueError('Duplicate RNA family output: '+instance)
        if instance == 'rna_families_18':
            if set(indexed) != set(native.FAMILIES_18):
                raise ValueError('RNA 18-family output differs from the expected family set')
            paths = [indexed[family] for family in native.FAMILIES_18]
        cm = Path(native.GROUND_TRUTH_DIR)/instance/'Rfam.cm'
        bundle_hash = sha256(cm)
        if bundle_hash not in headers:
            lengths, accession = {}, None
            with cm.open() as stream:
                for line in stream:
                    fields = line.split()
                    if fields and fields[0] == 'ACC':
                        accession = fields[1]
                    elif fields and fields[0] == 'CLEN' and accession:
                        lengths[accession] = int(fields[1])
            headers[bundle_hash] = lengths
        ga = native.parse_ga_thresholds(str(cm)) if instance == 'rna_families_18' else {}
        for path in paths:
            sequences = native.validate_fasta(str(path), path.stem)
            ids = [row[0] for row in sequences]
            if len(set(ids)) != len(ids):
                raise ValueError('RNA sequence identifiers must be unique: '+str(path))
            if any(not sequence for _, sequence in sequences):
                raise ValueError('Empty RNA sequence: '+str(path))
            jobs.append(dict(instance=instance, family=path.stem, fasta=str(path),
                             fasta_sha256=sha256(path), cm_bundle=str(cm), cm_bundle_sha256=bundle_hash,
                             sequence_ids=ids, ga_threshold=ga.get(path.stem),
                             cm_length=headers[bundle_hash].get(path.stem, 1),
                             total_bases=sum(len(s) for _, s in sequences),
                             length_min=min(len(s) for _, s in sequences),
                             length_max=max(len(s) for _, s in sequences)))
    return jobs


def _scores(native, path, ids):
    scores = native.parse_score_file(str(path))
    if set(scores) != set(ids) or len(scores) != len(ids):
        raise ValueError(f'Incomplete sequence scores: expected {len(ids)}, parsed {len(scores)}')
    if any(not math.isfinite(score) for score in scores.values()):
        raise ValueError('Nonfinite RNA bit score')
    return scores


def score_family(native, job, cache, executables, identity, threads, seconds, processes, on_start):
    """A failed attempt never replaces a successful checkpoint or its evidence."""
    extraction = cache/'models'/digest(dict(bundle=job['cm_bundle_sha256'], family=job['family'],
                                            cmfetch=identity['cmfetch_sha256']))
    with entry_lock(extraction/'lock', processes):
        model = extraction/'family.cm'
        manifest = extraction/'result.json'
        saved = json.loads(manifest.read_text()) if manifest.exists() else {}
        if not model.exists() or saved.get('sha256') != sha256(model):
            temporary = extraction/'family.cm.tmp'
            with temporary.open('w') as out, (extraction/'cmfetch.log').open('w') as err:
                processes.run([executables['cmfetch'], job['cm_bundle'], job['family']],
                              stdout=out, stderr=err, timeout=min(seconds, 120))
            temporary.replace(model)
            write_json(manifest, dict(sha256=sha256(model)))
        cm_hash = sha256(model)
    binding = dict(**identity, fasta_sha256=job['fasta_sha256'], cm_sha256=cm_hash,
                   family=job['family'], flags=['--notrunc', '--mxsize', '8192'], threads=threads)
    entry = cache/'scores'/digest(binding)
    with entry_lock(entry/'lock', processes):
        result_path = entry/'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            score_path = entry/result['score_file']
            if result['binding'] == binding and score_path.is_file() and sha256(score_path) == result['scores_sha256']:
                scores = _scores(native, score_path, job['sequence_ids'])
                return dict(result, cache_hit=True, cache_entry=str(entry),
                            mean_bit_score=float(native.np.mean(list(scores.values()))))
        attempt = entry/'attempts'/uuid.uuid4().hex
        attempt.mkdir(parents=True)
        command = [executables['cmalign'], '--notrunc', '--mxsize', '8192', '--sfile',
                   str(attempt/'scores.txt'), '--cpu', str(threads), str(model), job['fasta']]
        started = time.time()
        receipt = dict(binding=binding, started_at=started, command=command, status='running')
        write_json(attempt/'status.json', receipt)
        on_start(job, dict(started_at=started, attempt=str(attempt), threads=threads))
        try:
            with (attempt/'alignment.stk').open('w') as out, (attempt/'stderr.log').open('w') as err:
                execution = processes.run(command, stdout=out, stderr=err, timeout=seconds)
            execution.update(started_at=started, finished_at=time.time())
            scores = _scores(native, attempt/'scores.txt', job['sequence_ids'])
            if sha256(Path(job['fasta'])) != job['fasta_sha256'] or sha256(model) != cm_hash:
                raise ValueError('RNA scoring inputs changed during execution')
            result = dict(binding=binding, n_sequences=len(scores),
                          mean_bit_score=float(native.np.mean(list(scores.values()))),
                          scores_sha256=sha256(attempt/'scores.txt'),
                          score_file=str((attempt/'scores.txt').relative_to(entry)), execution=execution)
            write_json(attempt/'status.json', dict(receipt, status='completed', finished_at=time.time()))
            write_json(result_path, result)
            return dict(result, cache_hit=False, cache_entry=str(entry))
        except BaseException as exc:
            write_json(attempt/'status.json', dict(receipt, status='failed', finished_at=time.time(),
                                                  error=f'{type(exc).__name__}: {exc}'))
            raise


def partition_family(job, cache, chunk_size):
    """Preserve FASTA records verbatim, including IDs, case and descriptions."""
    if not chunk_size or len(job['sequence_ids']) <= chunk_size:
        return [job]
    raw = Path(job['fasta']).read_text()
    records = []
    for line in raw.splitlines(keepends=True):
        if line.startswith('>'):
            records.append(line)
        elif records:
            records[-1] += line
        elif line.strip():
            raise ValueError('Unexpected data before the first FASTA record')
    if len(records) != len(job['sequence_ids']):
        raise ValueError('FASTA partition disagrees with native validation')
    chunks = []
    for offset in range(0, len(records), chunk_size):
        text = ''.join(records[offset:offset+chunk_size])
        # Content-addressed input paths are independent of instance and run.
        directory = cache/'inputs'/digest(dict(fasta=job['fasta_sha256'],
                                                chunk_size=chunk_size, offset=offset))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/(job['family']+'.fa')
        temporary = directory/(uuid.uuid4().hex+'.tmp')
        temporary.write_text(text)
        temporary.replace(path)
        chunks.append(dict(job, fasta=str(path), fasta_sha256=sha256(path),
            sequence_ids=job['sequence_ids'][offset:offset+chunk_size],
            total_bases=sum(len(''.join(r.splitlines()[1:])) for r in records[offset:offset+chunk_size]),
            chunk_index=len(chunks)))
    if sha256(Path(job['fasta'])) != job['fasta_sha256']:
        raise ValueError('RNA FASTA changed during partitioning')
    return chunks


def merge_chunks(native, job, results, cache):
    """Rebuild the native sequence order; never average chunk means."""
    if len(results) == 1 and results[0]['binding']['fasta_sha256'] == job['fasta_sha256']:
        return results[0]
    rows, receipts = {}, []
    for result in results:
        path = Path(result['cache_entry'])/result['score_file']
        if sha256(path) != result['scores_sha256']:
            raise ValueError('RNA chunk scores changed before aggregation')
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.split()
            if len(fields) >= 8:
                if fields[1] in rows:
                    raise ValueError('Duplicate RNA score across chunks')
                rows[fields[1]] = fields
        receipts.append(dict(cache_entry=result['cache_entry'], scores_sha256=result['scores_sha256'],
                             score_file=result['score_file'], binding=result['binding']))
    if set(rows) != set(job['sequence_ids']):
        raise ValueError('RNA chunk aggregation lacks complete sequence coverage')
    binding = dict(fasta_sha256=job['fasta_sha256'], chunks=receipts, aggregation='native_input_order_v1')
    entry = cache/'aggregates'/digest(binding)
    entry.mkdir(parents=True, exist_ok=True)
    path = entry/(uuid.uuid4().hex+'.scores.txt')
    with path.open('w') as out:
        for index, name in enumerate(job['sequence_ids'], 1):
            fields = rows[name]
            fields[0] = str(index)
            out.write(' '.join(fields)+'\n')
    scores = _scores(native, path, job['sequence_ids'])
    return dict(binding=binding, cache_entry=str(entry), score_file=path.name,
                scores_sha256=sha256(path), n_sequences=len(scores),
                mean_bit_score=float(native.np.mean(list(scores.values()))),
                cache_hit=all(r['cache_hit'] for r in results), n_chunks=len(results),
                execution=dict(started_at=min(r['execution']['started_at'] for r in results),
                               finished_at=max(r['execution']['finished_at'] for r in results),
                               summed_chunk_seconds=sum(r['execution']['seconds'] for r in results)))


def run_evaluation(native, *, cache_root, progress_path, cpu_budget, threads=6, workers=4,
                   memory_mib=None, family_seconds=3600, chunk_size=64):
    if not math.isfinite(family_seconds) or family_seconds <= 0:
        raise ValueError('family_seconds must be positive and finite')
    if type(chunk_size) is not int or chunk_size < 0:
        raise ValueError('chunk_size must be a nonnegative integer (0 disables partitioning)')
    memory_mib = min(memory_mib, available_memory_mib()) if memory_mib is not None else int(available_memory_mib()*.8)
    resources = allocation(cpu_budget, threads, workers, memory_mib)
    executables = {name: shutil.which(name) for name in ('cmalign', 'cmfetch')}
    if not all(executables.values()):
        raise RuntimeError('Infernal cmalign and cmfetch must be on PATH')
    identity = dict(adapter_sha256=sha256(Path(__file__)), evaluator_sha256=sha256(Path(native.__file__)),
                    **{name+'_sha256': sha256(Path(path)) for name, path in executables.items()})
    jobs = families(native)
    progress = dict(status='running', started_at=time.time(), resources=resources, identity=identity,
                    scheduling=dict(policy='largest_chunk_first_v1', chunk_size=chunk_size,
                                    cost_proxy='CM length times chunk bases; not a runtime estimate',
                                    timeout_policy='family deadline from first chunk dispatch'),
                    total_families=len(jobs), completed={}, failed={}, running={}, completed_chunks={})
    write_json(progress_path, progress)
    progress_lock = threading.Lock()
    last_publish = 0.
    def publish(force=False):
        nonlocal last_publish
        now = time.monotonic()
        if force or now-last_publish >= 1.:
            progress['updated_at'] = time.time()
            write_json(progress_path, progress)
            last_publish = now

    def chunk_key(job):
        key = job['instance']+'/'+job['family']
        return key+f"/chunk_{job['chunk_index']:05d}" if 'chunk_index' in job else key

    def on_start(job, receipt):
        with progress_lock:
            progress['running'][chunk_key(job)] = receipt
            publish()
    cache = Path(cache_root).resolve()
    unique = {}
    for job in jobs:
        key = (job['family'], job['fasta_sha256'], job['cm_bundle_sha256'])
        unique.setdefault(key, []).append(job)
    work, results, deadlines = [], {}, {}
    for key, aliases in unique.items():
        chunks = partition_family(aliases[0], cache, chunk_size)
        results[key] = [None]*len(chunks)
        work.extend((key, index, chunk) for index, chunk in enumerate(chunks))
    work.sort(key=lambda row: row[2]['cm_length']*row[2]['total_bases'], reverse=True)
    progress['total_chunks'] = len(work)
    def execute(key, chunk, processes):
        with progress_lock:
            deadline = deadlines.setdefault(key, time.monotonic()+family_seconds)
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('RNA family deadline expired; completed chunks remain reusable')
        return score_family(native, chunk, cache, executables, identity, threads,
                            remaining, processes, on_start)

    with NativeProcesses().managed() as processes:
        executor = ThreadPoolExecutor(max_workers=resources['workers'])
        try:
            pending = {executor.submit(execute, key, chunk, processes): (key, index, chunk)
                       for key, index, chunk in work}
            for future in as_completed(pending):
                family_key, index, chunk = pending[future]
                aliases = unique[family_key]
                keys = [job['instance']+'/'+job['family'] for job in aliases]
                try:
                    result = future.result()
                    results[family_key][index] = result
                    with progress_lock:
                        progress['completed_chunks'][chunk_key(chunk)] = result
                    if all(row is not None for row in results[family_key]):
                        merged = merge_chunks(native, aliases[0], results[family_key], cache)
                        with progress_lock:
                            for key in keys:
                                progress['completed'][key] = dict(merged, shared_family_outputs=keys)
                except Exception as exc:
                    with progress_lock:
                        for key in keys:
                            progress['failed'][key] = f'{type(exc).__name__}: {exc}'
                with progress_lock:
                    progress['running'].pop(chunk_key(chunk), None)
                    publish()
        except BaseException:
            processes.stop()
            with progress_lock:
                progress['status'] = 'interrupted'
                publish(force=True)
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    if any(sha256(Path(path)) != identity[name+'_sha256'] for name, path in executables.items()):
        raise ValueError('Infernal executable changed during scoring')
    if any(sha256(Path(job['fasta'])) != job['fasta_sha256'] for job in jobs):
        raise ValueError('RNA scoring inputs changed during execution')
    progress['finished_at'] = time.time()
    progress['status'] = 'incomplete' if progress['failed'] else 'completed'
    write_json(progress_path, progress)
    if progress['failed']:
        raise RuntimeError(f"RNA scoring incomplete: {len(progress['failed'])}/{len(jobs)} families failed; "
                           'completed families are checkpointed in '+str(progress_path))
    metrics = {}
    for instance in native.INSTANCES:
        selected = [job for job in jobs if job['instance'] == instance]
        values, details = [], {}
        for job in selected:
            result = progress['completed'][instance+'/'+job['family']]
            mean = result['mean_bit_score']
            ga = job['ga_threshold']
            normalized = mean/ga if ga not in (None, 0) else mean
            values.append(normalized)
            details[job['family']] = dict(mean_bit_score=mean, n_sequences=result['n_sequences'],
                                          ga_threshold=ga, normalized_mean=normalized)
        metrics[instance] = dict(mean_bit_score=float(native.np.mean(values)),
                                 n_families_evaluated=len(values), per_family=details,
                                 n_sequences_evaluated=sum(row['n_sequences'] for row in details.values()),
                                 complete=True)
    return metrics
