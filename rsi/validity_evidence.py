"""Bounded NatureBench-style evidence for the selected scientific method only."""
import ast
import hashlib
import json
from pathlib import Path
import re

from judge.sources import SOURCE_SKIP_DIRS, SOURCE_SUFFIXES, clip_text_bytes
from rsi.common import encoded
from rsi.sandbox import assert_plain_tree


SOURCE_BYTES = 160000
TRACE_BYTES = 40000
RESULT_BYTES = 16000
OMIT_RESULT = {'details', 'per_sample', 'group_scores', 'selection', 'candidates',
               'predictions', 'trajectory', 'executions', 'test_samples'}
FITTING = re.compile(r'\b(?:fit|partial_fit|backward|optimize|train|minimize|least_squares|lstsq)\s*\(')


def executed_sources(actions, available):
    """Resolve saved script and python -m entry points without executing commands."""
    commands = '\n'.join(a.get('request', {}).get('command', a.get('request', {}).get('code', ''))
                         for a in actions)
    scripts = re.findall(r'(?:python[\d.]*|torchrun|bash|sh)\s+(?:-[\w-]+\s+)*[\"\']?([\w./-]+\.(?:py|bash|sh))', commands)
    modules = re.findall(r'python[\d.]*\s+(?:-[uBEOsSI]+\s+)*-m\s+([\w.]+)', commands)
    entries, missing = set(), set()
    for script in scripts:
        matches = {name for name in available if name == script.removeprefix('./')
                   or script.endswith('/'+name) or name.endswith('/'+script)}
        if len(matches) == 1:
            entries.update(matches)
        elif re.search(r'train|fit|learn|optim', script, re.I):
            missing.add(script)
    for module in modules:
        base = module.replace('.', '/')
        matches = {name for name in (base+'.py', base+'/__main__.py') if name in available}
        if matches:
            entries.update(matches)
        elif re.search(r'train|fit|learn|optim', module, re.I):
            missing.add('module:'+module)
    return entries, missing


def compact(value, depth=0):
    """Keep aggregate fields; never recursively include sample arrays or logs."""
    if isinstance(value, dict) and depth < 4:
        return {str(k): compact(v, depth + 1) for k, v in list(value.items())[:100]
                if k not in OMIT_RESULT and not isinstance(v, (list, tuple))}
    if value is None or type(value) in (int, float, bool):
        return value
    if isinstance(value, str):
        return clip_text_bytes(value, 1000)
    return None


def result_summary(value):
    result = {k: compact(value[k]) for k in ('artifact', 'attempt', 'task', 'method', 'status',
              'split', 'metric', 'score', 'samples', 'aggregate_improvement', 'raw_scores', 'metrics',
              'reused', 'scientific_reuse', 'fitted_artifact')
              if k in value}
    # A task can have arbitrarily many aggregate metrics too. The full score is
    # preserved by the evaluator; performance tables are not the validity proof.
    for key in ('raw_scores', 'metrics'):
        if len(encoded(result).encode()) > RESULT_BYTES:
            result.pop(key, None)
            result['metric_details_omitted'] = True
    return result


def read_checked(reference, directory, key='path'):
    path = Path(reference[key]).resolve()
    if not path.is_relative_to(Path(directory).resolve()):
        raise ValueError('Execution evidence escaped the experiment directory')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != reference['sha256']:
        kind = 'source' if key == 'source' else 'execution'
        raise ValueError('Host ' + kind + ' evidence changed: ' + str(path))
    return data


def read_receipt(reference, directory):
    return json.loads(read_checked(reference, directory))


def frozen_method(workspace):
    path = Path(workspace) / 'frozen_model.json'
    if not path.exists():
        return {}
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('Frozen method declaration exceeds 1 MiB')
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else {}


def execution(record, feedback, identity, directory):
    """Use the receipt captured at this artifact, before later experiments."""
    audit = feedback.get('training_audit') or {}
    if audit.get('host_execution'):
        report = read_receipt(audit['host_execution'], directory)
        run_id = record.get('run_id')
        runs = [r for r in report['runs'] if r['run_id'] == run_id]
        # Older synthetic/legacy records may omit run_id; the first receipt
        # entry is the current artifact's run, followed by its ancestors.
        if run_id is None:
            runs = report['runs'][:1]
        return [action for r in runs for action in r.get('actions', [])]
    actions = []
    for turn in record.get('trace', []):
        request, response = turn.get('request', {}), turn.get('response', {})
        if request.get('op') in ('bash', 'python') or str(request.get('op', '')).startswith('native_'):
            actions.append(dict(request=request, outcome={k: response[k] for k in
                ('returncode', 'reason', 'seconds', 'execution_audit') if k in response}))
        if request.get('op') == 'evaluate' and (response.get('artifact') == identity or
                (feedback.get('attempt') is not None and response.get('attempt') == feedback['attempt'])):
            break
    return actions


def source_paths(workspace):
    return {p.relative_to(workspace).as_posix(): p for p in sorted(workspace.rglob('*'))
            if p.is_file() and p.suffix.lower() in SOURCE_SUFFIXES
            and not any(part in SOURCE_SKIP_DIRS | {'.tmp'} for part in p.relative_to(workspace).parts[:-1])}


def dependencies(text, name, available):
    """Include local imports and literal source references, without execution."""
    result = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    if tree is None and Path(name).suffix in ('.sh', '.bash'):
        return executed_sources([dict(request=dict(command=text))], available)[0]
    if tree is not None:
        parent = Path(name).parent
        for node in ast.walk(tree):
            modules = []
            prefixes = (Path('.'), parent)
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                modules = [base, *(base + '.' + a.name if base else a.name for a in node.names)]
                if node.level:
                    prefix = parent
                    for _ in range(node.level - 1):
                        prefix = prefix.parent
                    prefixes = (prefix,)
            elif isinstance(node, ast.Call) and node.args:
                function = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
                if function in ('import_module', '__import__') and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, str):
                        modules = [node.args[0].value]
            for module in modules:
                if not module:
                    continue
                parts = module.split('.')
                for n in range(1, len(parts) + 1):
                    rel = Path(*parts[:n])
                    for candidate in (rel.with_suffix('.py'), rel / '__init__.py'):
                        for prefix in prefixes:
                            key = (prefix / candidate).as_posix()
                            if key in available:
                                result.add(key)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in available:
                    result.add(node.value)
    return result


def source_closure(roots, available, read_source):
    selected, pending = {}, sorted(roots)
    while pending:
        name = pending.pop(0)
        if name in selected or name not in available:
            continue
        text = read_source(name)
        selected[name] = text
        pending.extend(sorted(dependencies(text, name, available) - selected.keys()))
        pending.extend((parent/'__init__.py').as_posix() for parent in Path(name).parents
                       if (parent/'__init__.py').as_posix() in available)
    return selected


def method_sources(workspace, method, actions, *, self_contained=False,
                   available=None, read_source=None, reference_inference=None, selection=None):
    """Select entry points, observed fitting scripts and their local imports."""
    available = source_paths(workspace) if available is None else available
    read_source = read_source or (lambda name: available[name].read_text(errors='replace'))
    roots = set()
    if self_contained:
        roots.add('model.py')  # NatureBench Active executes only this saved program.
    else:
        roots.update(x for x in [method.get('inference_entry', 'predict.py'),
                                 method.get('training_entry'), *method.get('source_dependencies', []),
                                 *method.get('checkpoints', [])]
                     if isinstance(x, str) and x in available)
        if not method and not roots:
            roots.update(name for name in ('run.py', 'solution.py', 'main.py', 'train.py', 'model.py', 'policy.py')
                         if name in available)
    if reference_inference is None:
        selected = source_closure(roots, available, read_source)
    else:
        # Historical prediction code may belong to an abandoned architecture.
        # Use the selected snapshot's dependency names to identify fitting code.
        selected = {name: read_source(name) for name in sorted(reference_inference) if name in available}
    if not self_contained:
        executed, missing = executed_sources(actions, available)
        checkpoints = [Path(x).name for x in method.get('checkpoints', [])]
        shared = set(selected if reference_inference is None else reference_inference) - {
            method.get('inference_entry', 'predict.py')}
        inference_names = set(selected)
        for name in available:
            if name not in executed:
                continue
            text = read_source(name)
            related = source_closure([name], available, read_source)
            if (not method or any(FITTING.search(value) for value in related.values())
                    or any(checkpoint in text for checkpoint in checkpoints)):
                # Multiple models can have been tried in the origin round.
                # Prefer the fitting program using the selected architecture /
                # feature modules, rather than every script that saved a file
                # named model.pt. Single-file methods have no such imports.
                if not shared or related.keys() & shared or name in inference_names:
                    selected.update(related)
        if selection is not None:
            selection.update(executed=sorted(executed & selected.keys()), missing=sorted(missing))
    return selected, sorted(set(available) - selected.keys())


def selected_method_evidence(workspace, artifact, selected_feedback, record, runs, directory,
                             *, self_contained=False):
    """Trace selected checkpoints to inherited snapshots, not other candidates."""
    workspace = Path(workspace)
    assert_plain_tree(workspace)
    method = frozen_method(workspace)
    index = {a['id']: (r, a) for r in runs for a in r.get('artifacts', [])}
    # Follow unchanged weights to their first inherited snapshot. For newly
    # fitted weights, retain explicitly declared fitting parents as well.
    targets = [(record, artifact, selected_feedback, [])]
    seen, new_weights, ancestry_basis, ancestry_gaps = {artifact.get('id')}, set(), {}, []
    for owner, item, _, _ in targets:
        declaration = frozen_method(Path(item.get('workspace', workspace)))
        chain, chain_seen = [(owner, item)], {item.get('id')}
        parent = owner.get('artifact_parent')
        while parent in index and parent not in chain_seen:
            chain_seen.add(parent)
            parent_owner, ancestor = index[parent]
            chain.append((parent_owner, ancestor))
            parent = parent_owner.get('artifact_parent')
        needed = {}
        for name in declaration.get('checkpoints', []):
            if Path(name).suffix.lower() in SOURCE_SUFFIXES:
                continue
            digest = item.get('files', {}).get(name)
            matches = [(r, a) for r, a in chain if digest and digest in a.get('files', {}).values()]
            if matches and matches[-1][1]['id'] != item.get('id'):
                parent_owner, ancestor = matches[-1]
                needed.setdefault(ancestor['id'], (parent_owner, ancestor, [], 'matching_checkpoint_hash'))[2].append(name)
            else:
                new_weights.add(item.get('id'))
        if item.get('id') in new_weights:
            from rsi.store import safe_path
            provenance = safe_path(Path(item.get('workspace', workspace)),
                                   declaration.get('training_provenance', 'training_provenance.json'))
            if provenance.exists():
                if provenance.stat().st_size > 1024*1024:
                    raise ValueError('Training provenance exceeds 1 MiB')
                declared = json.loads(provenance.read_text()).get('parent_artifacts', [])
                for identity in declared:
                    if identity not in index:
                        ancestry_gaps.append('Missing declared fitting parent: ' + identity)
                        continue
                    parent_owner, ancestor = index[identity]
                    needed.setdefault(identity, (parent_owner, ancestor,
                        declaration.get('checkpoints', []), 'declared_fitting_parent'))
        for identity, (parent_owner, ancestor, checkpoints, basis) in needed.items():
            if identity in seen:
                continue
            seen.add(identity)
            ancestry_basis[identity] = basis
            feedback = next((e for e in parent_owner.get('evaluations', []) if e.get('artifact') == identity), {})
            targets.append((parent_owner, ancestor, feedback, checkpoints))
    sources, historical, traces, manifest, gaps = {}, [], [], [], []
    gaps.extend(ancestry_gaps)
    remaining, seen_sources, observed_cache = SOURCE_BYTES, set(), {}
    inline_only = []
    for owner, item, feedback, checkpoints in targets:
        path = Path(item.get('workspace', workspace))
        assert_plain_tree(path)
        actions = execution(owner, feedback, item.get('id'), directory)
        declaration = frozen_method(path)
        inference, _ = method_sources(path, declaration, [], self_contained=self_contained)
        selection = {}
        files, omitted = method_sources(path, declaration, actions, self_contained=self_contained, selection=selection)
        executed = set(selection.get('executed', []))
        missing = set(selection.get('missing', []))
        source_versions = {name: hashlib.sha256((path/name).read_bytes()).hexdigest() for name in files}
        for name, digest in source_versions.items():
            if item.get('files', {}).get(name, digest) != digest:
                raise ValueError('Selected source differs from saved artifact: ' + name)
        expected = ['model.py'] if self_contained else [declaration.get('inference_entry')]
        expected += [declaration.get('training_entry'), *declaration.get('source_dependencies', [])]
        for name in expected:
            if name and name not in files:
                gaps.append(f"{item.get('id')}: missing declared source {name}")
        file_fitting = any(FITTING.search(value) for value in files.values())
        file_fitting |= bool(declaration.get('training_entry') in files)
        # Availability of a producing program is distinct from proving learning;
        # the judge must assess classical/custom fitting too, not just API names.
        file_fitting |= bool(executed - inference.keys())
        inline_fitting = any(FITTING.search(a.get('request', {}).get('code',
            a.get('request', {}).get('command', ''))) for a in actions)
        trace = dict(artifact=item.get('id'), run_id=owner.get('run_id'),
                     selected_evaluation=result_summary(feedback), actions=[])
        for action in actions:
            trace['actions'].append(dict(request=action['request'], outcome={k: v for k, v in
                action.get('outcome', {}).items() if k in ('returncode', 'reason', 'seconds')}))
        traces.append(trace)
        included, clipped = [], []
        for name, text in files.items():
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen_sources:
                continue
            content = clip_text_bytes(text, min(60000, remaining))
            if not content and text:
                clipped.append(name)
                continue
            seen_sources.add(digest)
            remaining -= len(content.encode())
            included.append(name)
            if content != text:
                clipped.append(name)
            if not checkpoints:
                sources[name] = content
            else:
                historical.append(dict(artifact=item['id'], checkpoints=checkpoints, file=name,
                                       sha256=digest, source=content))
        # A training script can be edited or deleted after fitting. Preserve differing
        # host-observed versions of relevant files instead of trusting only
        # the source left in the submitted workspace.
        observed_files = []
        for action in actions:
            reference = action.get('outcome', {}).get('execution_audit')
            if not reference:
                continue
            observation = read_receipt(reference, directory)
            for phase in ('before', 'after'):
                reference = observation.get(phase, {})
                if not reference.get('path'):
                    continue
                observed = read_receipt(reference, directory)['files']

                def observed_source(name):
                    source = observed[name]
                    key = (source['source'], source['sha256'])
                    if key not in observed_cache:
                        observed_cache[key] = read_checked(source, directory, 'source').decode(errors='replace')
                    return observed_cache[key]

                selected, _ = method_sources(path, declaration, [action], self_contained=self_contained,
                    available=observed, read_source=observed_source, reference_inference=inference)
                observed_entries, _ = executed_sources([action], observed)
                executed.update(observed_entries & selected.keys())
                file_fitting |= any(FITTING.search(value) for value in selected.values())
                file_fitting |= bool((observed_entries & selected.keys()) - inference.keys())
                for name, text in selected.items():
                    source = observed[name]
                    if source['sha256'] in seen_sources:
                        continue
                    content = clip_text_bytes(text, min(60000, remaining))
                    if content != text:
                        clipped.append(name + ':' + phase)
                    if not content and text:
                        continue
                    remaining -= len(content.encode())
                    seen_sources.add(source['sha256'])
                    historical.append(dict(artifact=item.get('id'), file=name, phase=phase,
                        sha256=source['sha256'], source=content, evidence='host_observed_source',
                        observation=action['outcome']['execution_audit']['sha256']))
                    observed_files.append(name)
        for name in missing:
            normalized = name.removeprefix('module:').replace('.', '/') if name.startswith('module:') else name
            if not any(entry in (normalized+'.py', normalized+'/__main__.py', normalized)
                       or entry.endswith('/'+normalized) for entry in executed):
                gaps.append(f"{item.get('id')}: unavailable training entry {name}")
        for name in clipped:
            gaps.append(f"{item.get('id')}: source was clipped or omitted: {name}")
        checkpoint_files = {name: item.get('files', {}).get(name) for name in declaration.get('checkpoints', [])
                            if Path(name).suffix.lower() not in SOURCE_SUFFIXES}
        if inline_fitting and not file_fitting:
            inline_only.append(item.get('id'))
        if checkpoint_files and not (file_fitting or inline_fitting) and item.get('id') in new_weights and any(
                name in text for name in checkpoint_files for text in inference.values()):
            gaps.append(f"{item.get('id')}: checkpoint is loaded but its fitting source is unavailable")
        if not declaration.get('source_dependencies'):
            for name, text in files.items():
                if not name.endswith('.py'):
                    continue
                try:
                    tree = ast.parse(text)
                except SyntaxError:
                    gaps.append(f"{item.get('id')}: cannot inspect Python source {name}")
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and node.args:
                        function = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
                        if function in ('import_module', '__import__') and not isinstance(node.args[0], ast.Constant):
                            gaps.append(f"{item.get('id')}: dynamic dependency in {name}; declare source_dependencies")
        manifest.append(dict(artifact=item.get('id'), run_id=owner.get('run_id'),
            checkpoint_origin_for=checkpoints, ancestry_basis=ancestry_basis.get(item.get('id'), 'selected'),
            included_files=included,
            checkpoint_sha256=checkpoint_files, source_sha256=source_versions, executed_sources=sorted(executed),
            unselected_source_files=omitted, clipped_or_omitted_files=clipped,
            observed_source_files=sorted(set(observed_files))))
    trace_text = encoded(traces)
    trace_excerpt = trace_text
    if len(trace_text.encode()) > TRACE_BYTES:
        per_artifact = max(0, (TRACE_BYTES-len(traces)) // max(1, len(traces)))
        trace_excerpt = '\n'.join(clip_text_bytes(encoded(trace), per_artifact) for trace in traces)
        gaps.extend(f'{identity}: only inline fitting evidence is available and its trace was clipped' for identity in inline_only)
    evidence = dict(source_at_SCORE_ATTEMPT=sources, checkpoint_training_sources=historical,
        execution_trace=trace_excerpt,
        selected_method={k: method[k] for k in ('method', 'inference_entry', 'training_entry', 'checkpoints')
                         if k in method},
        evidence_scope=(
            'Judge only the method producing SCORE_ATTEMPT, primarily how its predictive behavior was '
            'trained, learned, fit or optimized from the provided data. Source is from the selected '
            'immutable snapshot and the needed fitting ancestors. Hash-matched inherited weights and '
            'declared fitting parents are labeled separately. Commands concern those snapshots, '
            'not the full research history. Local source '
            'selection follows entry points, observed fitting scripts and static imports; dynamic '
            'dependencies may be absent. Content-hash ancestry and declared membership do not prove '
            'actual fitting. Missing evidence alone is not invalidity. Apply the NatureBench criteria '
            'to the method that actually generated the selected output, not abandoned alternatives.'))
    evidence['method_evidence_links'] = [{k: item[k] for k in
        ('artifact', 'checkpoint_origin_for', 'ancestry_basis', 'checkpoint_sha256', 'source_sha256', 'executed_sources')}
        for item in manifest]
    return evidence, dict(scope='selected_method', artifacts=manifest,
        evidence_gaps=sorted(set(gaps)),
        source_budget_bytes=SOURCE_BYTES, source_bytes=SOURCE_BYTES-remaining,
        trace_bytes=len(evidence['execution_trace'].encode()), trace_clipped=evidence['execution_trace'] != trace_text)
