"""Input-only preparation for the irreversible frozen inference handoff."""
import csv
import json
from pathlib import Path

from benchmarks.naturebench_active.common import digest
from benchmarks.naturebench_active.common import connect, sha256


from benchmarks.task_names import original_id


def export_identifiers(evidence, db, destination, source_root):
    """Recover native output names from hashed, public, unlabeled input files."""
    task=evidence.manifest['task_id']; identifiers={}; sources={}
    if task not in ('LIFE_08','CHEM_05'):
        return
    def checked(relative):
        # Hash-locked packages prepared before the rename record original task IDs.
        original=original_id(task)+relative[len(task):]
        if db.execute('SELECT 1 FROM reference.sources WHERE path=?',(original,)).fetchone():relative=original
        expected=db.execute('SELECT sha256 FROM reference.sources WHERE path=?',(relative,)).fetchone()
        path=source_root/relative
        if expected is None or sha256(path)!=expected[0]:
            raise ValueError('Native input identifier source changed')
        sources[relative]=expected[0]
        return path
    if task=='LIFE_08':
        names=checked(task+'/problem/data/benchmark/id.txt').read_text().splitlines()
        for row in db.execute("SELECT sample_id,ordinal FROM reference.samples WHERE split='T'"):
            name=Path(names[row['ordinal']].strip()).stem
            if not name or '/' in name:raise ValueError('Invalid native DNA identifier')
            identifiers[row['sample_id']]=name
        if len(set(identifiers.values()))!=len(identifiers):raise ValueError('Duplicate DNA output name')
    else:
        variant=evidence.manifest['protocol']['repeated_split']
        from benchmarks.naturebench_active.adapters.multi_properties import require_split_zero
        require_split_zero(variant)
        for instance in ('buchwald_hartwig','suzuki'):
            relative=f'{task}/problem/data/{instance}/test_split_{variant}.csv'
            with checked(relative).open(newline='') as stream:rows=list(csv.DictReader(stream))
            released=list(db.execute("SELECT sample_id,ordinal,inputs FROM reference.samples WHERE split='T' AND instance=? ORDER BY ordinal",(instance,)))
            if len(rows)!=len(released):raise ValueError('Reaction input count changed')
            for row in released:
                original=rows[row['ordinal']];inputs=json.loads(row['inputs'])
                if any(original[k]!=v for k,v in inputs.items()):raise ValueError('Reaction input order changed')
                identifiers[row['sample_id']]=int(original['index'])
    (destination/'native_identifiers.json').write_text(json.dumps(dict(
        identifiers=identifiers, sources=sources, test_truth_read=False,
        repeated_split=evidence.manifest['protocol']['repeated_split']),sort_keys=True))


def prepare_records(task, directory):
    """Reconstruct exactly the features authorized by the one-way input release."""
    directory=Path(directory).resolve()
    with connect(directory/'inputs.sqlite') as db:
        if 'label' in [r[1] for r in db.execute('PRAGMA table_info(samples)')]:
            raise ValueError('Test release unexpectedly contains labels')
        records=[dict(r) for r in db.execute('SELECT sample_id,instance,inputs FROM samples ORDER BY instance,ordinal')]
    for row in records:row['inputs']=json.loads(row['inputs'])
    from benchmarks.naturebench_active.adapters.pending import FEATURE_TASKS, TASKS as PENDING_TASKS, load_resource_paths
    if task in PENDING_TASKS:
        paths=load_resource_paths(directory)
        if paths:
            for row in records:row['inputs']['resource_paths']=paths
    from benchmarks.naturebench_active.outputs import normalize_inputs
    normalization=normalize_inputs(task,records)
    if task=='LIFE_04':
        from benchmarks.naturebench_active.adapters.graph import enrich, network_features
        enrich(records,network_features(directory/'inputs.sqlite',scope='final'))
    elif task=='MAT_03':
        from benchmarks.naturebench_active.execution import enrich_public_inputs
        enrich_public_inputs(records,directory/'inputs.sqlite',task)
    files={}
    if task in ('MAT_02','LIFE_07','LIFE_16',*FEATURE_TASKS):
        for row in records:
            paths=sorted((directory/'files'/digest(row['sample_id'])).iterdir())
            if len(paths)!=(3 if task=='MAT_02' else 1):
                raise ValueError('Incomplete frozen feature attachments')
            row['inputs']['feature_paths']=[str(p) for p in paths]
            for path in paths:files[str(path.relative_to(directory))]=sha256(path)
            if task=='LIFE_16':
                from benchmarks.naturebench_active.adapters.noncoding import checked_feature
                checked_feature(directory/'files',row)
            if task in FEATURE_TASKS:
                from benchmarks.naturebench_active.adapters.pending import checked_feature
                checked_feature(directory/'files',row)
    normalization.update(public_asset_features=task in ('LIFE_04','MAT_03'),
                         sanitized_attachment_sha256=files)
    return records,normalization
