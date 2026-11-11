"""GTEx source decoding, public features, safe attachment access and scoring.
"""
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import numpy as np

from benchmarks.naturebench_active.common import digest

TASK = 'LIFE_16'
BINS = ('small', 'middle', 'large', 'huge')
SCALARS = ('phenotype_id', 'variant_id', 'tss_distance', 'maf', 'ma_samples', 'ma_count', 'bulk')
SEQUENCES = ('variant_51_seq', 'seq_between_variant_tss',
             'variant_51_seq_after_mutation', 'seq_between_variant_tss_after_mutation')
SIGNALS = ('atac_between', 'atac_variant_51')
HIDDEN = ('label', 'slope', 'pval_nominal', 'slope_se')


class NativeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Published bins mix pandas and NumPy serialization versions. Apply
        # pandas' compatibility remappings without changing the shared env.
        from pandas.compat.pickle_compat import _class_locations_map
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        module, name = _class_locations_map.get((module, name), (module, name))
        return super().find_class(module, name)


def read_native(path):
    """Decode trusted organizer source files; never use for Agent artifacts."""
    with Path(path).open('rb') as stream:
        return NativeUnpickler(stream).load()


def feature_arrays(row):
    arrays = {k: np.frombuffer(row[k].encode('ascii'), dtype=np.uint8) for k in SEQUENCES}
    for k in SIGNALS:
        arrays[k] = np.asarray(row[k], dtype=np.float64)
        if arrays[k].ndim != 1 or not np.isfinite(arrays[k]).all():
            raise ValueError('Invalid native ATAC input: ' + k)
    return arrays


def scalar_inputs(row):
    return {k: row[k].item() if hasattr(row[k], 'item') else row[k] for k in SCALARS}


def payload_digest(arrays):
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = arrays[k]
        h.update(k.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def group_keys(inputs):
    gene = inputs['phenotype_id'].split('.')[0]
    parts = inputs['variant_id'].split('_')
    if not gene or len(parts) != 5 or parts[-1] != 'b38':
        raise ValueError('Unknown gene/variant identifier schema')
    return 'gene:' + gene, 'site:' + parts[0] + ':' + str(int(parts[1]))


def checked_feature(base, row):
    from benchmarks.naturebench_active.common import sha256
    path = Path(base)/digest(row['sample_id'])/'features.npz'
    inputs = json.loads(row['inputs']) if isinstance(row['inputs'], str) else row['inputs']
    if sha256(path) != inputs['feature_sha256']:
        raise ValueError('Noncoding feature attachment changed')
    return path


def verify_feature(path, inputs):
    from benchmarks.naturebench_active.common import sha256
    if sha256(path) != inputs['feature_sha256']:
        raise ValueError('Noncoding feature attachment changed')
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != set(SEQUENCES + SIGNALS):
            raise ValueError('Forbidden or missing NPZ fields')
        arrays = {k: data[k] for k in data.files}
    if any(arrays[k].dtype != np.uint8 for k in SEQUENCES):
        raise ValueError('Sequence encoding changed')
    if any(arrays[k].dtype != np.float64 for k in SIGNALS):
        raise ValueError('ATAC precision changed')
    if payload_digest(arrays) != inputs['feature_payload_sha256']:
        raise ValueError('Native feature payload changed')
    if {k: len(arrays[k]) for k in SEQUENCES} != inputs['sequence_lengths']:
        raise ValueError('Sequence lengths changed')
    if {k: len(arrays[k]) for k in SIGNALS} != inputs['signal_lengths']:
        raise ValueError('Signal lengths changed')
    if set(inputs) != set(SCALARS + ('sequence_lengths','signal_lengths','feature_payload_sha256','feature_sha256')):
        raise ValueError('Unreviewed input metadata')


def export_features(evidence, db, destination):
    for r in db.execute("SELECT sample_id,inputs FROM reference.samples WHERE split='T'"):
        source = checked_feature(evidence.package/'organizer/features', r)
        target = Path(destination)/'files'/digest(r['sample_id'])/'features.npz'
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            checked_feature(Path(destination)/'files', r)
        else:
            shutil.copyfile(source, target)
            target.chmod(0o444)


def validate_prediction(value):
    if set(value) != {'predicted_label', 'predicted_probability', 'predicted_slope'}:
        raise ValueError('Noncoding prediction needs label, probability and slope')
    values = {k: float(v) for k, v in value.items()}
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError('Nonfinite noncoding prediction')
    if values['predicted_label'] not in (0, 1) or not 0 <= values['predicted_probability'] <= 1:
        raise ValueError('Invalid noncoding classification prediction')
    values['predicted_label'] = int(values['predicted_label'])
    return values


def score(records, values, source):
    import pandas as pd
    from benchmarks.naturebench_active.scoring.metrics import native_functions
    native = native_functions(source, ['calculate_metrics'])['calculate_metrics']
    instances = {}
    for instance in sorted({r['instance'] for r in records}):
        pairs = [(r, validate_prediction(v)) for r, v in zip(records, values) if r['instance'] == instance]
        frame = pd.DataFrame([{**r['label'], **v} for r, v in pairs])
        instances[instance] = dict(count=len(pairs), metrics=native(frame))
    names = ('AUC', 'Accuracy', 'F1-Score', 'Pearson Correlation', 'RMSE', 'R-squared')
    # The native primary is an unweighted mean over distance bins. Require all
    # four cohorts and a finite AUC in each; never hide a missing/single-class bin.
    complete = set(instances) == {'gtex_v8/' + b for b in BINS}
    macro = {k: float(np.mean([v['metrics'][k] for v in instances.values()]))
             if complete and all(v['metrics'][k] is not None and np.isfinite(v['metrics'][k]) for v in instances.values())
             else None for k in names}
    return dict(instances=instances, native_macro_metrics=macro,
                native_metric_functions=['calculate_metrics'], aggregation='unweighted mean of four distance bins')


OUTPUT_CONTRACT = '''Noncoding variant effect prediction, four independent distance bins.
Each output dictionary has exactly predicted_label (0/1), predicted_probability
(probability of label=1), predicted_slope (finite eQTL effect size).
Training targets are label and slope; association statistics pval_nominal and
slope_se are excluded entirely. Do not use phenotype_id, variant_id, sample_id,
ordinal, group_id, filenames or hashes as learned features. Tissue bulk, distance,
maf, allele counts, DNA sequences and ATAC profiles are allowed inputs.
For this task file reading is permitted ONLY for inputs.feature_paths, a list of
one input-only NPZ. Use np.load(path, allow_pickle=False). Four sequence arrays
(variant_51_seq, seq_between_variant_tss, variant_51_seq_after_mutation,
seq_between_variant_tss_after_mutation) are uint8 ASCII bytes; decode with
array.tobytes().decode('ascii'). atac_between and atac_variant_51 are float64.
All full sequences and signals are preserved; mutant and reference lengths can
differ in the native data. Pool inputs include feature digests and lengths.
Fit per-bin classifier and regressor using practical sequence composition,
reference/mutant contrasts, tissue and summary/positional ATAC features. Read each
NPZ once per prediction/fit feature pass, not once per estimator or CV fold.
group_id identifies global gene/site components shared across tissues and bins.
Choose your own internal validation and CV design. select_validation registers
exact acquired sample IDs and may replace or clear the selection during research.
Read current acquired and role columns after changing the selection.
For local experiments, fetch_data returns each record's relative files list; join
those paths to the supplied problem directory. For fit/predict, the host supplies
absolute inputs.feature_paths automatically. Do not guess directories from hashes.
The primary AUC is the unweighted mean of the four per-bin AUC values.
'''
