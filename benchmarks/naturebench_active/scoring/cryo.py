"""Native cryo-EM structure metrics on acquired held-out references only."""
from collections import defaultdict
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.distance import cdist


OUTPUT_CONTRACT='''Output exactly {"pdb": a complete PDB-format string with ATOM records for the predicted nucleic-acid structure}. This overrides generic target-name instructions. Fit a scientific model from all acquired training reference_mmcif strings and density maps, then infer the structure from sequences and density alone. Learn geometric templates, atom/bond distance statistics, or a density classifier from D0; do not merely emit a fixed helix or copy a training structure. Only inputs.feature_paths may be opened: one sanitized gzip MRC density map per record. You may use gzip, struct, Bio.PDB and scipy for parsing/numerics. mmCIF strings are labels in memory; parse using io.StringIO. MRC may have axis mapping, nonzero grid starts and origins; map voxels into the correct Cartesian coordinate frame. Map header annotations have been cleared. Full map arrays may be large: use deterministic subsampling/coarse grids for efficient fitting and tracing, use the assigned CUDA GPU for learned neural models and tensor workloads (CPU parsing/numerics remain available), and stay below 10 minutes for fitting and 10 minutes for inference on 4 validation maps. The model must store fitted parameters/templates, with no training file path dependencies at prediction. Predict a physically coherent connected nucleotide chain for each provided sequence, with reasonable phosphodiester distances and full backbone atoms; never flood the map with disconnected residues to exploit coverage metrics. Respect the supplied sequence lengths and identities, never use a structure/map identifier as a feature. Output exactly one chain per provided sequence, in input order, with one residue for every sequence position including ambiguous N positions. PDB ATOM columns must be correctly formatted (atom serial numbers are not scored and may wrap modulo 100000 in large structures), residue names A/U/T/C/G (or DNA DA/DT/DC/DG), distinct chains and globally increasing residue numbers, valid finite coordinates. Native metrics measure 5-Angstrom backbone and sequence recall; fitting uses only D0 labels. This shared-map D0 holdout does not separately validate the two Test-only map collections.'''


def cryo_score(task,records,values,source):
    from benchmarks.naturebench_active.scoring.metrics import native_functions
    names=['parse_cif','parse_pdb','compute_center_distance_matrix','compute_metrics']
    constants=['PHO_ATOMS','SUGAR_ATOMS','NUC_MAP','DISTANCE_CUTOFF','MAX_CUTOFF']
    native=native_functions(source,names,constants)
    native.update(defaultdict=defaultdict,cdist=cdist)
    atoms=native['PHO_ATOMS']+native['SUGAR_ATOMS']+['N1']
    results={}; per_sample=[]
    with tempfile.TemporaryDirectory(prefix='naturebench-cryo-validation-') as directory:
        directory=Path(directory)
        for row,prediction in zip(records,values):
            if set(prediction)!={'pdb'} or not isinstance(prediction['pdb'],str):raise ValueError('Cryo output must contain a PDB string')
            query=directory/'prediction.pdb';truth=directory/'acquired-validation.cif'
            query.write_text(prediction['pdb']);truth.write_text(row['label']['reference_mmcif'])
            q=native['parse_pdb'](str(query),atoms);t=native['parse_cif'](str(truth),atoms)
            if not q or not t:raise ValueError('Empty parsed nucleic-acid structure')
            for residue in q.values():
                if not np.isfinite(np.asarray(list(residue.values()),float)).all():raise ValueError('Nonfinite predicted atomic coordinates')
            if len(q)<max(2,len(t)*.05) and len(t)>=10:raise ValueError('Native minimum prediction coverage failed')
            declared=sum(len(s) for s in row['inputs']['sequences'])
            if len(q)!=declared:raise ValueError('Predicted residues must match input sequences, without metric-driven oversampling')
            center=native['compute_center_distance_matrix'](q,t)
            # Native compute_metrics never reads atom_dist; avoid its unused
            # O(atoms * residues^2) allocation without changing either metric.
            br,sr=native['compute_metrics'](q,t,center,None)
            results.setdefault(row['instance'],[]).append((br,sr))
            per_sample.append(dict(sample_id=row['sample_id'],backbone_recall=br,sequence_recall=sr,
                                   predicted_residues=len(q),reference_residues=len(t)))
    instances={instance:dict(count=len(rows),metrics=dict(backbone_recall=round(float(np.mean(rows,axis=0)[0]),6),
                    sequence_recall=round(float(np.mean(rows,axis=0)[1]),6)),
                    coverage_note='Shared-map D0 holdout; two terminal Test collections remain unevaluated') for instance,rows in results.items()}
    return dict(instances=instances,native_metric_functions=names,per_sample=per_sample)
