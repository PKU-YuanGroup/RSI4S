"""Native metric adapters for structured targets; validation truth stays here."""
import numpy as np
import pandas as pd

TASKS = {'LIFE_08', 'LIFE_06', 'CHEM_09', 'CHEM_10', 'LIFE_13'}
OUTPUT_CONTRACTS = {
 'LIFE_13': '''Predict exactly {"seq": amino-acid string} with length equal to len(inputs["coords"]["CA"]), using only the 20 standard letters ARNDCQEGHILKMFPSTWYV. Coordinates for N,CA,C,O can include None (missing); handle them and do not drop prediction residues. Fit a single shared residue-identity model across acquired CATH records; test-only instance names must not cause lookup failures. Build translation/rotation-invariant backbone geometry features (local distances/angles, residue neighborhood density), use relative position if helpful but never protein name. Train on all provided labeled residues where features are usable. Do not use any native sequence from Q or V. Native recovery is median per-protein percent; NSSR is mean per-protein positive BLOSUM62 score percent.''',
 'LIFE_08': '''Override the generic target dictionary requirement: predict exactly {"pwm": L-by-4 nested float list}, with L=len(inputs["X_dna"]). Every row is a probability distribution over A,C,G,T (nonnegative, sums to one). Do NOT predict Y_pwm or pwm_mask: label Y_pwm has shape (2,M,4) in a different alignment. For training, align inputs using dna_mask[0] with label Y_pwm[0][pwm_mask[0]] in order. The two masks have the same number of true positions. You may use Y_hard (co-crystal sequence, allowed input), all structural input arrays, and atom/DNA features. Generalize from fitted D0 complexes; do not merely copy Y_hard. Native metrics average per-complex MAE/RMSE (sum over bases, mean over valid primary-strand positions).''',
 'LIFE_06': '''Predict exactly {"labels1": list of positive-class probabilities, "labels2": list of positive-class probabilities}, with lengths equal to the two input sequences. Missing training residue labels are None and must be masked. Pool all valid labeled residues for fitting, using residue/window features and partner sequence features. Split CV by entire group_id, never split residues of the same complex across folds. Native metrics pool valid residues across both sides and all validation complexes.''',
 'CHEM_09': '''Predict exactly {"energy": scalar in eV, "forces": N-by-3 nested float list in eV/angstrom}, N=27 for the available system. Fit jointly to all acquired energies and forces; a fitted rotation/translation-invariant pair-distance potential and analytic negative gradient is a practical CPU option. Handle large constant energy offsets. The 300K internal holdout does not measure generalization to any of the other Test regimes. Avoid expensive large Gaussian process matrices over all force observations.''',
 'CHEM_10': '''Predict exactly {"positions": N-by-3 nested finite float list}, in angstrom and the original input atom order. No need to predict charges because the species and atom order are input. Fit a structural predictor to the acquired transition-state coordinates using the allowed reactant/product geometries; a fitted interpolation/residual model is acceptable as an explicitly limited baseline. Treat varying atom counts correctly. Scoring is native Kabsch-aligned RMSD allowing reflection, then mean/median over reactions.''',
}


def finite_array(value, shape, name, probability=False):
    array=np.asarray(value, dtype=float)
    if array.shape != tuple(shape) or not np.isfinite(array).all():
        raise ValueError(f'{name}: expected finite shape {tuple(shape)}, got {array.shape}')
    if probability and ((array < 0).any() or (array > 1).any()):
        raise ValueError(name+': probabilities outside [0,1]')
    return array


def structure_score(task, records, values, source):
    from benchmarks.naturebench_active.scoring.metrics import native_functions
    names=['calculate_metrics']
    if task == 'LIFE_06':names += ['parse_scores','parse_labels']
    if task == 'CHEM_10':names += ['kabsch_rmsd','compute_rmsd']
    if task == 'LIFE_13':names += ['seq_to_indices','calculate_recovery_rate','calculate_nssr']
    native=native_functions(source,names,['AA_ORDER'])
    if task == 'LIFE_13':
        from Bio.Align import substitution_matrices
        table=substitution_matrices.load('BLOSUM62');order=native['AA_ORDER']
        matrix=np.asarray([[table[a,b] for b in order] for a in order])
        native['AA_TO_IDX']={a:i for i,a in enumerate(order)}
        # This exact standard metric constant was compared to the native
        # eval_blosums.pth tensor; no sequence ground truth is accessed.
    instances={}
    for instance in sorted({r['instance'] for r in records}):
        pairs=[(r,v) for r,v in zip(records,values) if r['instance']==instance]
        if task == 'LIFE_08':
            per_sample={}
            for r,p in pairs:
                if set(p) != {'pwm'}:raise ValueError('DNA prediction requires only pwm')
                prediction=finite_array(p['pwm'],(len(r['inputs']['X_dna']),4),'pwm',True)
                if not np.allclose(prediction.sum(axis=1),1.,atol=1e-5):raise ValueError('PWM rows must sum to one')
                s=native['calculate_metrics'](prediction,np.asarray(r['label']['Y_pwm'],float),
                    np.asarray(r['inputs']['dna_mask'],bool),np.asarray(r['label']['pwm_mask'],bool))
                if s.get('error') or any(s[k] is None or not np.isfinite(s[k]) for k in ('MAE','RMSE')):
                    raise ValueError('No valid aligned DNA score for '+r['sample_id'])
                per_sample[r['sample_id']]=s
            metrics={k:float(np.mean([s[k] for s in per_sample.values()])) for k in ('MAE','RMSE')}
            instances[instance]=dict(count=len(pairs),metrics=metrics,per_sample=per_sample,
                                     aggregation='native unweighted mean of per-complex scores')
            continue
        if task == 'LIFE_13':
            truth={};pred={}
            for r,p in pairs:
                if set(p)!={'seq'} or not isinstance(p['seq'],str):raise ValueError('Inverse folding requires seq string')
                if len(p['seq'])!=len(r['inputs']['coords']['CA']) or set(p['seq'])-set(order):
                    raise ValueError('Inverse folding length/alphabet mismatch')
                truth[r['sample_id']]=r['label']['seq'];pred[r['sample_id']]=p['seq']
            metrics=native['calculate_metrics'](pred,truth,matrix)
        elif task == 'LIFE_06':
            truth=[];pred=[]
            for r,p in pairs:
                if set(p) != {'labels1','labels2'}:raise ValueError('PPI prediction requires labels1 and labels2')
                y={'pair_id':r['sample_id']};v={'pair_id':r['sample_id']}
                for side in ('1','2'):
                    scores=finite_array(p['labels'+side],(len(r['inputs']['sequence'+side]),),'ppi scores',True)
                    v['scores'+side]=';'.join(map(str,scores))
                    y['labels'+side]=';'.join('nan' if x is None else str(x) for x in r['label']['labels'+side])
                truth.append(y);pred.append(v)
            metrics=native['calculate_metrics'](pd.DataFrame(pred),pd.DataFrame(truth))
        elif task == 'CHEM_09':
            energies=[];forces=[]
            for r,p in pairs:
                if set(p) != {'energy','forces'}:raise ValueError('Energy prediction requires energy and forces')
                energies.append(float(finite_array(p['energy'],(),'energy')))
                forces.append(finite_array(p['forces'],(len(r['inputs']['positions']),3),'forces'))
            metrics=native['calculate_metrics'](np.asarray(energies),np.asarray(forces),
                np.asarray([r['label']['energy'] for r,_ in pairs]),np.asarray([r['label']['forces'] for r,_ in pairs]))
        else:
            positions=[]
            for r,p in pairs:
                if set(p) != {'positions'}:raise ValueError('Transition-state prediction requires positions only')
                positions.append(finite_array(p['positions'],(len(r['inputs']['reactant']['positions']),3),'transition state'))
            metrics=native['calculate_metrics'](positions,{'transition_state':{'positions':[r['label']['positions'] for r,_ in pairs]}})
        instances[instance]=dict(count=len(pairs),metrics=metrics)
        if task == 'CHEM_09':
            instances[instance]['coverage_note']='Acquired 300K holdout only; four native Test regimes are not four independently validated tasks'
    return dict(instances=instances,native_metric_functions=names)
