"""Task-specific scalar projections and mixed-validation native metrics."""
import numpy as np
import pandas as pd
from benchmarks.naturebench_active.adapters.nanoparticle import OUTPUT_CONTRACT as NANOPARTICLE_CONTRACT

TASKS={'LIFE_09','CHEM_05','MAT_05','MAT_03','MAT_02'}
TASKS.add('LIFE_04')
OUTPUT_CONTRACTS={
 'LIFE_04': '''Output exactly {"cancer_gene": probability in [0,1]}. Each acquired label is one gene shared across graphs; the record instance is only its canonical storage network, and its subset may have only one class. inputs.network_features maps each available network to 64 numeric multi-omics features, 64 neighbor_mean features, and degree. The host computed the unweighted neighbor means on the public undirected graph with self loops removed; no targets entered these features. All Test genes are absent from all development graphs. Map each acquired training gene's cancer_gene label into every network present in its network_features. Fit per-network predictors using those shared acquired labels, or a shared model with regularization; do not fit only the canonical storage subsets. Group all representations of one gene together for internal CV, so its annotation cannot cross training/CV folds. At prediction, output a single probability for the network named by the record's instance. Gene names, node_index, graph_asset and sample_id are identifiers, never numeric or categorical predictors. Features are fully present in each record; no file reads are needed. Official V here is the curated unique-gene cohort at canonical storage locations, not the full original transductive network validation. Native AUPRC and AUROC are computed separately for those eight retained network subsets.''',
 'MAT_02': '''For each MOF output exactly {"value": finite scalar}, in its native target units. Fit separate predictors for band_gap, h2_diffusivity, h2_uptake. Override the no-file-reading rule ONLY for the exact sanitized paths in inputs["feature_paths"], which the host grants to this isolated fitting/prediction process. These provide .graphdata (pickle of [structure_id, atomic_numbers, neighbor_indices, neighbor_distances, unique_atom_index_groups, unique_atom_counts, None]), .grid (three lines CELL_PARAMETERS/CELL_ANGLES/GRID_NUMBERS), and .griddata16 (pickled 1D numeric ndarray). The seventh graph field is always None; never treat it as a target. You may use pickle to decode these input-only files, and derive atom-composition, neighbor-distance, lattice and grid statistics/features. Do not read other paths, do not use structure_id as a predictor, and do not preserve training file dependencies in the fitted model. Return a pickleable model containing fitted parameters. Predict validation rows using only their own feature_paths; no training paths are accessible then. There are ~4.7k fitting and ~5.9k V structures, so cache repeated structure featurization and use efficient regularized/ensemble models.''',
 'LIFE_09': '''Task-specific override: for all classification instances output {"label": integer predicted class}, NOT a probability, because native metric is MCC. Detect binary vs multiclass using acquired labels (enhancers_types and splice_sites_all can be multiclass). For deepstarr output scalar Dev_log2_enrichment and Hk_log2_enrichment. Use sequence-derived features; region is an identifier and must not be a predictor. Efficient k-mer/positional sequence features and regularized models are appropriate CPU baselines; avoid repeated expensive hyperparameter search across 19 instances.''',
 'CHEM_05': '''Fit each of seven included instances on the provided fitting records. Small molecules use SMILES; reactions use either the four input component SMILES or the rxn reaction string (parse reaction SMILES by its separators); proteins use primary sequence and protein_length. Targets in protein labels are single-element lists; prediction may return the same target as a scalar or one-element list. Five instances have native V and reactions have a D0 group holdout; do not use any validation targets. This experiment fixes both reaction instances to native split 0. Only split 0 training and Test inputs are permitted; other splits and their histories are unavailable. Terminal reaction metrics are split 0 R-squared, not the official ten-split mean.''',
 'MAT_05': NANOPARTICLE_CONTRACT,
 'MAT_03': '''Task-specific override: output exactly {"b": probability of binding, "Binding (SiO2)": binding energy estimate, "BE": expected binding energy across all pairs, "load_norm_class": integer 0..47}. The BE training target is b*Binding (SiO2) from acquired labels; it may be modeled separately or combined with binding probability and energy. The loading class is an acquired target and must not be an input. Molecule/framework feature payloads are available in inputs["molecule_descriptors"] and inputs["framework_descriptors"] when present; raw molecule_features/framework_features strings are source references, never predictors. You can also use SMILES fingerprints and categorical Zeolite identity. The native score uses thresholded b F1, conditional energy RMSE, BE RMSE and per-molecule framework-rank correlation. Optimize CPU use over ~49k fitting and ~66k prediction rows; vectorize cached features per unique molecule/framework.''',
}


def scalar(value):
    a=np.asarray(value,dtype=float)
    if a.size!=1 or not np.isfinite(a).all():raise ValueError('Expected one finite scalar prediction')
    return float(a.reshape(-1)[0])


def scientific_score(task,records,values,source):
    from benchmarks.naturebench_active.scoring.metrics import native_functions
    if task=='LIFE_09':names=['calculate_classification_metrics','calculate_regression_metrics']
    elif task=='CHEM_05':names=['compute_rmse','compute_spearman','compute_r2']
    elif task=='MAT_05':names=['_compute_split_mse']
    else:names=['calculate_metrics']
    native=native_functions(source,names)
    instances={}
    for instance in sorted({r['instance'] for r in records}):
        pairs=[(r,p) for r,p in zip(records,values) if r['instance']==instance]
        if task=='MAT_05':
            truth={};pred=[]
            for r,p in pairs:
                if set(p)!={'target'}:raise ValueError('Nanoparticle prediction requires target only')
                x=np.asarray(r['label']['wavelength_spectrum_x'],float);y=np.asarray(r['label']['wavelength_spectrum_y'],float)
                truth[r['sample_id']]=float(np.log10(100+y[(np.abs(x)>300)&(np.abs(x)<450)].sum()))
                pred.append(dict(_id=r['sample_id'],target=scalar(p['target'])))
            metrics={'MSE_x1000':native['_compute_split_mse'](pd.DataFrame(pred),truth)}
        elif task=='MAT_03':
            truth=[];pred=[]
            for r,p in pairs:
                if set(p)!={'b','Binding (SiO2)','BE','load_norm_class'}:raise ValueError('Zeolite prediction target schema mismatch')
                b=scalar(p['b']);loading=scalar(p['load_norm_class'])
                if not 0<=b<=1 or not 0<=loading<=47 or loading!=int(loading):raise ValueError('Invalid binding probability or loading class')
                ids={k:r['inputs'][k] for k in ('SMILES','Zeolite')}
                truth.append({**ids,**{k:float(v) for k,v in r['label'].items()}})
                pred.append({**ids,'b_pred':int(b>=.5),'E_pred':scalar(p['Binding (SiO2)']),
                             'BE_pred':scalar(p['BE']),'load_class_pred':int(loading)})
            # Keep actual molecular and framework identities for native ranking.
            metrics=native['calculate_metrics'](pd.DataFrame(pred),pd.DataFrame(truth))
        else:
            target_names=set(pairs[0][0]['label'])
            if any(set(r['label'])!=target_names or set(p)!=target_names for r,p in pairs):raise ValueError('Prediction target schema mismatch')
            truth=pd.DataFrame([{k:scalar(v) for k,v in r['label'].items()} for r,_ in pairs])
            pred=pd.DataFrame([{k:scalar(v) for k,v in p.items()} for _,p in pairs])
            if task=='LIFE_04':
                if ((pred.values<0)|(pred.values>1)).any():raise ValueError('Cancer gene predictions require probabilities')
                metrics=native['calculate_metrics'](pred.values.ravel(),truth.values.ravel())
            elif task=='MAT_02':
                metrics=native['calculate_metrics'](truth.values.ravel(),pred.values.ravel())
            elif task=='LIFE_09':
                if instance=='deepstarr':metrics=native['calculate_regression_metrics'](pred,truth)
                else:
                    if not np.equal(pred['label'],np.floor(pred['label'])).all():raise ValueError('Genomic MCC requires integer class predictions')
                    metrics=native['calculate_classification_metrics'](pred,truth)
            else:
                y=truth.values.ravel();p=pred.values.ravel()
                if instance in ('esol','freesolv','lipophilicity'):key,func='RMSE','compute_rmse'
                elif instance in ('fluorescence','stability'):key,func="Spearman's rho",'compute_spearman'
                else:key,func='R-squared','compute_r2'
                metrics={key:round(native[func](p,y),6)}
        instances[instance]=dict(count=len(pairs),metrics=metrics)
        if task=='CHEM_05' and instance in ('buchwald_hartwig','suzuki'):
            instances[instance]['coverage_note']='One isolated split only; not the ten-split aggregate'
        if task=='MAT_05':
            instances[instance]['coverage_note']='Internal D0 holdout; independent IID/OOD Test metrics not computed'
        if task=='LIFE_04':
            instances[instance]['coverage_note']='Curated unique-gene V at canonical storage networks; development graphs exclude every Test gene'
    return dict(instances=instances,native_metric_functions=names)
