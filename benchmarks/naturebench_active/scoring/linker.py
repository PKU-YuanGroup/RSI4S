"""Native linker metrics with an explicitly acquired-training novelty reference."""
from collections import defaultdict
import os

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import QED, rdMolDescriptors, rdMolAlign, AllChem, rdShapeHelpers, MolStandardize
from rdkit.Chem.FeatMaps import FeatMaps

from benchmarks.naturebench_active.common import sha256


OUTPUT_CONTRACT='''Output exactly {"molecule": generated full-molecule SMILES, "linker": generated linker SMILES, "molblock": generated 3D full-molecule SDF mol block}. One candidate per input (K=1), in input order. This overrides the generic label target names: training molecule_molblock/linker_molblock are acquired targets, not prediction inputs. If a candidate cannot be constructed chemically, return empty strings explicitly; it counts as invalid, never omit a row. Nonempty candidates must be one connected, chemically valid molecule containing ALL supplied fragments attached at the specified anchors, with sensible 3D geometry and finite coordinates. Do not output disconnected input fragments, flood atoms, or repeat predictions to game validity/uniqueness. Learn connector length/motif distributions or a conditional generative model from ALL provided D0 records; do not merely emit a fixed alkyl chain, ignore labels, or memorize sample IDs. A practical CPU baseline may learn a compact motif library and a conditional model selecting motifs from fragment geometry and anchor chemistry, then connect/rebuild using RDKit; acquired training motifs are permitted. Preserve the supplied fragment coordinates by constrained embedding or deterministic placement followed by constrained refinement. Some fragments have open aromatic valence and require unsanitized RDKit parsing initially; invalid cases must be represented explicitly. Geom may have more than two anchors; pockets provide optional full protein environment features. Dataset names zinc/geom/pockets select instance models; final casf must use the trained zinc model unchanged. Native novelty will be compared ONLY to acquired fitting linker labels, not unqueried Q. K=1 uniqueness is not comparable to a multi-candidate generation benchmark. No raw data or file reads: mol blocks and pockets are provided in records. Fit at most 10 minutes for ~89k examples, infer at most 10 minutes for ~2.1k V examples; cache features, avoid fitting many large ensembles or expensive per-molecule docking.'''


def canonical_linkers(training_records):
    result=set()
    enumerator=MolStandardize.rdMolStandardize.TautomerEnumerator()
    for value in sorted({row['label']['linker'] for row in training_records}):
        mol=Chem.MolFromSmiles(value)
        if mol is None:result.add(value);continue
        Chem.RemoveStereochemistry(mol)
        try:value=Chem.MolToSmiles(enumerator.Canonicalize(mol))
        except Exception:pass
        result.add(value)
    return result


def consistent_smiles(mol, smiles):
    """Accept the SDF's exact canonical declaration before reparsing SMILES.

    RDKit can change the @OH permutation token when reparsing its own canonical
    SMILES for non-tetrahedral sulfur. Comparing two reparsed strings alone
    incorrectly rejects a declaration produced directly from the stored SDF.
    """
    canonical=Chem.MolToSmiles(Chem.RemoveHs(mol))
    if smiles==canonical:return True
    graph=Chem.MolFromSmiles(smiles)
    return graph is not None and canonical==Chem.MolToSmiles(Chem.RemoveHs(graph))


def linker_score(task,records,values,source,training_records):
    from benchmarks.naturebench_active.scoring.metrics import native_functions
    if training_records is None:raise ValueError('Linker novelty requires acquired fitting labels')
    RDLogger.DisableLog('rdApp.*')
    names=['_load_sa_scores','compute_sa_score','googles_ring_atoms','_init_sc_rdkit',
           'compute_sc_rdkit','topology_from_rdkit','is_valid','calculate_metrics']
    native=native_functions(source,names,['_KEEP_FEATURES'])
    native.update(os=os,Chem=Chem,QED=QED,rdMolDescriptors=rdMolDescriptors,rdMolAlign=rdMolAlign,
                  AllChem=AllChem,rdShapeHelpers=rdShapeHelpers,MolStandardize=MolStandardize,FeatMaps=FeatMaps,
                  RDConfig=RDConfig,nx=nx,isomorphism=isomorphism,defaultdict=defaultdict,
                  EVAL_DIR=str(source.parent),_fscores=None,_fdef=None,_fmParams=None)
    train_linkers=canonical_linkers(training_records)
    instances={}
    for instance in sorted({r['instance'] for r in records}):
        pairs=[(r,p) for r,p in zip(records,values) if r['instance']==instance]
        data=[];predicted=[];truth={};truth_order=[];failed=0
        for row,pred in pairs:
            if set(pred)!={'molecule','linker','molblock'} or any(not isinstance(x,str) for x in pred.values()):
                raise ValueError('Linker prediction requires three string fields')
            # Match native SDMolSupplier sanitization and hydrogen handling.
            mol=Chem.MolFromMolBlock(pred['molblock'],sanitize=True,removeHs=True) if pred['molblock'] else None
            graph=Chem.MolFromSmiles(pred['molecule']) if pred['molecule'] else None
            if graph is None:
                failed+=1
            else:
                if len(Chem.GetMolFrags(graph))!=1:raise ValueError('Disconnected generated molecule')
                if mol is None or not mol.GetNumConformers():raise ValueError('Valid molecule lacks 3D coordinates')
                if not np.isfinite(mol.GetConformer().GetPositions()).all():raise ValueError('Nonfinite generated coordinates')
                check=Chem.Mol(mol)
                Chem.SanitizeMol(check)
                if not consistent_smiles(check,pred['molecule']):
                    raise ValueError('Generated SMILES and SDF connectivity disagree')
            data.append(dict(pred_molecule=pred['molecule'],pred_linker=pred['linker'],fragments=row['inputs']['fragments']))
            predicted.append(mol)
            true_smiles=row['label']['molecule']
            truth_order.append(true_smiles)
            truth[true_smiles]=Chem.MolFromMolBlock(row['label']['molecule_molblock'],sanitize=True,removeHs=True)
        metrics=native['calculate_metrics'](data,predicted,truth,train_linkers,truth_order)
        instances[instance]=dict(count=len(pairs),metrics=metrics,explicit_failed_candidates=failed,
            coverage_note='One candidate per input; novelty reference contains acquired fitting linkers only. CASF is Test-only; pocket variants share the full-pocket development instance.')
    return dict(instances=instances,native_metric_functions=names,novelty_reference='acquired_fitting_labels_only',
                training_linker_types=len(train_linkers),candidate_count_per_input=1,
                metric_constant_sha256=sha256(source.parent/'shared/fpscores.pkl.gz'))
