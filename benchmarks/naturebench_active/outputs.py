"""Native output serialization and input-only validation of frozen predictions.

Only selected prediction loaders are compiled from evaluator source. No native
module imports, ground-truth loaders, or evaluation entrypoints are executed.
"""
import ast
import csv
import json
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from benchmarks.naturebench_active.common import sha256
from benchmarks.naturebench_active.scoring.metrics import aligned_predictions
from collections import defaultdict
from benchmarks.naturebench_active.scoring.scientific import scalar
from benchmarks.naturebench_active.scoring.structure import finite_array
from benchmarks.naturebench_active.adapters.resp import TASK as RESP_TASK, INSTANCE as RESP_INSTANCE

EXTRA_TASKS={'LIFE_15','LIFE_10','CHEM_03',
       'CHEM_05','MAT_02','MAT_03',
       'LIFE_06','LIFE_08','LIFE_04',
       'LIFE_07','CHEM_07','LIFE_16'}
EXTRA_TASKS.add(RESP_TASK)
from benchmarks.naturebench_active.adapters.pending import TASKS as PENDING_TASKS
EXTRA_TASKS.update(PENDING_TASKS)
from benchmarks.naturebench_active.adapters.plm_interact import TASK as PLM_TASK
EXTRA_TASKS.add(PLM_TASK)


SUPPORTED = {'LIFE_12', 'CHEM_09',
             'CHEM_10', 'LIFE_13',
             'MAT_05', 'LIFE_09',
             'LIFE_02','LIFE_14',
             'MAT_04','CHEM_02'}
SUPPORTED |= EXTRA_TASKS


def normalize_inputs(task, records):
    """Lossless layout conversion, with no fitted parameters or truth access."""
    converted=0
    if task=='LIFE_13':
        for row in records:
            coords=row['inputs']['coords']
            if isinstance(coords,list):
                if not coords or any(len(residue)!=4 or any(len(atom)!=3 for atom in residue) for residue in coords):
                    raise ValueError('Expected packed backbone layout [residue, N/CA/C/O, xyz]')
                row['inputs']['coords']={atom:[residue[i] for residue in coords] for i,atom in enumerate(('N','CA','C','O'))}
                converted+=1
    return dict(packed_backbones_transposed=converted, learned_parameters=False, test_truth_read=False)


def prediction_loaders(source, output, names):
    tree = ast.parse(source.read_bytes())
    namespace = dict(os=os, np=np, pd=pd, json=json, pickle=pickle,
                     ValidationError=ValueError, OUTPUT_DIR=str(output))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id != 'OUTPUT_DIR':
                    namespace[target.id] = value
    if 'AA_ORDER' in namespace:
        namespace['VALID_AAS'] = set(namespace['AA_ORDER'])
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError('Missing native prediction loader')
    referenced={n.id for function in selected for n in ast.walk(function) if isinstance(n,ast.Name)}
    if referenced & {'GROUND_TRUTH_DIR','load_ground_truth'}:
        raise ValueError('Native prediction loader requires forbidden Test truth; use input-only format checks')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def csv_file(path, columns, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(task, records, predictions, output, source_root, *, native_checks=True,native_context=None):
    if task not in SUPPORTED:
        raise NotImplementedError('Native final output adapter pending: ' + task)
    if any('label' in r for r in records):
        raise ValueError('Frozen serialization must receive inputs only')
    from benchmarks.naturebench_active.adapters.multi_properties import TASK as PROPERTY_TASK, require_split_zero, evaluation_protocol
    if task == PROPERTY_TASK:
        require_split_zero((native_context or {}).get('repeated_split'))
    values = aligned_predictions(records, predictions)
    output = Path(output)
    source = Path(source_root)/task/'evaluation/evaluator.py'
    counts = {};check_methods={}
    for instance in sorted({r['instance'] for r in records}):
        pairs = [(r,p) for r,p in zip(records,values) if r['instance'] == instance]
        out = output/instance
        out.mkdir(parents=True, exist_ok=True)
        if task in PENDING_TASKS:
            from benchmarks.naturebench_active.scoring.pending import write_instance as write_pending
            check_methods[instance]=write_pending(task,instance,pairs,out)
            counts[instance]=len(pairs)
            continue
        if task in EXTRA_TASKS:
            check_methods[instance]=write_instance(task,instance,pairs,out,output,source,native_context or {},native_checks)
            counts[instance]=len(pairs)
            continue
        if task == 'LIFE_12':
            csv_file(out/'predictions.csv', ['Accession','pHopt_pred'],
                     ({'Accession':r['inputs']['Accession'],'pHopt_pred':p['pHopt']} for r,p in pairs))
            names = ['load_and_validate']
        elif task=='LIFE_02':
            cyclase=instance=='cyclase_peptide_pair'
            key='label' if cyclase else 'Score'
            # Match validation: accept numeric strings, reject nonfinite values
            # and out-of-range classification probabilities before thresholding.
            csv_file(out/'predictions.csv',[key],
                     ({key:int(probability(p[key])>=.5) if cyclase else scalar(p[key])} for _,p in pairs))
            names=['load_and_validate_cyclase' if cyclase else 'load_and_validate_ubonodin']
        elif task=='LIFE_14':
            csv_file(out/'predictions.csv',['name','prediction'],
                     ({'name':r['inputs']['name'],'prediction':float(next(iter(p.values())))} for r,p in pairs))
            names=['load_and_validate']
        elif task=='MAT_04':
            np.save(out/'output.npy',np.asarray([next(iter(p.values())) for _,p in pairs],float))
            names=['load_and_validate']
        elif task=='CHEM_02':
            columns=['smiles',*sorted(pairs[0][1])]
            csv_file(out/'output.csv',columns,({'smiles':r['inputs']['smiles'],**p} for r,p in pairs))
            regression=instance in ('esol','freesolv','lipo','qm7','qm8','qm9')
            names=['load_and_validate_regression' if regression else 'load_and_validate_classification']
        elif task == 'CHEM_09':
            for r,p in pairs:
                if np.shape(p['forces']) != np.shape(r['inputs']['positions']):
                    raise ValueError('Energy/force output atom count mismatch')
            np.savez(out/'predictions.npz', energies=np.asarray([p['energy'] for _,p in pairs],float),
                     forces=np.asarray([p['forces'] for _,p in pairs],float))
            names = ['load_and_validate']
        elif task == 'CHEM_10':
            positions = []
            for r,p in pairs:
                pos = np.asarray(p['positions'],float)
                if pos.shape != np.shape(r['inputs']['reactant']['positions']):
                    raise ValueError('Transition-state output atom count mismatch')
                positions.append(pos)
            with (out/'predictions.pkl').open('wb') as f:
                pickle.dump(dict(predicted_ts_positions=positions),f)
            names = ['load_and_validate']
        elif task == 'LIFE_13':
            rows = []
            for r,p in pairs:
                if len(p['seq']) != len(r['inputs']['coords']['CA']):
                    raise ValueError('Designed sequence length differs from input backbone')
                rows.append(dict(name=r['inputs']['name'],seq=p['seq']))
            (out/'predictions.json').write_text(json.dumps(rows,allow_nan=False))
            names = ['load_and_validate']
        elif task == 'MAT_05':
            if instance not in ('sunset_1/test_iid','sunset_1/test_ood'):
                raise ValueError('Unknown nanoparticle final split')
            out = output/'sunset_1'; out.mkdir(parents=True,exist_ok=True)
            filename = instance.split('/')[-1]+'_pred.csv'
            def identifier(r):
                value = r['inputs']['_id']
                if isinstance(value,dict):
                    value = value.get('oid',value.get('$oid'))
                if not isinstance(value,str) or not value:
                    raise ValueError('Missing nanoparticle input identifier')
                return value
            csv_file(out/filename, ['_id','target'],
                     ({'_id':identifier(r),'target':p['target']} for r,p in pairs))
            names = ['_load_pred_csv']
        else:
            columns = ['Dev_log2_enrichment','Hk_log2_enrichment'] if instance == 'deepstarr' else ['label']
            csv_file(out/'predictions.csv',columns,({k:p[k] for k in columns} for _,p in pairs))
            names = ['load_and_validate_regression' if instance == 'deepstarr' else 'load_and_validate_classification']
        if native_checks:
            loaders = prediction_loaders(source,output,names)
            if task == 'MAT_05':
                loaders[names[0]](str(out/filename),len(pairs))
            else:
                loaders[names[0]](instance)
        counts[instance] = len(pairs)
        check_methods[instance]='native prediction loader'
    report = dict(instances=counts, native_prediction_format_checked=native_checks,
                native_source_sha256=sha256(source), test_truth_read=False,check_methods=check_methods,
                files={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    if task == PROPERTY_TASK:
        report['evaluation_protocol'] = evaluation_protocol()
        # Input-only checks can cover an instance subset; terminal scoring
        # independently requires all seven instances before claiming completeness.
        from benchmarks.naturebench_active.adapters.multi_properties import INSTANCES
        report['evaluation_protocol']['experiment_coverage_complete'] = set(counts) == set(INSTANCES)
    return report


def probability(value):
    result=scalar(value)
    if not 0<=result<=1:raise ValueError('Prediction is not a probability')
    return result


def sole(prediction):
    if len(prediction)!=1:raise ValueError('Expected one scalar target')
    return scalar(next(iter(prediction.values())))


def write_instance(task,instance,pairs,out,output,source,context,native_checks):
    names=[]; checks='input-only schema, identifiers, count, finite values'
    if task==PLM_TASK:
        from benchmarks.naturebench_active.adapters.plm_interact import probability as plm_probability, INSTANCES
        if instance not in INSTANCES:raise ValueError('Unknown PLM-interact instance')
        csv_file(out/'predictions.csv',['prediction_score'],
                 ({'prediction_score':plm_probability(p)} for _,p in pairs))
        # The upstream loader opens y_ref.csv for its row-count check. Here
        # count/order come exclusively from the provided unlabeled records.
    elif task==RESP_TASK:
        from benchmarks.naturebench_active.adapters.resp import probabilities
        if instance!=RESP_INSTANCE:raise ValueError('Unknown RESP instance')
        rows=[]
        for row,prediction in pairs:
            p=probabilities(prediction)
            rows.append(dict(sample_id=row['sample_id'],p_RH01=p[0],p_RH02=p[1],p_RH03=p[2],
                             predicted_class=int(p.argmax())))
        csv_file(out/'predictions.csv',['sample_id','p_RH01','p_RH02','p_RH03','predicted_class'],rows)
    elif task=='LIFE_16':
        from benchmarks.naturebench_active.adapters.noncoding import validate_prediction, BINS
        if instance not in {'gtex_v8/'+b for b in BINS}:raise ValueError('Unknown noncoding bin')
        keys=['phenotype_id','variant_id','bulk']
        ids=[tuple(r['inputs'][k] for k in keys) for r,_ in pairs]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate noncoding output identity')
        columns=keys+['predicted_label','predicted_probability','predicted_slope']
        csv_file(out/'predictions.csv',columns,
                 ({**{k:r['inputs'][k] for k in keys},**validate_prediction(p)} for r,p in pairs))
    elif task=='LIFE_15':
        ids=[r['inputs']['index'] for r,_ in pairs]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate essentiality input ID')
        csv_file(out/'predictions.csv',['index','prediction'],
                 ({'index':r['inputs']['index'],'prediction':probability(sole(p))} for r,p in pairs))
    elif task=='LIFE_10':
        ids=[r['inputs']['variant'] for r,_ in pairs]
        if len(set(ids))!=len(ids) or any(not isinstance(x,str) or not x for x in ids):
            raise ValueError('Invalid variant input identifiers')
        with (out/'predictions.tsv').open('w',newline='') as stream:
            writer=csv.writer(stream,delimiter='\t');writer.writerow(['variant','score'])
            writer.writerows((r['inputs']['variant'],scalar(p['score'])) for r,p in pairs)
    elif task=='CHEM_03':
        regression=instance in ('esol','lipophilicity','freesolv')
        multiple=instance in ('sider','tox21','toxcast')
        columns=sorted(pairs[0][1]) if multiple else ['prediction']
        rows=[]
        for _,p in pairs:
            if multiple:
                if sorted(p)!=columns:raise ValueError('Multitarget schema changed')
                rows.append({k:probability(p[k]) for k in columns})
            else:rows.append({'prediction':sole(p) if regression else probability(sole(p))})
        csv_file(out/'predictions.csv',columns,rows)
    elif task=='CHEM_05':
        reaction=instance in ('buchwald_hartwig','suzuki')
        filename=f"predictions_split_{context['repeated_split']}.csv" if reaction else 'predictions.csv'
        ids=context.get('identifiers',{})
        csv_file(out/filename,['index','prediction'],
            ({'index':ids[r['sample_id']] if reaction else i,'prediction':sole(p)} for i,(r,p) in enumerate(pairs)))
    elif task=='MAT_02':
        values={r['inputs']['structure_id']:scalar(p['value']) for r,p in pairs}
        if len(values)!=len(pairs):raise ValueError('Duplicate MOF input ID')
        (out/'predictions.json').write_text(json.dumps(values,allow_nan=False))
    elif task=='MAT_03':
        rows=[]
        for r,p in pairs:
            loading=scalar(p['load_norm_class'])
            if loading!=int(loading) or not 0<=loading<=47:raise ValueError('Invalid loading class')
            rows.append(dict(SMILES=r['inputs']['SMILES'],Zeolite=r['inputs']['Zeolite'],
                             b_pred=int(probability(p['b'])>=.5),E_pred=scalar(p['Binding (SiO2)']),
                             BE_pred=scalar(p['BE']),load_class_pred=int(loading)))
        csv_file(out/'predictions.csv',['SMILES','Zeolite','b_pred','E_pred','BE_pred','load_class_pred'],rows)
    elif task=='LIFE_06':
        rows=[]
        for r,p in pairs:
            row={'pair_id':r['inputs']['pair_id']}
            for side in ('1','2'):
                scores=finite_array(p['labels'+side],(len(r['inputs']['sequence'+side]),),'PPI scores',True)
                row['scores'+side]=';'.join(map(str,scores))
            rows.append(row)
        if len({r['pair_id'] for r in rows})!=len(rows):raise ValueError('Duplicate PPI input ID')
        csv_file(out/'predictions.csv',['pair_id','scores1','scores2'],rows)
        names=['load_and_validate']
    elif task=='LIFE_08':
        loader=prediction_loaders(source,output,['load_and_validate']) if native_checks else None
        for r,p in pairs:
            length=len(r['inputs']['V_dna']);value=finite_array(p['pwm'],(length,4),'PWM',True)
            if not np.allclose(value.sum(axis=1),1.,atol=1e-5):raise ValueError('PWM row sum differs from one')
            name=context['identifiers'][r['sample_id']]
            if not isinstance(name,str) or '/' in name or name in ('','.','..'):raise ValueError('Unsafe DNA output name')
            np.save(out/(name+'.npy'),value)
            if loader:loader['load_and_validate'](name,instance,length)
        checks='native prediction loader; input-derived names and dimensions'
    elif task=='LIFE_04':
        np.save(out/'predictions.npy',np.asarray([probability(p['cancer_gene']) for _,p in pairs],float))
        if native_checks:prediction_loaders(source,output,['load_and_validate'])['load_and_validate'](instance,len(pairs))
        checks='native prediction loader; input-derived count'
    elif task=='LIFE_07':
        from benchmarks.naturebench_active.scoring.metrics import native_functions
        from benchmarks.naturebench_active.adapters.pdb import fixed_columns
        native=native_functions(source,['parse_pdb'],['PHO_ATOMS','SUGAR_ATOMS','NUC_MAP'])
        native.update(defaultdict=defaultdict)
        atoms=native['PHO_ATOMS']+native['SUGAR_ATOMS']+['N1']
        ids=set();formatting={}
        for r,p in pairs:
            name=r['inputs']['target_id']
            if '/' in name or name in ids:raise ValueError('Invalid cryo input ID')
            ids.add(name);path=out/(name+'.pdb')
            text,format_report=fixed_columns(p['pdb']);path.write_text(text);formatting[name]=format_report
            residues=native['parse_pdb'](str(path),atoms)
            sequences=r['inputs']['sequences'];declared=sum(map(len,sequences))
            atom_residues={(line[21],line[22:26]) for line in text.splitlines() if line.startswith('ATOM  ')}
            ambiguous=sum(base not in 'ACGTU' for sequence in sequences for base in sequence)
            # Native NUC_MAP omits N/DN. Preserve ambiguous input residues and
            # their coordinates without fabricating a known nucleotide type.
            if len(atom_residues)!=declared or not declared-ambiguous<=len(residues)<=declared:
                raise ValueError('Cryo residue count differs from sequences')
            format_report.update(declared_input_residues=declared,native_parsed_residues=len(residues),
                                 ambiguous_input_residues=ambiguous,ignored_by_native_parser=declared-len(residues))
            for residue in residues.values():
                if not np.isfinite(np.asarray(list(residue.values()),float)).all():raise ValueError('Nonfinite cryo coordinates')
        (out/'pdb_formatting.json').write_text(json.dumps(formatting,sort_keys=True))
        checks='native PDB parser; input identifiers, sequence lengths, finite coordinates; fixed-width rounding recorded separately'
    elif task=='CHEM_07':
        from rdkit import Chem
        from benchmarks.naturebench_active.scoring.linker import consistent_smiles
        smi=[];sdf=[]
        for r,p in pairs:
            if set(p)!={'molecule','linker','molblock'}:raise ValueError('Linker schema changed')
            if not p['molecule']:
                if p['linker'] or p['molblock']:raise ValueError('Partial failed linker candidate')
                smi.append(r['inputs']['fragments']+' INVALID')
                sdf.append(Chem.MolToMolBlock(Chem.MolFromSmiles('')))
                continue
            graph=Chem.MolFromSmiles(p['molecule']);mol=Chem.MolFromMolBlock(p['molblock'],removeHs=False)
            if graph is None or len(Chem.GetMolFrags(graph))!=1 or mol is None or not mol.GetNumConformers():
                raise ValueError('Invalid nonempty linker candidate')
            if not np.isfinite(mol.GetConformer().GetPositions()).all():raise ValueError('Nonfinite linker coordinates')
            if not consistent_smiles(mol,p['molecule']):
                raise ValueError('Linker SMILES/SDF mismatch')
            smi.append(' '.join([r['inputs']['fragments'],p['molecule'],p['linker']]).rstrip())
            sdf.append(p['molblock'].removesuffix('$$$$\n'))
        (out/'output.smi').write_text('\n'.join(smi)+'\n')
        (out/'output.sdf').write_text(''.join(block.rstrip()+'\n$$$$\n' for block in sdf))
        if native_checks:
            loader=prediction_loaders(source,output,['load_and_validate']);loader['Chem']=Chem
            data,molecules=loader['load_and_validate'](instance)
            if len(data)!=len(pairs) or len(molecules)!=len(pairs):raise ValueError('Native linker output count mismatch')
        checks='native prediction loader; input order, connected chemistry, SMILES/SDF consistency'
    else:raise NotImplementedError(task)
    if native_checks and names:
        prediction_loaders(source,output,names)[names[0]](instance)
        checks='native prediction loader plus input identifiers and dimensions'
    return checks
