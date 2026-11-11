"""Native optimization success on official Val with public pretrained oracles.

The evaluator's metric/oracle definitions are compiled without its Test path
configuration or entrypoint. Its x_test.csv argument is an ephemeral Val-only
projection. Native source files and Test scorers are never edited.
"""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from benchmarks.naturebench_active.common import sha256


def native_namespace(source, oracle_root):
    import torch
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, QED
    namespace = dict(os=os, np=np, pd=pd, pickle=pickle, torch=torch, nn=torch.nn,
                     F=torch.nn.functional, Chem=Chem, DataStructs=DataStructs, AllChem=AllChem,
                     QED=QED, ValidationError=ValueError, GROUND_TRUTH_DIR=str(oracle_root))
    tree = ast.parse(Path(source).read_bytes())
    definitions = []
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name not in ('run_evaluation','error_result','ValidationError'):
            definitions.append(node)
        elif isinstance(node,ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError,TypeError):
                continue
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id not in ('GROUND_TRUTH_DIR','OUTPUT_DIR','DATA_DIR','EVAL_DIR','BASE_DIR'):
                    namespace[target.id] = value
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(source),'exec'),namespace)
    # The metric/oracle definitions stay native; the trusted validation process
    # sees only its assigned GPU (or CPU) through CUDA_VISIBLE_DEVICES.
    return namespace


def oracle_root(records):
    resources = records[0]['inputs'].get('resource_paths',{})
    key = 'drd2_qed/clf_py36.pkl'
    if key not in resources:
        raise ValueError('Prepared public optimization oracles are required')
    root = Path(resources[key]).parents[1]
    for relative,path in resources.items():
        if Path(path) != root/relative:
            raise ValueError('Inconsistent oracle resource root')
    return root


def score(records, values, source):
    from benchmarks.naturebench_active.adapters.pending import OPTIMIZATION
    from benchmarks.naturebench_active.scoring.pending import validate_prediction, write_instance
    for row,value in zip(records,values):
        validate_prediction(OPTIMIZATION,row,value)
    root = oracle_root(records)
    native = native_namespace(source, root)
    results = {}
    with tempfile.TemporaryDirectory(prefix='rsi-val-optimization-') as temp:
        directory = Path(temp)
        for instance in sorted({r['instance'] for r in records}):
            all_rows = [(r,v) for r,v in zip(records,values) if r['instance']==instance]
            selected = [(r,v) for r,v in all_rows if instance!='drd2_qed' or r['inputs']['task_name']=='drd2 qed']
            if not selected:
                raise ValueError('Official Val has no rows for the native joint optimization metric')
            out = directory/instance/'output';out.mkdir(parents=True)
            data = directory/instance/'data';data.mkdir()
            pd.DataFrame([r['inputs'] for r,_ in selected]).drop(columns=['resource_paths'],errors='ignore').to_csv(data/'x_test.csv',index=False)
            write_instance(OPTIMIZATION,instance,selected,out)
            if instance=='drd2_qed':
                metrics = native['evaluate_drd2_qed'](str(out),str(data))
            else:
                metrics = native['evaluate_admet'](str(out),str(data),str(root))
            if any(not math.isfinite(float(value)) for value in metrics.values()):
                raise ValueError('Nonfinite native optimization validation score')
            results[instance] = dict(count=len(selected),official_validation_records=len(all_rows),metrics=metrics)
    return dict(instances=results,test_truth_read=False,test_score_computed=False,
                native_metric_source=str(source),native_metric_sha256=sha256(source),
                validation_scope='official Val; DRD2/QED joint-task rows and all ADMET conditions; no Test-derived conditions')


def isolated_score(records, predictions, source, directory, *, python, seconds, gpu):
    from benchmarks.naturebench_active.scoring.metrics import aligned_predictions
    if seconds <= 0:
        raise TimeoutError('Research deadline exhausted before optimization scoring')
    directory = Path(directory)/'native_validation';directory.mkdir()
    inputs,output = directory/'input.json',directory/'metrics.json'
    inputs.write_text(json.dumps({'records':records,'values':aligned_predictions(records,predictions)},allow_nan=False))
    environment = {**os.environ,'CUDA_VISIBLE_DEVICES':'' if gpu is None else str(gpu),
                   'DGLBACKEND':'pytorch','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    with (directory/'scoring.log').open('w') as log:
        proc = subprocess.run([str(python),'-m',__name__,'--source',str(source),'--input',str(inputs),'--output',str(output)],
                              env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=seconds,check=False)
    if proc.returncode:
        raise RuntimeError('Native optimization validation failed: ' + str(directory/'scoring.log'))
    return json.loads(output.read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text())
    Path(args.output).write_text(json.dumps(score(payload['records'],payload['values'],Path(args.source)),allow_nan=False))
