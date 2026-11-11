"""Execute environment prerequisites before the research clock and model calls."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from benchmarks.naturebench_active.adapters.pending import POCKET, OPTIMIZATION, RNA_GENERATION
from benchmarks.naturebench_active.common import sha256
from benchmarks.naturebench_active.scoring.process import NativeProcesses
from rsi.common import ROOT, write_json


def worker(task, source, directory, gpu, cpus):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    os.sched_setaffinity(0,cpus)
    checks={}
    def check(name, function):
        started=time.monotonic()
        try:checks[name]=dict(success=True,details=function(),seconds=time.monotonic()-started)
        except Exception as exc:checks[name]=dict(success=False,error=f'{type(exc).__name__}: {exc}',seconds=time.monotonic()-started)
        write_json(directory/'environment.json',dict(task=task,python=sys.executable,gpu=gpu,cpu_affinity=cpus,
                                                    checks=checks,success=all(c['success'] for c in checks.values())))
    def cuda():
        import torch
        if gpu is None or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError('Exactly one assigned CUDA device must be available to scientific workers')
        x=torch.tensor([2.],device='cuda',requires_grad=True);loss=(x*x).sum();loss.backward()
        assert x.grad.item()==4.
        uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).strip()
        return dict(torch=torch.__version__,cuda=torch.version.cuda,visible=os.environ['CUDA_VISIBLE_DEVICES'],
                    gpu_uuid=uuid,gradient=4.)
    check('cuda_execution',cuda)
    if task==OPTIMIZATION:
        def dgl_cuda():
            import dgl
            import dgl.function as fn
            import torch
            graph=dgl.graph(([0,1],[1,0]),device='cuda')
            values=torch.ones((2,2),device='cuda',requires_grad=True)
            graph.ndata['x']=values;graph.update_all(fn.copy_u('x','m'),fn.sum('m','h'))
            graph.ndata['h'].sum().backward()
            assert values.grad.sum().item()==4.
            return dict(version=dgl.__version__,device=str(graph.device),cuda_message_passing=True)
        check('dgl_cuda_execution',dgl_cuda)
    if task==POCKET:
        def chemistry():
            import meeko
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit.Contrib.SA_Score import sascorer
            molecule=Chem.AddHs(Chem.MolFromSmiles('CCO'))
            assert AllChem.EmbedMolecule(molecule,randomSeed=42)==0
            setup=meeko.MoleculePreparation().prepare(molecule)[0]
            text,success,error=meeko.PDBQTWriterLegacy.write_string(setup)
            if not success or 'ATOM' not in text:raise RuntimeError(error or 'Meeko failed to write prepared atoms')
            (directory/'ethanol.pdbqt').write_text(text)
            return dict(meeko=meeko.__version__,pdbqt_sha256=sha256(directory/'ethanol.pdbqt'),
                        sa_score=float(sascorer.calculateScore(molecule)))
        check('meeko_parameters_and_ligand_preparation',chemistry)
        def qvina():
            source_binary=Path(source)/task/'evaluation/qvina02'
            executable=directory/'qvina02';shutil.copy2(source_binary,executable);executable.chmod(0o700)
            with (directory/'qvina-help.log').open('w') as log,NativeProcesses().managed() as processes:
                processes.run([str(executable),'--help'],stdout=log,stderr=subprocess.STDOUT,timeout=15)
            return dict(source_sha256=sha256(source_binary),executed_copy_sha256=sha256(executable),cpu_native=True)
        check('qvina_executable',qvina)
    if task==RNA_GENERATION:
        def infernal():
            hashes={}
            for name in ('cmalign','cmfetch'):
                executable=shutil.which(name)
                if not executable:raise RuntimeError('Missing '+name+' in science environment PATH')
                with (directory/(name+'-help.log')).open('w') as log,NativeProcesses().managed() as processes:
                    processes.run([executable,'-h'],stdout=log,stderr=subprocess.STDOUT,timeout=15)
                hashes[name]=dict(path=executable,sha256=sha256(Path(executable)))
            return dict(executables=hashes,backend='CPU; Infernal has no CUDA backend')
        check('infernal_executables',infernal)
    def numpy_native():
        # Exercise the actual runner compatibility path, without any task truth.
        evaluator=directory/'numpy_probe.py'
        evaluator.write_text('''import pickle
import numpy as np
def run_evaluation():
    restore=pickle.loads(b'cnumpy._core.numeric\\n_frombuffer\\n.')
    values=restore(np.array([1.,2.,3.]).tobytes(),np.dtype('float64'),(3,),'C')
    assert values.tolist()==[1.,2.,3.]
    return {'probe':{'sum':float(values.sum())}}
''')
        environment=dict(os.environ,OUTPUT_DIR=str(directory));environment.pop('RSI_NATIVE_OPTIONS',None)
        with (directory/'numpy-probe.log').open('w') as log,NativeProcesses().managed() as processes:
            processes.run([sys.executable,str(ROOT/'benchmarks/naturebench_active/scoring/native_runner.py'),str(evaluator)],
                          stdout=log,stderr=subprocess.STDOUT,env=environment,timeout=30)
        return dict(native_runner_sha256=sha256(ROOT/'benchmarks/naturebench_active/scoring/native_runner.py'),
                    numpy2_pickle_reference_decoded=True)
    check('native_numpy_compatibility',numpy_native)
    return json.loads((directory/'environment.json').read_text())


def check_environment(task, source, directory, *, python=sys.executable,gpu,resources,seconds=180):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    cpus=resources['cpu_affinity'];threads=resources.get('threads',len(cpus))
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),DGLBACKEND='pytorch',
                     PATH=str(Path(python).parent)+os.pathsep+os.environ.get('PATH',''),
                     OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS=str(threads))
    command=[python,'-m','benchmarks.naturebench_active.preflight','--worker','--task',task,
             '--source',str(source),'--directory',str(directory),'--gpu',str(gpu),'--cpus',','.join(map(str,cpus))]
    started=time.monotonic()
    source_hash=sha256(Path(__file__))
    try:
        with (directory/'worker.log').open('w') as log,NativeProcesses().managed() as processes:
            processes.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=environment,timeout=seconds)
        result=json.loads((directory/'environment.json').read_text())
        if not result['success']:raise RuntimeError('Environment prerequisites failed: '+str(result['checks']))
        if sha256(Path(__file__))!=source_hash:raise RuntimeError('Preflight source changed during execution')
        result.update(preflight_seconds=time.monotonic()-started,preflight_source_sha256=sha256(Path(__file__)),
                      research_clock_started=False,test_truth_read=False)
        write_json(directory/'environment.json',result)
        return result
    except BaseException as exc:
        report=directory/'environment.json'
        checks=json.loads(report.read_text()).get('checks',{}) if report.exists() else {}
        failed={name:row.get('error') for name,row in checks.items() if not row.get('success')}
        tail=(directory/'worker.log').read_text(errors='replace')[-2400:]
        write_json(directory/'failure.json',dict(error=f'{type(exc).__name__}: {exc}',seconds=time.monotonic()-started,
            checks=str(report),failed_checks=failed,log_tail=tail))
        if not isinstance(exc,Exception):raise
        raise RuntimeError('Environment preflight failed before research: '+
                           (json.dumps(failed) if failed else tail or str(exc))) from exc


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true',required=True)
    for name in ('task','source','directory'):p.add_argument('--'+name,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--cpus',required=True)
    a=p.parse_args();result=worker(a.task,a.source,a.directory,a.gpu,list(map(int,a.cpus.split(','))))
    sys.exit(0 if result['success'] else 1)
