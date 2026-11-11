"""Independent prediction scoring; no baseline trainer is imported."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

from .source import WORK, DATA, ROOT, COUNTS, NativeReader
from .membership import connect, native_row
from .registry import score_pairs, regression


def validate_labels(task, prediction):
    p = prediction
    if not np.isfinite(p).all() or not np.equal(p,np.floor(p)).all() or (p<0).any():
        raise ValueError('Segmentation requires finite nonnegative integer labels')
    if task == 'MED_02':
        if not np.isin(p, [0, 1, 2, 4]).all():
            raise ValueError('BraTS requires native labels 0/1/2/4')
    elif task == 'MED_05':
        if p.ndim != 3 or p.shape[-1] != 2:
            raise ValueError('CoNIC requires H×W×2')
        inst, typ = p[...,0], p[...,1]
        if (typ>6).any() or np.any((inst==0)!=(typ==0)):
            raise ValueError('Invalid CoNIC background/type')
        for ident in np.unique(inst):
            if ident and len(np.unique(typ[inst==ident])) != 1:
                raise ValueError('Instance has inconsistent type')
    elif (p > {'RS_02':14,'RS_01':1,'MED_01':3,'MED_03':2,'MED_06':2,'MED_07':28,'MED_08':1}[task]).any():
        raise ValueError('Class outside task label range')


def selected_members(index, pool=None):
    """Read only public validation labels or (organizer final path) original T."""
    db = connect(index)
    try:
        if pool is None:
            sql = "SELECT * FROM members WHERE split='T' ORDER BY ordinal"
        else:
            db.execute('ATTACH DATABASE ? AS pool',(str(pool),))
            sql = '''SELECT m.* FROM members m JOIN pool.samples s USING(sample_id)
                     WHERE s.role='val' AND s.acquired=1 ORDER BY m.ordinal'''
        yield from db.execute(sql)
    finally:
        db.close()


def regression_truths(task, members, reader):
    ys, groups, chemistry = [], [], []
    # Cache numeric label columns by native table. Avoid millions of pandas
    # iloc/DataFrame allocations for daily CAMELS targets.
    cache_key, column = None, None
    for member in members:
        row = native_row(task,member)
        selector = row['label_selector'][0]
        path = row['label_path'][0]
        if task in ('ENV_02','ENV_04'):
            key = (path,json.dumps(selector.get('read_csv',{}),sort_keys=True))
            if key != cache_key:
                column = reader.table(path,selector.get('read_csv',{}))[selector['columns'][0]].to_numpy(float)
                cache_key = key
            y = [column[selector['row']]]
        else:
            y = reader.load(path,selector).to_numpy(float).ravel()
        ys.append(np.log10(y) if task=='ENV_01' else y)
        groups.append(member['group_id'])
        if task == 'ENV_05':
            chemistry.append(member['sample_id'].split(':')[0])
    return np.asarray(ys,dtype=np.float32 if task=='ENV_02' else float),np.asarray(groups),np.asarray(chemistry)


def luna_score(run, members, reader, scratch):
    sys.path.insert(0,str(WORK/'evaluation/official/luna'))
    import noduleCADEvaluationLUNA16 as official
    header = ['seriesuid','coordX','coordY','coordZ','diameter_mm']
    included,excluded,uids,predicted = [header],[header],[],[]
    for i,member in enumerate(members):
        row = native_row('MED_04',member)
        uid = row['sample_id']
        uids.append(uid)
        for target,path,selector in zip([included,excluded],row['label_path'],row['label_selector']):
            for item in reader.load(path,selector):
                target.append([item[k] for k in header])
        with np.load(run/'predictions'/f'{i:06d}.npz',allow_pickle=False) as data:
            p = data['prediction_0']
            if str(data['sample_id']) != uid or p.ndim != 2 or p.shape[1] != 4 or len(p)>100:
                raise ValueError('Invalid LUNA ID, shape or candidate count')
            if not np.isfinite(p).all() or np.any((p[:,3]<0)|(p[:,3]>1)):
                raise ValueError('Invalid LUNA coordinate/confidence')
            predicted.extend([[uid,*point.tolist()] for point in p])
    scratch.mkdir(parents=True,exist_ok=True)
    submission = scratch/'prediction.csv'
    with submission.open('w',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['seriesuid','coordX','coordY','coordZ','probability'])
        writer.writerows(predicted)
    nodules = official.collectNoduleAnnotations(included,excluded,uids)
    result = official.evaluateCAD(uids,str(submission),str(scratch),nodules,'prediction',
                                  maxNumberOfCADMarks=100,performBootstrapping=False)
    fps,sens = result[:2]
    samples = np.interp([.125,.25,.5,1,2,4,8],fps,sens)
    if not np.isfinite(samples).all():
        raise ValueError('Nonfinite official LUNA result')
    return dict(metric='CPM',score=float(samples.mean()),sensitivity=samples.tolist(),
                bootstrap=False,max_marks=100)


def score(task, run, index, data_root=DATA, pool=None, scratch=None):
    run = Path(run)
    meta = json.loads((run/'predictions.json').read_text())
    split = 'T' if pool is None else ('V' if COUNTS[task][3] else 'validation')
    with connect(index) as db:
        if pool is None:
            count = db.execute("SELECT count(*) FROM members WHERE split='T'").fetchone()[0]
        else:
            db.execute('ATTACH DATABASE ? AS pool',(str(pool),))
            count = db.execute("SELECT count(*) FROM pool.samples WHERE role='val' AND acquired=1").fetchone()[0]
    if count == 0:
        raise ValueError('Declare internal validation from acquired labels before requesting a score')
    if meta.get('task') != task or meta.get('split') != split or meta.get('count') != count:
        raise ValueError(f'Prediction metadata must declare task={task}, split={split}, count={count}')
    # Bind the score to the ordered IDs actually consumed by the native scorer.
    # No labels enter this receipt and no extra pass over the dataset is needed.
    import hashlib
    identity = hashlib.sha256()
    membership = dict(count=0, algorithm='sha256-json-sample-id-lines-v1')
    def observed_members():
        for member in selected_members(index, pool):
            identity.update((json.dumps(member['sample_id'], ensure_ascii=True)+'\n').encode())
            membership['count'] += 1
            yield member
    members = observed_members()
    with NativeReader(data_root) as reader:
        if task.startswith('ENV_'):
            y,g,chem = regression_truths(task,members,reader)
            prediction = np.load(run/'predictions/values.npy',allow_pickle=False,mmap_mode='r')
            expected = (*y.shape,99) if task=='ENV_03' else y.shape
            if prediction.shape != expected:
                raise ValueError(f'Wrong prediction shape: {prediction.shape}; expected {expected}')
            result = regression(task,prediction,y,g)
            if task=='ENV_05':
                from sklearn.metrics import root_mean_squared_error
                result['chemistry_rmse_Ah'] = {c:float(root_mean_squared_error(y[chem==c],prediction[chem==c]))
                                               for c in np.unique(chem)}
        else:
            filenames = {p.name for p in (run/'predictions').glob('*.npz')}
            if filenames != {f'{i:06d}.npz' for i in range(count)}:
                raise ValueError('Missing or extra image prediction files')
            if task == 'MED_04':
                result = luna_score(run,members,reader,Path(scratch or run/'official_luna'))
            else:
                from .images import decode
                def pairs():
                    for i,member in enumerate(members):
                        row = native_row(task,member)
                        with np.load(run/'predictions'/f'{i:06d}.npz',allow_pickle=False) as data:
                            if str(data['sample_id']) != row['sample_id']:
                                raise ValueError('Wrong sample ID/order')
                            for j,part in enumerate(decode(row,reader,labels=True)):
                                p,t = data[f'prediction_{j}'],part['y']
                                if p.shape != t.shape:
                                    raise ValueError('Invalid native prediction shape')
                                validate_labels(task,p)
                                if part['affine'] is not None:
                                    a = data[f'affine_{j}']
                                    if a.shape != (4,4) or not np.isfinite(a).all() or not np.allclose(part['affine'],a,atol=1e-4):
                                        raise ValueError('Wrong affine')
                                yield p,t,row['group_id']
                result = score_pairs(task,pairs())
    if membership['count'] != count:
        raise ValueError('Native scorer did not consume the complete declared membership')
    membership['sample_ids_sha256'] = identity.hexdigest()
    result.update(task=task,split=split,samples=count,membership=membership)
    return result


def final_submission(task, run, index, data_root=DATA):
    """Organizer CLI, original frozen submission layout and one-shot semantics."""
    run = Path(run).resolve()
    if not run.is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('Submission must live under personal runs')
    freeze = json.loads((run/'frozen_model.json').read_text())
    if freeze.get('task') != task or not freeze.get('method'):
        raise ValueError('Require frozen task/method declaration')
    output = run/'metrics.json'
    with output.open('x') as stream:
        json.dump(dict(status='scoring',task=task,method=freeze['method']),stream)
    start = time.monotonic()
    try:
        result = score(task,run,index,data_root)
        result.update(status='completed',method=freeze['method'],test_samples=result['samples'],
                      elapsed_seconds=time.monotonic()-start)
    except Exception as exc:
        output.write_text(json.dumps(dict(status='failed',task=task,error=repr(exc))))
        raise
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result
