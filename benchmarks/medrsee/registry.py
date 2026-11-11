"""Official metric adapters. Prediction arrays remain on their native grids."""
import importlib.util, sys
import numpy as np
from .source import WORK

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,WORK/'evaluation/official'/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def dice(pred,true):
    import torch
    from monai.metrics import compute_dice
    p=torch.as_tensor(pred.astype(np.float32))[None,None]
    t=torch.as_tensor(true.astype(np.float32))[None,None]
    return float(compute_dice(p,t,include_background=True,ignore_empty=False).item())

def score_pairs(task,pairs):
    """Each item: (prediction, truth, group). Callers validate IDs and geometry."""
    if task=='MED_02':
        from .brats_metrics import region_dice
        details = [dict(group=g, **region_dice(p, t)) for p, t, g in pairs]
        if not details:
            raise ValueError('No BraTS cases selected')
        return dict(metric='region_dice', score=float(np.mean([x['score'] for x in details])),
                    regions={k:float(np.mean([x['regions'][k] for x in details]))
                             for k in ('WT','TC','ET')}, details=details)
    if task=='RS_02':
        pp=[];tt=[]
        for p,t,g in pairs:
            valid=t>=0;pp.append(p[valid].ravel());tt.append(t[valid].ravel())
        out=module('mados_metrics','mados/utils/metrics.py').Evaluation(np.concatenate(pp),np.concatenate(tt))
        return {'metric':'macro_f1','score':float(out['macroF1']),'details':{k:float(v) for k,v in out.items()}}
    if task=='RS_01':
        official=module('landslide_metrics','landslide/utils/tools.py')
        totals=np.zeros((4,2,1))
        for p,t,g in pairs:
            vals=official.eval_image(p.ravel(),t.ravel(),2);totals+=np.array(vals[:4])
        tp,fp,tn,fn=totals[:,1,0]
        return dict(metric='landslide_f1',score=float(2*tp/(2*tp+fp+fn+1e-14)),tp=int(tp),fp=int(fp),fn=int(fn))
    if task=='MED_05':
        sys.path.insert(0,str(WORK/'evaluation/official/conic'))
        from metrics.stats_utils import get_multi_pq_info
        stats=np.zeros((6,4))
        for p,t,g in pairs:stats+=np.asarray(get_multi_pq_info(t,p),float)
        tp,fp,fn,iou=stats.T
        pq=tp/(tp+.5*fp+.5*fn+1e-6)*iou/(tp+1e-6)
        return dict(metric='mPQ+',score=float(pq.mean()),per_class=pq.tolist(),statistics=stats.tolist())
    if task=='MED_01':
        official=module('acdc_metrics','acdc.py')
        groups={};details=[]
        for p,t,g in pairs:
            values=[float(v) for v in official.metrics(t,p,(1,1,1))[::3]]
            groups.setdefault(g,[]).append(float(np.mean(values)))
            details.append(dict(group=g,labels=[3,1,2],dice=values))
        group_scores={g:float(np.mean(v)) for g,v in groups.items()}
        return dict(metric='three_structure_dice',score=float(np.mean(list(group_scores.values()))),group_scores=group_scores,details=details)
    groups={};all_scores=[]
    verse=module('verse_metrics','verse_compat.py') if task=='MED_07' else None
    for p,t,g in pairs:
        if task=='MED_07':
            labels=sorted(set(np.unique(t).tolist())|set(np.unique(p).tolist()))
            labels=[v for v in labels if v>0]
        else:labels={'MED_01':[1,2,3],'MED_03':[1,2],'MED_06':[1,2],'MED_08':[1]}[task]
        values=[]
        for c in labels:
            pp=p>=1 if task=='MED_03' and c==1 else p==c
            tt=t>=1 if task=='MED_03' and c==1 else t==c
            values.append(float(verse.compute_dice(pp,tt)) if verse else dice(pp,tt))
        if not values:raise ValueError('No foreground identities')
        groups.setdefault(g,[]).append(float(np.mean(values)))
        all_scores.append(dict(group=g,labels=labels,dice=values))
    means={g:float(np.mean(v)) for g,v in groups.items()}
    return dict(metric='patient_mean_dice' if task=='MED_08' else 'structure_dice',score=float(np.mean(list(means.values()))),group_scores=means,details=all_scores)

def regression(task,pred,true,groups):
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_pinball_loss, r2_score
    if not np.isfinite(pred).all() or (task!='ENV_03' and not np.isfinite(true).all()):raise ValueError('Nonfinite metric input')
    if task=='ENV_01':return dict(metric='log10_chla_rmse',score=float(root_mean_squared_error(true,pred)))
    if task=='ENV_03':
        assert pred.shape==(*true.shape,99),(pred.shape,true.shape)
        valid=np.isfinite(true)
        if not valid.any():raise ValueError('No observed target hours')
        values=[mean_pinball_loss(true[valid],pred[...,i][valid],alpha=(i+1)/100) for i in range(99)]
        return dict(metric='mean_pinball_loss',score=float(np.mean(values)),per_quantile=[float(v) for v in values],observed_hours=int(valid.sum()),missing_target_hours=int((~valid).sum()))
    values={}
    for g in np.unique(groups):
        t=true[groups==g].ravel();p=pred[groups==g].ravel()
        if task=='ENV_02':
            den=float(np.sum((t-t.mean())**2))
            if den<=0:raise ValueError('Undefined NSE for constant basin')
            values[str(g)]=float(r2_score(t,p,force_finite=False))
        else:values[str(g)]=float(mean_absolute_error(t,p))
    return dict(metric='basin_macro_nse' if task=='ENV_02' else 'group_equal_mae',score=float(np.mean(list(values.values()))),group_scores=values,mae=float(mean_absolute_error(true,pred)))
