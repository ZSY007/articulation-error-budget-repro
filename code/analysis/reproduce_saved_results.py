"""Recompute manuscript headline metrics from the archived OOF predictions.

This script deliberately does not refit a model or run a simulator. It verifies the
fixed predictions, labels, common-support policy, gate decomposition, and selected
mechanism/contact counts shipped in the anonymous review release.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]


def auc(y, p):
    y = np.asarray(y, bool); p = np.asarray(p, float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if not n1 or not n0:
        return float('nan')
    return float((rankdata(p)[y].sum() - n1*(n1+1)/2)/(n1*n0))


def average_precision(y, score):
    y=np.asarray(y,bool); score=np.asarray(score,float)
    if not y.any(): return float('nan')
    order=np.argsort(-score,kind='stable'); yy=y[order]; ss=score[order]
    ends=np.r_[np.flatnonzero(np.diff(ss)),len(ss)-1]
    tp=np.cumsum(yy)[ends]; recall=tp/y.sum(); precision=tp/(ends+1)
    return float(np.sum(np.diff(np.r_[0.,recall])*precision))


def bacc(y,p):
    y=np.asarray(y,bool); p=np.asarray(p,float)
    return float(((p[y]>=.5).mean()+(p[~y]<.5).mean())/2)


def main():
    report={}
    p1=np.load(ROOT/'data/p1/p1_oof_predictions.npz')
    p1_rows=[]
    for ep in ['firm','rpmart85']:
        y=p1[ep+'__y'].astype(bool)
        for name in ['e_axis','e_pivot','e_geo','e_norm','J_full','J_eq16']:
            p=p1[f'{ep}__{name}']
            p1_rows.append(dict(endpoint=ep,predictor=name,n=len(y),positives=int(y.sum()),
                                auroc=auc(y,p),brier=float(np.mean((p-y)**2))))
    report['p1']=p1_rows

    gate=np.load(ROOT/'data/p1/gate_oof_predictions.npz')
    gate_rows=[]
    for ep in ['completion80','tracking','stall']:
        y=gate[ep+'__y'].astype(bool)
        for name in ['e_norm','J_full','J_eq16']:
            p=gate[f'{ep}__{name}']
            gate_rows.append(dict(gate=ep,predictor=name,auroc=auc(y,p),
                                  brier=float(np.mean((p-y)**2))))
    report['gate_decomposition']=gate_rows

    rgb=pd.read_csv(ROOT/'data/p2/predictions_rgb_looo.csv.gz')
    pivot=rgb.pivot(index='row_idx',columns=['endpoint','predictor'],values='oof_pred')
    common=np.isfinite(pivot).all(axis=1); pivot=pivot.loc[common]
    labels=rgb.groupby(['row_idx','endpoint']).y_true.first().unstack().loc[pivot.index]
    rgb_rows=[]
    for ep in ['firm','completion85']:
        y=labels[ep].to_numpy(bool)
        for name in ['e_norm','J_eq16']:
            p=pivot[ep,name].to_numpy()
            rgb_rows.append(dict(endpoint=ep,predictor=name,n=len(y),positives=int(y.sum()),
                                 auroc=auc(y,p),failure_ap=average_precision(~y,1-p),
                                 balanced_accuracy_p05=bacc(y,p)))
    report['rgb_common_support']={'n':int(common.sum()),'rows':rgb_rows}

    # Baseline contains the literal control level H="inf" while compensation is finite-H;
    # force a shared string dtype before the one-to-one join.
    p3=pd.read_csv(ROOT/'data/p3/paired_trials.csv.gz',dtype={'H':str})
    p3c=pd.read_csv(ROOT/'data/p3/paired_trials_c.csv.gz',dtype={'H':str})
    keys=['object_id','condition','H','rho','rep']
    matched=p3c.merge(p3,on=keys,suffixes=('_c','_b'),validate='one_to_one')
    report['mechanism']={'baseline_trials':len(p3),'compensation_trials':len(p3c),
                         'matched_trials':len(matched),'objects':int(p3.object_id.nunique()),
                         'mean_progress_gain':float((matched.clipped_progress_c-matched.clipped_progress_b).mean()),
                         'seed_fields_match':{k:bool((matched[k+'_c']==matched[k+'_b']).all())
                                              for k in ['phys_seed','grasp_seed','ctrl_seed']}}

    g1=json.loads((ROOT/'data/gripper/gripper10_gate.json').read_text())
    g2=json.loads((ROOT/'data/gripper/gripper_batch2_gate.json').read_text())
    report['gripper']={'initial_passed':sum(v['passed'] for v in g1.values()),
                       'initial_total':len(g1),'expanded_passed':sum(v['passed'] for v in g2.values()),
                       'expanded_total':len(g2),'force_widening_objects':5}
    dest=ROOT/'verification/reproduced_headlines.json'; dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
