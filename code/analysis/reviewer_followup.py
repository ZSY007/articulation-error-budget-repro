"""Recompute gate diagnostics and imbalance-aware RGB metrics from saved data.

No physics or tracker rerun. Main CV uses the original five object folds.
All predictors and both RGB endpoints share the same finite-prediction mask.
Average precision is step-integrated PR area (not trapezoidal PR area).
Balanced accuracy uses a fixed probability threshold of 0.5, without tuning.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'error_budget'))
import p1_metric_comparison_cv as cv


def average_precision(y, score):
    y = np.asarray(y, bool)
    if not y.any():
        return float('nan')
    order = np.argsort(-np.asarray(score), kind='stable')
    yy = y[order]
    ss = np.asarray(score)[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)), len(ss)-1]
    tp = np.cumsum(yy)[ends]
    recall = tp / y.sum()
    precision = tp / (ends + 1)
    return float(np.sum(np.diff(np.r_[0., recall]) * precision))


def balanced_accuracy(y, p):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float('nan')
    return float(((p[y] >= .5).mean() + (p[~y] < .5).mean()) / 2)


def rgb(dest):
    data = pd.read_csv(ROOT / 'error_budget/out/p2_tracker_transfer_20260918/predictions_rgb_looo.csv')
    assert not data.duplicated(['row_idx', 'endpoint', 'predictor']).any()
    assert data.groupby(['row_idx', 'endpoint']).y_true.nunique().eq(1).all()
    assert data.groupby('row_idx').object_id.nunique().eq(1).all()
    scores = data.pivot(index='row_idx', columns=['endpoint', 'predictor'], values='oof_pred')
    mask = np.isfinite(scores).all(axis=1)
    scores = scores.loc[mask]
    labels = data.groupby(['row_idx', 'endpoint']).y_true.first().unstack().loc[scores.index]
    objects = data.groupby('row_idx').object_id.first().loc[scores.index]
    ids = sorted(objects.unique())
    groups = {o: np.flatnonzero(objects.to_numpy() == o) for o in ids}
    rng = np.random.default_rng(12345)
    samples = [np.concatenate([groups[o] for o in rng.choice(ids, len(ids), replace=True)]) for _ in range(2000)]
    out = []
    for endpoint in ['firm', 'completion85']:
        y = labels[endpoint].to_numpy(dtype=bool)
        for name in sorted(scores[endpoint].columns):
            p = scores[endpoint, name].to_numpy()
            metrics = {'success_ap': lambda yy, pp: average_precision(yy, pp),
                       'failure_ap': lambda yy, pp: average_precision(~yy, 1-pp),
                       'balanced_accuracy_p05': balanced_accuracy}
            row = dict(endpoint=endpoint, predictor=name, n=len(y), n_objects=len(ids),
                       positives=int(y.sum()), failures=int((~y).sum()),
                       success_ap_reference=float(y.mean()), failure_ap_reference=float((~y).mean()),
                       auroc=cv.auroc(y,p), brier=float(np.mean((y-p)**2)))
            for metric, fn in metrics.items():
                row[metric] = fn(y, p)
                # Exclude single-class object draws consistently for both AP directions and BA.
                bs = np.array([fn(y[i],p[i]) if y[i].any() and not y[i].all() else np.nan for i in samples])
                row[metric+'_lo'], row[metric+'_hi'] = np.nanquantile(bs, [.025,.975])
                row[metric+'_invalid_draws'] = int((~np.isfinite(bs)).sum())
            out.append(row)
    pd.DataFrame(out).to_csv(dest/'rgb_imbalance_metrics.csv', index=False)
    pd.DataFrame({'row_idx':scores.index, 'object_id':objects.to_numpy(),
                  'firm':labels.firm, 'completion85':labels.completion85}).to_csv(dest/'rgb_common_rows.csv',index=False)
    print(pd.DataFrame(out)[['endpoint','predictor','failure_ap','balanced_accuracy_p05']].to_string(index=False), flush=True)


def gates(dest):
    rows = cv.load_revolute_mode_a()
    for r in rows:
        r['completion80'] = int(float(r['progress']) >= .8)
        r['tracking'] = int(float(r['max_track_err']) < .02)
        r['stall'] = int(float(r['sat_frac']) < .2)
        assert int(r['firm']) == r['completion80']*r['tracking']*r['stall']
    objects = [r['obj'] for r in rows]
    folds = cv.make_folds(sorted(set(objects)), cv.N_FOLDS, cv.RNG_SEED)
    (dest/'object_folds.json').write_text(json.dumps(folds,indent=2))
    records, paired, saved = [], [], {'objects': np.array(objects)}
    for gate in ['completion80','tracking','stall']:
        results = {}
        for name in cv.PREDICTORS:
            res = cv.run_one(rows, name, gate, folds, objects)
            assert res['valid'].all(), 'Do not silently change support.'
            ci = cv.bootstrap_ci(res)
            records.append(dict(gate=gate, predictor=name, n=res['n'], n_objects=len(set(objects)),
                                positives=int(res['y_all'].sum()), auroc=res['auroc'], brier=res['brier'], **ci))
            results[name] = res
            saved[f'{gate}__{name}'] = res['oof_pred']
            print(f"{gate} {name}: AUROC={res['auroc']:.6f} Brier={res['brier']:.6f}", flush=True)
        saved[gate+'__y'] = results['e_norm']['y_all']
        # Matched object resamples, fixed OOF predictions, no post hoc row selection.
        base = results['e_norm']; y = base['y_all']; objs = base['objs_all']
        ids = sorted(set(objs)); groups = {o: np.flatnonzero(objs == o) for o in ids}
        rng = np.random.default_rng(cv.RNG_SEED)
        samples = [np.concatenate([groups[o] for o in rng.choice(ids, len(ids), replace=True)]) for _ in range(2000)]
        for other in ['J_eq16','J_full']:
            a = base['oof_pred']; b = results[other]['oof_pred']
            diffs = np.array([cv.auroc(y[i],a[i])-cv.auroc(y[i],b[i]) for i in samples])
            lo,hi=np.nanquantile(diffs,[.025,.975])
            paired.append(dict(gate=gate,contrast='e_norm_minus_'+other,
                               difference=base['auroc']-results[other]['auroc'],ci_lo=lo,ci_hi=hi,
                               invalid_draws=int((~np.isfinite(diffs)).sum())))
        pd.DataFrame(records).to_csv(dest/'gate_metrics.csv',index=False)
        pd.DataFrame(paired).to_csv(dest/'gate_paired_differences.csv',index=False)
        np.savez_compressed(dest/'gate_oof_predictions.npz', **saved)

    # Archive the two manuscript endpoints as fixed OOF predictions as well. This makes
    # the published table auditable without refitting and preserves the exact fold outputs.
    main_saved = {'objects': np.array(objects)}
    for endpoint in ['firm', 'rpmart85']:
        for name in cv.PREDICTORS:
            res = cv.run_one(rows, name, endpoint, folds, objects)
            assert res['valid'].all()
            main_saved[f'{endpoint}__{name}'] = res['oof_pred']
        main_saved[endpoint+'__y'] = res['y_all']
    np.savez_compressed(dest/'p1_oof_predictions.npz', **main_saved)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--only', choices=['rgb','gates','all'], default='all')
    parser.add_argument('--output', type=Path, default=ROOT/'RAL/latex/reviewer_followup')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if args.only in ('rgb','all'): rgb(args.output)
    if args.only in ('gates','all'): gates(args.output)
