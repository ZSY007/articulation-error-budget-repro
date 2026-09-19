"""Item B core (user request, 2026-09-16): object-grouped cross-validated
comparison of five scalar articulation-error metrics as predictors of
manipulation success --

    e_axis (axis_err, deg)   e_pivot (pivot_err, cm)   e_geo (e_geo_nominal_cm)
    e_norm (e_norm_nominal_cm)   J_full (Heppert tangent similarity)

against two success criteria -- firm (PRIMARY endpoint: progress>=0.80 AND
max_track_err<2cm AND sat_frac<0.20) and rpmart85 (SECONDARY: progress>=0.85
only). This is the analysis that decides whether e_norm has a real
methodological edge over axis/pivot error and over the literature's own
Heppert J, or whether the paper should fall back to J's interpretability and
move its main contribution to the reference-update mechanism instead
(docs/方向调整与补实验审阅_2026-09-16.md SS4.B).

Scope: REVOLUTE objects only, mode A only (the real estimate/real execution
pairs -- the condition a fielded system actually faces; B/B2's synthetic
random directions are a distribution shift the paper is not claiming to
predict). Prismatic is reported separately and briefly: e_axis/e_geo/e_norm/
J_full nearly coincide there (no rotational "running ahead/behind" absorption
for a straight slide -- see the script's tail), so a 5-way comparison is not
informative on that subset.

Method per (predictor, criterion):
  1. Object-grouped K-fold (K=5, objects shuffled once with a fixed seed,
     so folds are identical across predictors/criteria -- a fair comparison).
  2. Per fold: fit a 1-D logistic regression (predictor -> P(success)) on the
     TRAINING objects' trials only; predict on the held-out objects' trials.
     Out-of-fold predictions cover every trial exactly once.
  3. Report, pooled over all out-of-fold predictions: AUROC (rank-based, no
     sklearn dependency), Brier score, a 10-bin calibration table.
  4. Threshold-transfer error: for target reliabilities {0.50, 0.80, 0.90},
     invert each fold's TRAIN-only logistic fit for the predictor threshold
     x* hitting that target, then measure the ACTUAL success rate achieved
     on the held-out fold at that same raw x* -- both pooled and per held-out
     object (the per-object spread is the direct answer to "does a fixed
     threshold transfer across objects").
  5. Object-level bootstrap CI (1000 resamples of the OBJECT set, not trials)
     on pooled AUROC and Brier, using the already-computed out-of-fold
     predictions re-weighted by each bootstrap object's multiplicity.

    conda activate cv
    python error_budget/p1_metric_comparison_cv.py
    -> out/p1_metric_comparison_{firm,rpmart85}.csv, out/p1_metric_comparison.md
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import optimize, stats

OUT = Path(__file__).resolve().parent / "out"
RNG_SEED = 12345
N_FOLDS = 5
N_BOOT = 1000
PREDICTORS = {  # name -> (csv column, higher_is_worse)
    "e_axis": ("axis_err", True),
    "e_pivot": ("pivot_err", True),
    "e_geo": ("e_geo_nominal_cm", True),
    "e_norm": ("e_norm_nominal_cm", True),
    "J_full": ("j_full", False),
    "J_eq16": ("j_eq16", False),
}
CRITERIA = ("firm", "rpmart85")
TARGETS = (0.50, 0.80, 0.90)


def load_revolute_mode_a():
    rows = []
    with (OUT / "p1_tangent_all.csv").open() as f:
        for r in csv.DictReader(f):
            if r["kind"] != "revolute" or r["mode"] != "A":
                continue
            rows.append(r)
    return rows


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U), ties handled via average rank."""
    y = np.asarray(y, bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(score)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_logistic_1d(x: np.ndarray, y: np.ndarray, worse_high: bool):
    """Univariate logistic P(success)=sigmoid(a*z+b), z=(x-mu)/sd standardized.
    Returns (predict_fn(x)->prob, invert_fn(target_prob)->x). worse_high flips
    the sign convention so a is expected >0 in standardized units for a "good"
    predictor (higher standardized z => higher P(success))."""
    mu, sd = float(np.mean(x)), float(np.std(x) + 1e-9)
    z = (x - mu) / sd
    if worse_high:
        z = -z

    def nll(theta):
        a, b = theta
        logits = a * z + b
        # log(1+exp(logits)) via logaddexp for stability
        ll = y * logits - np.logaddexp(0, logits)
        return -np.sum(ll)

    res = optimize.minimize(nll, x0=np.array([0.0, 0.0]), method="Nelder-Mead",
                            options=dict(xatol=1e-6, fatol=1e-6, maxiter=2000))
    a, b = res.x

    def predict(xq):
        zq = (np.asarray(xq, float) - mu) / sd
        if worse_high:
            zq = -zq
        logits = np.clip(a * zq + b, -35.0, 35.0)  # avoid exp overflow; saturates at ~1e-15/1-1e-15
        return 1.0 / (1.0 + np.exp(-logits))

    def invert(target_p):
        # a*z+b = logit(target_p)  ->  z = (logit(target_p)-b)/a
        target_p = np.clip(target_p, 1e-6, 1 - 1e-6)
        logit_t = np.log(target_p / (1 - target_p))
        if abs(a) < 1e-9:
            return float("nan")
        z_star = (logit_t - b) / a
        if worse_high:
            z_star = -z_star
        return float(z_star * sd + mu)

    return predict, invert


def make_folds(objects: list[str], k: int, seed: int) -> list[list[str]]:
    objs = sorted(objects)
    rng = np.random.default_rng(seed)
    rng.shuffle(objs)
    return [objs[i::k] for i in range(k)]


def run_one(rows, pred_name, crit, folds, obj_of_row):
    col, worse_high = PREDICTORS[pred_name]
    x_all = np.array([float(r[col]) for r in rows])
    y_all = np.array([int(r[crit]) for r in rows], dtype=bool)
    objs_all = np.array(obj_of_row)

    oof_pred = np.full(len(rows), np.nan)
    fold_transfer = {t: [] for t in TARGETS}          # pooled |achieved-target| per fold
    per_obj_achieved = {t: {} for t in TARGETS}        # target -> {obj: achieved_rate}

    for held_out in folds:
        held_out = set(held_out)
        train_mask = ~np.isin(objs_all, list(held_out))
        test_mask = ~train_mask
        if train_mask.sum() < 10 or test_mask.sum() == 0:
            continue
        predict, invert = fit_logistic_1d(x_all[train_mask], y_all[train_mask], worse_high)
        oof_pred[test_mask] = predict(x_all[test_mask])

        for t in TARGETS:
            x_star = invert(t)
            if not np.isfinite(x_star):
                continue
            if worse_high:
                sel = x_all[test_mask] <= x_star
            else:
                sel = x_all[test_mask] >= x_star
            y_test = y_all[test_mask]
            if sel.sum() > 0:
                achieved = float(y_test[sel].mean())
                fold_transfer[t].append(abs(achieved - t))
            # per-object achieved rate at this threshold (only objects with >=5 selected trials)
            test_objs = objs_all[test_mask]
            for o in held_out:
                om = (test_objs == o) & sel
                if om.sum() >= 5:
                    per_obj_achieved[t].setdefault(o, []).append(float(y_test[om].mean()))

    valid = np.isfinite(oof_pred)
    return dict(
        pred_name=pred_name, crit=crit,
        n=int(valid.sum()),
        auroc=auroc(y_all[valid], oof_pred[valid]),  # oof_pred is P(success) already
        brier=float(np.mean((oof_pred[valid] - y_all[valid]) ** 2)),
        oof_pred=oof_pred, y_all=y_all, objs_all=objs_all, valid=valid,
        transfer_mae={t: (float(np.mean(v)) if v else float("nan"))
                     for t, v in fold_transfer.items()},
        per_obj_achieved=per_obj_achieved,
    )


def bootstrap_ci(result, n_boot=N_BOOT, seed=RNG_SEED):
    valid = result["valid"]
    y = result["y_all"][valid]
    p = result["oof_pred"][valid]
    objs = result["objs_all"][valid]
    uniq_objs = np.array(sorted(set(objs)))
    idx_by_obj = {o: np.where(objs == o)[0] for o in uniq_objs}
    rng = np.random.default_rng(seed)
    aucs, briers = [], []
    for _ in range(n_boot):
        sample_objs = rng.choice(uniq_objs, size=len(uniq_objs), replace=True)
        idx = np.concatenate([idx_by_obj[o] for o in sample_objs])
        aucs.append(auroc(y[idx], p[idx]))
        briers.append(float(np.mean((p[idx] - y[idx]) ** 2)))
    aucs = np.array([a for a in aucs if np.isfinite(a)])
    briers = np.array(briers)
    return dict(auroc_lo=float(np.percentile(aucs, 2.5)), auroc_hi=float(np.percentile(aucs, 97.5)),
               brier_lo=float(np.percentile(briers, 2.5)), brier_hi=float(np.percentile(briers, 97.5)))


def calibration_table(result, n_bins=10):
    valid = result["valid"]
    y, p = result["y_all"][valid], result["oof_pred"][valid]
    order = np.argsort(p)
    y, p = y[order], p[order]
    bins = np.array_split(np.arange(len(p)), n_bins)
    rows = []
    for b in bins:
        if len(b) == 0:
            continue
        rows.append((float(p[b].mean()), float(y[b].mean()), len(b)))
    return rows


def main():
    rows = load_revolute_mode_a()
    print(f"loaded {len(rows)} revolute mode-A trials from "
         f"{len(set(r['obj'] for r in rows))} objects")
    obj_of_row = [r["obj"] for r in rows]
    objects = sorted(set(obj_of_row))
    folds = make_folds(objects, N_FOLDS, RNG_SEED)
    print(f"object-grouped {N_FOLDS}-fold CV, fold sizes: {[len(f) for f in folds]}")

    md = [f"# P1 item-B metric comparison (object-grouped {N_FOLDS}-fold CV, "
         f"REVOLUTE mode-A trials, n_objects={len(objects)})\n\n"]
    all_results = {}
    for crit in CRITERIA:
        md.append(f"## {crit}{'  (PRIMARY)' if crit == 'firm' else '  (secondary)'}\n\n")
        md.append("| metric | AUROC | 95% CI | Brier | 95% CI | "
                  "transfer MAE @50/80/90 |\n")
        md.append("|---|---|---|---|---|---|\n")
        rows_csv = []
        for pred_name in PREDICTORS:
            res = run_one(rows, pred_name, crit, folds, obj_of_row)
            ci = bootstrap_ci(res)
            tm = res["transfer_mae"]
            tm_str = "/".join(f"{tm[t]:.3f}" if np.isfinite(tm[t]) else "NA" for t in TARGETS)
            md.append(f"| {pred_name} | {res['auroc']:.3f} | "
                     f"[{ci['auroc_lo']:.3f},{ci['auroc_hi']:.3f}] | "
                     f"{res['brier']:.4f} | [{ci['brier_lo']:.4f},{ci['brier_hi']:.4f}] | "
                     f"{tm_str} |\n")
            rows_csv.append(dict(metric=pred_name, criterion=crit, n=res["n"],
                                 auroc=res["auroc"], auroc_lo=ci["auroc_lo"], auroc_hi=ci["auroc_hi"],
                                 brier=res["brier"], brier_lo=ci["brier_lo"], brier_hi=ci["brier_hi"],
                                 transfer_mae_50=tm[0.50], transfer_mae_80=tm[0.80], transfer_mae_90=tm[0.90]))
            all_results[(pred_name, crit)] = res
            print(f"  {crit:10s} {pred_name:8s}  AUROC={res['auroc']:.3f} "
                 f"[{ci['auroc_lo']:.3f},{ci['auroc_hi']:.3f}]  Brier={res['brier']:.4f}  "
                 f"transferMAE={tm_str}")
        csv_p = OUT / f"p1_metric_comparison_{crit}.csv"
        with csv_p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader(); w.writerows(rows_csv)
        md.append("\n")

        # per-object threshold-transfer dispersion at target=0.80 (most policy-relevant)
        md.append(f"### {crit}: per-object achieved rate at the train-fitted 80%-target "
                  f"threshold (dispersion = does ONE threshold transfer across objects)\n\n")
        md.append("| metric | median | IQR | objects within +-0.15 of 0.80 |\n")
        md.append("|---|---|---|---|\n")
        for pred_name in PREDICTORS:
            res = all_results[(pred_name, crit)]
            vals = [np.mean(v) for v in res["per_obj_achieved"][0.80].values()]
            if not vals:
                md.append(f"| {pred_name} | NA | NA | NA |\n")
                continue
            vals = np.array(vals)
            within = int(np.sum(np.abs(vals - 0.80) <= 0.15))
            md.append(f"| {pred_name} | {np.median(vals):.3f} | "
                     f"[{np.percentile(vals,25):.3f},{np.percentile(vals,75):.3f}] | "
                     f"{within}/{len(vals)} |\n")
        md.append("\n")

        md.append(f"### {crit}: calibration table (10 out-of-fold bins, pred vs actual)\n\n")
        for pred_name in PREDICTORS:
            res = all_results[(pred_name, crit)]
            ct = calibration_table(res)
            md.append(f"**{pred_name}**: " + "  ".join(
                f"({p_:.2f}->{a:.2f},n={n})" for p_, a, n in ct) + "\n\n")

    md_p = OUT / "p1_metric_comparison.md"
    md_p.write_text("".join(md))
    print(f"\n-> {md_p}")


if __name__ == "__main__":
    main()
