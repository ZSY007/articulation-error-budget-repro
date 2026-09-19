"""Item B follow-up (user request, 2026-09-17): decompose `firm` into its three
actual gate components plus a continuous force diagnostic, and run the SAME
object-grouped CV comparison (error_budget/p1_metric_comparison_cv.py) against
each separately, instead of only the blended `firm`/`rpmart85` labels.

`firm = progress>=0.80 AND max_track_err<0.02 AND sat_frac<0.20` (pilot_object.py).
Decomposed into three binary targets, scored exactly like p1_metric_comparison_cv.py
scores firm/rpmart85 (object-grouped 5-fold CV, AUROC/Brier, object-level bootstrap
CI) -- NOT three new metrics, three new labels on the SAME 73-object/74752-trial
revolute mode-A corpus:

  Y1 completion       progress >= 0.80                      ("did the joint get there")
  Y2 tracking_ok       max_track_err < 2cm                   ("did it stay on path")
  Y3 quality_ok        sat_frac < 0.20                        ("did it avoid stalling")

Y4 (force/impulse) has no natural binary cutoff in firm's own definition -- peak_force
is an auxiliary diagnostic, not one of firm's three gates -- so it is reported as a
continuous outcome (object-grouped bootstrap CI on Spearman rho vs. each predictor),
not forced into the same AUROC framework as a fabricated threshold would be.

The hypothesis under test: if e_norm's firm-AUROC advantage (0.998 vs J_full's 0.975,
p1_metric_comparison_cv.py) is really about EXECUTION FIDELITY specifically, e_norm
should win Y2 (tracking) by a wide margin and Y3 (contact/stall) should follow it;
if J_full's rpmart85 advantage is really about REACHABILITY, J_full should win Y1
(completion) even more clearly than it wins the blended rpmart85 label.

    conda activate cv
    python error_budget/p1_criterion_decomposition_cv.py
    -> out/p1_criterion_decomposition_{Y1_completion,Y2_tracking,Y3_quality}.csv
    -> out/p1_criterion_decomposition_force.csv (continuous, Y4)
    -> out/p1_criterion_decomposition.md
"""
from __future__ import annotations

import csv
import glob
from pathlib import Path

import numpy as np
from scipy import stats

from p1_metric_comparison_cv import (
    PREDICTORS, N_FOLDS, RNG_SEED, N_BOOT,
    auroc, fit_logistic_1d, make_folds, bootstrap_ci,
)

OUT = Path(__file__).resolve().parent / "out"

TARGETS = {
    "Y1_completion": "progress_ge_80",
    "Y2_tracking":   "track_ok_2cm",
    "Y3_quality":    "quality_ok_20pct",
}


def load_revolute_mode_a_with_force():
    """Same corpus as p1_metric_comparison_cv.py (revolute, mode A, out/p1_tangent_
    all.csv) but joined against the original per-trial CSVs for `peak_force`, which
    p1_tangent_similarity_offline.py did not carry over."""
    tan_rows = []
    with (OUT / "p1_tangent_all.csv").open() as f:
        for r in csv.DictReader(f):
            if r["kind"] == "revolute" and r["mode"] == "A":
                tan_rows.append(r)

    # index peak_force by (obj, selector, estimator, src_row, rep, mode)
    force_idx = {}
    for fp in sorted(glob.glob(str(OUT / "p0_*_flow_gaussian_L1_*_se3.csv"))):
        name = Path(fp).stem
        parts = name.split("_")
        obj = parts[1]
        selector = "crossfit" if "crossfit" in parts else "heuristic"
        for row in csv.DictReader(open(fp)):
            key = (obj, selector, row["estimator"], row["src_row"], row["rep"], row["mode"])
            force_idx[key] = float(row["peak_force"])

    rows = []
    missing = 0
    for r in tan_rows:
        key = (r["obj"], r["selector"], r["estimator"], r["src_row"], r["rep"], r["mode"])
        pf = force_idx.get(key)
        if pf is None:
            missing += 1
            continue
        r["peak_force"] = pf
        r["progress_ge_80"] = int(float(r["progress"]) >= 0.80)
        r["track_ok_2cm"] = int(float(r["max_track_err"]) < 0.02)
        r["quality_ok_20pct"] = int(float(r["sat_frac"]) < 0.20)
        rows.append(r)
    print(f"joined {len(rows)} rows ({missing} unmatched, dropped)")
    return rows


def continuous_correlation(rows, pred_name, outcome_col="peak_force", n_boot=N_BOOT, seed=RNG_SEED):
    col, worse_high = PREDICTORS[pred_name]
    x = np.array([float(r[col]) for r in rows])
    y = np.array([float(r[outcome_col]) for r in rows])
    objs = np.array([r["obj"] for r in rows])
    rho, p = stats.spearmanr(x, y)

    uniq_objs = np.array(sorted(set(objs)))
    idx_by_obj = {o: np.where(objs == o)[0] for o in uniq_objs}
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_boot):
        sample_objs = rng.choice(uniq_objs, size=len(uniq_objs), replace=True)
        idx = np.concatenate([idx_by_obj[o] for o in sample_objs])
        r_b, _ = stats.spearmanr(x[idx], y[idx])
        if np.isfinite(r_b):
            rhos.append(r_b)
    rhos = np.array(rhos)
    return dict(pred_name=pred_name, rho=float(rho), p=float(p),
               rho_lo=float(np.percentile(rhos, 2.5)), rho_hi=float(np.percentile(rhos, 97.5)))


def main():
    rows = load_revolute_mode_a_with_force()
    obj_of_row = [r["obj"] for r in rows]
    objects = sorted(set(obj_of_row))
    folds = make_folds(objects, N_FOLDS, RNG_SEED)
    print(f"object-grouped {N_FOLDS}-fold CV, {len(objects)} objects, fold sizes {[len(f) for f in folds]}")

    from p1_metric_comparison_cv import run_one  # after sys.path setup in that module

    md = ["# Item B follow-up: decomposing `firm` into its gate components "
         "(object-grouped CV, REVOLUTE mode-A, 2026-09-17)\n\n",
         "firm = progress>=0.80 AND max_track_err<2cm AND sat_frac<0.20. Each gate scored "
         "separately, same predictors/folds/bootstrap as the blended firm/rpmart85 CV.\n\n"]

    summary_auroc = {t: {} for t in TARGETS}
    for label, col in TARGETS.items():
        md.append(f"## {label} (`{col}`)\n\n")
        md.append("| metric | AUROC | 95% CI | Brier | 95% CI |\n|---|---|---|---|---|\n")
        rows_csv = []
        for pred_name in PREDICTORS:
            res = run_one(rows, pred_name, col, folds, obj_of_row)
            ci = bootstrap_ci(res)
            summary_auroc[label][pred_name] = res["auroc"]
            md.append(f"| {pred_name} | {res['auroc']:.3f} | "
                     f"[{ci['auroc_lo']:.3f},{ci['auroc_hi']:.3f}] | {res['brier']:.4f} | "
                     f"[{ci['brier_lo']:.4f},{ci['brier_hi']:.4f}] |\n")
            rows_csv.append(dict(metric=pred_name, target=label, n=res["n"],
                                 auroc=res["auroc"], auroc_lo=ci["auroc_lo"], auroc_hi=ci["auroc_hi"],
                                 brier=res["brier"], brier_lo=ci["brier_lo"], brier_hi=ci["brier_hi"]))
            print(f"  {label:16s} {pred_name:8s}  AUROC={res['auroc']:.3f} "
                 f"[{ci['auroc_lo']:.3f},{ci['auroc_hi']:.3f}]  Brier={res['brier']:.4f}")
        csv_p = OUT / f"p1_criterion_decomposition_{label}.csv"
        with csv_p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader(); w.writerows(rows_csv)
        md.append("\n")

    md.append("## Y4_force (`peak_force`, continuous -- no natural firm-native threshold)\n\n")
    md.append("| metric | Spearman rho | 95% CI |\n|---|---|---|\n")
    rows_csv = []
    for pred_name in PREDICTORS:
        res = continuous_correlation(rows, pred_name)
        md.append(f"| {pred_name} | {res['rho']:+.3f} (p={res['p']:.4f}) | "
                 f"[{res['rho_lo']:+.3f},{res['rho_hi']:+.3f}] |\n")
        rows_csv.append(res)
        print(f"  Y4_force         {pred_name:8s}  rho={res['rho']:+.3f}  "
             f"[{res['rho_lo']:+.3f},{res['rho_hi']:+.3f}]")
    csv_p = OUT / "p1_criterion_decomposition_force.csv"
    with csv_p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader(); w.writerows(rows_csv)

    md.append("\n## Summary: which metric wins which gate\n\n")
    md.append("| target | winner | e_norm AUROC | J_full AUROC |\n|---|---|---|---|\n")
    for label in TARGETS:
        a = summary_auroc[label]
        winner = max(a, key=a.get)
        md.append(f"| {label} | **{winner}** | {a['e_norm']:.3f} | {a['J_full']:.3f} |\n")

    md_p = OUT / "p1_criterion_decomposition.md"
    md_p.write_text("".join(md))
    print(f"\n-> {md_p}")


if __name__ == "__main__":
    main()
