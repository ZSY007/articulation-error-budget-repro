"""方案3 E3-B analysis (docs/实验方案3_判据与参考更新机制验证_2026-09-18.md §6):
reads `p3_delay_mechanism_replay.py`'s output (paired_trials.csv, reset_events.csv)
and reports the three pre-specified primary results:

  1. continuous-progress object-macro MAE of the finite-block ideal-recursion
     model, vs. the asymptotic 1-d/H model and a constant-progress baseline
     (GT condition only -- this is a statement about the REFERENCE mechanism,
     not about estimation error).
  2. paired progress difference between d=0 and rho in {0.20, 0.40} (same
     object/rep, same H) -- reported for progress AND for the tracking (T)
     and stall (Q) gates, not completion alone.
  3. (E3-C compensation comparison -- only if E3-C was run; skipped otherwise.)

Plus an offline tau re-scoring (C_tau/T/Q/S_tau for tau in {0.60,0.70,0.80,0.90})
reusing the same convention as p3_delay_criterion_rescore.py's E3-A analysis, so
the archive (E3-A) and the new paired trials (E3-B) are read the same way.

All bootstrap CIs are 2,000-draw OBJECT-cluster (not per-trial), per §6: "重复
seed是对象内条件，不按8,832个独立样本计算显著性."

    conda activate cv
    python error_budget/p3_delay_mechanism_replay.py --procs 16   # once, first
    python error_budget/p3_delay_mechanism_analyze.py
    -> out/p3_delay_mechanism_20260918/{progress_model_comparison.csv,
       paired_effects.csv, report.md}
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
RUN_DIR = OUT / "p3_delay_mechanism_20260918"
STEP_DEG = 0.5
TAUS = [0.60, 0.70, 0.80, 0.90]
TRACK_ERR_LIMIT = 0.02
SAT_FRAC_LIMIT = 0.20
N_BOOT = 2000
RNG_SEED = 12345


def load_trials():
    with open(RUN_DIR / "paired_trials.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("N", "d", "rep"):
            r[k] = int(r[k]) if r[k] not in ("", "nan") else None
        for k in ("Q_target_deg", "raw_progress", "clipped_progress", "max_track_err",
                  "mean_track_err", "sat_frac", "peak_force", "rho"):
            try:
                r[k] = float(r[k])
            except (ValueError, TypeError):
                r[k] = float("nan")
    return rows


def add_criteria(rows):
    for r in rows:
        t = int(r["max_track_err"] < TRACK_ERR_LIMIT)
        q = int(r["sat_frac"] < SAT_FRAC_LIMIT)
        r["T"] = t
        r["Q"] = q
        for tau in TAUS:
            c = int(r["clipped_progress"] >= tau)
            r[f"S_{tau:.2f}"] = c * t * q
    return rows


def finite_block_predicted_progress(N, H, d, Q_target_deg):
    """q[0]=0; for each block start b=0,H,2H,...<N: anchor=q[max(0,b-d)];
    for k=b+1..min(b+H,N): q[k]=anchor+(k-b)*Delta_q. Returns predicted
    progress = q[N]/Q_target (Delta_q and Q_target both in degrees, so units
    cancel; STEP_DEG matches the replay driver's fixed command increment)."""
    if H is None:  # Hinf: open-loop, q[N] = N*Delta_q directly (no resets)
        return (N * STEP_DEG) / Q_target_deg
    q = np.zeros(N + 1)
    b = 0
    while b < N:
        anchor = q[max(0, b - d)]
        k_end = min(b + H, N)
        for k in range(b + 1, k_end + 1):
            q[k] = anchor + (k - b) * STEP_DEG
        b += H
    return float(q[N]) / Q_target_deg


def asymptotic_predicted_progress(H, d, p0):
    if H is None:
        return p0
    return p0 * max(0.0, 1.0 - d / H)


def object_cluster_bootstrap_ci(values_by_object, n_boot=N_BOOT, seed=RNG_SEED):
    objs = sorted(values_by_object.keys())
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(objs, size=len(objs), replace=True)
        vals = [v for o in sample for v in values_by_object[o]]
        if vals:
            means.append(float(np.mean(vals)))
    means = np.array(means)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    rows = add_criteria(load_trials())
    print(f"loaded {len(rows)} trials")
    objects = sorted(set(r["object_id"] for r in rows))

    # -------- primary result 1: finite-block model vs asymptotic vs constant --------
    gt_rows = [r for r in rows if r["condition"] == "gt"]
    p0_by_obj = {}
    for obj in objects:
        base = [r for r in gt_rows if r["object_id"] == obj and r["H"] != "Hinf" and r["rho"] == 0.0]
        if base:
            p0_by_obj[obj] = float(np.mean([r["clipped_progress"] for r in base]))

    model_rows = []
    err_finite, err_asym, err_const = {}, {}, {}
    for r in gt_rows:
        obj = r["object_id"]
        if obj not in p0_by_obj:
            continue
        H = None if r["H"] == "Hinf" else int(r["H"])
        d = r["d"]
        pred_finite = finite_block_predicted_progress(r["N"], H, d, r["Q_target_deg"])
        pred_asym = asymptotic_predicted_progress(H, d, p0_by_obj[obj])
        pred_const = p0_by_obj[obj]
        actual = r["clipped_progress"]
        model_rows.append(dict(object_id=obj, H=r["H"], rho=r["rho"], d=d, rep=r["rep"],
                               actual_progress=actual, finite_block_pred=pred_finite,
                               asymptotic_pred=pred_asym, constant_pred=pred_const,
                               finite_block_abs_err=abs(actual - pred_finite),
                               asymptotic_abs_err=abs(actual - pred_asym),
                               constant_abs_err=abs(actual - pred_const)))
        err_finite.setdefault(obj, []).append(abs(actual - pred_finite))
        err_asym.setdefault(obj, []).append(abs(actual - pred_asym))
        err_const.setdefault(obj, []).append(abs(actual - pred_const))
    with open(RUN_DIR / "progress_model_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(model_rows[0].keys()))
        w.writeheader(); w.writerows(model_rows)
    print(f"-> progress_model_comparison.csv ({len(model_rows)} rows)")

    def obj_macro_mae(err_by_obj):
        per_obj = {o: float(np.mean(v)) for o, v in err_by_obj.items()}
        return float(np.mean(list(per_obj.values()))), per_obj

    mae_finite, per_obj_finite = obj_macro_mae(err_finite)
    mae_asym, _ = obj_macro_mae(err_asym)
    mae_const, _ = obj_macro_mae(err_const)
    ci_finite = object_cluster_bootstrap_ci({o: v for o, v in err_finite.items()})

    # -------- primary result 2: paired d=0 vs rho=0.20/0.40, progress + T + Q --------
    paired_rows = []
    for condition in ("gt", "axis", "pivot"):
        cond_rows = [r for r in rows if r["condition"] == condition and r["H"] != "Hinf"]
        for H in (20, 40):
            base = {(r["object_id"], r["rep"]): r for r in cond_rows if int(r["H"]) == H and r["rho"] == 0.0}
            for rho_test in (0.20, 0.40):
                test = {(r["object_id"], r["rep"]): r for r in cond_rows if int(r["H"]) == H and r["rho"] == rho_test}
                diffs_p, diffs_t, diffs_q = {}, {}, {}
                for key, r0 in base.items():
                    r1 = test.get(key)
                    if r1 is None:
                        continue
                    obj = key[0]
                    diffs_p.setdefault(obj, []).append(r1["clipped_progress"] - r0["clipped_progress"])
                    diffs_t.setdefault(obj, []).append(r1["T"] - r0["T"])
                    diffs_q.setdefault(obj, []).append(r1["Q"] - r0["Q"])
                if not diffs_p:
                    continue
                mean_p, ci_p = (float(np.mean([v for vv in diffs_p.values() for v in vv])),
                                object_cluster_bootstrap_ci(diffs_p))
                mean_t, ci_t = (float(np.mean([v for vv in diffs_t.values() for v in vv])),
                                object_cluster_bootstrap_ci(diffs_t))
                mean_q, ci_q = (float(np.mean([v for vv in diffs_q.values() for v in vv])),
                                object_cluster_bootstrap_ci(diffs_q))
                paired_rows.append(dict(condition=condition, H=H, rho_test=rho_test,
                                        n_objects=len(diffs_p),
                                        progress_diff_mean=mean_p, progress_diff_ci_lo=ci_p[0], progress_diff_ci_hi=ci_p[1],
                                        tracking_T_diff_mean=mean_t, tracking_T_diff_ci_lo=ci_t[0], tracking_T_diff_ci_hi=ci_t[1],
                                        stall_Q_diff_mean=mean_q, stall_Q_diff_ci_lo=ci_q[0], stall_Q_diff_ci_hi=ci_q[1]))
    with open(RUN_DIR / "paired_effects.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
        w.writeheader(); w.writerows(paired_rows)
    print(f"-> paired_effects.csv ({len(paired_rows)} rows)")

    # -------- report --------
    lines = ["# 方案3 E3-B analysis (2026-09-18)\n\n"]
    lines.append("## Primary result 1: progress model comparison (GT condition)\n\n")
    lines.append(f"- finite-block ideal-recursion model: object-macro MAE = {mae_finite:.4f}, "
                 f"95% object-cluster bootstrap CI [{ci_finite[0]:.4f},{ci_finite[1]:.4f}]\n")
    lines.append(f"- asymptotic 1-d/H model: object-macro MAE = {mae_asym:.4f}\n")
    lines.append(f"- constant-progress (null) baseline: object-macro MAE = {mae_const:.4f}\n")
    verdict1 = ("PASS (<=5 percentage points, this study's own working threshold)"
               if ci_finite[1] <= 0.05 else "FAIL threshold, report as-is, not re-fit")
    lines.append(f"- pre-specified threshold (finite-block MAE 95% upper bound <= 0.05): {verdict1}\n\n")

    lines.append("## Primary result 2: paired progress/tracking/stall diffs, d=0 -> rho\n\n")
    lines.append("| condition | H | rho | n_obj | Δprogress | 95% CI | ΔT (tracking gate) | 95% CI | ΔQ (stall gate) | 95% CI |\n")
    lines.append("|---|---|---|---|---:|---|---:|---|---:|---|\n")
    for r in paired_rows:
        lines.append(f"| {r['condition']} | {r['H']} | {r['rho_test']} | {r['n_objects']} | "
                     f"{r['progress_diff_mean']:.4f} | [{r['progress_diff_ci_lo']:.4f},{r['progress_diff_ci_hi']:.4f}] | "
                     f"{r['tracking_T_diff_mean']:.4f} | [{r['tracking_T_diff_ci_lo']:.4f},{r['tracking_T_diff_ci_hi']:.4f}] | "
                     f"{r['stall_Q_diff_mean']:.4f} | [{r['stall_Q_diff_ci_lo']:.4f},{r['stall_Q_diff_ci_hi']:.4f}] |\n")

    lines.append("\n## Allowed conclusion, per doc §6\n\n")
    gt_20 = next((r for r in paired_rows if r["condition"] == "gt" and r["rho_test"] == 0.40), None)
    if gt_20:
        t_bounded = abs(gt_20["tracking_T_diff_ci_lo"]) <= 0.05 and abs(gt_20["tracking_T_diff_ci_hi"]) <= 0.05
        q_bounded = abs(gt_20["stall_Q_diff_ci_lo"]) <= 0.05 and abs(gt_20["stall_Q_diff_ci_hi"]) <= 0.05
        if gt_20["progress_diff_mean"] < 0 and t_bounded and q_bounded:
            lines.append("- GT condition at rho=0.40: progress drops, tracking/stall gate diffs both stay within "
                         "+-0.05 -> supports 'primarily progress loss, these two gates roughly preserved.'\n")
        else:
            lines.append("- GT condition at rho=0.40: tracking/stall gate diff CI does NOT stay fully within "
                         "+-0.05 (or progress did not drop) -> CI crosses the threshold, report as uncertain, "
                         "not as 'no effect.'\n")

    with open(RUN_DIR / "report.md", "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("".join(lines))
    print("-> report.md")


if __name__ == "__main__":
    main()
