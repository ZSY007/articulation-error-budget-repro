"""Paired significance test for Table V-1's headline claim (reviewer-simulation
item 7, 2026-09-18): "e_norm's AUROC falls outside J's 95% CI" (checked via two
INDEPENDENT bootstrap CIs) is not the same as a formal PAIRED test on the same
resampled object sets. This computes the paired object-level bootstrap
distribution of AUROC(e_norm) - AUROC(J) (and Brier(J) - Brier(e_norm)) directly,
reusing p1_metric_comparison_cv.py's own out-of-fold predictions and folds
(same seed, same CV split -- not re-derived).

    conda activate cv
    python error_budget/p1_paired_significance.py
    -> out/p1_paired_significance.md
"""
import numpy as np

from p1_metric_comparison_cv import (
    CRITERIA, N_FOLDS, RNG_SEED, load_revolute_mode_a, make_folds, run_one, auroc,
)

OUT_MD = "out/p1_paired_significance.md"
N_BOOT = 2000
PAIRS = [("e_norm", "J_full"), ("e_norm", "J_eq16")]


def paired_bootstrap(res_a, res_b, n_boot=N_BOOT, seed=RNG_SEED):
    valid = res_a["valid"] & res_b["valid"]
    y = res_a["y_all"][valid]
    pa = res_a["oof_pred"][valid]
    pb = res_b["oof_pred"][valid]
    objs = res_a["objs_all"][valid]
    uniq = np.array(sorted(set(objs)))
    idx_by_obj = {o: np.where(objs == o)[0] for o in uniq}
    rng = np.random.default_rng(seed)
    d_auc, d_brier = [], []
    for _ in range(n_boot):
        sample_objs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_obj[o] for o in sample_objs])
        auc_a = auroc(y[idx], pa[idx])
        auc_b = auroc(y[idx], pb[idx])
        if np.isfinite(auc_a) and np.isfinite(auc_b):
            d_auc.append(auc_a - auc_b)
        brier_a = float(np.mean((pa[idx] - y[idx]) ** 2))
        brier_b = float(np.mean((pb[idx] - y[idx]) ** 2))
        d_brier.append(brier_b - brier_a)  # positive = a (e_norm) has lower Brier
    d_auc = np.array(d_auc)
    d_brier = np.array(d_brier)
    point_auc = res_a["auroc"] - res_b["auroc"]
    p_two_sided = 2 * min((d_auc <= 0).mean(), (d_auc >= 0).mean())
    return dict(
        point_diff_auroc=point_auc,
        ci_lo=float(np.percentile(d_auc, 2.5)), ci_hi=float(np.percentile(d_auc, 97.5)),
        p_value_approx=float(min(p_two_sided, 1.0)),
        frac_boot_favoring_a=float((d_auc > 0).mean()),
        brier_diff_median=float(np.median(d_brier)),
        brier_diff_ci=(float(np.percentile(d_brier, 2.5)), float(np.percentile(d_brier, 97.5))),
    )


def main():
    rows = load_revolute_mode_a()
    obj_of_row = [r["obj"] for r in rows]
    objects = sorted(set(obj_of_row))
    folds = make_folds(objects, N_FOLDS, RNG_SEED)

    lines = ["# Paired object-level bootstrap significance test (e_norm vs J_full, e_norm vs J_eq16)\n\n",
            f"n_boot={N_BOOT}, same folds/seed as Table V-1 (`p1_metric_comparison_cv.py`, "
            f"RNG_SEED={RNG_SEED}). Positive AUROC diff / positive Brier diff both favor e_norm.\n"
            f"`J_full` = `J_cfg` (this paper's original implementation); `J_eq16` = the "
            f"formula-corrected implementation (Section III-B' correction note, 2026-09-18).\n\n"]
    for name_a, name_b in PAIRS:
        lines.append(f"# {name_a} vs {name_b}\n\n")
        for crit in CRITERIA:
            res_a = run_one(rows, name_a, crit, folds, obj_of_row)
            res_b = run_one(rows, name_b, crit, folds, obj_of_row)
            r = paired_bootstrap(res_a, res_b)
            lines.append(f"## {crit}\n\n")
            lines.append(f"- point estimate: AUROC({name_a}) - AUROC({name_b}) = {r['point_diff_auroc']:.4f}\n")
            lines.append(f"- paired bootstrap 95% CI of the difference: "
                         f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]\n")
            lines.append(f"- approx two-sided p-value bound (bootstrap sign test, {N_BOOT} draws -- "
                        f"a value of exactly 0 means NO reversal was observed, bounding p<1/{N_BOOT}, "
                        f"not measuring it exactly): {r['p_value_approx']:.4g}\n")
            lines.append(f"- fraction of bootstrap draws favoring {name_a}: {r['frac_boot_favoring_a']:.3f}\n")
            lines.append(f"- Brier({name_b}) - Brier({name_a}) median [95% CI]: {r['brier_diff_median']:.4f} "
                         f"[{r['brier_diff_ci'][0]:.4f}, {r['brier_diff_ci'][1]:.4f}]\n\n")
            print(f"{name_a} vs {name_b} | {crit}: diff={r['point_diff_auroc']:.4f} CI=[{r['ci_lo']:.4f},{r['ci_hi']:.4f}] "
                 f"p~{r['p_value_approx']:.4g} frac_favoring_{name_a}={r['frac_boot_favoring_a']:.3f}")

    with open(OUT_MD, "w") as f:
        f.writelines(lines)
    print(f"-> {OUT_MD}")


if __name__ == "__main__":
    main()
