"""方案2 §4-6: E2-A (RGB-internal LOOO calibration) and E2-B (source-only frozen
transfer), per docs/实验方案2_结构化跟踪误差与预算迁移_2026-09-18.md. Reads the
common-nominal tables written by p2_tracker_transfer_manifest.py (must be run
first). Reuses `fit_logistic_1d`/`auroc` from p1_metric_comparison_cv.py
UNCHANGED (frozen per the doc -- "拟合规则冻结使用项目现有 fit_logistic_1d(),
不因目标结果临时加入高阶项、选择阈值或更换模型").

    conda activate cv
    python error_budget/p2_tracker_transfer_manifest.py   # once, first
    python error_budget/p2_tracker_transfer_eval.py
    -> out/p2_tracker_transfer_20260918/{predictions_rgb_looo.csv,
       predictions_source_frozen.csv, metrics.csv, paired_ci.csv, coverage.csv,
       acceptance.csv, calibration.png/.pdf, report.md}
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_metric_comparison_cv import fit_logistic_1d, auroc                    # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
RUN_DIR = OUT / "p2_tracker_transfer_20260918"
N_BOOT = 2000
RNG_SEED = 12345

PREDICTOR_COLS = {   # name -> (csv column, worse_high)
    "axis_err": ("axis_err_deg", True),
    "pivot_err": ("pivot_err_cm", True),
    "e_geo": ("e_geo_common_nominal_cm", True),
    "e_norm": ("e_norm_common_nominal_cm", True),
    "J_cfg": ("j_cfg", False),
    "J_eq16": ("j_eq16", False),
}
ENDPOINTS = ("firm", "completion85")
AQUA = "#1baf7a"
ORANGE = "#eb6834"
GRID = "#e1e0d9"
TEXT = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------- E2-A: LOOO --
def run_looo(rows, pred_name, crit):
    col, worse_high = PREDICTOR_COLS[pred_name]
    x_all = np.array([float(r[col]) for r in rows])
    y_all = np.array([int(r[crit]) for r in rows], dtype=bool)
    objs_all = np.array([r["object_id"] for r in rows])
    objects = sorted(set(objs_all))
    oof_pred = np.full(len(rows), np.nan)
    const_pred = np.full(len(rows), np.nan)
    abstain_reason = {}

    for held in objects:
        train_mask = objs_all != held
        test_mask = objs_all == held
        y_train, x_train = y_all[train_mask], x_all[train_mask]
        if len(set(y_train.tolist())) < 2:
            abstain_reason[held] = "single_class_train"
            continue
        if np.std(x_train) < 1e-9:
            abstain_reason[held] = "zero_variance_train"
            continue
        predict, _invert = fit_logistic_1d(x_train, y_train, worse_high)
        p_test = predict(x_all[test_mask])
        if len(p_test) > 1 and np.all((p_test < 1e-6) | (p_test > 1 - 1e-6)):
            abstain_reason[held] = "saturated_fit"
            continue
        oof_pred[test_mask] = p_test
        const_pred[test_mask] = float(y_train.mean())

    valid = np.isfinite(oof_pred)
    return dict(pred_name=pred_name, crit=crit, x_all=x_all, y_all=y_all, objs_all=objs_all,
               oof_pred=oof_pred, const_pred=const_pred, valid=valid,
               abstain_reason=abstain_reason, n_objects=len(objects))


# ----------------------------------------------------- E2-B: source-frozen --
def fit_source_frozen(source_rows, pred_name, crit):
    col, worse_high = PREDICTOR_COLS[pred_name]
    x_train = np.array([float(r[col]) for r in source_rows])
    y_train = np.array([int(r[crit]) for r in source_rows], dtype=bool)
    predict, invert = fit_logistic_1d(x_train, y_train, worse_high)
    return dict(predict=predict, invert=invert, const_rate=float(y_train.mean()),
               n_train=len(source_rows), col=col, worse_high=worse_high)


def apply_frozen(frozen, target_rows):
    col = frozen["col"]
    x = np.array([float(r[col]) for r in target_rows])
    p_hat = frozen["predict"](x)
    return p_hat


# ------------------------------------------------------------- bootstrap CI --
def object_cluster_bootstrap(objs_all, n_boot=N_BOOT, seed=RNG_SEED):
    """Returns n_boot resamplings of the unique object list (with replacement),
    shared across all predictor/endpoint comparisons in one draw."""
    uniq = np.array(sorted(set(objs_all)))
    rng = np.random.default_rng(seed)
    return uniq, [rng.choice(uniq, size=len(uniq), replace=True) for _ in range(n_boot)]


def resample_indices(objs_all, sample_objs, idx_by_obj):
    return np.concatenate([idx_by_obj[o] for o in sample_objs])


def bootstrap_metric_series(result, uniq_objs, samples):
    """AUROC/Brier bootstrap distribution for one run_looo/frozen-prediction
    result, using the SAME object resamplings as every other predictor (so
    paired differences are valid)."""
    valid = result["valid"]
    y = result["y_all"][valid]
    p = result["oof_pred"][valid]
    objs = result["objs_all"][valid]
    idx_by_obj = {o: np.where(objs == o)[0] for o in uniq_objs if o in objs}
    aucs, briers = [], []
    for sample_objs in samples:
        idx_parts = [idx_by_obj[o] for o in sample_objs if o in idx_by_obj]
        if not idx_parts:
            aucs.append(np.nan); briers.append(np.nan); continue
        idx = np.concatenate(idx_parts)
        if len(idx) == 0 or len(set(y[idx].tolist())) < 2:
            aucs.append(np.nan)
        else:
            aucs.append(auroc(y[idx], p[idx]))
        briers.append(float(np.mean((p[idx] - y[idx]) ** 2)) if len(idx) else np.nan)
    return np.array(aucs), np.array(briers)


def ci_report(arr, lo_pct, hi_pct):
    a = arr[np.isfinite(arr)]
    n_invalid = int(np.sum(~np.isfinite(arr)))
    if len(a) == 0:
        return dict(lo=float("nan"), hi=float("nan"), n_invalid=n_invalid, n_valid=0)
    return dict(lo=float(np.percentile(a, lo_pct)), hi=float(np.percentile(a, hi_pct)),
               n_invalid=n_invalid, n_valid=len(a))


# --------------------------------------------------------------- reliability --
def reliability_bins_equal_width(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        n = int(sel.sum())
        if n == 0:
            rows.append((0.5 * (lo + hi), None, None, 0))
        else:
            rows.append((0.5 * (lo + hi), float(p[sel].mean()), float(y[sel].mean()), n))
    return rows


def plot_reliability(ax, y, p, color, label):
    bins = reliability_bins_equal_width(y, p)
    xs, ys, sizes = [], [], []
    for _, pm, am, n in bins:
        if n == 0:
            continue
        xs.append(pm); ys.append(am); sizes.append(20 + 60 * n / max(b[3] for b in bins))
    ax.plot(xs, ys, color=color, lw=1.6, zorder=3)
    ax.scatter(xs, ys, color=color, s=sizes, zorder=4, label=label)


def main():
    src_rows = load_rows(RUN_DIR / "source_execution_common_nominal.csv")
    tgt_rows = load_rows(RUN_DIR / "target_execution_common_nominal.csv")
    print(f"source rows: {len(src_rows)}, target rows: {len(tgt_rows)}")

    # ---------------- E2-A ----------------
    looo_results = {}
    for pred_name in PREDICTOR_COLS:
        for crit in ENDPOINTS:
            looo_results[(pred_name, crit)] = run_looo(tgt_rows, pred_name, crit)

    with open(RUN_DIR / "predictions_rgb_looo.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["predictor", "endpoint", "row_idx", "object_id", "y_true", "oof_pred",
                   "const_pred", "valid"])
        for (pred_name, crit), res in looo_results.items():
            for i in range(len(res["y_all"])):
                w.writerow([pred_name, crit, i, res["objs_all"][i], int(res["y_all"][i]),
                          res["oof_pred"][i] if np.isfinite(res["oof_pred"][i]) else "",
                          res["const_pred"][i] if np.isfinite(res["const_pred"][i]) else "",
                          int(res["valid"][i])])
    print(f"-> predictions_rgb_looo.csv")

    # ---------------- E2-B ----------------
    frozen_models = {}
    for pred_name in PREDICTOR_COLS:
        for crit in ENDPOINTS:
            frozen_models[(pred_name, crit)] = fit_source_frozen(src_rows, pred_name, crit)

    frozen_target_results = {}
    for (pred_name, crit), frozen in frozen_models.items():
        p_hat = apply_frozen(frozen, tgt_rows)
        y_all = np.array([int(r[crit]) for r in tgt_rows], dtype=bool)
        objs_all = np.array([r["object_id"] for r in tgt_rows])
        frozen_target_results[(pred_name, crit)] = dict(
            pred_name=pred_name, crit=crit, oof_pred=p_hat, y_all=y_all, objs_all=objs_all,
            valid=np.ones(len(tgt_rows), dtype=bool), const_pred=np.full(len(tgt_rows), frozen["const_rate"]),
            const_rate_source=frozen["const_rate"], n_train=frozen["n_train"])

    with open(RUN_DIR / "predictions_source_frozen.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["predictor", "endpoint", "row_idx", "object_id", "y_true", "p_hat",
                   "source_const_rate"])
        for (pred_name, crit), res in frozen_target_results.items():
            for i in range(len(res["y_all"])):
                w.writerow([pred_name, crit, i, res["objs_all"][i], int(res["y_all"][i]),
                          res["oof_pred"][i], res["const_rate_source"]])
    print(f"-> predictions_source_frozen.csv")

    # ---------------- bootstrap: object-cluster, shared draws ----------------
    uniq_objs, samples = object_cluster_bootstrap(
        np.array([r["object_id"] for r in tgt_rows]), N_BOOT, RNG_SEED)

    boot_A, boot_B = {}, {}
    for key, res in looo_results.items():
        boot_A[key] = bootstrap_metric_series(res, uniq_objs, samples)
    for key, res in frozen_target_results.items():
        boot_B[key] = bootstrap_metric_series(res, uniq_objs, samples)

    # ---------------- common-support fix (found by an independent audit, 2026-09-18) ----------------
    # BUG (now fixed): the paired bootstrap below originally let bootstrap_metric_series drop
    # each predictor's OWN abstained (LOOO-saturated) objects independently before computing
    # AUROC, then subtracted the two series across the SAME resampled object list. Same object
    # RESAMPLING is not the same as same EVALUATION ROWS when predictors abstain on different
    # objects -- e_norm/firm alone abstains on 5/12 objects (saturated_fit), so "paired" AUROC
    # differences were actually computed over different row sets per predictor, silently. Fixed
    # by restricting every PAIRED primary comparison to the intersection of objects valid across
    # ALL SIX predictors and BOTH endpoints simultaneously -- the only rows a true paired
    # comparison can use. Marginal per-predictor numbers in metrics.csv (each predictor's own
    # coverage) are unaffected and still valid on their own terms.
    all_abstaining_objects = set()
    for (pred_name, crit), res in looo_results.items():
        all_abstaining_objects |= set(res["abstain_reason"].keys())
    common_objects = np.array(sorted(set(uniq_objs.tolist()) - all_abstaining_objects))
    print(f"[common-support fix] {len(common_objects)}/{len(uniq_objs)} objects valid across all "
         f"6 predictors x 2 endpoints (excluded: {sorted(all_abstaining_objects)})")

    def restrict_to_common(res):
        objs = res["objs_all"]
        mask = res["valid"] & np.isin(objs, common_objects)
        return dict(y_all=res["y_all"][mask], oof_pred=res["oof_pred"][mask], objs_all=objs[mask])

    # re-sample WITHIN the common object pool only, so every draw is itself all-common
    _rng_common = np.random.default_rng(RNG_SEED)
    common_samples = [_rng_common.choice(common_objects, size=len(common_objects), replace=True)
                      for _ in range(N_BOOT)]

    def bootstrap_common(res_common):
        idx_by_obj = {o: np.where(res_common["objs_all"] == o)[0] for o in common_objects}
        aucs = []
        for sample_objs in common_samples:
            idx = np.concatenate([idx_by_obj[o] for o in sample_objs])
            y, p = res_common["y_all"][idx], res_common["oof_pred"][idx]
            aucs.append(auroc(y, p) if len(set(y.tolist())) > 1 else np.nan)
        return np.array(aucs)

    boot_A_common = {key: bootstrap_common(restrict_to_common(res)) for key, res in looo_results.items()}
    n_common_rows = len(restrict_to_common(looo_results[("e_norm", "firm")])["y_all"])
    print(f"[common-support fix] {n_common_rows} common rows "
         f"({n_common_rows/len(tgt_rows)*100:.1f}% coverage)")

    # ---------------- metrics.csv ----------------
    metrics_rows = []
    for domain, results, boots in (("A_rgb_internal_looo", looo_results, boot_A),
                                   ("B_source_frozen", frozen_target_results, boot_B)):
        for (pred_name, crit), res in results.items():
            valid = res["valid"]
            y, p = res["y_all"][valid], res["oof_pred"][valid]
            objs = res["objs_all"][valid]
            pooled_auc = auroc(y, p) if len(set(y.tolist())) > 1 else float("nan")
            pooled_brier = float(np.mean((p - y) ** 2)) if len(p) else float("nan")
            const_brier = float(np.mean((res["const_pred"][valid] - y) ** 2)) if len(p) else float("nan")
            # object-macro brier + per-object AUROC (only both-classes objects)
            obj_briers, obj_aucs, n_auc_objs = [], [], 0
            for o in sorted(set(objs.tolist())):
                m = objs == o
                obj_briers.append(float(np.mean((p[m] - y[m]) ** 2)))
                if len(set(y[m].tolist())) > 1:
                    obj_aucs.append(auroc(y[m], p[m])); n_auc_objs += 1
            aucs_b, briers_b = boots[(pred_name, crit)]
            ci95 = ci_report(aucs_b, 2.5, 97.5)
            metrics_rows.append(dict(
                domain=domain, predictor=pred_name, endpoint=crit,
                n_valid=int(valid.sum()), n_total=len(valid),
                coverage=float(valid.mean()),
                n_pos=int(y.sum()), n_neg=int((~y).sum()),
                pooled_auroc=pooled_auc, pooled_brier=pooled_brier,
                object_macro_brier=float(np.mean(obj_briers)) if obj_briers else float("nan"),
                object_auroc_mean=float(np.mean(obj_aucs)) if obj_aucs else float("nan"),
                n_objects_with_both_classes=n_auc_objs,
                const_baseline_brier=const_brier,
                brier_skill_vs_const=(1 - pooled_brier / const_brier) if const_brier else float("nan"),
                auroc_ci95_lo=ci95["lo"], auroc_ci95_hi=ci95["hi"], auroc_boot_n_invalid=ci95["n_invalid"],
            ))
    with open(RUN_DIR / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        w.writeheader(); w.writerows(metrics_rows)
    print(f"-> metrics.csv ({len(metrics_rows)} rows)")

    # ---------------- paired_ci.csv: 3 primary + secondary pairs ----------------
    paired_rows = []

    # primary: A domain, e_norm vs J_eq16, firm & completion85, diff-of-diffs.
    # FIXED (see common-support note above): computed on the object intersection
    # valid for all 6 predictors x 2 endpoints, not each predictor's own valid set.
    enorm_firm_c = restrict_to_common(looo_results[("e_norm", "firm")])
    jeq16_firm_c = restrict_to_common(looo_results[("J_eq16", "firm")])
    enorm_c85_c = restrict_to_common(looo_results[("e_norm", "completion85")])
    jeq16_c85_c = restrict_to_common(looo_results[("J_eq16", "completion85")])

    d_firm = boot_A_common[("e_norm", "firm")] - boot_A_common[("J_eq16", "firm")]
    d_c85 = boot_A_common[("e_norm", "completion85")] - boot_A_common[("J_eq16", "completion85")]
    d_of_d = d_firm - d_c85
    point_firm = auroc(enorm_firm_c["y_all"], enorm_firm_c["oof_pred"]) - auroc(jeq16_firm_c["y_all"], jeq16_firm_c["oof_pred"])
    point_c85 = auroc(enorm_c85_c["y_all"], enorm_c85_c["oof_pred"]) - auroc(jeq16_c85_c["y_all"], jeq16_c85_c["oof_pred"])
    for name, arr, point in [("e_norm_minus_J_eq16__firm__AUROC_diff", d_firm, point_firm),
                             ("e_norm_minus_J_eq16__completion85__AUROC_diff", d_c85, point_c85),
                             ("diff_of_diffs__firm_minus_completion85", d_of_d, point_firm - point_c85)]:
        ci95 = ci_report(arr, 2.5, 97.5)
        ci_bonf = ci_report(arr, 0.8333, 99.1667)
        paired_rows.append(dict(comparison=name, family="primary_bonferroni_3way", domain="A_rgb_internal_looo",
                                point_estimate=point,
                                ci95_lo=ci95["lo"], ci95_hi=ci95["hi"],
                                ci_bonferroni98333_lo=ci_bonf["lo"], ci_bonferroni98333_hi=ci_bonf["hi"],
                                n_boot_invalid=ci95["n_invalid"],
                                n_common_objects=len(common_objects), n_common_rows=n_common_rows,
                                common_coverage=n_common_rows / len(tgt_rows)))

    # Secondary descriptive pairs use the same support rule as the primary family. Domain A
    # uses the all-predictor/all-endpoint common object pool; Domain B has no abstention and
    # therefore uses all target rows. This avoids retaining a known-invalid historical table
    # merely because it is secondary.
    for domain, boots, results in (("A_rgb_internal_looo", boot_A, looo_results),
                                   ("B_source_frozen", boot_B, frozen_target_results)):
        for crit in ENDPOINTS:
            if domain == "A_rgb_internal_looo":
                base_a = boot_A_common[("e_norm", crit)]
                base_res = restrict_to_common(results[("e_norm", crit)])
            else:
                base_a = boots[("e_norm", crit)][0]
                rr = results[("e_norm", crit)]; vv = rr["valid"]
                base_res = dict(y_all=rr["y_all"][vv], oof_pred=rr["oof_pred"][vv])
            pa = auroc(base_res["y_all"], base_res["oof_pred"])
            for other in PREDICTOR_COLS:
                if other == "e_norm":
                    continue
                if domain == "A_rgb_internal_looo":
                    other_boot = boot_A_common[(other, crit)]
                    other_res = restrict_to_common(results[(other, crit)])
                else:
                    other_boot = boots[(other, crit)][0]
                    rr = results[(other, crit)]; vv = rr["valid"]
                    other_res = dict(y_all=rr["y_all"][vv], oof_pred=rr["oof_pred"][vv])
                arr = base_a - other_boot
                po = auroc(other_res["y_all"], other_res["oof_pred"])
                ci95 = ci_report(arr, 2.5, 97.5)
                paired_rows.append(dict(comparison=f"e_norm_minus_{other}", family="secondary_descriptive",
                                        domain=domain, endpoint=crit if domain else None,
                                        point_estimate=pa - po, ci95_lo=ci95["lo"], ci95_hi=ci95["hi"],
                                        ci_bonferroni98333_lo=float("nan"), ci_bonferroni98333_hi=float("nan"),
                                        n_boot_invalid=ci95["n_invalid"],
                                        common_support_corrected=True,
                                        n_common_objects=(len(common_objects) if domain == "A_rgb_internal_looo" else len(uniq_objs)),
                                        n_common_rows=(n_common_rows if domain == "A_rgb_internal_looo" else len(tgt_rows))))
    with open(RUN_DIR / "paired_ci.csv", "w", newline="") as f:
        fieldnames = sorted({k for r in paired_rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(paired_rows)
    print(f"-> paired_ci.csv ({len(paired_rows)} rows)")

    # ---------------- coverage.csv ----------------
    coverage_rows = []
    for domain, rows_ in (("target_rgb", tgt_rows), ("source_flow_gaussian", src_rows)):
        for obj in sorted(set(r["object_id"] for r in rows_)):
            sub = [r for r in rows_ if r["object_id"] == obj]
            for crit in ENDPOINTS:
                y = [int(r[crit]) for r in sub]
                coverage_rows.append(dict(domain=domain, object_id=obj, endpoint=crit,
                                          n=len(sub), n_pos=sum(y), n_neg=len(y) - sum(y),
                                          pos_rate=sum(y) / len(y) if y else float("nan")))
    with open(RUN_DIR / "coverage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(coverage_rows[0].keys()))
        w.writeheader(); w.writerows(coverage_rows)
    print(f"-> coverage.csv")

    # ---------------- acceptance.csv (E2-B) ----------------
    acceptance_rows = []
    for (pred_name, crit), res in frozen_target_results.items():
        for r_thresh in (0.80, 0.90):
            accepted = res["oof_pred"] >= r_thresh
            n_exec = int(accepted.sum())
            if n_exec == 0:
                acceptance_rows.append(dict(predictor=pred_name, endpoint=crit, r_threshold=r_thresh,
                                            n_objects_accepted=0, n_estimates_accepted=0,
                                            n_executions_accepted=0, actual_success_rate="abstain_NA",
                                            mean_predicted_prob="abstain_NA", pred_minus_actual="abstain_NA"))
                continue
            objs_acc = set(res["objs_all"][accepted].tolist())
            actual_rate = float(res["y_all"][accepted].mean())
            mean_pred = float(res["oof_pred"][accepted].mean())
            acceptance_rows.append(dict(predictor=pred_name, endpoint=crit, r_threshold=r_thresh,
                                        n_objects_accepted=len(objs_acc),
                                        n_estimates_accepted=n_exec,  # each row here is one (cell,rep); estimates==unique cells computed separately below
                                        n_executions_accepted=n_exec,
                                        actual_success_rate=actual_rate, mean_predicted_prob=mean_pred,
                                        pred_minus_actual=mean_pred - actual_rate))
    with open(RUN_DIR / "acceptance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(acceptance_rows[0].keys()))
        w.writeheader(); w.writerows(acceptance_rows)
    print(f"-> acceptance.csv")

    # primary E2-B pass/fail check: e_norm -> firm, r=0.90
    primary = frozen_target_results[("e_norm", "firm")]
    accepted = primary["oof_pred"] >= 0.90
    n_exec = int(accepted.sum())
    n_objs = len(set(primary["objs_all"][accepted].tolist()))
    pass_verdict = "INSUFFICIENT_DATA"
    diff_ci = None
    if n_exec > 0:
        # bootstrap the (pred-actual) diff at r=0.90 over accepted rows, object-cluster
        acc_idx = np.where(accepted)[0]
        acc_objs = primary["objs_all"][acc_idx]
        uniq_acc = np.array(sorted(set(acc_objs.tolist())))
        idx_by_obj = {o: acc_idx[acc_objs == o] for o in uniq_acc}
        rng = np.random.default_rng(RNG_SEED)
        diffs = []
        for _ in range(N_BOOT):
            samp = rng.choice(uniq_acc, size=len(uniq_acc), replace=True)
            idx = np.concatenate([idx_by_obj[o] for o in samp])
            diffs.append(float(primary["oof_pred"][idx].mean() - primary["y_all"][idx].mean()))
        diffs = np.array(diffs)
        diff_ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
        n_estimates_acc = len(set((o, c) for o, c in zip(
            primary["objs_all"][acc_idx],
            [tgt_rows[i]["corruption"] + "_L" + tgt_rows[i]["level"] for i in acc_idx])))
        thresh_pass = (abs(diff_ci[0]) <= 0.05 and abs(diff_ci[1]) <= 0.05
                      and n_objs >= 6 and n_estimates_acc >= 24)
        # Paired Brier difference: both row-wise losses are reweighted by the same object draw.
        # The historical implementation subtracted one fixed full-set baseline from a
        # resampled model Brier, which understated the resampling variation.
        model_loss = (primary["oof_pred"] - primary["y_all"]) ** 2
        const_loss = (primary["const_pred"] - primary["y_all"]) ** 2
        idx_by_obj_full = {o: np.where(primary["objs_all"] == o)[0] for o in uniq_objs}
        brier_diff = np.array([float((model_loss[
            np.concatenate([idx_by_obj_full[o] for o in sample_objs])
        ] - const_loss[
            np.concatenate([idx_by_obj_full[o] for o in sample_objs])
        ]).mean()) for sample_objs in samples])
        brier_ci = ci_report(brier_diff, 2.5, 97.5)
        brier_pass = brier_ci["hi"] < 0
        pass_verdict = "PASS" if (thresh_pass and brier_pass) else "FAIL"
    else:
        thresh_pass = False; brier_pass = False; n_estimates_acc = 0

    # ---------------- calibration figure ----------------
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    panels = [("A_rgb_internal_looo firm", looo_results[("e_norm", "firm")], looo_results[("J_eq16", "firm")]),
             ("A_rgb_internal_looo completion85", looo_results[("e_norm", "completion85")], looo_results[("J_eq16", "completion85")]),
             ("B_source_frozen firm", frozen_target_results[("e_norm", "firm")], frozen_target_results[("J_eq16", "firm")]),
             ("B_source_frozen completion85", frozen_target_results[("e_norm", "completion85")], frozen_target_results[("J_eq16", "completion85")])]
    for ax, (title, resA, resB) in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, ls="--", zorder=1)
        for res, color, label in [(resA, AQUA, "e_norm"), (resB, ORANGE, "J_eq16")]:
            v = res["valid"]
            plot_reliability(ax, res["y_all"][v], res["oof_pred"][v], color, label)
        ax.set_title(title, fontsize=9.5, color=TEXT)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7.5, colors=MUTED)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("Protocol-2 reliability diagrams (10 equal-width bins; empty bins not connected)",
                fontsize=11.5, color=TEXT, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RUN_DIR / "calibration.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(RUN_DIR / "calibration.pdf", facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"-> calibration.png/.pdf")

    # ---------------- report.md ----------------
    lines = ["# 方案2 E2-A/E2-B results (2026-09-18, common-support fix applied same evening)\n\n"]
    lines.append(f"**Common-support correction**: the 3 primary paired comparisons below use only "
                f"the {len(common_objects)}/{len(uniq_objs)} objects valid across ALL 6 predictors x "
                f"2 endpoints simultaneously ({n_common_rows} rows, {n_common_rows/len(tgt_rows)*100:.1f}% "
                f"coverage) -- an earlier version of this script let each predictor drop its own "
                f"abstained objects independently before pairing, which is not a valid paired "
                f"comparison when abstention patterns differ by predictor (caught by an independent "
                f"audit). Excluded objects: {sorted(all_abstaining_objects)}.\n\n")
    lines.append("## Primary comparisons (Bonferroni 98.333% CI, family-wise 5% over 3 tests)\n\n")
    lines.append("| comparison | point | 95% CI | Bonferroni 98.333% CI |\n|---|---:|---|---|\n")
    for r in paired_rows:
        if r["family"] == "primary_bonferroni_3way":
            lines.append(f"| {r['comparison']} | {r['point_estimate']:.4f} | "
                         f"[{r['ci95_lo']:.4f},{r['ci95_hi']:.4f}] | "
                         f"[{r['ci_bonferroni98333_lo']:.4f},{r['ci_bonferroni98333_hi']:.4f}] |\n")
    lines.append("\n## E2-B primary test: e_norm -> firm, r=0.90\n\n")
    lines.append(f"- accepted: {n_exec} executions, {n_objs} objects"
                 f"{', ' + str(n_estimates_acc) + ' estimates' if n_exec else ''}\n")
    if n_exec:
        lines.append(f"- mean predicted - actual success rate: 95% CI [{diff_ci[0]:.4f},{diff_ci[1]:.4f}] "
                     f"(pass needs both bounds within +-0.05, n_objects>=6, n_estimates>=24)\n")
        lines.append(f"- threshold sub-check (CI within +-0.05 and coverage): {'PASS' if thresh_pass else 'FAIL'}\n")
        lines.append(f"- Brier(frozen) - Brier(source const baseline) 95% CI upper bound < 0: "
                     f"{'PASS' if brier_pass else 'FAIL'}\n")
    lines.append(f"- **operational verdict: {pass_verdict}** (5-percentage-point threshold is this "
                 f"study's own working standard, not a field-standard safety guarantee)\n")
    lines.append("\nSee metrics.csv / paired_ci.csv / coverage.csv / acceptance.csv / "
                 "predictions_rgb_looo.csv / predictions_source_frozen.csv / calibration.png "
                 "for full numeric detail.\n")
    with open(RUN_DIR / "report.md", "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("".join(lines))
    print(f"-> report.md")


if __name__ == "__main__":
    main()
