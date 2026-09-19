"""E3-A (方案3 S3): archive re-scoring of the existing P4e delay/horizon stress trials.

Status: offline analysis only, no new physics execution. Does not depend on the
J_eq16 recompute. Reuses two disjoint archives written by earlier P4e runs:

  - p4e_feedback_noise_stress_trials_critical.csv  (8 calibration objects, 14 H/d
    cells, axis+pivot sweeps x16 reps each -- the "calibration" pool)
  - p4e_feedback_noise_stress_trials_p3c_dH.csv    (8 disjoint objects, 5 H/d
    cells -- the frozen "held-out" pool; must not be used to fit anything)

Per docs/实验方案3_判据与参考更新机制验证_2026-09-18.md §3:
  - Recompute C_tau/T/Q/S_tau for tau in {0.60,0.70,0.80,0.90} directly from the
    archived continuous fields (progress, max_track_err, sat_frac). Verify
    S_0.80 reproduces the archived `firm` column exactly (sanity check, not an
    assumption).
  - Compare continuous progress against candidate reference models, fit ONLY on
    the calibration pool, evaluated (not refit) on the disjoint pool:
      * constant-progress: progress_hat = P0 (the d=0 baseline mean), i.e. delay
        has no effect -- the null the other models must beat.
      * asymptotic:        progress_hat = P0 * (1 - d/H)
    The finite-block ideal-recursion model (needs per-trial N and initial state,
    which this archive does not record -- see doc §3.2) is explicitly NOT fit
    here; it is deferred to new-trace data (E3-B/E3-C), per the doc's own
    fallback for archives missing N/initial-state.
  - Report 50% success crossings per (H, tau) only when the observed S_tau rate
    across delay actually brackets 0.5; otherwise mark censored/unidentifiable
    rather than extrapolating.

Outputs (written to out/p3a_*):
  p3a_archive_rescore.csv        per-cell (pool,object,sweep,H,d) C_tau/T/Q/S_tau
                                  rates + continuous progress median/quartiles
  p3a_per_object_crossings.csv   50% S_tau crossings per (pool,H,tau), bracketed
                                  or censored
  p3a_progress_model_comparison.csv  per-cell |actual - predicted| for both
                                  candidate models, calibration vs disjoint
  p3a_summary.md                  human-readable summary of what the archive can
                                  and cannot support (per doc §3.3)
"""
import numpy as np
import pandas as pd

OUT = "out"
TAUS = [0.60, 0.70, 0.80, 0.90]
TRACK_ERR_LIMIT = 0.02
SAT_FRAC_LIMIT = 0.20


def load(path):
    df = pd.read_csv(f"{OUT}/{path}")
    # horizon like "H16", "H22", ... ; keep numeric H for d/H ratio (no Hinf in these two files)
    df["H_num"] = df["horizon"].str.replace("H", "", regex=False).astype(int)
    df["d_over_H"] = df["delay"] / df["H_num"]
    return df


def add_criteria(df):
    for tau in TAUS:
        c = (df["progress"] >= tau).astype(int)
        t = (df["max_track_err"] < TRACK_ERR_LIMIT).astype(int)
        q = (df["sat_frac"] < SAT_FRAC_LIMIT).astype(int)
        df[f"C_{tau:.2f}"] = c
        df[f"T_{tau:.2f}"] = t  # identical across tau, kept per-tau for output symmetry
        df[f"Q_{tau:.2f}"] = q
        df[f"S_{tau:.2f}"] = c * t * q
    return df


def sanity_check_firm(df, label):
    s080 = df["S_0.80"]
    mismatch = (s080 != df["firm"]).sum()
    n = len(df)
    print(f"[{label}] S_0.80 vs archived firm: {n - mismatch}/{n} match, {mismatch} mismatch")
    if mismatch:
        bad = df.loc[s080 != df["firm"], ["object_id", "horizon", "delay", "sweep", "rep", "progress", "max_track_err", "sat_frac", "firm"]]
        print(bad.head(10).to_string())
    return mismatch


def per_cell_table(df, pool_name):
    rows = []
    group_cols = ["object_id", "sweep", "horizon", "H_num", "delay", "d_over_H"]
    for keys, g in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["pool"] = pool_name
        row["n"] = len(g)
        row["progress_median"] = g["progress"].median()
        row["progress_q25"] = g["progress"].quantile(0.25)
        row["progress_q75"] = g["progress"].quantile(0.75)
        row["progress_clipped_at_1.5"] = bool((g["progress"] >= 1.499).any())
        for tau in TAUS:
            t = f"{tau:.2f}"
            row[f"C_{t}_rate"] = g[f"C_{t}"].mean()
            row[f"T_{t}_rate"] = g[f"T_{t}"].mean()
            row[f"Q_{t}_rate"] = g[f"Q_{t}"].mean()
            row[f"S_{t}_rate"] = g[f"S_{t}"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def crossings(cell_df, pool_name):
    """50% S_tau crossing in delay, per (pool, object_id, sweep, horizon), per tau.
    Only reported when observed cells bracket 0.5; otherwise censored."""
    rows = []
    for tau in TAUS:
        t = f"{tau:.2f}"
        for keys, g in cell_df.groupby(["object_id", "sweep", "horizon", "H_num"]):
            g = g.sort_values("delay")
            rates = g[f"S_{t}_rate"].values
            delays = g["delay"].values
            row = {
                "pool": pool_name, "object_id": keys[0], "sweep": keys[1],
                "horizon": keys[2], "H_num": keys[3], "tau": tau,
                "n_delay_points": len(delays),
                "min_rate": rates.min() if len(rates) else np.nan,
                "max_rate": rates.max() if len(rates) else np.nan,
            }
            if len(rates) < 2 or rates.max() < 0.5 or rates.min() > 0.5:
                row["crossing_delay"] = np.nan
                row["status"] = "censored_no_bracket"
            else:
                # linear-interpolate the first bracketing pair (rates are
                # non-increasing in delay in expectation, but archive is noisy;
                # walk points in delay order and take first sign change)
                found = None
                for i in range(len(rates) - 1):
                    if (rates[i] - 0.5) * (rates[i + 1] - 0.5) <= 0 and rates[i] != rates[i + 1]:
                        frac = (0.5 - rates[i]) / (rates[i + 1] - rates[i])
                        found = delays[i] + frac * (delays[i + 1] - delays[i])
                        break
                if found is None:
                    row["crossing_delay"] = np.nan
                    row["status"] = "censored_nonmonotone"
                else:
                    row["crossing_delay"] = found
                    row["status"] = "bracketed"
            rows.append(row)
    return pd.DataFrame(rows)


def fit_models(calib_cells):
    """Fit P0 (d=0 baseline mean progress) per (object,sweep,H) on the
    calibration pool only. Returns a lookup dict."""
    baseline = calib_cells[calib_cells["delay"] == 0]
    p0 = baseline.groupby(["object_id", "sweep"])["progress_median"].mean().to_dict()
    # fallback: if an object/sweep has no delay==0 cell in calib (shouldn't
    # happen for critical pool given H16,d=0 exists), use its min-delay cell
    if not p0:
        raise RuntimeError("no delay=0 baseline found in calibration pool")
    return p0


def evaluate_models(cells, p0_lookup, pool_name):
    rows = []
    global_p0 = np.mean(list(p0_lookup.values()))
    for _, r in cells.iterrows():
        key = (r["object_id"], r["sweep"])
        p0 = p0_lookup.get(key, global_p0)  # disjoint objects use global P0 (frozen, no per-object refit)
        actual = r["progress_median"]
        const_pred = p0
        asym_pred = p0 * max(0.0, 1.0 - r["d_over_H"])
        rows.append({
            "pool": pool_name, "object_id": r["object_id"], "sweep": r["sweep"],
            "horizon": r["horizon"], "delay": r["delay"], "d_over_H": r["d_over_H"],
            "progress_actual_median": actual,
            "p0_used": p0, "p0_is_object_specific": key in p0_lookup,
            "const_pred": const_pred, "const_abs_err": abs(actual - const_pred),
            "asym_pred": asym_pred, "asym_abs_err": abs(actual - asym_pred),
        })
    return pd.DataFrame(rows)


def main():
    crit = add_criteria(load("p4e_feedback_noise_stress_trials_critical.csv"))
    p3c = add_criteria(load("p4e_feedback_noise_stress_trials_p3c_dH.csv"))

    sanity_check_firm(crit, "critical")
    sanity_check_firm(p3c, "p3c_dH")

    crit_cells = per_cell_table(crit, "critical_calibration")
    p3c_cells = per_cell_table(p3c, "p3c_disjoint")
    all_cells = pd.concat([crit_cells, p3c_cells], ignore_index=True)
    all_cells.to_csv(f"{OUT}/p3a_archive_rescore.csv", index=False)

    cross_crit = crossings(crit_cells, "critical_calibration")
    cross_p3c = crossings(p3c_cells, "p3c_disjoint")
    all_cross = pd.concat([cross_crit, cross_p3c], ignore_index=True)
    all_cross.to_csv(f"{OUT}/p3a_per_object_crossings.csv", index=False)

    p0_lookup = fit_models(crit_cells)
    model_crit = evaluate_models(crit_cells, p0_lookup, "critical_calibration_INSAMPLE")
    model_p3c = evaluate_models(p3c_cells, p0_lookup, "p3c_disjoint_FROZEN")
    all_model = pd.concat([model_crit, model_p3c], ignore_index=True)
    all_model.to_csv(f"{OUT}/p3a_progress_model_comparison.csv", index=False)

    # summary
    const_mae_crit = model_crit["const_abs_err"].mean()
    asym_mae_crit = model_crit["asym_abs_err"].mean()
    const_mae_p3c = model_p3c["const_abs_err"].mean()
    asym_mae_p3c = model_p3c["asym_abs_err"].mean()
    const_mae_p3c_p95 = model_p3c["const_abs_err"].quantile(0.95)
    asym_mae_p3c_p95 = model_p3c["asym_abs_err"].quantile(0.95)

    n_bracketed = (all_cross["status"] == "bracketed").sum()
    n_total = len(all_cross)

    lines = []
    lines.append("# E3-A archive re-scoring summary (offline, no new physics)\n")
    lines.append(f"- critical (calibration) pool: {crit.object_id.nunique()} objects, "
                  f"{len(crit)} trials, {crit_cells.shape[0]} (object,sweep,H,d) cells\n")
    lines.append(f"- p3c_dH (disjoint) pool: {p3c.object_id.nunique()} objects, "
                  f"{len(p3c)} trials, {p3c_cells.shape[0]} (object,sweep,H,d) cells\n")
    lines.append(f"- S_0.80 vs archived `firm`: see stdout of this run for exact match counts "
                  f"(sanity check that the tau=0.80 recompute reproduces the original label)\n")
    lines.append("\n## Candidate progress models (fit on calibration ONLY, frozen before disjoint check)\n")
    lines.append(f"- constant-progress (null: delay has no effect), calibration in-sample MAE = {const_mae_crit:.4f}\n")
    lines.append(f"- asymptotic 1-d/H, calibration in-sample MAE = {asym_mae_crit:.4f}\n")
    lines.append(f"- constant-progress, DISJOINT frozen-test MAE = {const_mae_p3c:.4f} (95th pct abs err = {const_mae_p3c_p95:.4f})\n")
    lines.append(f"- asymptotic 1-d/H, DISJOINT frozen-test MAE = {asym_mae_p3c:.4f} (95th pct abs err = {asym_mae_p3c_p95:.4f})\n")
    lines.append("- finite-block ideal-recursion model: NOT fit here -- this archive has no per-trial "
                  "N (command step count) or initial state, only final progress/track-err/sat-frac "
                  "(per doc §3.2's own fallback for archives missing N/initial-state). Left for new-trace "
                  "data (E3-B/E3-C) which will log q/command/anchor per step.\n")
    lines.append(f"\n## 50% S_tau crossings\n")
    lines.append(f"- {n_bracketed}/{n_total} (pool,object,sweep,H,tau) combinations have an observed bracket "
                  f"around 0.5; the rest are reported censored (no crossing claimed) rather than extrapolated.\n")
    lines.append("\n## Allowed conclusions at this stage (per doc §3.3)\n")
    lines.append("- If asymptotic/constant model MAE stays small on the frozen disjoint pool: consistent with "
                  "delay degrading progress roughly as expected, but this is NOT yet a causal claim about "
                  "reference-reset mechanics -- that needs E3-B's paired new trials with step-level traces.\n")
    lines.append("- Check T_rate/Q_rate alongside C_rate per cell in p3a_archive_rescore.csv: if tracking/stall "
                  "gates also degrade materially with delay (not just completion), the completion-only "
                  "explanation is insufficient and must be stated as a mixed mechanism.\n")
    lines.append("- Crossings that move with tau are a necessary criterion-sensitivity check, not by themselves "
                  "proof of reference-backtracking causing the loss.\n")

    with open(f"{OUT}/p3a_summary.md", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("".join(lines))
    print("Wrote:", f"{OUT}/p3a_archive_rescore.csv", f"{OUT}/p3a_per_object_crossings.csv",
          f"{OUT}/p3a_progress_model_comparison.csv", f"{OUT}/p3a_summary.md")


if __name__ == "__main__":
    main()
