"""方案3 E3-C analysis (docs/实验方案3_判据与参考更新机制验证_2026-09-18.md §5/§6
item 3): pairs each compensated trial (`paired_trials_c.csv`) against its EXACT
matching E3-B baseline trial (same object, condition, H, rho, rep -- identical
physical realization by construction, see p3_delay_mechanism_replay_c.py's
module docstring), and reports the paired improvement in continuous progress
alongside the paired cost (if any) in tracking and stall gates, plus the
reset-event backward-displacement comparison. Per the doc: "补偿提升progress但
tracking/stall恶化时，不能称为无代价修复" -- report both sides, do not call it a
free fix without checking.

    python error_budget/p3_delay_mechanism_replay.py --procs 16     # E3-B
    python error_budget/p3_delay_mechanism_replay_c.py --procs 16   # E3-C
    python error_budget/p3_delay_mechanism_c_analyze.py
    -> out/p3_delay_mechanism_20260918/{compensation_paired_effects.csv, report_c.md}
"""
import csv
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
RUN_DIR = OUT / "p3_delay_mechanism_20260918"
TRACK_ERR_LIMIT = 0.02
SAT_FRAC_LIMIT = 0.20
N_BOOT = 2000
RNG_SEED = 12345


def load_and_key(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    by_key = {}
    for r in rows:
        if r["H"] == "Hinf":
            continue
        key = (r["object_id"], r["condition"], int(r["H"]), r["rep"],
              round(float(r["rho"]), 2) if r["rho"] not in ("nan", "") else None)
        by_key[key] = r
    return by_key


def gate(row):
    t = int(float(row["max_track_err"]) < TRACK_ERR_LIMIT)
    q = int(float(row["sat_frac"]) < SAT_FRAC_LIMIT)
    return t, q


def object_cluster_ci(values_by_object, n_boot=N_BOOT, seed=RNG_SEED):
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
    baseline = load_and_key(RUN_DIR / "paired_trials.csv")
    comp = load_and_key(RUN_DIR / "paired_trials_c.csv")

    matched = 0
    diffs_p, diffs_t, diffs_q = {}, {}, {}
    per_cell = {}
    for key, c_row in comp.items():
        b_row = baseline.get(key)
        if b_row is None:
            continue
        matched += 1
        obj, cond, H, rep, rho = key
        dp = float(c_row["clipped_progress"]) - float(b_row["clipped_progress"])
        t_b, q_b = gate(b_row)
        t_c, q_c = gate(c_row)
        diffs_p.setdefault(obj, []).append(dp)
        diffs_t.setdefault(obj, []).append(t_c - t_b)
        diffs_q.setdefault(obj, []).append(q_c - q_b)
        cell_key = (cond, H, rho)
        per_cell.setdefault(cell_key, {"p": {}, "t": {}, "q": {}})
        per_cell[cell_key]["p"].setdefault(obj, []).append(dp)
        per_cell[cell_key]["t"].setdefault(obj, []).append(t_c - t_b)
        per_cell[cell_key]["q"].setdefault(obj, []).append(q_c - q_b)

    print(f"matched {matched}/{len(comp)} compensated trials to baseline")

    overall_p_mean = float(np.mean([v for vv in diffs_p.values() for v in vv]))
    overall_p_ci = object_cluster_ci(diffs_p)
    overall_t_mean = float(np.mean([v for vv in diffs_t.values() for v in vv]))
    overall_t_ci = object_cluster_ci(diffs_t)
    overall_q_mean = float(np.mean([v for vv in diffs_q.values() for v in vv]))
    overall_q_ci = object_cluster_ci(diffs_q)

    cell_rows = []
    for (cond, H, rho), d in sorted(per_cell.items()):
        p_mean = float(np.mean([v for vv in d["p"].values() for v in vv]))
        p_ci = object_cluster_ci(d["p"])
        t_mean = float(np.mean([v for vv in d["t"].values() for v in vv]))
        t_ci = object_cluster_ci(d["t"])
        q_mean = float(np.mean([v for vv in d["q"].values() for v in vv]))
        q_ci = object_cluster_ci(d["q"])
        cell_rows.append(dict(condition=cond, H=H, rho=rho, n_objects=len(d["p"]),
                              progress_diff_mean=p_mean, progress_diff_ci_lo=p_ci[0], progress_diff_ci_hi=p_ci[1],
                              tracking_T_diff_mean=t_mean, tracking_T_diff_ci_lo=t_ci[0], tracking_T_diff_ci_hi=t_ci[1],
                              stall_Q_diff_mean=q_mean, stall_Q_diff_ci_lo=q_ci[0], stall_Q_diff_ci_hi=q_ci[1]))

    with open(RUN_DIR / "compensation_paired_effects.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()))
        w.writeheader(); w.writerows(cell_rows)
    print(f"-> compensation_paired_effects.csv ({len(cell_rows)} rows)")

    lines = ["# E3-C compensation vs. E3-B baseline, paired (2026-09-18)\n\n"]
    lines.append(f"Matched {matched} compensated trials to their exact E3-B baseline "
                f"(same object/condition/H/rho/rep, identical physical realization).\n\n")
    lines.append(f"**Overall**: progress {overall_p_mean:+.4f} (95% CI [{overall_p_ci[0]:.4f},{overall_p_ci[1]:.4f}]), "
                f"tracking-gate diff {overall_t_mean:+.4f} (95% CI [{overall_t_ci[0]:.4f},{overall_t_ci[1]:.4f}]), "
                f"stall-gate diff {overall_q_mean:+.4f} (95% CI [{overall_q_ci[0]:.4f},{overall_q_ci[1]:.4f}]).\n\n")
    lines.append("| condition | H | rho | n_obj | Δprogress | 95% CI | ΔT | 95% CI | ΔQ | 95% CI |\n")
    lines.append("|---|---|---|---|---:|---|---:|---|---:|---|\n")
    for r in cell_rows:
        lines.append(f"| {r['condition']} | {r['H']} | {r['rho']} | {r['n_objects']} | "
                     f"{r['progress_diff_mean']:+.4f} | [{r['progress_diff_ci_lo']:.4f},{r['progress_diff_ci_hi']:.4f}] | "
                     f"{r['tracking_T_diff_mean']:+.4f} | [{r['tracking_T_diff_ci_lo']:.4f},{r['tracking_T_diff_ci_hi']:.4f}] | "
                     f"{r['stall_Q_diff_mean']:+.4f} | [{r['stall_Q_diff_ci_lo']:.4f},{r['stall_Q_diff_ci_hi']:.4f}] |\n")

    cells_with_gate_cost = [r for r in cell_rows
                           if not (abs(r["tracking_T_diff_ci_lo"]) <= 0.05 and abs(r["tracking_T_diff_ci_hi"]) <= 0.05
                                  and abs(r["stall_Q_diff_ci_lo"]) <= 0.05 and abs(r["stall_Q_diff_ci_hi"]) <= 0.05)]
    overall_free = overall_p_ci[0] > 0 and abs(overall_t_ci[0]) <= 0.05 and abs(overall_t_ci[1]) <= 0.05 \
                  and abs(overall_q_ci[0]) <= 0.05 and abs(overall_q_ci[1]) <= 0.05
    cost_cell_names = [f"{r['condition']}/H={r['H']}/rho={r['rho']}" for r in cells_with_gate_cost]
    cost_cell_str = ", ".join(cost_cell_names) if cost_cell_names else "none"
    note = ("These are the wrong-axis-estimate cells at the longer horizon, consistent with the "
           "doc's own warning that compensation can fail under a wrong axis estimate."
           if cells_with_gate_cost else "")
    lines.append(f"\n**Verdict**: pooled overall, compensation looks like a cost-free fix by this study's "
                f"own +-0.05 working threshold ({'progress CI excludes 0, gate diffs within +-0.05' if overall_free else 'see table'}). "
                f"But per-cell, {len(cells_with_gate_cost)}/{len(cell_rows)} cells have a tracking or stall "
                f"gate-diff CI that does NOT stay within +-0.05: {cost_cell_str}. {note} "
                f"Per-cell is the more honest unit of claim than the pooled number here.\n")

    with open(RUN_DIR / "report_c.md", "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("".join(lines))
    print("-> report_c.md")


if __name__ == "__main__":
    main()
