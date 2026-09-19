"""Item D closeout, part 2 (user request, 2026-09-17): is there a "true" grasp-
widening constant, or does the ratio keep climbing with wrist force budget?

Reuses eval_gripper_10obj.py's exact trial generation (same stable_seed scheme, same
5 gate-passing objects, same delta grid, same firm_like() completion+contact-held
criterion) but sweeps wrist_f_max across a grid from near the matched (~23-90N,
object-specific) end up through the previously-tested 600N "generous" arm, instead of
just those two endpoints. The point-constraint (firm_point) baseline does not depend
on the gripper's f_max at all, so it is loaded from the existing
out/gripper10_sweep.csv rather than re-run.

    conda activate cv
    python error_budget/p1_gripper_force_sweep.py --procs 6 --reps 10
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import multiprocessing as mp

from eval_gripper_10obj import (
    OBJS, JOINT_OVERRIDE, DELTAS, N_SWEEP, OUT, firm_like, stable_seed,
    _gripper_worker,
)

FORCE_LEVELS = (50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 450.0, 600.0)
TRACK_BUDGET_CM = 2.0  # matches firm's own tracking budget; the force sweep is about
                        # f_max, not a repeat of the tracking-budget sensitivity check


def load_baseline_point_crossings():
    """firm_point condition from the existing gripper10_sweep.csv -- unaffected by
    the gripper's f_max, so it is loaded rather than re-run."""
    rows = list(csv.DictReader(open(OUT / "gripper10_sweep.csv")))
    by_obj_delta = {}
    for r in rows:
        if r["condition"] != "firm_point":
            continue
        key = (r["obj"], float(r["delta_deg"]))
        succ = firm_like(float(r["progress"]), float(r["max_track_err"]),
                         float(r["quality_frac"]), TRACK_BUDGET_CM)
        by_obj_delta.setdefault(key, []).append(int(succ))
    return by_obj_delta


def crossing(xs, ys):
    if ys[0] < 0.5:
        return 0.0, "baseline_below_50"
    for i in range(1, len(xs)):
        if ys[i] < 0.5:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            frac = (y0 - 0.5) / (y0 - y1) if y0 != y1 else 0.5
            return float(x0 + frac * (x1 - x0)), "observed"
    return float(xs[-1]), "right_censored_above_max"


def crossing_from_binary(by_obj_delta, obj):
    by_delta = {d: v for (o, d), v in by_obj_delta.items() if o == obj}
    xs = sorted(by_delta)
    ys = [np.mean(by_delta[x]) for x in xs]
    return crossing(xs, ys)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args(argv)

    gate_result = json.loads((OUT / "gripper10_gate.json").read_text())
    passed_objs = [o for o in OBJS if gate_result.get(o, {}).get("passed")]
    print(f"force-sweeping {len(passed_objs)} gate-passing objects: {passed_objs}")

    point_by_obj_delta = load_baseline_point_crossings()
    point_cross = {obj: crossing_from_binary(point_by_obj_delta, obj)[0] for obj in passed_objs}

    all_rows = []
    for f_max in FORCE_LEVELS:
        jobs = []
        for obj in passed_objs:
            joint = JOINT_OVERRIDE.get(obj, "joint_0")
            for i, delta in enumerate(DELTAS):
                for t in range(args.reps):
                    seed = stable_seed(f"sweep_d{i}", obj, t)
                    init_s = float(np.random.default_rng(seed + 1).uniform(0, 5))
                    mass_s = float(np.random.default_rng(seed + 2).uniform(0.7, 1.5))
                    jobs.append((obj, joint, float(delta), seed, init_s, mass_s, f_max))
        with mp.Pool(args.procs) as pool:
            rows = pool.map(_gripper_worker, jobs)
        for r in rows:
            r["f_max_level"] = f_max
        all_rows.extend(rows)
        print(f"  f_max={f_max:.0f}N: {len(rows)} trials done")

    out_csv = OUT / "gripper_force_sweep.csv"
    with out_csv.open("w", newline="") as f:
        fieldnames = ["obj", "estimator", "delta_deg", "seed", "f_max_level", "progress",
                     "max_track_err", "contact_lost_frac", "peak_force", "wrist_f_max"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(all_rows)
    print(f"-> {out_csv}  ({len(all_rows)} trials)")

    print("\n=== widening ratio (completion+contact-held) vs wrist force budget ===")
    lines = ["# Item D force sweep: is there a true widening constant? (2026-09-17)", "",
             f"> point-constraint baseline loaded from out/gripper10_sweep.csv "
             f"(condition=firm_point), reps={args.reps} per new force level.", "",
             "| f_max (N) | " + " | ".join(passed_objs) + " | median widening |",
             "|---:|" + "---:|" * (len(passed_objs) + 1)]
    for f_max in FORCE_LEVELS:
        ratios = []
        cells = []
        for obj in passed_objs:
            by_delta = {}
            for r in all_rows:
                if r["obj"] != obj or r["f_max_level"] != f_max:
                    continue
                # completion + contact-held, same definition run_gripper_trial itself
                # uses for its `task_done` field (progress>=0.80 and contact_lost_frac<=0.20)
                succ = int(r["progress"] >= 0.80 and r["contact_lost_frac"] <= 0.20)
                by_delta.setdefault(r["delta_deg"], []).append(succ)
            xs = sorted(by_delta); ys = [np.mean(by_delta[x]) for x in xs]
            cg, status = crossing(xs, ys)
            ratio = cg / point_cross[obj] if point_cross[obj] > 0 else float("nan")
            ratios.append(ratio)
            flag = ">" if status == "right_censored_above_max" else ""
            cells.append(f"{flag}{ratio:.2f}x")
        valid = [r for r in ratios if not np.isnan(r)]
        med = np.median(valid) if valid else float("nan")
        lines.append(f"| {f_max:.0f} | " + " | ".join(cells) + f" | {med:.2f}x |")
        print(f"  f_max={f_max:4.0f}N  median widening={med:.2f}x  per-object=[" +
             ", ".join(f"{c}" for c in cells) + "]")

    md_p = OUT / "gripper_force_sweep.md"
    md_p.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {md_p}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
