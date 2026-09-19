"""P4-a H_closed follow-up: quantify HOW MUCH the closed-loop re-anchored controller
widens the axis/pivot tolerance relative to the open-loop controller, using the same
wide multiplier grid and isotonic 50%-crossing method P3(a) used (not just the {0.7,
1.0,1.3} spot-check in p4a_controller_interventions.py, which showed most cells already
back at the firm ceiling by 1.3x -- not enough resolution to state a widening factor).

Reuses P3(a)'s frozen 8 objects/predictions (out/p3_predictions_preregistered.csv) and
the _init_worker/_one workers from p4a_controller_interventions.py (open vs
reanchor=True), but with the FULL P3(a) multiplier grid. reps=16 (half of P3(a)'s 32)
to keep this a reasonably fast follow-up rather than a second full-scale study.

    conda activate cv
    python error_budget/p4a_closed_loop_widening.py --procs 16 --reps 16
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_object as PO                                        # noqa: E402
from p4a_controller_interventions import (                       # noqa: E402
    PREDICTIONS, FROZEN_SHA256, _init_worker, _one, ARMS, suppress_native_output,
)

OUT = Path(__file__).resolve().parent / "out"
MULTIPLIERS_WIDE = (0.0, .25, .4, .55, .7, .85, 1.0, 1.15, 1.3, 1.5, 1.8, 2.2, 3.0)


def isotonic_nonincreasing(values, weights=None):
    """Pool-adjacent-violators fit constrained to y[i] >= y[i+1] (same algorithm as
    p3_validate_preregistered.py::isotonic_nonincreasing)."""
    y = np.asarray(values, float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, float)
    blocks = []
    for i, (yi, wi) in enumerate(zip(y, w)):
        blocks.append([i, i, yi * wi, wi])
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left[2] / left[3] >= right[2] / right[3] - 1e-15:
                break
            blocks[-2:] = [[left[0], right[1], left[2] + right[2], left[3] + right[3]]]
    fit = np.empty_like(y)
    for start, end, total, weight in blocks:
        fit[start:end + 1] = total / weight
    return fit


def crossing50(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if y[0] <= .5:
        return float(x[0]), "baseline_at_or_below_50"
    for i in range(1, len(x)):
        if y[i] <= .5:
            if y[i - 1] == y[i]:
                return float(x[i]), "observed"
            frac = (y[i - 1] - .5) / (y[i - 1] - y[i])
            return float(x[i - 1] + frac * (x[i] - x[i - 1])), "observed"
    return float(x[-1]), "right_censored_above_max"


def stable_seed(namespace, obj, rep, seed_base=3_100_000):
    payload = f"p3|{namespace}|{obj}|{rep}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 1_000_000
    return seed_base + offset


def run_config(predictions, config_name, reanchor, reps, procs):
    cfg = dict(ARMS["none"]) if not reanchor else dict(ARMS["closed_reanchor"])
    rows = []
    for pred in predictions:
        obj, joint = pred["object_id"], pred["joint"]
        spec = PO.ObjSpec(obj, joint)
        import pybullet as p
        with suppress_native_output():
            dev, sgn = PO.selfcheck(spec, save=False)
            cid = p.connect(p.DIRECT)
            try:
                bid, jm, lm = PO.build_scene(spec)
                p.resetJointState(bid, jm[spec.joint], 0.0)
                axis, pivot = PO.gt_axis_pivot_world(bid, jm, spec)
                handle = PO.handle_world(bid, lm[spec.child], spec.h_local)
            finally:
                p.disconnect(cid)
        tasks = []
        for sweep, col in (("axis", "axis_tol50_pred_deg"), ("pivot", "pivot_tol50_pred_cm")):
            predicted = float(pred[col])
            for multiplier in MULTIPLIERS_WIDE:
                for rep in range(reps):
                    tasks.append((sweep, predicted * multiplier, multiplier, rep,
                                  stable_seed("physics", obj, rep), stable_seed(f"direction_{sweep}", obj, rep)))
        initargs = (obj, joint, sgn, axis.tolist(), pivot.tolist(), handle.tolist(),
                    spec.target_deg, spec.radius, spec.f_max, spec.substeps,
                    cfg["step_deg"], cfg["substeps_mult"], cfg["f_max_mult"], cfg["ctrl_noise_mm"],
                    reanchor)
        ctx = mp.get_context("spawn")
        with ctx.Pool(procs, initializer=_init_worker, initargs=initargs) as pool:
            measured = pool.map(_one, tasks)
        for row in measured:
            row.update(object_id=obj, config=config_name)
        rows.extend(measured)
        print(f"  [{config_name}] {obj}: {len(measured)} trials done", flush=True)
    return rows


def crossings_for(rows):
    out = {}
    for obj in sorted({r["object_id"] for r in rows}):
        for sweep in ("axis", "pivot"):
            sub = sorted([r for r in rows if r["object_id"] == obj and r["sweep"] == sweep],
                        key=lambda r: r["value"])
            by_val = {}
            for r in sub:
                by_val.setdefault(r["value"], []).append(int(r["firm"]))
            vals = sorted(by_val)
            firm = [np.mean(by_val[v]) for v in vals]
            fit = isotonic_nonincreasing(firm)
            actual, status = crossing50(vals, fit)
            out[(obj, sweep)] = (actual, status)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--reps", type=int, default=16)
    args = ap.parse_args(argv)

    current_sha = hashlib.sha256(PREDICTIONS.read_bytes()).hexdigest().upper()
    if current_sha != FROZEN_SHA256:
        raise ValueError(f"prediction CSV changed: {current_sha} != {FROZEN_SHA256}")
    predictions = list(csv.DictReader(PREDICTIONS.open(encoding="utf-8")))

    print("=== open-loop (none) wide sweep ===")
    open_rows = run_config(predictions, "open", False, args.reps, args.procs)
    print("=== closed-loop (reanchor) wide sweep ===")
    closed_rows = run_config(predictions, "closed", True, args.reps, args.procs)

    OUT.mkdir(exist_ok=True)
    with (OUT / "p4a_closed_loop_widening_trials.csv").open("w", newline="", encoding="utf-8") as f:
        allrows = open_rows + closed_rows
        w = csv.DictWriter(f, fieldnames=sorted({k for r in allrows for k in r}))
        w.writeheader(); w.writerows(allrows)

    open_cross = crossings_for(open_rows)
    closed_cross = crossings_for(closed_rows)
    lines = ["# P4-a H_closed: quantified tolerance widening (2026-09-13)", "",
             f"> Wide multiplier grid {MULTIPLIERS_WIDE}, reps={args.reps}, P3(a)'s frozen 8 objects. "
             "50% crossing found via isotonic regression + linear interpolation, same method as "
             "p3_validate_preregistered.py.", "",
             "| object | sweep | open 50% (x pred) | closed 50% (x pred) | widening factor |",
             "|---|---|---:|---:|---:|"]
    factors = []
    for obj in sorted({o for o, s in open_cross}):
        for sweep in ("axis", "pivot"):
            ov, ostat = open_cross[(obj, sweep)]
            cv, cstat = closed_cross[(obj, sweep)]
            factor = cv / ov if ov > 0 else float("nan")
            factors.append(factor)
            flag_o = ">" if ostat == "right_censored_above_max" else ""
            flag_c = ">" if cstat == "right_censored_above_max" else ""
            lines.append(f"| {obj} | {sweep} | {flag_o}{ov:.2f} | {flag_c}{cv:.2f} | {flag_c or flag_o}{factor:.2f}x |")
    lines += ["", f"**Median widening factor: {np.median(factors):.2f}x** "
                   f"(range {np.min(factors):.2f}x-{np.max(factors):.2f}x across "
                   f"{len(factors)} object x sweep cells; '>' marks a right-censored crossing "
                   "at the top of the tested grid, i.e. the true factor is even larger).", ""]
    (OUT / "p4a_closed_loop_widening.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    mp.freeze_support()
    main()
