"""Joint H x delay tolerance-BUDGET validation (2026-09-18, user-requested follow-up
to a proposed design heuristic). V-B's two curves were each measured separately:
widening(H) is a MAGNITUDE-sweep 50%-crossing curve at delay=0; the d/H curve is a
FIRM-RATE-at-fixed-magnitude curve as a function of delay. A naive multiplicative
combination -- tolerable e_norm budget(H,d) ~= budget(H=inf) * widening(H) *
delay_penalty(d/H) -- was proposed as a design heuristic but is an untested
extrapolation across two different response variables (a crossing magnitude and a
success rate), not a validated result. This script tests it directly and
self-containedly: full magnitude sweeps (same grid/method as p4e_horizon_sweep.py)
at H in {16,32}, delay in {0, ...} chosen to give matched d/H ratios across the two
horizons, so the actual 50%-crossing budget under real (H,delay) combinations can be
compared against the naive product prediction on its own terms, not assumed.

    conda activate cv   (local) / ee900-prime (JARVIS)
    python error_budget/p4f_joint_hd_budget_validate.py --procs 16 --reps 8
    -> out/p4f_joint_hd_trials.csv, out/p4f_joint_hd_validate.md
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
import pilot_object as PO                                             # noqa: E402
import p0_basis as PB                                                 # noqa: E402
from p0_direction_preserving_replay import CRITERIA                    # noqa: E402
from p4a_controller_interventions import (                             # noqa: E402
    suppress_native_output, _perpendicular,
)
from p4a_closed_loop_widening import (                                 # noqa: E402
    MULTIPLIERS_WIDE, isotonic_nonincreasing, crossing50,
)

OUT = Path(__file__).resolve().parent / "out"
PRED_PATH = OUT / "p3_predictions_preregistered.csv"
PRED_SHA = "359E9FD095F7C53D12D60D8E89562BE0B4583F178ADFEB1F544212F287D7F328"
AXIS_COL, PIVOT_COL = "axis_tol50_pred_deg", "pivot_tol50_pred_cm"
SEED_BASE = 3_400_000  # distinct block, same SHA256-derived scheme as p3/p4 scripts

# H in {16,32}, delay chosen so both horizons hit the SAME two nonzero rho values
# (0.125, 0.1875) plus their own delay=0 reference -- self-contained, no reuse of
# other scripts' partial sweeps at mismatched horizon grids.
CELLS = [
    ("H16_d0", 16, 0), ("H16_d2", 16, 2), ("H16_d3", 16, 3),
    ("H32_d0", 32, 0), ("H32_d4", 32, 4), ("H32_d6", 32, 6),
]

_SPEC = None
_SGN = 1.0
_AXIS = None
_PIVOT = None
_HANDLE = None
_HORIZON = 1
_DELAY = 0


def stable_seed(namespace, obj, rep, seed_base=SEED_BASE):
    payload = f"p4f|{namespace}|{obj}|{rep}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 1_000_000
    return seed_base + offset


def _init_worker(obj, joint, sgn, axis, pivot, handle, target_deg, radius,
                 f_max, substeps, horizon, delay):
    global _SPEC, _SGN, _AXIS, _PIVOT, _HANDLE, _HORIZON, _DELAY
    _SPEC = PO.ObjSpec(obj, joint)
    _SPEC.target_deg = target_deg
    _SPEC.radius = radius
    _SPEC.f_max = f_max
    _SPEC.substeps = substeps
    _SGN = float(sgn)
    _AXIS = np.asarray(axis, float)
    _PIVOT = np.asarray(pivot, float)
    _HANDLE = np.asarray(handle, float)
    _HORIZON = horizon
    _DELAY = delay


def _one(task):
    sweep, value, multiplier, rep, phys_seed, direction_seed = task
    d = _perpendicular(_AXIS, direction_seed)
    est_axis, est_pivot = _AXIS.copy(), _PIVOT.copy()
    if sweep == "axis":
        est_axis = PO.rot_about_axis(d, np.radians(value)) @ _AXIS
        est_axis /= np.linalg.norm(est_axis)
    else:
        est_pivot = _PIVOT + d * (value / 100.0)

    phys_rng = np.random.default_rng(phys_seed)
    mass_scale = float(phys_rng.uniform(.5, 2.0))
    damping_scale = float(phys_rng.uniform(.5, 2.0))
    init_state_deg = float(phys_rng.uniform(0.0, 5.0))
    exec_seed = phys_seed + 7_000_000
    with suppress_native_output():
        result = PO.run_trial(
            _SPEC, est_axis, est_pivot, init_state_deg,
            np.random.default_rng(exec_seed), sgn=_SGN,
            mass_scale=mass_scale, damping_scale=damping_scale,
            grasp_jitter_cm=1.0, init_state_deg=init_state_deg,
            ctrl_noise_mm=1.0, f_max=_SPEC.f_max, horizon=_HORIZON,
            obs_delay_steps=_DELAY, obs_noise_mm=0.0)
    for name, fn in CRITERIA.items():
        result[name] = int(fn(result))
    return dict(sweep=sweep, value=value, multiplier=multiplier, rep=rep, **result)


def run_config(predictions, config_name, horizon, delay, reps, procs):
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
        for sweep, col in (("axis", AXIS_COL), ("pivot", PIVOT_COL)):
            predicted = float(pred[col])
            for multiplier in MULTIPLIERS_WIDE:
                for rep in range(reps):
                    tasks.append((sweep, predicted * multiplier, multiplier, rep,
                                 stable_seed("physics", obj, rep),
                                 stable_seed(f"direction_{sweep}", obj, rep)))
        initargs = (obj, joint, sgn, axis.tolist(), pivot.tolist(), handle.tolist(),
                   spec.target_deg, spec.radius, spec.f_max, spec.substeps, horizon, delay)
        ctx = mp.get_context("spawn")
        with ctx.Pool(procs, initializer=_init_worker, initargs=initargs) as pool:
            measured = pool.map(_one, tasks)
        for row in measured:
            row.update(object_id=obj, config=config_name, horizon=horizon, delay=delay)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--reps", type=int, default=8)
    args = ap.parse_args()

    current_sha = hashlib.sha256(PRED_PATH.read_bytes()).hexdigest().upper()
    if current_sha != PRED_SHA:
        raise ValueError(f"prediction CSV changed: {current_sha} != {PRED_SHA}")
    predictions = list(csv.DictReader(PRED_PATH.open(encoding="utf-8")))

    all_rows = []
    crossings = {}
    for name, h, d in CELLS:
        print(f"=== {name}: H={h} delay={d} d/H={d/h:.4f} ===")
        rows = run_config(predictions, name, h, d, args.reps, args.procs)
        all_rows.extend(rows)
        crossings[name] = crossings_for(rows)

    OUT.mkdir(exist_ok=True)
    trials_p = OUT / "p4f_joint_hd_trials.csv"
    with trials_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in all_rows for k in r}))
        w.writeheader(); w.writerows(all_rows)
    print(f"-> {trials_p} ({len(all_rows)} trials)")

    # per-cell median crossing (across object/sweep), and ratio vs that horizon's own d=0
    lines = ["# Joint H x delay tolerance-budget validation (2026-09-18)", ""]
    cell_medians = {}
    for name, h, d in CELLS:
        vals = [v for (v, status) in crossings[name].values() if np.isfinite(v)]
        cell_medians[name] = float(np.median(vals)) if vals else float("nan")
        lines.append(f"- {name} (H={h}, d={d}, d/H={d/h:.4f}): median 50%-crossing = "
                    f"{cell_medians[name]:.3f} (n={len(vals)}/16 object-sweep cells)")
    lines.append("")

    lines.append("## Naive-product prediction vs. measured (both self-contained in this run)")
    lines.append("")
    lines.append("| H | d | d/H | measured crossing | measured ratio vs H,d=0 | naive multiplicative prediction* |")
    lines.append("|---|---|---:|---:|---:|---|")
    for h in (16, 32):
        base_name = f"H{h}_d0"
        base = cell_medians[base_name]
        for name, hh, d in CELLS:
            if hh != h:
                continue
            ratio = cell_medians[name] / base if base > 0 else float("nan")
            note = "(reference cell, ratio=1 by definition)" if d == 0 else ""
            lines.append(f"| {h} | {d} | {d/h:.4f} | {cell_medians[name]:.3f} | {ratio:.3f} | {note} |")
    lines.append("")
    lines.append("*A naive product prediction needs an independently-measured delay_penalty(d/H) "
                "function from a DIFFERENT response variable (fixed-magnitude firm rate, not a "
                "crossing magnitude); this run does not assume one -- it reports the measured ratio "
                "directly so any proposed penalty function can be checked against it after the fact, "
                "rather than baking an assumed normalization into this script.")
    lines.append("")
    lines.append("## Cross-H consistency at matched d/H (tests the ratio, not absolute H/d, hypothesis)")
    lines.append("")
    r1 = cell_medians["H16_d2"] / cell_medians["H16_d0"] if cell_medians["H16_d0"] > 0 else float("nan")
    r2 = cell_medians["H32_d4"] / cell_medians["H32_d0"] if cell_medians["H32_d0"] > 0 else float("nan")
    r3 = cell_medians["H16_d3"] / cell_medians["H16_d0"] if cell_medians["H16_d0"] > 0 else float("nan")
    r4 = cell_medians["H32_d6"] / cell_medians["H32_d0"] if cell_medians["H32_d0"] > 0 else float("nan")
    lines.append(f"- d/H=0.125: H16,d=2 ratio={r1:.3f} vs H32,d=4 ratio={r2:.3f} (gap {abs(r1-r2):.3f})")
    lines.append(f"- d/H=0.1875: H16,d=3 ratio={r3:.3f} vs H32,d=6 ratio={r4:.3f} (gap {abs(r3-r4):.3f})")

    md_p = OUT / "p4f_joint_hd_validate.md"
    md_p.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {md_p}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
