"""P4-a: executor-side interventions to explain the object/sweep dependence of the
executor gain g = e_exec/e_norm found in the P3(a) posthoc diagnostic (2026-09-13).

See docs/条件误差预算_P3c-P4-E5_方案_2026-09-13.md SS3. This reuses P3(a)'s FROZEN 8
objects and predictions (out/p3_predictions_preregistered.csv) -- not P3(c)'s -- so it
does not depend on the P3(c) run. It reuses the EXACT SAME stable_seed() scheme (same
namespace strings, same default seed_base) as p3_validate_preregistered.py, so the
"none" (unintervened) arm at multiplier in {0.7, 1.0, 1.3} is bit-for-bit reproducible
from the existing out/p3_holdout_trials.csv -- it does not need to be rerun, and a
--verify-baseline flag checks that claim against a fresh run of a few cells.

Implemented arms (parameter-only interventions on `pilot_object.run_trial`):
  step_0.25   / step_1.0     -- PO.STEP_DEG override (H_lag: per-step scheduling lag)
  substeps_x2                -- spec.substeps override (H_settle: settling time)
  fmax_x0.5   / fmax_x2      -- f_max override passed to run_trial (H_force)
  noise_0                    -- ctrl_noise_mm=0 (H_noise: control-noise floor)

H_closed (per-step re-anchoring of the commanded arc to the observed handle position,
`closed_reanchor` arm) is implemented via `pilot_object.run_trial(..., reanchor=True)`
(2026-09-13) -- see that function's docstring for the exact mechanism and why it
differs from the open-loop default.

PO.STEP_DEG is a module-level constant mutated once per worker process inside
_init_worker (each `multiprocessing.get_context("spawn")` worker is a fresh
interpreter, so this mutation is process-local and cannot leak into other arms run
sequentially in the parent process, or into concurrently running scripts).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import multiprocessing as mp
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out"
PREDICTIONS = OUT / "p3_predictions_preregistered.csv"
FROZEN_SHA256 = "359E9FD095F7C53D12D60D8E89562BE0B4583F178ADFEB1F544212F287D7F328"
MULTIPLIERS = (0.7, 1.0, 1.3)
SEED_BASE = 3_100_000  # matches p3_validate_preregistered.py's default exactly

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_basis as PB           # noqa: E402
import pilot_object as PO       # noqa: E402
from p0_direction_preserving_replay import CRITERIA  # noqa: E402

ARMS = {
    "none": dict(step_deg=0.5, substeps_mult=1.0, f_max_mult=1.0, ctrl_noise_mm=1.0, reanchor=False),
    "step_0.25": dict(step_deg=0.25, substeps_mult=1.0, f_max_mult=1.0, ctrl_noise_mm=1.0, reanchor=False),
    "step_1.0": dict(step_deg=1.0, substeps_mult=1.0, f_max_mult=1.0, ctrl_noise_mm=1.0, reanchor=False),
    "substeps_x2": dict(step_deg=0.5, substeps_mult=2.0, f_max_mult=1.0, ctrl_noise_mm=1.0, reanchor=False),
    "fmax_x0.5": dict(step_deg=0.5, substeps_mult=1.0, f_max_mult=0.5, ctrl_noise_mm=1.0, reanchor=False),
    "fmax_x2": dict(step_deg=0.5, substeps_mult=1.0, f_max_mult=2.0, ctrl_noise_mm=1.0, reanchor=False),
    "noise_0": dict(step_deg=0.5, substeps_mult=1.0, f_max_mult=1.0, ctrl_noise_mm=0.0, reanchor=False),
    "closed_reanchor": dict(step_deg=0.5, substeps_mult=1.0, f_max_mult=1.0, ctrl_noise_mm=1.0, reanchor=True),
}
NOT_IMPLEMENTED_ARMS = []

_SPEC = None
_SGN = 1.0
_AXIS = None
_PIVOT = None
_HANDLE = None
_F_MAX_MULT = 1.0
_CTRL_NOISE = 1.0
_REANCHOR = False


@contextmanager
def suppress_native_output():
    null_fd = os.open(os.devnull, os.O_WRONLY)
    saved = [os.dup(1), os.dup(2)]
    try:
        os.dup2(null_fd, 1); os.dup2(null_fd, 2)
        yield
    finally:
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        os.close(saved[0]); os.close(saved[1]); os.close(null_fd)


def stable_seed(namespace, obj, rep, seed_base=SEED_BASE):
    payload = f"p3|{namespace}|{obj}|{rep}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 1_000_000
    return seed_base + offset


def _init_worker(obj, joint, sgn, axis, pivot, handle, target_deg, radius,
                 f_max, substeps, step_deg, substeps_mult, f_max_mult, ctrl_noise_mm, reanchor):
    global _SPEC, _SGN, _AXIS, _PIVOT, _HANDLE, _F_MAX_MULT, _CTRL_NOISE, _REANCHOR
    _SPEC = PO.ObjSpec(obj, joint)
    _SPEC.target_deg = target_deg
    _SPEC.radius = radius
    _SPEC.f_max = f_max
    _SPEC.substeps = max(1, int(round(substeps * substeps_mult)))
    _SGN = float(sgn)
    _AXIS = np.asarray(axis, float)
    _PIVOT = np.asarray(pivot, float)
    _HANDLE = np.asarray(handle, float)
    _F_MAX_MULT = float(f_max_mult)
    _CTRL_NOISE = float(ctrl_noise_mm)
    _REANCHOR = bool(reanchor)
    PO.STEP_DEG = float(step_deg)  # process-local: fresh interpreter under spawn context


def _perpendicular(axis, seed):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    rng = np.random.default_rng(seed)
    while True:
        d = rng.normal(size=3); d -= (d @ axis) * axis
        n = np.linalg.norm(d)
        if n > 1e-8:
            return d / n


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
    init_theta = _SGN * np.radians(init_state_deg)
    handle_start = (_PIVOT + PO.rot_about_axis(_AXIS, init_theta) @ (_HANDLE - _PIVOT))
    plan_total = _SGN * np.radians(_SPEC.target_deg) - _SGN * np.radians(init_state_deg)
    n_steps = max(1, int(abs(np.degrees(plan_total)) / PO.STEP_DEG))
    step_sign = np.sign(plan_total) if plan_total != 0 else 1.0
    rel_angles = step_sign * np.arange(1, n_steps + 1) * np.radians(PO.STEP_DEG)
    e_geo_cm = PB.geometric_path_deviation_cm(_AXIS, _PIVOT, est_axis, est_pivot, handle_start, rel_angles)
    e_norm_cm = PB.normal_path_deviation_cm(_AXIS, _PIVOT, est_axis, est_pivot, handle_start, rel_angles)
    tangential_share = 1.0 - (e_norm_cm / e_geo_cm) ** 2 if e_geo_cm > 1e-9 else 0.0
    exec_seed = phys_seed + 7_000_000
    with suppress_native_output():
        result = PO.run_trial(
            _SPEC, est_axis, est_pivot, init_state_deg,
            np.random.default_rng(exec_seed), sgn=_SGN,
            mass_scale=mass_scale, damping_scale=damping_scale,
            grasp_jitter_cm=1.0, init_state_deg=init_state_deg,
            ctrl_noise_mm=_CTRL_NOISE, f_max=_SPEC.f_max * _F_MAX_MULT, reanchor=_REANCHOR)
    result["task_done"] = int(result["progress"] >= .80)
    for name, fn in CRITERIA.items():
        result[name] = int(fn(result))
    return dict(sweep=sweep, value=value, multiplier=multiplier, rep=rep,
                phys_seed=phys_seed, direction_seed=direction_seed,
                mass_scale=mass_scale, damping_scale=damping_scale,
                init_state_deg=init_state_deg,
                e_geo_nominal_cm=e_geo_cm, e_norm_cm=e_norm_cm,
                tangential_share=tangential_share,
                e_exec_cm=result["max_track_err"] * 100.0, **result)


def run_arm(arm_name, predictions, reps, procs):
    cfg = ARMS[arm_name]
    rows = []
    for pred in predictions:
        obj, joint = pred["object_id"], pred["joint"]
        spec = PO.ObjSpec(obj, joint)
        with suppress_native_output():
            dev, sgn = PO.selfcheck(spec, save=False)
        if dev >= 1e-3:
            raise ValueError(f"{obj} selfcheck failed: {dev * 1000:.3f} mm")
        import pybullet as p
        with suppress_native_output():
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
            for multiplier in MULTIPLIERS:
                for rep in range(reps):
                    tasks.append((sweep, predicted * multiplier, multiplier, rep,
                                  stable_seed("physics", obj, rep), stable_seed(f"direction_{sweep}", obj, rep)))
        initargs = (obj, joint, sgn, axis.tolist(), pivot.tolist(), handle.tolist(),
                    spec.target_deg, spec.radius, spec.f_max, spec.substeps,
                    cfg["step_deg"], cfg["substeps_mult"], cfg["f_max_mult"], cfg["ctrl_noise_mm"],
                    cfg["reanchor"])
        ctx = mp.get_context("spawn")
        with ctx.Pool(procs, initializer=_init_worker, initargs=initargs) as pool:
            measured = pool.map(_one, tasks)
        for row in measured:
            row.update(object_id=obj, category=pred["category"], arm=arm_name)
        rows.extend(measured)
        print(f"  [{arm_name}] {obj}: {len(measured)} trials done", flush=True)
    return rows


def write_csv(path, rows):
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def summarize(all_rows_by_arm):
    """g = e_exec/e_norm per row; report median g and object-to-object IQR of g per arm."""
    lines = ["# P4-a executor interventions: does anything shrink the object-dependence of g "
             "(2026-09-13)", "",
             "> g = e_exec/e_norm (rows with e_norm>0.3cm). Frozen G_EXEC=1.175 comes from the "
             "'none' arm computed on the P3(a) 6656-trial holdout at ALL multipliers; this script "
             "reruns 'none' at only {0.7,1.0,1.3} for a same-N comparison across arms.", "",
             "| arm | n | median g | g IQR (obj-level medians) | Delta IQR vs none |",
             "|---|---:|---:|---:|---:|"]
    obj_iqr = {}
    for arm, rows in all_rows_by_arm.items():
        eg = np.array([float(r["e_norm_cm"]) for r in rows])
        ee = np.array([float(r["e_exec_cm"]) for r in rows])
        objs = np.array([r["object_id"] for r in rows])
        keep = eg > 0.3
        g = ee[keep] / eg[keep]
        objs_k = objs[keep]
        per_obj_median = {o: float(np.median(g[objs_k == o])) for o in sorted(set(objs_k))}
        vals = np.array(list(per_obj_median.values()))
        iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        obj_iqr[arm] = iqr
        delta_str = "--" if arm == "none" else f"{iqr - obj_iqr['none']:+.3f}"
        lines.append(f"| {arm} | {len(rows)} | {np.median(g):.3f} | {iqr:.3f} | {delta_str} |")
    lines += ["", "## Per-object median g by arm", "",
             "| object | " + " | ".join(all_rows_by_arm.keys()) + " |",
             "|---:|" + "---:|" * len(all_rows_by_arm)]
    objects = sorted({r["object_id"] for rows in all_rows_by_arm.values() for r in rows})
    for o in objects:
        cells = []
        for arm, rows in all_rows_by_arm.items():
            eg = np.array([float(r["e_norm_cm"]) for r in rows if r["object_id"] == o])
            ee = np.array([float(r["e_exec_cm"]) for r in rows if r["object_id"] == o])
            keep = eg > 0.3
            cells.append(f"{np.median(ee[keep] / eg[keep]):.3f}" if keep.sum() else "n/a")
        lines.append(f"| {o} | " + " | ".join(cells) + " |")
    lines += ["", "## Reading rule (fixed before running)", "",
             "The arm with the smallest object-level g-IQR is the best-supported explanation for "
             "g's object dependence. If NO arm meaningfully shrinks the IQR relative to 'none', that "
             "is a legitimate result: g's object dependence is geometric (tangential-absorption / "
             "lever-arm), not a controller artifact, and must be reported as such rather than forcing "
             "one of the arms into an explanation.", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS.keys()))
    ap.add_argument("--objects", nargs="+")
    ap.add_argument("--reps", type=int, default=32)
    ap.add_argument("--procs", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args(argv)

    current_sha = hashlib.sha256(PREDICTIONS.read_bytes()).hexdigest().upper()
    if current_sha != FROZEN_SHA256:
        raise ValueError(f"P3(a) prediction CSV changed: {current_sha} != {FROZEN_SHA256}")
    predictions = list(csv.DictReader(PREDICTIONS.open(encoding="utf-8")))
    if args.objects:
        wanted = set(args.objects)
        predictions = [p for p in predictions if p["object_id"] in wanted]

    all_rows_by_arm = {}
    for arm in args.arms:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; choices: {sorted(ARMS)}")
        print(f"=== arm {arm} ===", flush=True)
        rows = run_arm(arm, predictions, args.reps, args.procs)
        write_csv(OUT / f"p4a_{arm}.csv", rows)
        all_rows_by_arm[arm] = rows
        print(f"-> out/p4a_{arm}.csv ({len(rows)} rows)", flush=True)

    if len(all_rows_by_arm) > 1:
        report = summarize(all_rows_by_arm)
        (OUT / "p4a_summary.md").write_text(report, encoding="utf-8")
        print("-> out/p4a_summary.md")
        print(report)


if __name__ == "__main__":
    mp.freeze_support()
    main()
