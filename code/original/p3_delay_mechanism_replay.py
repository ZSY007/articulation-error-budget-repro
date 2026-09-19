"""方案3 E3-B (docs/实验方案3_判据与参考更新机制验证_2026-09-18.md §4): the minimal
PAIRED mechanism experiment for the "reference reset causes progress loss" claim.
A NEW driver, not a reuse of `pilot_object.py::run_trial`'s or the old
`p4e_feedback_noise_stress.py::run_cell`'s seed/RNG-consumption scheme -- both
draw control noise SEQUENTIALLY from one RNG stream inside the step loop, so the
number of reset events (which differs by H) desyncs which draw lands on which
step across different H/rho conditions for "the same" seed. That breaks the
doc's pairing requirement (§4.2, §6: same object/rep must share the identical
physical realization across every H/d/estimate-condition cell). Fixed here by
PRE-GENERATING each (object, rep)'s full control-noise stream and grasp-jitter
draw ONCE, up front, indexed by step -- not re-drawn per H/d condition.

Also adds the full per-step trace and reset-event log this project's archives
never recorded (§4.3): step k, block_start, q, grasp position, anchor,
command, track error, force, stall, seeds, N, Q_target, raw/clipped progress;
and per reset event, old anchor / stale point / current actual point / last &
new command, plus a tangential/normal decomposition of the reference's
backward displacement against the TRUE joint's tangent direction (an
equivalent-q-loss number for the GT condition; a separate normal component for
the perturbed-axis condition, so a spatial distance is never silently read as
a pure progress loss).

Matrix (§4.1): 8 p3c objects x 3 estimate conditions (GT / axis-perturbed /
pivot-perturbed, frozen magnitude from p3c_predictions_preregistered.csv) x
H in {20,40} x rho=d/H in {0,.05,...,.50} (d=round(rho*H) exactly) x 16 reps
= 8,448, plus H-infinity (open-loop) control x 8 x 3 x 16 = 384 -> 8,832 total.
tau-rescoring is OFFLINE only (reuses p3_delay_criterion_rescore.py's C/T/Q/S_tau
logic against this run's own archive) -- no extra executions per tau.

    conda activate cv   (local) / ee900-prime (JARVIS, see jarvis_p3b_replay.sh)
    python error_budget/p3_delay_mechanism_replay.py --procs 16
    -> out/p3_delay_mechanism_20260918/{paired_trials.csv, reset_events.csv,
       traces/trace_<obj>.csv, seed_and_physics_manifest.csv, protocol.json}
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_object as PO                                                     # noqa: E402
from p1_tangent_similarity_offline import JOINT_OVERRIDE                      # noqa: E402
from p4a_controller_interventions import _perpendicular                       # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
RUN_DIR = OUT / "p3_delay_mechanism_20260918"
TRACE_DIR = RUN_DIR / "traces"
EXPERIMENT_VERSION = "v1"
STEP_DEG = 0.5
H_VALUES = [20, 40]
RHO_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
CONDITIONS = ["gt", "axis", "pivot"]
REPS = 16
GRASP_JITTER_CM = 1.0
CTRL_NOISE_MM = 1.0

P3C_PRED_PATH = OUT / "p3c_predictions_preregistered.csv"
P3C_AXIS_COL = "axis_tol50_pred_deg_m0"
P3C_PIVOT_COL = "pivot_tol50_pred_cm_m0"


def derive_seed(payload: str) -> int:
    h = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 31)


def load_p3c_objects():
    rows = list(csv.DictReader(P3C_PRED_PATH.open()))
    return {r["object_id"]: dict(axis_tol_deg=float(r[P3C_AXIS_COL]),
                                 pivot_tol_cm=float(r[P3C_PIVOT_COL])) for r in rows}


def scene_geometry(obj):
    joint = JOINT_OVERRIDE.get(obj, "joint_0")
    spec = PO.ObjSpec(obj, joint)
    devmax, sgn = PO.selfcheck(spec, save=False)
    cid = p.connect(p.DIRECT)
    try:
        bid, jm, lm = PO.build_scene(spec)
        p.resetJointState(bid, jm[spec.joint], 0.0)
        gt_axis_w, gt_pivot_w = PO.gt_axis_pivot_world(bid, jm, spec)
    finally:
        p.disconnect(cid)
    tgt = sgn * np.radians(spec.target_deg)
    n_steps = max(1, int(abs(np.degrees(tgt)) / STEP_DEG))
    return dict(obj=obj, spec=spec, sgn=sgn, gt_axis_w=np.asarray(gt_axis_w, float),
               gt_pivot_w=np.asarray(gt_pivot_w, float), n_steps=n_steps)


def build_estimate(geo, condition, axis_tol_deg, pivot_tol_cm):
    """Perturbation direction/magnitude fixed per (object, condition) -- NOT
    per rep, NOT per H/rho (doc §4.2: "估计扰动方向独立于H/d固定")."""
    if condition == "gt":
        return geo["gt_axis_w"].copy(), geo["gt_pivot_w"].copy()
    dseed = derive_seed(f"p3b|{EXPERIMENT_VERSION}|dir|{condition}|{geo['obj']}")
    d = _perpendicular(geo["gt_axis_w"], dseed)
    if condition == "axis":
        ea = PO.rot_about_axis(d, np.radians(axis_tol_deg)) @ geo["gt_axis_w"]
        return ea / np.linalg.norm(ea), geo["gt_pivot_w"].copy()
    ep = geo["gt_pivot_w"] + d * (pivot_tol_cm / 100.0)
    return geo["gt_axis_w"].copy(), ep


def pregenerate_noise(obj, rep, n_steps):
    phys_seed = derive_seed(f"p3b|{EXPERIMENT_VERSION}|phys|{obj}|{rep}")
    grasp_seed = derive_seed(f"p3b|{EXPERIMENT_VERSION}|grasp|{obj}|{rep}")
    ctrl_seed = derive_seed(f"p3b|{EXPERIMENT_VERSION}|ctrl|{obj}|{rep}")
    obs_seed = derive_seed(f"p3b|{EXPERIMENT_VERSION}|obs|{obj}|{rep}")  # reserved; obs_noise=0 this round
    phys_rng = np.random.default_rng(phys_seed)
    mass_scale = float(phys_rng.uniform(0.5, 2.0))
    damping_scale = float(phys_rng.uniform(0.5, 2.0))
    grasp_rng = np.random.default_rng(grasp_seed)
    grasp_jitter = grasp_rng.normal(0, GRASP_JITTER_CM / 100.0, 3)
    ctrl_rng = np.random.default_rng(ctrl_seed)
    ctrl_noise_stream = ctrl_rng.normal(0, CTRL_NOISE_MM / 1000.0, (n_steps, 3))
    return dict(phys_seed=phys_seed, grasp_seed=grasp_seed, ctrl_seed=ctrl_seed, obs_seed=obs_seed,
               mass_scale=mass_scale, damping_scale=damping_scale,
               grasp_jitter=grasp_jitter, ctrl_noise_stream=ctrl_noise_stream)


def run_traced_trial(geo, est_axis, est_pivot, H, d_delay, noise, want_trace=True):
    """Mirrors pilot_object.py::run_trial's verified physics/controller math
    exactly (same scene build, same constraint, same command formula, same
    stall/track-error definitions) -- the only changes are (a) control noise
    read from a pregenerated, step-indexed array instead of drawn live, so it
    stays IDENTICAL across H/rho conditions sharing this (object,rep), and
    (b) full per-step + reset-event tracing. init_state_deg=0 and
    est_state_deg=0 (q0=0 both true and estimated) are this experiment's own
    fixed convention (§4.1), not parameters."""
    spec, sgn = geo["spec"], geo["sgn"]
    f_max = spec.f_max
    cid = p.connect(p.DIRECT)
    try:
        bid, jm, lm = PO.build_scene(spec, noise["mass_scale"], noise["damping_scale"])
        jidx, ci = jm[spec.joint], lm[spec.child]
        p.resetJointState(bid, jidx, 0.0)
        for _ in range(20):
            p.stepSimulation()
        h_local = spec.h_local + noise["grasp_jitter"]
        h_now = PO.handle_world(bid, ci, h_local)
        t_d, R_d = PO.link_world(bid, ci)
        handle_in_door = R_d.T @ (h_now - t_d)
        c = p.createConstraint(bid, ci, -1, -1, p.JOINT_POINT2POINT, [0, 0, 0],
                               handle_in_door.tolist(), h_now.tolist())
        p.changeConstraint(c, h_now.tolist(), maxForce=f_max)

        tgt = sgn * np.radians(spec.target_deg)
        n_steps = max(1, int(abs(np.degrees(tgt)) / STEP_DEG))
        assert n_steps == geo["n_steps"], "n_steps drifted from geometry precompute"
        s = np.sign(tgt) if tgt != 0 else 1.0
        track_err, stall, forces = [], 0, []
        q_prev = p.getJointState(bid, jidx)[0]
        gpos_history = [h_now]
        anchor = h_now
        block_start = 1
        cmd_prev = h_now.copy()
        step_rows, reset_rows = [], []

        for k in range(1, n_steps + 1):
            if H is not None and (k - block_start) >= H:
                delayed_idx = max(0, len(gpos_history) - 1 - d_delay)
                observed = gpos_history[delayed_idx]
                old_anchor = anchor
                actual_point_now = gpos_history[-1]
                radius_vec = actual_point_now - geo["gt_pivot_w"]
                radius = float(np.linalg.norm(radius_vec))
                tangent_raw = np.cross(geo["gt_axis_w"], radius_vec)
                tn = float(np.linalg.norm(tangent_raw))
                backward_vec = actual_point_now - observed
                if tn > 1e-9 and radius > 1e-9:
                    tangent_dir = tangent_raw / tn
                    tang_cm = float(np.dot(backward_vec, tangent_dir)) * 100.0
                    equiv_q_loss_deg = float(np.degrees(np.dot(backward_vec, tangent_dir) / radius))
                    normal_vec = backward_vec - np.dot(backward_vec, tangent_dir) * tangent_dir
                    normal_cm = float(np.linalg.norm(normal_vec)) * 100.0
                else:
                    tang_cm = equiv_q_loss_deg = normal_cm = float("nan")
                reset_rows.append(dict(
                    step=k, block_start=block_start, observation_index=delayed_idx,
                    old_anchor_x=old_anchor[0], old_anchor_y=old_anchor[1], old_anchor_z=old_anchor[2],
                    stale_point_x=observed[0], stale_point_y=observed[1], stale_point_z=observed[2],
                    actual_point_x=actual_point_now[0], actual_point_y=actual_point_now[1], actual_point_z=actual_point_now[2],
                    last_command_x=cmd_prev[0], last_command_y=cmd_prev[1], last_command_z=cmd_prev[2],
                    tangential_backward_cm=tang_cm, equiv_q_loss_deg=equiv_q_loss_deg,
                    normal_component_cm=normal_cm,
                ))
                anchor = observed
                block_start = k
            local_k = k - block_start + 1
            ang = s * local_k * np.radians(STEP_DEG)
            cmd = est_pivot + PO.rot_about_axis(est_axis, float(ang)) @ (anchor - est_pivot)
            cmd = cmd + noise["ctrl_noise_stream"][k - 1]
            p.changeConstraint(c, cmd.tolist(), maxForce=f_max)
            for _ in range(spec.substeps):
                p.stepSimulation()
            tt, RR = PO.link_world(bid, ci)
            gpos = tt + RR @ handle_in_door
            te = float(np.linalg.norm(gpos - cmd))
            track_err.append(te)
            f = float(np.linalg.norm(p.getConstraintState(c)[:3]))
            forces.append(f)
            q_now = p.getJointState(bid, jidx)[0]
            is_stall = int(f >= f_max and abs(q_now - q_prev) < 0.3 * np.radians(STEP_DEG))
            stall += is_stall
            if want_trace:
                step_rows.append(dict(step=k, block_start=block_start, q_rad=q_now,
                                      gpos_x=gpos[0], gpos_y=gpos[1], gpos_z=gpos[2],
                                      anchor_x=anchor[0], anchor_y=anchor[1], anchor_z=anchor[2],
                                      command_x=cmd[0], command_y=cmd[1], command_z=cmd[2],
                                      track_err=te, force=f, stall=is_stall))
            q_prev = q_now
            gpos_history.append(gpos)
            cmd_prev = cmd

        gt_final = p.getJointState(bid, jidx)[0]
        raw_progress = float(gt_final / tgt) if tgt != 0 else float("nan")
        te_arr = np.asarray(track_err)
        sat = stall / max(1, n_steps)
        summary = dict(N=n_steps, Q_target_deg=float(np.degrees(tgt)),
                       raw_progress=raw_progress, clipped_progress=float(np.clip(raw_progress, 0, 1.5)),
                       max_track_err=float(te_arr.max()), mean_track_err=float(te_arr.mean()),
                       sat_frac=float(sat), peak_force=float(np.max(forces)))
        return summary, step_rows, reset_rows
    finally:
        p.disconnect(cid)


# ---------------------------------------------------------------- workers --
_CTX = None  # dict: obj -> dict(geo, estimates{cond:(ea,ep)}, noise[rep])


def _init_worker(ctx):
    global _CTX
    _CTX = ctx


def _one(task):
    obj, condition, H, rho, rep, is_hinf = task
    entry = _CTX[obj]
    geo = entry["geo"]
    ea, ep = entry["estimates"][condition]
    noise = entry["noise"][rep]
    if is_hinf:
        H_val, d = None, 0
    else:
        H_val = H
        d = int(round(rho * H))
    summary, step_rows, reset_rows = run_traced_trial(geo, ea, ep, H_val, d, noise, want_trace=True)
    trial_row = dict(object_id=obj, condition=condition,
                     H=("Hinf" if is_hinf else H), rho=(float("nan") if is_hinf else rho),
                     d=d, rep=rep,
                     phys_seed=noise["phys_seed"], grasp_seed=noise["grasp_seed"],
                     ctrl_seed=noise["ctrl_seed"], **summary)
    for r in step_rows:
        r["object_id"] = obj; r["condition"] = condition
        r["H"] = ("Hinf" if is_hinf else H); r["rho"] = (float("nan") if is_hinf else rho)
        r["d"] = d; r["rep"] = rep
    for r in reset_rows:
        r["object_id"] = obj; r["condition"] = condition
        r["H"] = ("Hinf" if is_hinf else H); r["rho"] = (float("nan") if is_hinf else rho)
        r["d"] = d; r["rep"] = rep
    return trial_row, step_rows, reset_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--objects", default="", help="comma-separated subset for a partial/smoke run")
    args = ap.parse_args()

    RUN_DIR.mkdir(exist_ok=True)
    TRACE_DIR.mkdir(exist_ok=True)

    p3c = load_p3c_objects()
    objs = sorted(p3c.keys())
    if args.objects:
        objs = [o for o in objs if o in set(args.objects.split(","))]
    print(f"[p3b] objects: {objs}")

    ctx = {}
    manifest_rows = []
    for obj in objs:
        geo = scene_geometry(obj)
        estimates = {cond: build_estimate(geo, cond, p3c[obj]["axis_tol_deg"], p3c[obj]["pivot_tol_cm"])
                    for cond in CONDITIONS}
        noise = {rep: pregenerate_noise(obj, rep, geo["n_steps"]) for rep in range(REPS)}
        ctx[obj] = dict(geo=geo, estimates=estimates, noise=noise)
        for rep in range(REPS):
            n = noise[rep]
            manifest_rows.append(dict(object_id=obj, rep=rep, n_steps=geo["n_steps"],
                                      phys_seed=n["phys_seed"], grasp_seed=n["grasp_seed"],
                                      ctrl_seed=n["ctrl_seed"], obs_seed_reserved=n["obs_seed"],
                                      mass_scale=n["mass_scale"], damping_scale=n["damping_scale"]))
        print(f"  [{obj}] n_steps={geo['n_steps']}, estimates built for {list(estimates.keys())}")

    with open(RUN_DIR / "seed_and_physics_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader(); w.writerows(manifest_rows)
    print(f"-> seed_and_physics_manifest.csv ({len(manifest_rows)} rows)")

    tasks = []
    for obj in objs:
        for condition in CONDITIONS:
            for rep in range(REPS):
                for H in H_VALUES:
                    for rho in RHO_VALUES:
                        tasks.append((obj, condition, H, rho, rep, False))
                tasks.append((obj, condition, None, None, rep, True))  # Hinf, once
    n_want = len(objs) * 3 * REPS * (len(H_VALUES) * len(RHO_VALUES) + 1)
    print(f"[p3b] {len(tasks)} total trials (want {n_want} for this object set)")

    ctx_ctx = mp.get_context("spawn")
    trial_f = open(RUN_DIR / "paired_trials.csv", "w", newline="")
    trial_writer = None
    reset_f = open(RUN_DIR / "reset_events.csv", "w", newline="")
    reset_writer = None
    trace_files = {}  # obj -> (file, writer)

    n_done = 0
    with ctx_ctx.Pool(args.procs, initializer=_init_worker, initargs=(ctx,)) as pool:
        for trial_row, step_rows, reset_rows in pool.imap_unordered(_one, tasks, chunksize=4):
            if trial_writer is None:
                trial_writer = csv.DictWriter(trial_f, fieldnames=list(trial_row.keys()))
                trial_writer.writeheader()
            trial_writer.writerow(trial_row)

            if reset_rows and reset_writer is None:
                reset_writer = csv.DictWriter(reset_f, fieldnames=list(reset_rows[0].keys()))
                reset_writer.writeheader()
            for r in reset_rows:
                reset_writer.writerow(r)

            obj = trial_row["object_id"]
            if obj not in trace_files:
                tf = open(TRACE_DIR / f"trace_{obj}.csv", "w", newline="")
                trace_files[obj] = (tf, None)
            tf, tw = trace_files[obj]
            if step_rows:
                if tw is None:
                    tw = csv.DictWriter(tf, fieldnames=list(step_rows[0].keys()))
                    tw.writeheader()
                    trace_files[obj] = (tf, tw)
                for r in step_rows:
                    tw.writerow(r)

            n_done += 1
            if n_done % 200 == 0:
                print(f"  [p3b] {n_done}/{len(tasks)} trials done")

    trial_f.close(); reset_f.close()
    for tf, _ in trace_files.values():
        tf.close()
    print(f"-> paired_trials.csv, reset_events.csv, traces/trace_<obj>.csv ({n_done} trials)")

    protocol = dict(date="2026-09-18", protocol_doc="docs/实验方案3_判据与参考更新机制验证_2026-09-18.md",
                    experiment_version=EXPERIMENT_VERSION, objects=objs, h_values=H_VALUES,
                    rho_values=RHO_VALUES, conditions=CONDITIONS, reps=REPS, step_deg=STEP_DEG,
                    n_trials=n_done)
    with open(RUN_DIR / "protocol.json", "w") as f:
        json.dump(protocol, f, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
