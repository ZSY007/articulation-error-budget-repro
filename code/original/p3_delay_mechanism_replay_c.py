"""方案3 E3-C (docs/实验方案3_判据与参考更新机制验证_2026-09-18.md §5): optional
directed-recovery compensation, added since E3-B confirmed the expected reset
loss mechanism (finite-block model MAE=0.0046; tracking/stall gates diff
exactly 0.000 while progress drops -- see p3_delay_mechanism_analyze.py's
report). Adds a "delay-forward-compensation" condition: instead of anchoring
the new reference block at the stale observed point directly (E3-B's
baseline), project that stale point FORWARD along the CONTROLLER's own
estimated axis by the exact number of commanded steps it is known to be
behind:

    anchor_comp = phat + Rot(ahat, s * d_eff * Delta_q) @ (stale_anchor - phat)
    d_eff = actual steps between the stale reading and the current true
            observation (accounts for early-trajectory clipping in
            gpos_history, not just the nominal delay `d`)

This uses ONLY the estimated axis/pivot and the KNOWN delay -- never the true
current position, which would silently recreate a no-delay oracle rather than
test a deployable compensation. Reuses E3-B's exact geometry/estimate/noise
construction (same EXPERIMENT_VERSION, same seed namespaces) so the physical
realization for a given (object, rep) is IDENTICAL to E3-B's -- the doc's own
"use the same physical configuration and noise stream as B" requirement,
satisfied by construction rather than by copying files.

Matrix (§5): 8 p3c objects x {axis, pivot} estimate conditions (GT excluded --
compensation has nothing to correct for a perfect estimate) x H in {20,40} x
rho in {0.20,0.40} x 16 reps = 1,024 trials. Baseline for comparison is E3-B's
own matching rows (`paired_trials.csv`), not re-executed here.

    conda activate cv   (local) / ee900-prime (JARVIS, see jarvis_p3c_replay.sh)
    python error_budget/p3_delay_mechanism_replay.py --procs 16   # E3-B first
    python error_budget/p3_delay_mechanism_replay_c.py --procs 16
    -> out/p3_delay_mechanism_20260918/{paired_trials_c.csv, reset_events_c.csv,
       traces_c/trace_<obj>.csv}
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_object as PO                                                     # noqa: E402
from p3_delay_mechanism_replay import (                                       # noqa: E402
    RUN_DIR, STEP_DEG, load_p3c_objects, scene_geometry, build_estimate,
    pregenerate_noise, REPS,
)

TRACE_DIR_C = RUN_DIR / "traces_c"
H_VALUES = [20, 40]
RHO_VALUES = [0.20, 0.40]
CONDITIONS = ["axis", "pivot"]  # GT excluded: nothing to compensate for


def run_traced_trial_compensated(geo, est_axis, est_pivot, H, d_delay, noise, want_trace=True):
    """Identical to p3_delay_mechanism_replay.run_traced_trial except at each
    reset: the new anchor is the delay-forward-COMPENSATED stale point, not
    the raw stale point. See module docstring for the formula and the
    no-oracle-cheating constraint."""
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
                d_eff = (len(gpos_history) - 1) - delayed_idx
                old_anchor = anchor
                actual_point_now = gpos_history[-1]

                # compensation: project the stale point forward by d_eff
                # commanded steps, using the CONTROLLER's own est_axis/pivot --
                # never the true axis, never the true current position.
                ang_comp = s * d_eff * np.radians(STEP_DEG)
                anchor_comp = est_pivot + PO.rot_about_axis(est_axis, float(ang_comp)) @ (observed - est_pivot)

                radius_vec = actual_point_now - geo["gt_pivot_w"]
                radius = float(np.linalg.norm(radius_vec))
                tangent_raw = np.cross(geo["gt_axis_w"], radius_vec)
                tn = float(np.linalg.norm(tangent_raw))
                backward_vec = actual_point_now - anchor_comp
                if tn > 1e-9 and radius > 1e-9:
                    tangent_dir = tangent_raw / tn
                    tang_cm = float(np.dot(backward_vec, tangent_dir)) * 100.0
                    equiv_q_loss_deg = float(np.degrees(np.dot(backward_vec, tangent_dir) / radius))
                    normal_vec = backward_vec - np.dot(backward_vec, tangent_dir) * tangent_dir
                    normal_cm = float(np.linalg.norm(normal_vec)) * 100.0
                else:
                    tang_cm = equiv_q_loss_deg = normal_cm = float("nan")
                reset_rows.append(dict(
                    step=k, block_start=block_start, observation_index=delayed_idx, d_eff=d_eff,
                    old_anchor_x=old_anchor[0], old_anchor_y=old_anchor[1], old_anchor_z=old_anchor[2],
                    stale_point_x=observed[0], stale_point_y=observed[1], stale_point_z=observed[2],
                    compensated_anchor_x=anchor_comp[0], compensated_anchor_y=anchor_comp[1], compensated_anchor_z=anchor_comp[2],
                    actual_point_x=actual_point_now[0], actual_point_y=actual_point_now[1], actual_point_z=actual_point_now[2],
                    last_command_x=cmd_prev[0], last_command_y=cmd_prev[1], last_command_z=cmd_prev[2],
                    tangential_backward_cm=tang_cm, equiv_q_loss_deg=equiv_q_loss_deg,
                    normal_component_cm=normal_cm,
                ))
                anchor = anchor_comp
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


_CTX = None


def _init_worker(ctx):
    global _CTX
    _CTX = ctx


def _one(task):
    obj, condition, H, rho, rep = task
    entry = _CTX[obj]
    geo = entry["geo"]
    ea, ep = entry["estimates"][condition]
    noise = entry["noise"][rep]
    d = int(round(rho * H))
    summary, step_rows, reset_rows = run_traced_trial_compensated(geo, ea, ep, H, d, noise, want_trace=True)
    trial_row = dict(object_id=obj, condition=condition, H=H, rho=rho, d=d, rep=rep,
                     phys_seed=noise["phys_seed"], grasp_seed=noise["grasp_seed"],
                     ctrl_seed=noise["ctrl_seed"], **summary)
    for r in step_rows:
        r.update(object_id=obj, condition=condition, H=H, rho=rho, d=d, rep=rep)
    for r in reset_rows:
        r.update(object_id=obj, condition=condition, H=H, rho=rho, d=d, rep=rep)
    return trial_row, step_rows, reset_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()

    RUN_DIR.mkdir(exist_ok=True)
    TRACE_DIR_C.mkdir(exist_ok=True)

    p3c = load_p3c_objects()
    objs = sorted(p3c.keys())
    print(f"[p3c] objects: {objs}")

    ctx = {}
    for obj in objs:
        geo = scene_geometry(obj)
        estimates = {cond: build_estimate(geo, cond, p3c[obj]["axis_tol_deg"], p3c[obj]["pivot_tol_cm"])
                    for cond in CONDITIONS}
        noise = {rep: pregenerate_noise(obj, rep, geo["n_steps"]) for rep in range(REPS)}
        ctx[obj] = dict(geo=geo, estimates=estimates, noise=noise)
        print(f"  [{obj}] n_steps={geo['n_steps']}")

    tasks = [(obj, cond, H, rho, rep)
            for obj in objs for cond in CONDITIONS for H in H_VALUES for rho in RHO_VALUES
            for rep in range(REPS)]
    print(f"[p3c] {len(tasks)} total trials (want {len(objs)*2*2*2*REPS}=1024)")

    mp_ctx = mp.get_context("spawn")
    trial_f = open(RUN_DIR / "paired_trials_c.csv", "w", newline="")
    trial_writer = None
    reset_f = open(RUN_DIR / "reset_events_c.csv", "w", newline="")
    reset_writer = None
    trace_files = {}

    n_done = 0
    with mp_ctx.Pool(args.procs, initializer=_init_worker, initargs=(ctx,)) as pool:
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
                trace_files[obj] = (open(TRACE_DIR_C / f"trace_{obj}.csv", "w", newline=""), None)
            tf, tw = trace_files[obj]
            if step_rows:
                if tw is None:
                    tw = csv.DictWriter(tf, fieldnames=list(step_rows[0].keys()))
                    tw.writeheader()
                    trace_files[obj] = (tf, tw)
                for r in step_rows:
                    tw.writerow(r)
            n_done += 1
            if n_done % 100 == 0:
                print(f"  [p3c] {n_done}/{len(tasks)} trials done")

    trial_f.close(); reset_f.close()
    for tf, _ in trace_files.values():
        tf.close()
    print(f"-> paired_trials_c.csv, reset_events_c.csv, traces_c/trace_<obj>.csv ({n_done} trials)")


if __name__ == "__main__":
    main()
