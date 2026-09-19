"""E4 grasp-dependence supplement (Section V-C follow-up): extend the single-object
(35059) real-gripper-vs-firm-grasp pilot to 10 NEW held-out objects from the expanded
library, all with a named handle/knob/bar visual mesh (not the door-edge fallback,
which is not a graspable geometry for a real parallel-jaw gripper).

Generic grasp geometry (no per-object hand-tuning, unlike pilot_gripper.py/
pilot_gripper_45677.py which each had a manually-picked world-frame APPROACH vector):
  outward_world  = direction from the moving link's visual-mesh AABB center to the
                   handle point, in world frame -- the direction the handle protrudes
                   away from the door/drawer body, i.e. into free space.
  APPROACH_world = -outward_world (gripper starts in free space beyond the handle and
                   drives back toward the body, matching pilot_gripper.py's convention)
  PINCH_world    = normalize(cross(APPROACH_world, a_w))  (a_w = GT hinge axis) --
                   perpendicular to both travel direction and hinge axis; correct for
                   the common case of a handle bar mounted parallel to the hinge axis,
                   wrong for a bar mounted perpendicular to it. We do NOT hand-verify
                   this per object -- instead every object must pass a GATE (see below)
                   before its number counts.

pr2_gripper.urdf convention (read directly from the asset, not guessed): local +x =
forward/approach, local +y = finger separation/pinch axis, local +z completes the
right-handed frame. The rotation matrix [APPROACH_world | PINCH_world | UP_world] (as
columns) is converted to a quaternion with scipy.spatial.transform.Rotation.

GATE (pre-registered before running the delta sweep, to avoid silently keeping only
the objects that happen to work): at delta_deg=0 (GT axis/pivot, ideal case), run 15
trials. An object must reach >=70% task_done to enter the delta-sweep comparison --
this is deliberately looser than pilot_object.py's single-object G1 gate (90%) because
the grasp GEOMETRY here is a generic heuristic applied cold to 10 diverse meshes, not
a hand-tuned rig; objects that fail are reported as "heuristic grasp failed", not
excluded silently and not force-fit by retuning thresholds.

2026-09-17 item D fixes (docs/方向调整与补实验审阅_2026-09-16.md SS3.3, SS4.D -- four
issues found by direct code audit, all fixed here, not just documented):
  1. Seeds were Python hash(obj) -- NOT reproducible across processes/runs (str hash
     is randomized per interpreter unless PYTHONHASHSEED is fixed). Replaced with a
     sha256-based stable_seed(), same pattern as p4a_controller_interventions.py.
  2. Gripper orientation tracked GT axis a_w during the sweep even though POSITION
     tracked the (possibly wrong) est_axis -- an undisclosed oracle on the gripper
     side with no equivalent on the point-executor side. Both sides now rotate by
     est_axis consistently: what the "controller" believes is what BOTH conditions
     are commanded with.
  3. `held` checked contact between the WHOLE gripper body and the WHOLE object body
     (bodyB=bid), not finger-vs-moving-handle-link specifically -- a contact anywhere
     on the object (e.g. the static frame) would count. Now filtered to
     linkIndexB == the moving child link.
  4. The two conditions used DIFFERENT success definitions entirely (gripper:
     progress+held; point: PO.run_trial's hardcoded 2cm/20%-sat firm) and different
     force caps (gripper wrist: 600N fixed; point: spec.f_max, radius-scaled, ~50-90N)
     with no sensitivity check. Both trial functions now return RAW telemetry
     (progress, max_track_err, contact/sat fraction, peak force) instead of a single
     bool, so `firm_like()` applies the IDENTICAL threshold function to both, and the
     tracking-error budget is swept at {1, 2, 4} sim-cm (not asserted to hold only at
     2cm). Force caps are reported explicitly per condition, matched-cap is the
     default comparison; the original generous 600N wrist cap is kept as a labeled
     secondary arm, not silently baked into the headline number.

    conda activate cv
    python error_budget/eval_gripper_10obj.py --gate     # step 1: gate only, fast
    python error_budget/eval_gripper_10obj.py --sweep     # step 2: full sweep on objects that passed
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation

import pilot_object as PO

OUT = PO.OUT
OBJS = ["46889", "47315", "48271", "10144", "10849", "7119", "41004", "45091", "10143", "10347"]
JOINT_OVERRIDE = {"45091": "joint_1"}

GRIP_FORCE = 150.0
CLOSE_ANGLE = 0.36
FINGER_FRICTION = 2.5
DELTAS = [0, 5, 10, 15, 20, 30, 45]
N_GATE = 15
N_SWEEP = 10
GATE_THRESH = 0.70
WRIST_FMAX_GENEROUS = 600.0   # original arm's fixed wrist-constraint force cap
TRACK_BUDGETS_CM = (1.0, 2.0, 4.0)   # firm_like() sensitivity sweep, item D


def stable_seed(namespace, obj, rep, seed_base=91_000):
    """sha256-based, reproducible across processes/runs/interpreters -- replaces the
    old `hash(obj)` (Python string hash is randomized per-process by default)."""
    payload = f"gripper10|{namespace}|{obj}|{rep}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 1_000_000
    return seed_base + offset


def firm_like(progress, track_err_m, quality_frac, track_budget_cm=2.0, quality_frac_budget=0.20):
    """Identical success definition applied to BOTH conditions' raw telemetry --
    the point executor's `quality_frac` is sat_frac (force-saturated/stalled-step
    fraction); the gripper's is contact_lost_frac (fraction of steps with no
    finger-to-moving-handle-link contact). Both are "fraction of steps where
    execution quality broke down", just from different physical causes."""
    return bool(progress >= 0.80 and track_err_m < track_budget_cm / 100.0
               and quality_frac < quality_frac_budget)


def link_mesh_aabb(urdf_path: Path, link_name: str):
    root = ET.parse(urdf_path).getroot()
    link = next(l for l in root.findall("link") if l.get("name") == link_name)
    lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
    for vis in link.findall("visual"):
        mesh = vis.find("geometry/mesh")
        if mesh is None:
            continue
        o = vis.find("origin")
        off = (np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
               if o is not None else np.zeros(3))
        verts = [[float(x) for x in ln.split()[1:4]]
                 for ln in (urdf_path.parent / mesh.get("filename")).read_text().splitlines()
                 if ln.startswith("v ")]
        v = np.asarray(verts) + off
        lo = np.minimum(lo, v.min(0)); hi = np.maximum(hi, v.max(0))
    return lo, hi


def grasp_frame(spec: PO.ObjSpec, bid, jm, lm):
    """Returns (h0, a_w, p_w, outward_world, g_orn_quat)."""
    ci = lm[spec.child]
    a_w, p_w = PO.gt_axis_pivot_world(bid, jm, spec)
    h0 = PO.handle_world(bid, ci, spec.h_local)
    lo, hi = link_mesh_aabb(spec.urdf, spec.child)
    center_local = (lo + hi) / 2.0
    t_d, R_d = PO.link_world(bid, ci)
    outward_local = spec.h_local - center_local
    n = np.linalg.norm(outward_local)
    if n < 1e-6:
        outward_local = np.array([1.0, 0.0, 0.0])
    else:
        outward_local = outward_local / n
    outward_world = R_d @ outward_local
    outward_world = outward_world / np.linalg.norm(outward_world)
    approach_world = -outward_world
    pinch_world = np.cross(approach_world, a_w)
    pn = np.linalg.norm(pinch_world)
    if pn < 1e-6:  # approach nearly parallel to hinge axis -- degenerate, pick arbitrary perpendicular
        tmp = np.array([1.0, 0.0, 0.0]) if abs(approach_world[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        pinch_world = np.cross(approach_world, tmp)
        pn = np.linalg.norm(pinch_world)
    pinch_world = pinch_world / pn
    up_world = np.cross(approach_world, pinch_world)
    Rmat = np.column_stack([approach_world, pinch_world, up_world])
    quat = Rotation.from_matrix(Rmat).as_quat()  # (x,y,z,w), matches pybullet convention
    return h0, a_w, p_w, outward_world, quat.tolist()


def run_gripper_trial(obj, joint, delta_deg, seed, init_state_deg=0.0, mass_scale=1.0,
                      wrist_f_max=None):
    rng = np.random.default_rng(seed)
    spec = PO.ObjSpec(obj, joint)
    _, sgn = PO.selfcheck(spec, save=False)  # sets spec.target_deg/substeps/f_max too
    wrist_f_max = spec.f_max if wrist_f_max is None else wrist_f_max  # matched-cap default
    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        bid, jm, lm = PO.build_scene(spec, mass_scale=mass_scale)
        jidx, ci = jm[spec.joint], lm[spec.child]
        p.resetJointState(bid, jidx, sgn * np.radians(init_state_deg))
        for _ in range(20):
            p.stepSimulation()

        h0, a_w, p_w, outward_world, g_orn = grasp_frame(spec, bid, jm, lm)
        if delta_deg:
            d = rng.normal(size=3); d -= (d @ a_w) * a_w; d /= np.linalg.norm(d)
            est_axis = PO.rot_about_axis(d, np.radians(delta_deg)) @ a_w
            est_axis /= np.linalg.norm(est_axis)
        else:
            est_axis = a_w
        est_pivot = p_w

        g_base = h0 + 0.19 * outward_world
        grip = p.loadURDF("pr2_gripper.urdf", g_base.tolist(), g_orn)
        for j in range(p.getNumJoints(grip)):
            p.changeDynamics(grip, j, lateralFriction=FINGER_FRICTION, spinningFriction=0.2)
        p.changeDynamics(bid, ci, lateralFriction=1.2)
        lg = next(j for j in range(p.getNumJoints(grip)) if p.getJointInfo(grip, j)[1] == b"left_gripper_joint")
        rg = next(j for j in range(p.getNumJoints(grip)) if p.getJointInfo(grip, j)[1] == b"right_gripper_joint")
        wc = p.createConstraint(grip, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                g_base.tolist(), childFrameOrientation=g_orn)
        p.changeConstraint(wc, g_base.tolist(), g_orn, maxForce=wrist_f_max)

        for a_, jj in ((0.0, lg), (0.0, rg)):
            p.setJointMotorControl2(grip, jj, p.POSITION_CONTROL, targetPosition=a_, force=GRIP_FORCE)
        for _ in range(40):
            p.stepSimulation()
        for jj in (lg, rg):
            p.setJointMotorControl2(grip, jj, p.POSITION_CONTROL, targetPosition=CLOSE_ANGLE, force=GRIP_FORCE)
        for _ in range(140):
            p.stepSimulation()

        tgt = sgn * np.radians(spec.target_deg)
        n_steps = max(1, int(abs(np.degrees(tgt - sgn * np.radians(init_state_deg))) / PO.STEP_DEG))
        contact_lost_steps = 0
        track_err, forces = [], []
        # item D fix #2: orientation now driven by est_axis (what the controller
        # believes), matching the point executor -- not a_w (GT), which was an
        # undisclosed oracle on this side only.
        for k in range(1, n_steps + 1):
            ang = sgn * k * np.radians(PO.STEP_DEG)
            newpos = est_pivot + PO.rot_about_axis(est_axis, float(ang)) @ (g_base - est_pivot)
            newpos = newpos + rng.normal(0, 0.001, 3)
            g_orn_k = p.getQuaternionFromAxisAngle(est_axis.tolist(), float(ang))
            g_orn_k = p.multiplyTransforms([0, 0, 0], g_orn_k, [0, 0, 0], g_orn)[1]
            p.changeConstraint(wc, newpos.tolist(), g_orn_k, maxForce=wrist_f_max)
            for jj in (lg, rg):
                p.setJointMotorControl2(grip, jj, p.POSITION_CONTROL, targetPosition=CLOSE_ANGLE, force=GRIP_FORCE)
            for _ in range(spec.substeps):
                p.stepSimulation()
            actual_pos, _ = p.getBasePositionAndOrientation(grip)
            track_err.append(float(np.linalg.norm(np.asarray(actual_pos) - newpos)))
            f = np.linalg.norm(p.getConstraintState(wc)[:3])
            forces.append(float(f))
            # item D fix #3: finger<->MOVING-HANDLE-LINK contact only, not whole body
            cps = p.getContactPoints(bodyA=grip, bodyB=bid)
            finger_handle_contact = any(cp[4] == ci for cp in cps)  # cp[4] = linkIndexB
            if not finger_handle_contact:
                contact_lost_steps += 1

        contact_lost_frac = contact_lost_steps / max(1, n_steps)
        held = contact_lost_frac <= 0.20  # legacy definition, kept for comparability
        q_final = p.getJointState(bid, jidx)[0]
        progress = (q_final - sgn * np.radians(init_state_deg)) / (tgt - sgn * np.radians(init_state_deg))
        te = np.asarray(track_err)
        return dict(obj=obj, delta_deg=delta_deg, seed=seed, held=int(held),
                   progress=float(np.clip(progress, 0, 1.6)),
                   max_track_err=float(te.max()), mean_track_err=float(te.mean()),
                   contact_lost_frac=float(contact_lost_frac),
                   peak_force=float(np.max(forces)), wrist_f_max=float(wrist_f_max),
                   task_done=int(progress >= 0.80 and held))  # legacy field, kept for comparability
    finally:
        p.disconnect(cid)


def run_firm_trial(obj, joint, delta_deg, seed, init_state_deg=0.0, mass_scale=1.0):
    rng = np.random.default_rng(seed)
    spec = PO.ObjSpec(obj, joint)
    _, sgn = PO.selfcheck(spec, save=False)  # sets spec.target_deg/substeps/f_max too
    cid = p.connect(p.DIRECT)
    try:
        bid, jm, lm = PO.build_scene(spec, mass_scale=mass_scale)
        a_w, p_w = PO.gt_axis_pivot_world(bid, jm, spec)
    finally:
        p.disconnect(cid)
    if delta_deg:
        d = rng.normal(size=3); d -= (d @ a_w) * a_w; d /= np.linalg.norm(d)
        est_axis = PO.rot_about_axis(d, np.radians(delta_deg)) @ a_w
        est_axis /= np.linalg.norm(est_axis)
    else:
        est_axis = a_w
    r = PO.run_trial(spec, est_axis, p_w, 0.0, rng, sgn=sgn, mass_scale=mass_scale,
                     init_state_deg=init_state_deg)
    return dict(obj=obj, delta_deg=delta_deg, seed=seed,
               progress=r["progress"], max_track_err=r["max_track_err"],
               sat_frac=r["sat_frac"], peak_force=r["peak_force"],
               f_max=float(spec.f_max), task_done=int(r["success"]))  # legacy field


def prep_spec(obj):
    joint = JOINT_OVERRIDE.get(obj, "joint_0")
    spec = PO.ObjSpec(obj, joint)
    dev, sgn = PO.selfcheck(spec, save=False)
    spec._sgn = sgn
    return spec, dev


def _gripper_worker(args):
    obj, joint, delta, seed, init_s, mass_s, wrist_f_max = args
    return run_gripper_trial(obj, joint, delta, seed, init_s, mass_s, wrist_f_max)


def _firm_worker(args):
    obj, joint, delta, seed, init_s, mass_s = args
    return run_firm_trial(obj, joint, delta, seed, init_s, mass_s)


def gate(procs=6):
    specs = {}
    for obj in OBJS:
        spec, dev = prep_spec(obj)
        specs[obj] = spec
        status = "PASS" if dev < 1e-3 else "FAIL"
        print(f"[{obj}] selfcheck dev={dev*1000:.3f}mm sgn={spec._sgn:+.0f} target={spec.target_deg:.1f}deg  [{status}]")

    jobs = []
    for obj in OBJS:
        joint = JOINT_OVERRIDE.get(obj, "joint_0")
        for t in range(N_GATE):
            seed = stable_seed("gate", obj, t)
            init_s = float(np.random.default_rng(seed + 1).uniform(0, 5))
            mass_s = float(np.random.default_rng(seed + 2).uniform(0.7, 1.5))
            jobs.append((obj, joint, 0.0, seed, init_s, mass_s, None))

    with mp.Pool(procs) as pool:
        rows = pool.map(_gripper_worker, jobs)

    by_obj = {}
    for r in rows:
        by_obj.setdefault(r["obj"], []).append(r["task_done"])
    result = {}
    print("\n" + "=" * 60)
    print("GATE results (delta=0, ideal axis, n=15/object, threshold >=0.70):")
    for obj in OBJS:
        rate = float(np.mean(by_obj[obj]))
        passed = rate >= GATE_THRESH
        result[obj] = dict(gate_rate=rate, passed=passed)
        print(f"  {obj}: task_done rate = {rate:.3f}  [{'PASS' if passed else 'FAIL -- excluded from sweep'}]")

    (OUT / "gripper10_gate.json").write_text(json.dumps(result, indent=2))
    print(f"\n-> {OUT / 'gripper10_gate.json'}")
    n_pass = sum(1 for v in result.values() if v["passed"])
    print(f"\n{n_pass}/{len(OBJS)} objects passed the generic grasp-geometry gate.")


def sweep(procs=6):
    gate_path = OUT / "gripper10_gate.json"
    if not gate_path.exists():
        print("run --gate first"); return
    gate_result = json.loads(gate_path.read_text())
    passed_objs = [o for o in OBJS if gate_result.get(o, {}).get("passed")]
    print(f"Sweeping {len(passed_objs)} gate-passing objects: {passed_objs}")

    jobs_g, jobs_g_generous, jobs_f = [], [], []
    for obj in passed_objs:
        joint = JOINT_OVERRIDE.get(obj, "joint_0")
        for i, delta in enumerate(DELTAS):
            for t in range(N_SWEEP):
                seed = stable_seed(f"sweep_d{i}", obj, t)
                init_s = float(np.random.default_rng(seed + 1).uniform(0, 5))
                mass_s = float(np.random.default_rng(seed + 2).uniform(0.7, 1.5))
                # matched-cap (primary) and generous-cap (secondary, item D fix #4)
                # gripper arms share the SAME seed/init/mass draw -- only f_max differs.
                jobs_g.append((obj, joint, float(delta), seed, init_s, mass_s, None))
                jobs_g_generous.append((obj, joint, float(delta), seed, init_s, mass_s,
                                       WRIST_FMAX_GENEROUS))
                jobs_f.append((obj, joint, float(delta), seed, init_s, mass_s))

    with mp.Pool(procs) as pool:
        rows_g = pool.map(_gripper_worker, jobs_g)
    with mp.Pool(procs) as pool:
        rows_gg = pool.map(_gripper_worker, jobs_g_generous)
    with mp.Pool(procs) as pool:
        rows_f = pool.map(_firm_worker, jobs_f)

    out_csv = OUT / "gripper10_sweep.csv"
    fieldnames = ["obj", "delta_deg", "seed", "condition", "progress", "max_track_err",
                 "quality_frac", "peak_force", "f_max_used"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_g:
            w.writerow(dict(obj=r["obj"], delta_deg=r["delta_deg"], seed=r["seed"],
                            condition="gripper_matched_fmax", progress=r["progress"],
                            max_track_err=r["max_track_err"], quality_frac=r["contact_lost_frac"],
                            peak_force=r["peak_force"], f_max_used=r["wrist_f_max"]))
        for r in rows_gg:
            w.writerow(dict(obj=r["obj"], delta_deg=r["delta_deg"], seed=r["seed"],
                            condition="gripper_generous_fmax", progress=r["progress"],
                            max_track_err=r["max_track_err"], quality_frac=r["contact_lost_frac"],
                            peak_force=r["peak_force"], f_max_used=r["wrist_f_max"]))
        for r in rows_f:
            w.writerow(dict(obj=r["obj"], delta_deg=r["delta_deg"], seed=r["seed"],
                            condition="firm_point", progress=r["progress"],
                            max_track_err=r["max_track_err"], quality_frac=r["sat_frac"],
                            peak_force=r["peak_force"], f_max_used=r["f_max"]))
    print(f"-> {out_csv}  ({len(rows_g) + len(rows_gg) + len(rows_f)} trials)")

    def crossing(xs, ys):
        if ys[0] < 0.5:
            return 0.0
        for i in range(1, len(xs)):
            if ys[i] < 0.5:
                x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
                frac = (y0 - 0.5) / (y0 - y1) if y0 != y1 else 0.5
                return x0 + frac * (x1 - x0)
        return float(xs[-1])  # never crosses -> right-censored at grid max

    def curves_and_crossing(rows_by_cond, obj, track_budget_cm):
        out = {}
        for cond, rows in rows_by_cond.items():
            by_delta = {}
            for r in rows:
                if r["obj"] != obj:
                    continue
                succ = firm_like(r["progress"], r["max_track_err"], r["quality_frac"], track_budget_cm)
                by_delta.setdefault(r["delta_deg"], []).append(int(succ))
            xs = sorted(by_delta)
            ys = [np.mean(by_delta[x]) for x in xs]
            out[cond] = crossing(xs, ys)
        return out

    print("\n" + "=" * 78)
    print(f"per-object 50%-crossing delta_deg, firm_like() threshold sensitivity "
         f"{TRACK_BUDGETS_CM} sim-cm:")
    all_rows = dict(gripper_matched_fmax=[dict(obj=r["obj"], delta_deg=r["delta_deg"],
                                                progress=r["progress"], max_track_err=r["max_track_err"],
                                                quality_frac=r["contact_lost_frac"]) for r in rows_g],
                    firm_point=[dict(obj=r["obj"], delta_deg=r["delta_deg"],
                                     progress=r["progress"], max_track_err=r["max_track_err"],
                                     quality_frac=r["sat_frac"]) for r in rows_f])
    widenings_by_budget = {b: [] for b in TRACK_BUDGETS_CM}
    for obj in passed_objs:
        line = f"  {obj}: "
        for b in TRACK_BUDGETS_CM:
            cross = curves_and_crossing(all_rows, obj, b)
            ratio = cross["gripper_matched_fmax"] / cross["firm_point"] if cross["firm_point"] > 0 else float("nan")
            widenings_by_budget[b].append(ratio)
            line += f"[{b:.0f}cm: gripper={cross['gripper_matched_fmax']:.1f} firm={cross['firm_point']:.1f} ratio={ratio:.2f}x]  "
        print(line)

    print("\nmedian widening ratio by tracking-error budget (matched f_max, firm_like() both sides):")
    for b in TRACK_BUDGETS_CM:
        valid = [w for w in widenings_by_budget[b] if not np.isnan(w)]
        print(f"  {b:.0f}cm budget: median={np.median(valid):.2f}x  (n={len(valid)} objects)")

    print("\n(compare: single-object 35059 pilot reported ~8-10x, generous 600N wrist "
         "cap; see gripper10_sweep.csv condition=gripper_generous_fmax for that arm "
         "under the new matched success definition)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.gate:
        gate(a.procs)
    if a.sweep:
        sweep(a.procs)
    if not a.gate and not a.sweep:
        print("pass --gate and/or --sweep")


if __name__ == "__main__":
    main()
