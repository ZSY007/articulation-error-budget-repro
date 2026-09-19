"""E4 Day-1 gate for a NEW revolute object (generalises pilot_35059.py).

Given --object <id>, auto-resolves from the URDF:
  * the revolute joint (default joint_0), its child (moving) link + parent link
  * the GT world-frame axis + a point on it (URDF joint origin propagated through
    the live parent link frame -- no ground_truth/*.json needed)
  * the grasp point: a visual named handle/knob/bar/grip on the moving link if
    present (mesh AABB + <visual><origin>), else the moving-link corner farthest
    from the hinge axis (a door edge)
Then:
  * COORDINATE SELF-CHECK -- predicted vs simulated handle arc < 1 mm
  * G1 -- GT params + ideal grasp, randomised, success rate >= 90 %

    conda activate cv
    python error_budget/pilot_object.py --object 10905
    python error_budget/pilot_object.py --object 40147 --trials 40

Pure CPU, DIRECT. Reuses pilot_35059's controller by importing run_trial-style
logic locally so this stays standalone.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out"
HANDLE_HINT = ("handle", "knob", "bar", "grip", "pull")

STEP_DEG = 0.5
STEP_M = 0.004         # 4 mm pull step for prismatic (~1 deg-equiv arc length)
SUBSTEPS = 24          # base, per ~0.7 m arc; scaled up for larger doors (below)
F_MAX = 50.0
DOOR_MASS = 2.0
REF_RADIUS = 0.70     # 35059's handle arc radius -- SUBSTEPS/F_MAX calibrated here
BODY_MASS = 20.0
JOINT_DAMPING = 0.5


def rot_about_axis(a, theta):
    a = a / (np.linalg.norm(a) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


# --------------------------------------------------------------------------- URDF
class ObjSpec:
    def __init__(self, obj_id: str, joint_name: str):
        self.id = obj_id
        self.urdf = ROOT / "partnet-mobility-dataset" / obj_id / "mobility.urdf"
        root = ET.parse(self.urdf).getroot()
        j = next(j for j in root.findall("joint") if j.get("name") == joint_name)
        assert j.get("type") in ("revolute", "continuous", "prismatic"), \
            f"{joint_name} not revolute/prismatic"
        self.kind = j.get("type")          # "revolute" | "continuous" | "prismatic"
        self.joint = joint_name
        self.child = j.find("child").get("link")
        self.parent = j.find("parent").get("link")
        o = j.find("origin")
        self.origin = np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
        rpy = o.get("rpy")
        self.rpy = np.array([float(v) for v in rpy.split()]) if rpy else np.zeros(3)
        self.axis_j = np.array([float(v) for v in j.find("axis").get("xyz").split()])
        lim = j.find("limit")
        self.lo = float(lim.get("lower", -np.pi)) if lim is not None else -np.pi
        self.hi = float(lim.get("upper", np.pi)) if lim is not None else np.pi
        self.h_local, self.h_src = self._grasp_point(root)
        self.target_deg = 60.0        # refined in selfcheck once the open direction (sgn) is known
        self.target_m = 0.0           # prismatic equivalent of target_deg, filled by selfcheck
        self.radius = None       # filled by selfcheck; scales f_max / substeps
        self.f_max = F_MAX
        self.substeps = SUBSTEPS

    def _calibrate(self, radius):
        # a bigger door has more leverage on the point constraint and covers more
        # ground per degree -> needs more force to stay on the arc and finer steps
        # to keep angular velocity (hence transient load) comparable to 35059.
        k = max(1.0, radius / REF_RADIUS)
        self.radius = radius
        self.f_max = F_MAX * k
        self.substeps = int(round(SUBSTEPS * k))

    def _rpy_R(self):
        r, pt, y = self.rpy
        cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(pt), np.sin(pt),
                                  np.cos(y), np.sin(y))
        return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                         [-sp, cp * sr, cp * cr]])

    def _mesh_aabb(self, fname: str):
        v = [[float(x) for x in ln.split()[1:4]]
             for ln in (self.urdf.parent / fname).read_text().splitlines()
             if ln.startswith("v ")]
        v = np.asarray(v)
        return v.min(0), v.max(0)

    def _grasp_point(self, root):
        """(point in child-link frame, source string)."""
        link = next(l for l in root.findall("link") if l.get("name") == self.child)
        for vis in link.findall("visual"):
            nm = (vis.get("name") or "").lower()
            if any(h in nm for h in HANDLE_HINT):
                o = vis.find("origin")
                xyz = (np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
                       if o is not None else np.zeros(3))
                mesh = vis.find("geometry/mesh")
                if mesh is not None:
                    lo, hi = self._mesh_aabb(mesh.get("filename"))
                    return (lo + hi) / 2.0 + xyz, f"visual '{vis.get('name')}'"
                return xyz, f"visual '{vis.get('name')}' (origin only)"
        # no handle mesh -> door edge: child-link AABB corner farthest from the hinge
        lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
        for vis in link.findall("visual"):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            o = vis.find("origin")
            off = (np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
                   if o is not None else np.zeros(3))
            a, b = self._mesh_aabb(mesh.get("filename"))
            lo = np.minimum(lo, a + off); hi = np.maximum(hi, b + off)
        # 8 corners; hinge line in child frame passes through origin_child (= 0 at
        # rest, child frame is defined at the joint) along axis_j
        corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        a = self.axis_j / np.linalg.norm(self.axis_j)
        d = corners - (corners @ a)[:, None] * a          # perp component from hinge
        # push slightly inward from the extreme corner so the grasp isn't on a vertex
        far = corners[np.argmax(np.linalg.norm(d, axis=1))]
        return far * 0.92, "door edge (no handle mesh)"


def _lmap(bid):
    return {p.getJointInfo(bid, j)[12].decode(): j for j in range(p.getNumJoints(bid))}


def _jmap(bid):
    return {p.getJointInfo(bid, j)[1].decode(): j for j in range(p.getNumJoints(bid))}


def link_world(bid, idx):
    if idx == -1:
        pos, orn = p.getBasePositionAndOrientation(bid)
    else:
        st = p.getLinkState(bid, idx, computeForwardKinematics=True)
        pos, orn = st[4], st[5]
    return np.asarray(pos), np.asarray(p.getMatrixFromQuaternion(orn)).reshape(3, 3)


def build_scene(spec: ObjSpec, mass_scale=1.0, damping_scale=1.0):
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    bid = p.loadURDF(str(spec.urdf), useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
    jm, lm = _jmap(bid), _lmap(bid)
    p.changeDynamics(bid, lm[spec.child], mass=DOOR_MASS * mass_scale,
                     jointDamping=JOINT_DAMPING * damping_scale, lateralFriction=0.6)
    if spec.parent in lm:
        p.changeDynamics(bid, lm[spec.parent], mass=BODY_MASS)
    p.setJointMotorControl2(bid, jm[spec.joint], p.VELOCITY_CONTROL, force=0.0)
    p.changeDynamics(bid, jm[spec.joint], jointDamping=JOINT_DAMPING * damping_scale)
    # lock every other movable joint at 0 (matches --freeze_extra_dofs renders)
    for name, ji in jm.items():
        if name != spec.joint and p.getJointInfo(bid, ji)[2] != p.JOINT_FIXED:
            p.resetJointState(bid, ji, 0.0)
            p.setJointMotorControl2(bid, ji, p.POSITION_CONTROL, targetPosition=0.0, force=1e4)
    return bid, jm, lm


def gt_axis_pivot_world(bid, jm, spec: ObjSpec):
    parent_idx = p.getJointInfo(bid, jm[spec.joint])[16]
    t_wp, R_wp = link_world(bid, parent_idx)
    R_j = spec._rpy_R()
    a_w = R_wp @ R_j @ spec.axis_j
    p_w = R_wp @ spec.origin + t_wp
    return a_w / np.linalg.norm(a_w), p_w


def handle_world(bid, child_idx, h_local):
    t, R = link_world(bid, child_idx)
    return t + R @ h_local


def _selfcheck_prismatic(spec: ObjSpec, save=True):
    """Prismatic equivalent of selfcheck(): predicted handle path is a straight
    line h0 + sgn*a_w*q (no pivot), ported from pilot_45677.py's approach."""
    cid = p.connect(p.DIRECT)
    try:
        bid, jm, lm = build_scene(spec)
        ci = lm[spec.child]
        a_w, _ = gt_axis_pivot_world(bid, jm, spec)
        lim = max(abs(spec.lo), abs(spec.hi))
        qs = np.linspace(0.0, lim, 30)
        p.resetJointState(bid, jm[spec.joint], 0.0)
        h0 = handle_world(bid, ci, spec.h_local)
        actual = []
        for q in qs:
            p.resetJointState(bid, jm[spec.joint], float(q))
            actual.append(handle_world(bid, ci, spec.h_local))
        actual = np.array(actual)
        best = None
        for sgn in (1.0, -1.0):
            pred = np.array([h0 + sgn * a_w * q for q in qs])
            dev = np.linalg.norm(actual - pred, axis=1)
            if best is None or dev.max() < best[0]:
                best = (dev.max(), dev, sgn)
        devmax, dev, sgn = best
        spec.target_m = 0.7 * lim
        print(f"[{spec.id}] joint {spec.joint} (prismatic)  child {spec.child}  parent {spec.parent}")
        print(f"   grasp point: {spec.h_src}   travel limit {lim*100:.1f} cm")
        print(f"   GT world slide axis {a_w.round(4)}  (sign {sgn:+.0f})")
        print(f"   self-check dev  max {devmax*1000:.3f} mm  mean {dev.mean()*1000:.3f} mm   "
              f"[< 1 mm]  {'PASS' if devmax < 1e-3 else 'FAIL'}")
        if save:
            try:
                import matplotlib; matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.2))
                ax.plot(qs * 100, dev * 1000, "o-", ms=3)
                ax.axhline(1.0, color="r", ls=":")
                ax.set_xlabel("joint position (cm)"); ax.set_ylabel("dev (mm)")
                ax.set_title(f"{spec.id} (prismatic)  max {devmax*1000:.3f} mm")
                fig.tight_layout(); fig.savefig(OUT / f"selfcheck_{spec.id}.png", dpi=110)
                print(f"   -> {OUT / f'selfcheck_{spec.id}.png'}")
            except Exception as e:  # noqa: BLE001
                print(f"   (plot skipped: {e})")
        return float(devmax), sgn
    finally:
        p.disconnect(cid)


# --------------------------------------------------------------------------- checks
def selfcheck(spec: ObjSpec, save=True) -> float:
    if spec.kind == "prismatic":
        return _selfcheck_prismatic(spec, save)
    cid = p.connect(p.DIRECT)
    try:
        bid, jm, lm = build_scene(spec)
        ci = lm[spec.child]
        a_w, p_w = gt_axis_pivot_world(bid, jm, spec)
        sweep = min(np.radians(160.0), max(abs(spec.lo), abs(spec.hi)))
        thetas = np.linspace(0.0, sweep, 40)
        p.resetJointState(bid, jm[spec.joint], 0.0)
        h0 = handle_world(bid, ci, spec.h_local)
        actual = []
        for th in thetas:
            p.resetJointState(bid, jm[spec.joint], float(th))
            actual.append(handle_world(bid, ci, spec.h_local))
        actual = np.array(actual)
        best = None
        for sgn in (1.0, -1.0):
            pred = np.array([p_w + rot_about_axis(a_w, sgn * th) @ (h0 - p_w) for th in thetas])
            dev = np.linalg.norm(actual - pred, axis=1)
            if best is None or dev.max() < best[0]:
                best = (dev.max(), dev, pred, sgn)
        devmax, dev, pred, sgn = best
        r = np.linalg.norm(h0 - p_w); spec._calibrate(r)
        # target: 0.7 of the limit in the OPENING direction, capped 60, so a few
        # deg of +init_state never drives the command past the joint limit.
        open_lim = np.degrees(abs(spec.hi if sgn > 0 else spec.lo))
        spec.target_deg = min(60.0, 0.7 * open_lim)
        print(f"[{spec.id}] joint {spec.joint}  child {spec.child}  parent {spec.parent}")
        print(f"   grasp point: {spec.h_src}   arc radius {r*100:.1f} cm")
        print(f"   GT world axis {a_w.round(4)}  pivot {p_w.round(4)}  (sign {sgn:+.0f})")
        print(f"   sweep {np.degrees(sweep):.0f} deg  |  self-check dev  max {devmax*1000:.3f} mm  "
              f"mean {dev.mean()*1000:.3f} mm   [< 1 mm]  {'PASS' if devmax < 1e-3 else 'FAIL'}")
        if save:
            try:
                import matplotlib; matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                e1 = np.cross(a_w, [0, 0, 1.0])
                e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 1e-6 else np.array([1.0, 0, 0])
                e2 = np.cross(a_w, e1)
                A = np.c_[(actual - p_w) @ e1, (actual - p_w) @ e2]
                Pr = np.c_[(pred - p_w) @ e1, (pred - p_w) @ e2]
                fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
                ax[0].plot(A[:, 0], A[:, 1], "o-", ms=3, label="simulated")
                ax[0].plot(Pr[:, 0], Pr[:, 1], "x--", ms=4, label="predicted (GT axis)")
                ax[0].scatter([0], [0], c="k", marker="+", s=120)
                ax[0].set_aspect("equal"); ax[0].legend(fontsize=8)
                ax[0].set_title(f"{spec.id} handle path")
                ax[1].plot(np.degrees(thetas), dev * 1000, "o-", ms=3)
                ax[1].axhline(1.0, color="r", ls=":")
                ax[1].set_xlabel("joint angle (deg)"); ax[1].set_ylabel("dev (mm)")
                ax[1].set_title(f"max {devmax*1000:.3f} mm")
                fig.tight_layout(); fig.savefig(OUT / f"selfcheck_{spec.id}.png", dpi=110)
                print(f"   -> {OUT / f'selfcheck_{spec.id}.png'}")
            except Exception as e:  # noqa: BLE001
                print(f"   (plot skipped: {e})")
        return float(devmax), sgn
    finally:
        p.disconnect(cid)


def run_trial(spec, est_axis, est_pivot, est_state_deg, rng, sgn=1.0,
              mass_scale=1.0, damping_scale=1.0, grasp_jitter_cm=0.0,
              init_state_deg=0.0, ctrl_noise_mm=1.0, f_max=None, reanchor=False,
              horizon=None, obs_delay_steps=0, obs_noise_mm=0.0):
    """reanchor=False (default) / horizon=None or horizon>=n_steps: open-loop -- each
    step's command is the nominal est_axis/est_pivot rotation applied to the FIXED
    start handle position h_now, at the cumulative commanded angle. Original behavior.

    reanchor=True / horizon=1 (P4-a H_closed, EXPERIMENTS.md SS P4 /
    docs/条件误差预算_P3c-P4-E5_方案_2026-09-13.md SS3): each step instead applies only
    ONE step's rotation to the ACTUAL observed handle position from the end of the
    previous step, rather than to h_now. A perpendicular (normal-direction) axis/pivot
    error then only misplaces the command by a single step's worth of rotation at a
    time instead of compounding over the whole trajectory.

    horizon=H (2026-09-16, item C, EXPERIMENTS.md SS P-Horizon): generalizes both of
    the above to an intermediate reference-update interval. Steps are grouped into
    blocks of H; within a block, the command accumulates rotation from the block's own
    anchor position (exactly like the open-loop scheme, but reset every H steps
    instead of never); at each block boundary the anchor is reset to the ACTUAL
    observed position at that point (exactly like reanchor, but only every H steps
    instead of every step). H=1 reduces exactly to reanchor=True; H>=n_steps reduces
    exactly to reanchor=False -- both are special cases of the same formula, not a
    separate code path, so a monotonic sweep over H is a genuine one-parameter family,
    not three different controllers glued together. `reanchor` is kept only for
    backward-compatible callers; when `horizon` is given it takes precedence.
    All variants use identical GT feedback (gpos from the real constraint state) for a
    fair contrast; horizon/reanchor is the only change to what the CONTROLLER does
    with that feedback.

    obs_delay_steps / obs_noise_mm (2026-09-17, item C feedback-noise stress test,
    EXPERIMENTS.md SS P4-e-noise): the horizon mechanism's re-anchoring read is the
    ONLY place any variant consumes observed state -- the open-loop path (horizon=
    None) never reads gpos at all, so it is mechanically UNAFFECTED by these two
    parameters (a useful invariant: any degradation under delay/noise can only come
    from the finite-horizon variants, not from a shifted baseline). obs_delay_steps
    reads the anchor from `obs_delay_steps` steps further back in the trajectory than
    the immediately-preceding step (delay=0 reduces exactly to the existing gpos_prev
    behavior); obs_noise_mm adds independent Gaussian noise to that read (0 = no extra
    rng draw at all, so the obs_delay_steps=0/obs_noise_mm=0 default is bit-identical
    to this function before these parameters existed).
    """
    if horizon is None:
        horizon = 1 if reanchor else None  # None = open-loop (infinite horizon)
    f_max = f_max if f_max is not None else spec.f_max
    cid = p.connect(p.DIRECT)
    try:
        bid, jm, lm = build_scene(spec, mass_scale, damping_scale)
        jidx, ci = jm[spec.joint], lm[spec.child]
        p.resetJointState(bid, jidx, sgn * np.radians(init_state_deg))
        for _ in range(20):
            p.stepSimulation()
        h_local = spec.h_local + (rng.normal(0, grasp_jitter_cm / 100.0, 3)
                                  if grasp_jitter_cm else 0.0)
        h_now = handle_world(bid, ci, h_local)
        t_d, R_d = link_world(bid, ci)
        handle_in_door = R_d.T @ (h_now - t_d)
        c = p.createConstraint(bid, ci, -1, -1, p.JOINT_POINT2POINT, [0, 0, 0],
                               handle_in_door.tolist(), h_now.tolist())
        p.changeConstraint(c, h_now.tolist(), maxForce=f_max)

        tgt = sgn * np.radians(spec.target_deg)
        plan_total = tgt - sgn * np.radians(est_state_deg)
        n_steps = max(1, int(abs(np.degrees(plan_total)) / STEP_DEG))
        s = np.sign(plan_total) if plan_total != 0 else 1.0
        nominal_dq = np.radians(STEP_DEG)
        track_err, stall, forces = [], 0, []
        q_prev = p.getJointState(bid, jidx)[0]
        gpos_prev = h_now
        gpos_history = [h_now]     # gpos_history[i] = actual position after step i (i=0 -> h_now)
        anchor = h_now             # reset every `horizon` steps; fixed at h_now if horizon is None
        block_start = 1            # step index (1-based) at which the current anchor was set
        for k in range(1, n_steps + 1):
            if horizon is not None and (k - block_start) >= horizon:
                delayed_idx = max(0, len(gpos_history) - 1 - obs_delay_steps)
                observed = gpos_history[delayed_idx]
                if obs_noise_mm:
                    observed = observed + rng.normal(0, obs_noise_mm / 1000.0, 3)
                anchor = observed   # (possibly delayed/noisy) observed position -- new block starts here
                block_start = k
            local_k = k - block_start + 1        # 1..horizon within the current block
            ang = s * local_k * np.radians(STEP_DEG)
            cmd = est_pivot + rot_about_axis(est_axis, float(ang)) @ (anchor - est_pivot)
            cmd = cmd + rng.normal(0, ctrl_noise_mm / 1000.0, 3)
            p.changeConstraint(c, cmd.tolist(), maxForce=f_max)
            for _ in range(spec.substeps):
                p.stepSimulation()
            tt, RR = link_world(bid, ci)
            gpos = tt + RR @ handle_in_door
            track_err.append(np.linalg.norm(gpos - cmd))
            f = np.linalg.norm(p.getConstraintState(c)[:3])
            forces.append(f)
            q_now = p.getJointState(bid, jidx)[0]
            if f >= f_max and abs(q_now - q_prev) < 0.3 * nominal_dq:
                stall += 1
            q_prev = q_now
            gpos_prev = gpos
            gpos_history.append(gpos)
        gt_final = p.getJointState(bid, jidx)[0]
        prog = (gt_final - sgn * np.radians(init_state_deg)) / (tgt - sgn * np.radians(init_state_deg))
        te = np.asarray(track_err)
        sat = stall / max(1, n_steps)
        return dict(success=bool(prog >= 0.80 and te.max() < 0.02 and sat < 0.20),
                    progress=float(np.clip(prog, 0, 1.5)), max_track_err=float(te.max()),
                    mean_track_err=float(te.mean()), sat_frac=float(sat),
                    peak_force=float(np.max(forces)), n_steps=int(n_steps))
    finally:
        p.disconnect(cid)


def run_trial_prismatic(spec, est_axis, est_state_m, rng, sgn=1.0,
                        mass_scale=1.0, damping_scale=1.0, grasp_jitter_cm=0.0,
                        init_state_m=0.0, ctrl_noise_mm=1.0, f_max=None,
                        horizon=None, obs_delay_steps=0, obs_noise_mm=0.0):
    """Prismatic sibling of run_trial(): straight pull along est_axis, no pivot.
    Ported from pilot_45677.py (which hardcoded a single drawer) so any
    prismatic ObjSpec can use it. est_state_m / init_state_m are metres.

    horizon/obs_delay_steps/obs_noise_mm (2026-09-17, item C prismatic follow-up,
    EXPERIMENTS.md SS P4-e-prismatic): same generalization as run_trial() -- command
    accumulates est_axis*STEP_M from an anchor point that resets to the (possibly
    delayed/noisy) observed position every `horizon` steps instead of never
    (horizon=None) or every step (horizon=1). No pivot to reconstruct here, so the
    formula is simpler: `anchor + est_axis*(local_k*STEP_M)`. Semantics and defaults
    are identical to run_trial()'s -- see that function's docstring for the exact
    mechanism and the bit-identical backward-compatibility guarantee.
    """
    f_max = f_max if f_max is not None else spec.f_max
    cid = p.connect(p.DIRECT)
    try:
        bid, jm, lm = build_scene(spec, mass_scale, damping_scale)
        jidx, ci = jm[spec.joint], lm[spec.child]
        p.resetJointState(bid, jidx, sgn * init_state_m)
        for _ in range(20):
            p.stepSimulation()
        h_local = spec.h_local + (rng.normal(0, grasp_jitter_cm / 100.0, 3)
                                  if grasp_jitter_cm else 0.0)
        h_now = handle_world(bid, ci, h_local)
        t_d, R_d = link_world(bid, ci)
        handle_in_door = R_d.T @ (h_now - t_d)
        c = p.createConstraint(bid, ci, -1, -1, p.JOINT_POINT2POINT, [0, 0, 0],
                               handle_in_door.tolist(), h_now.tolist())
        p.changeConstraint(c, h_now.tolist(), maxForce=f_max)

        tgt = sgn * spec.target_m
        plan_total = tgt - sgn * est_state_m
        n_steps = max(1, int(abs(plan_total) / STEP_M))
        s = np.sign(plan_total) if plan_total != 0 else 1.0
        track_err, stall, forces = [], 0, []
        q_prev = p.getJointState(bid, jidx)[0]
        gpos_prev = h_now
        gpos_history = [h_now]
        anchor = h_now
        block_start = 1
        for k in range(1, n_steps + 1):
            if horizon is not None and (k - block_start) >= horizon:
                delayed_idx = max(0, len(gpos_history) - 1 - obs_delay_steps)
                observed = gpos_history[delayed_idx]
                if obs_noise_mm:
                    observed = observed + rng.normal(0, obs_noise_mm / 1000.0, 3)
                anchor = observed
                block_start = k
            local_k = k - block_start + 1
            cmd = anchor + est_axis * (s * local_k * STEP_M)
            cmd = cmd + rng.normal(0, ctrl_noise_mm / 1000.0, 3)
            p.changeConstraint(c, cmd.tolist(), maxForce=f_max)
            for _ in range(spec.substeps):
                p.stepSimulation()
            tt, RR = link_world(bid, ci)
            gpos = tt + RR @ handle_in_door
            track_err.append(np.linalg.norm(gpos - cmd))
            f = np.linalg.norm(p.getConstraintState(c)[:3])
            forces.append(f)
            q_now = p.getJointState(bid, jidx)[0]
            if f >= f_max and abs(q_now - q_prev) < 0.3 * STEP_M:
                stall += 1
            q_prev = q_now
            gpos_prev = gpos
            gpos_history.append(gpos)
        gt_final = p.getJointState(bid, jidx)[0]
        prog = (gt_final - sgn * init_state_m) / (tgt - sgn * init_state_m)
        te = np.asarray(track_err)
        sat = stall / max(1, n_steps)
        return dict(success=bool(prog >= 0.80 and te.max() < 0.02 and sat < 0.20),
                    progress=float(np.clip(prog, 0, 1.5)), max_track_err=float(te.max()),
                    mean_track_err=float(te.mean()), sat_frac=float(sat),
                    peak_force=float(np.max(forces)), n_steps=int(n_steps))
    finally:
        p.disconnect(cid)


def g1(spec, n_trials, sgn, seed=0):
    cid = p.connect(p.DIRECT)
    try:
        bid, jm, _ = build_scene(spec)
        a_w, p_w = gt_axis_pivot_world(bid, jm, spec)
    finally:
        p.disconnect(cid)
    rng = np.random.default_rng(seed)
    rows = [run_trial(spec, a_w, p_w, 0.0, rng, sgn=sgn,
                      mass_scale=float(rng.uniform(0.5, 2.0)),
                      damping_scale=float(rng.uniform(0.5, 2.0)),
                      grasp_jitter_cm=1.0, init_state_deg=float(rng.uniform(0.0, 5.0)),
                      ctrl_noise_mm=1.0)
            for _ in range(n_trials)]
    s = np.array([r["success"] for r in rows])
    sr, n, z = s.mean(), len(s), 1.96
    cc = (sr + z * z / (2 * n)) / (1 + z * z / n)
    hw = z * np.sqrt(sr * (1 - sr) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    prog = np.array([r["progress"] for r in rows])
    te = np.array([r["max_track_err"] for r in rows]) * 100
    print(f"[{spec.id}] G1 (n={n}, target {spec.target_deg:.0f} deg): success {sr:.3f}  "
          f"Wilson95 [{cc-hw:.3f},{cc+hw:.3f}]  [gate >= 0.90]  "
          f"{'PASS' if (cc-hw >= 0.90 or sr >= 0.90) else 'FAIL'}")
    print(f"   progress med {np.median(prog):.2f} min {prog.min():.2f}  |  "
          f"track err med {np.median(te):.2f} p90 {np.percentile(te,90):.2f} cm")
    fails = [i for i, x in enumerate(s) if not x]
    if fails:
        print("   fails: " + ", ".join(
            f"#{i}(p{rows[i]['progress']:.2f},te{rows[i]['max_track_err']*100:.1f},sat{rows[i]['sat_frac']:.2f})"
            for i in fails[:8]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--joint", default="joint_0")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    spec = ObjSpec(a.object, a.joint)
    dev, sgn = selfcheck(spec)
    if dev >= 1e-3:
        print("self-check FAILED -- not running G1.")
        return
    g1(spec, a.trials, sgn, a.seed)


if __name__ == "__main__":
    main()
