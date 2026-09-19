"""Offline Heppert Tangent Similarity J for every trial already produced by
prime/scripts/jarvis_p1_direction_replay.sh (out/p0_<obj>_flow_gaussian_L1_
<selector>_se3.csv), WITHOUT re-running any PyBullet trial.

Why this is possible offline: every row already carries (axis_err, pivot_err,
state_err_signed, trial_seed, direction_seed, b2_seed, b2_phi) -- the exact
seeds p0_direction_preserving_replay.py used to draw est_axis_B/B2 -- plus the
source estimate's raw (est_axis, est_pivot) vectors live in the sibling
perseed_errors_<obj>_p0_<selector>_se3.csv (indexed by src_row). Reconstructing
est_axis/est_pivot per mode is therefore a deterministic replay of the same
RNG draws p0_direction_preserving_replay.py made -- no physics needed, just the
geometry functions in p0_basis.py plus one lightweight (no-trial) PyBullet
scene build per object to recover R_frame / gt_axis_w / gt_pivot_w / joint
limits (mirrors p0_direction_preserving_replay.py::main() steps [2/4]).

Computes, per (obj, estimator, selector, mode) row:
  J_full  -- Heppert Eq.16 over the object's FULL URDF joint range [0, open_lim]
             (revolute) or the single cosine cos(e_axis) (prismatic) -- the
             literature-comparable number.
  J_exec  -- same integrand but over the trial's actual commanded range (from
             the reconstructed est_state to target) -- diagnostic only.

    conda activate cv
    python error_budget/p1_tangent_similarity_offline.py
    -> out/p1_tangent_<obj>_<selector>.csv (per object) + out/p1_tangent_all.csv
"""
from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_perseed                                                    # noqa: E402
import pilot_object as PO                                             # noqa: E402
import p0_basis as PB                                                 # noqa: E402
from p0_direction_preserving_replay import build_random_direction_estimate  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
JOINT_OVERRIDE = {"10068": "joint_1", "45841": "joint_1", "10489": "joint_1",
                  "10620": "joint_1", "10685": "joint_1", "11231": "joint_1",
                  "7304": "joint_1", "45087": "joint_1"}


def _scene_geometry(obj: str):
    """One lightweight (no-trial) PyBullet build -- mirrors
    p0_direction_preserving_replay.py::main() steps [2/4], minus the replay."""
    joint = JOINT_OVERRIDE.get(obj, "joint_0")
    spec = PO.ObjSpec(obj, joint)
    devmax, sgn = PO.selfcheck(spec, save=False)
    cid = p.connect(p.DIRECT)
    try:
        bid, jm, lm = PO.build_scene(spec)
        p.resetJointState(bid, jm[spec.joint], 0.0)
        handle_rest_w = PO.handle_world(bid, lm[spec.child], spec.h_local)
        parent_idx = p.getJointInfo(bid, jm[spec.joint])[16]
        t_parent_w, R_parent_w = PO.link_world(bid, parent_idx)
        gt_axis_w, gt_pivot_w = PO.gt_axis_pivot_world(bid, jm, spec)
    finally:
        p.disconnect(cid)

    PRIME = Path(__file__).resolve().parent.parent / "prime"
    sys.path.insert(0, str(PRIME))
    from prime.data.dataset import load_clip                          # noqa: E402
    rd = PRIME / "outputs" / "renders" / f"{obj}_circular_1dof"
    if not (rd / "geometry.npz").exists():
        rd = PRIME / "outputs" / "renders" / f"{obj}_circular"
    clip = load_clip(str(rd), n_points=4096)
    parent_r = clip.link_poses[spec.parent][0].numpy().astype(float)
    R_frame, translation_frame = PB.frame_transform(
        parent_r[:3, :3], parent_r[:3, 3], R_parent_w, t_parent_w)

    open_lim = abs(spec.hi if sgn > 0 else spec.lo)   # rad (revolute) or m (prismatic)
    return dict(spec=spec, sgn=sgn, gt_axis_w=gt_axis_w, gt_pivot_w=gt_pivot_w,
               handle_rest_w=handle_rest_w, R_frame=R_frame,
               translation_frame=translation_frame, open_lim=open_lim,
               prismatic=(spec.kind == "prismatic"))


def _reconstruct_axis_pivot(geo, src_row, mode, trial_seed, b2_phi):
    est_axis_r = PB.parse_vec(src_row["est_axis"])
    est_pivot_r = PB.parse_vec(src_row["est_pivot"])
    gt_axis_r = PB.parse_vec(src_row["gt_axis"])
    gt_pivot_r = PB.parse_vec(src_row["gt_pivot"])
    axis_err_deg = float(src_row["axis_err"])
    pivot_err_cm = (float(src_row["pivot_err"])
                    if src_row["pivot_err"] not in ("", "nan") else 0.0)
    gt_axis_w, gt_pivot_w = geo["gt_axis_w"], geo["gt_pivot_w"]

    if geo["prismatic"]:
        est_axis_A = PB.apply_axis_deviation(geo["R_frame"], gt_axis_r, est_axis_r, gt_axis_w)
        if mode == "A":
            return est_axis_A, None
        if mode == "C":
            return gt_axis_w, None
        if mode == "B":
            dir_rng = np.random.default_rng(trial_seed + 3_000_000)
            ea, _ = build_random_direction_estimate(gt_axis_w, gt_pivot_w, axis_err_deg, 0.0, dir_rng)
            return ea, None
        if mode == "B2":
            ea, _ = PB.rotate_axis_line_about_true_axis(gt_axis_w, gt_pivot_w, est_axis_A, gt_pivot_w, b2_phi)
            return ea, None
        raise ValueError(mode)

    est_axis_A, est_pivot_A = PB.apply_relative_deviation(
        geo["R_frame"], gt_axis_r, gt_pivot_r, est_axis_r, est_pivot_r, gt_axis_w, gt_pivot_w,
        translation_frame=geo["translation_frame"])
    if mode == "A":
        return est_axis_A, est_pivot_A
    if mode == "C":
        return gt_axis_w, gt_pivot_w
    if mode == "B":
        dir_rng = np.random.default_rng(trial_seed + 3_000_000)
        return build_random_direction_estimate(gt_axis_w, gt_pivot_w, axis_err_deg, pivot_err_cm, dir_rng)
    if mode == "B2":
        return PB.rotate_axis_line_about_true_axis(gt_axis_w, gt_pivot_w, est_axis_A, est_pivot_A, b2_phi)
    raise ValueError(mode)


def process_object(obj: str, selector: str) -> list[dict]:
    trial_p = OUT / f"p0_{obj}_flow_gaussian_L1_{selector}_se3.csv"
    perseed_p = OUT / f"perseed_errors_{obj}_p0_{selector}_se3.csv"
    if not trial_p.exists() or not perseed_p.exists():
        return []
    src_rows = list(csv.DictReader(perseed_p.open()))
    trial_rows = list(csv.DictReader(trial_p.open()))
    geo = _scene_geometry(obj)
    spec, sgn = geo["spec"], geo["sgn"]

    out = []
    for row in trial_rows:
        ri = int(row["src_row"])
        src_row = src_rows[ri]
        if src_row.get("failed") == "1" or not src_row.get("est_axis"):
            continue
        trial_seed = int(row["trial_seed"])
        b2_phi = float(row["b2_phi"])
        ea, ep = _reconstruct_axis_pivot(geo, src_row, row["mode"], trial_seed, b2_phi)

        if geo["prismatic"]:
            j_full = PB.tangent_similarity_prismatic(geo["gt_axis_w"], ea)
            j_exec = j_full  # no q-dependence for prismatic; kept for schema symmetry
            j_eq16 = j_full  # prismatic has no path-point ambiguity (no pivot term); identical
        else:
            j_full = PB.tangent_similarity_revolute(
                geo["gt_axis_w"], geo["gt_pivot_w"], ea, ep, geo["handle_rest_w"],
                0.0, geo["open_lim"])
            # 2026-09-18: `j_full` above is J_cfg (evaluates the estimated twist at an
            # ESTIMATED path point) -- kept unchanged for continuity with every existing
            # table. `j_eq16` is the actual Heppert et al. Eq.16 formula (both twists at
            # the TRUE path point), added alongside, not a replacement.
            j_eq16 = PB.tangent_similarity_revolute_eq16(
                geo["gt_axis_w"], geo["gt_pivot_w"], ea, ep, geo["handle_rest_w"],
                0.0, geo["open_lim"])
            # exec-range: same reconstruction e_norm_nominal_cm's callers use for
            # rel_angles' upper bound (est_state = init_state + state_err_signed,
            # native units already matched to prismatic/revolute upstream).
            init_state = float(row["init_state_deg"])
            state_err_signed = float(row["state_err_signed"]) if row["mode"] != "C" else 0.0
            est_state = init_state + state_err_signed
            q0 = sgn * np.radians(min(init_state, est_state))
            q1 = sgn * np.radians(spec.target_deg)
            qlo, qhi = (q0, q1) if q1 > q0 else (q1, q0)
            if qhi - qlo < 1e-9:
                qhi = qlo + 1e-6
            j_exec = PB.tangent_similarity_revolute(
                geo["gt_axis_w"], geo["gt_pivot_w"], ea, ep, geo["handle_rest_w"], qlo, qhi)

        out.append(dict(obj=obj, kind=spec.kind, estimator=row["estimator"], selector=selector,
                        src_row=ri, rep=row["rep"], mode=row["mode"],
                        axis_err=row["axis_err"], pivot_err=row["pivot_err"],
                        state_err_signed=row["state_err_signed"],
                        j_full=round(j_full, 6), j_exec=round(j_exec, 6), j_eq16=round(j_eq16, 6),
                        e_geo_nominal_cm=row["e_geo_nominal_cm"],
                        e_norm_nominal_cm=row["e_norm_nominal_cm"],
                        progress=row["progress"], max_track_err=row["max_track_err"],
                        sat_frac=row["sat_frac"], firm=row["firm"], rpmart85=row["rpmart85"],
                        task_done=row["task_done"]))
    return out


def main():
    trial_files = sorted(glob.glob(str(OUT / "p0_*_flow_gaussian_L1_*_se3.csv")))
    seen = set()
    for fp in trial_files:
        name = Path(fp).stem
        parts = name.split("_")
        obj = parts[1]
        selector = "crossfit" if "crossfit" in parts else "heuristic"
        if (obj, selector) in seen:
            continue
        seen.add((obj, selector))

    all_rows = []
    n_obj_ok, n_obj_fail = 0, 0
    for obj, selector in sorted(seen, key=lambda x: (len(x[0]), x)):
        try:
            rows = process_object(obj, selector)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {obj}/{selector}: {type(e).__name__}: {e}")
            n_obj_fail += 1
            continue
        if not rows:
            continue
        n_obj_ok += 1
        obj_p = OUT / f"p1_tangent_{obj}_{selector}.csv"
        with obj_p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        all_rows.extend(rows)
        c_j = [r["j_full"] for r in rows if r["mode"] == "C"]
        print(f"  {obj:8s} {selector:10s}  {len(rows):5d} rows  "
             f"mode-C j_full sanity: min={min(c_j):.4f} max={max(c_j):.4f} (want ~1.0)")

    if all_rows:
        all_p = OUT / "p1_tangent_all.csv"
        with all_p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
        print(f"\n-> {all_p}  ({len(all_rows)} rows, {n_obj_ok} objects ok, {n_obj_fail} skipped)")


if __name__ == "__main__":
    main()
