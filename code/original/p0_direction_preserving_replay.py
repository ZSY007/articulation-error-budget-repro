"""P0 (EXPERIMENTS.md SS P-X): direction-preserving E5 replay.

Replays each per-seed estimate four ways and reports firm/task-done/e_exec for all
four, per object x estimator x preset:

  A  real estimate    -- the estimator's ACTUAL (axis, pivot) transported into the
                          PyBullet frame via p0_basis (direction preserved).
  B  random direction -- same axis_err/pivot_err MAGNITUDES, direction re-drawn
                          uniformly at random (this is what e4e5_object.py's
                          e5_replay() currently does for every reported E5 number).
  B2 coherent random azimuth -- rigidly rotates A's estimated axis line about the
                          true axis, preserving its internal line geometry.
  C  GT                -- true (axis, pivot, state); the execution floor for this
                          object/controller/physical-perturbation draw.

Physical (task, not perceptual) randomization -- mass_scale, damping_scale,
init_state_deg, grasp-jitter/ctrl-noise stream -- is drawn ONCE per trial index and
reused across A/B/B2/C. A preserves the complete estimated axis line using parent
SE(3); B retains the historical scalar sampler at the URDF pivot, with zero offset
along the GT axis. Their pivot-anchor conventions are not standardized, so A/B is
a full-estimate vs legacy-scalar comparison, not yet a direction-only intervention.
state_err_signed is shared by A/B; C uses zero state error.

    conda activate cv
    python error_budget/p0_direction_preserving_replay.py --obj 35059 --preset flow_gaussian --level 1 --reps 8
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_perseed                                                     # noqa: E402
import pilot_object as PO                                              # noqa: E402
import p0_basis as PB                                                  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"

CRITERIA = {
    "firm":      lambda x: bool(x["progress"] >= 0.80 and x["max_track_err"] < 0.02 and x["sat_frac"] < 0.20),
    "rpmart85":  lambda x: bool(x["progress"] >= 0.85),
}


def _perp_dir(axis, rng):
    """Same as e4e5_object.py::_perp_dir -- used ONLY for mode B (random direction),
    so B reproduces exactly what the existing E5 pipeline has been reporting."""
    a = axis / np.linalg.norm(axis)
    while True:
        v = rng.normal(size=3); v = v - (v @ a) * a
        n = np.linalg.norm(v)
        if n > 1e-6:
            return v / n


def rot_about_axis(a, theta):
    return PO.rot_about_axis(a, theta)


def build_random_direction_estimate(gt_axis_w, gt_pivot_w, axis_err_deg, pivot_err_cm, rng):
    """Mode B: same magnitudes as the real estimate, independently random directions
    -- the current e4e5_object.py behavior."""
    est_axis_w = gt_axis_w.copy()
    if axis_err_deg:
        d = _perp_dir(gt_axis_w, rng)
        est_axis_w = rot_about_axis(d, np.radians(axis_err_deg)) @ gt_axis_w
        est_axis_w = est_axis_w / np.linalg.norm(est_axis_w)
    est_pivot_w = gt_pivot_w.copy()
    if pivot_err_cm:
        d = _perp_dir(gt_axis_w, rng)
        est_pivot_w = gt_pivot_w + d * (pivot_err_cm / 100.0)
    return est_axis_w, est_pivot_w


def one_trial_all_modes(spec, gt_axis_w, gt_pivot_w, handle_rest_w, R_frame,
                        est_axis_r, est_pivot_r, gt_axis_r, gt_pivot_r,
                        axis_err_deg, pivot_err_cm, state_err_signed,
                        trial_seed, f_max, sgn=1.0, *, translation_frame):
    """Run modes A/B/B2/C for one (estimate, physical-perturbation draw) pair,
    sharing the physical randomization across all four. Returns a dict of
    per-mode telemetry plus the physical draws actually used (for the CSV)."""
    prismatic = (spec.kind == "prismatic")
    phys_rng = np.random.default_rng(trial_seed)          # physical/task randomization ONLY
    mass_scale = float(phys_rng.uniform(0.5, 2.0))
    damping_scale = float(phys_rng.uniform(0.5, 2.0))
    # init_state_deg carries native units by joint kind (degrees revolute, metres
    # prismatic) -- same mixed-unit convention gen_perseed.py already uses for
    # state_err_signed (see its header note); e4e5_object.py::_one() draws the
    # prismatic range as 0-2cm (0.02 m), not 0-5 (which would be a degrees-sized
    # jitter nonsensically reused as metres).
    init_state_deg = float(phys_rng.uniform(0.0, 0.02 if prismatic else 5.0))
    exec_seed = trial_seed + 7_000_000                     # separate stream: grasp jitter + ctrl noise

    if prismatic:
        # mode A: real estimate, direction-preserving (headless direction, no pivot)
        est_axis_A = PB.apply_axis_deviation(R_frame, gt_axis_r, est_axis_r, gt_axis_w)
        # mode B: legacy scalar reconstruction -- axis-tilt direction only (pivot_err_cm
        # forced to 0 so build_random_direction_estimate draws exactly the one _perp_dir
        # sample the axis needs and returns gt_pivot_w unchanged, which is discarded).
        dir_rng = np.random.default_rng(trial_seed + 3_000_000)
        est_axis_B, _ = build_random_direction_estimate(
            gt_axis_w, gt_pivot_w, axis_err_deg, 0.0, dir_rng)
        # mode B2: rotate A's estimated direction about the TRUE axis by a random
        # angle -- for a headless direction this is the direct azimuthal analogue of
        # the revolute B2 (the "point" argument/output is unused for prismatic, so a
        # fixed dummy point is passed and its rotated output discarded).
        phi_rng = np.random.default_rng(trial_seed + 5_000_000)
        phi = float(phi_rng.uniform(0.0, 2 * np.pi))
        est_axis_B2, _ = PB.rotate_axis_line_about_true_axis(
            gt_axis_w, gt_pivot_w, est_axis_A, gt_pivot_w, phi)
        est_axis_C = gt_axis_w
        est_pivot_A = est_pivot_B = est_pivot_B2 = est_pivot_C = None
    else:
        # mode A: real estimate, direction-preserving
        est_axis_A, est_pivot_A = PB.apply_relative_deviation(
            R_frame, gt_axis_r, gt_pivot_r, est_axis_r, est_pivot_r, gt_axis_w, gt_pivot_w,
            translation_frame=translation_frame)

        # mode B: legacy scalar reconstruction -- axis-tilt direction and pivot-offset
        # direction drawn INDEPENDENTLY at the URDF pivot (uses ITS OWN rng draw, seeded
        # off trial_seed but a distinct stream). Kept for continuity with existing E5
        # numbers; NOT a clean direction-only intervention against A (see module
        # docstring and docs/P0_10905_手持重放核查_2026-09-12.md SS5) -- A and B differ in
        # more than direction (e.g. the pivot's position along the axis is whatever A's
        # real estimate says, not independently controlled the way B's is).
        dir_rng = np.random.default_rng(trial_seed + 3_000_000)
        est_axis_B, est_pivot_B = build_random_direction_estimate(
            gt_axis_w, gt_pivot_w, axis_err_deg, pivot_err_cm, dir_rng)

        # mode B2 (2026-09-12): rigidly rotate A's ACTUAL estimated line about the TRUE
        # axis by a random angle -- preserves axis_err/pivot_err exactly AND preserves
        # the joint tilt/pivot geometry of one coherent line (unlike B's two independent
        # draws), so B2 vs A is the strict "same estimate, random azimuth" contrast the
        # audit report calls for. Own rng stream, independent of B's and of physics/exec.
        phi_rng = np.random.default_rng(trial_seed + 5_000_000)
        phi = float(phi_rng.uniform(0.0, 2 * np.pi))
        est_axis_B2, est_pivot_B2 = PB.rotate_axis_line_about_true_axis(
            gt_axis_w, gt_pivot_w, est_axis_A, est_pivot_A, phi)

        # mode C: GT
        est_axis_C, est_pivot_C = gt_axis_w, gt_pivot_w

        # Pre-execution nominal geometry metric.  It deliberately excludes the
        # physical/controller randomization so it can be compared with e_exec
        # (max_track_err) instead of silently containing the same nuisance draws.
        init_theta = sgn * np.radians(init_state_deg)
        handle_start = (gt_pivot_w + rot_about_axis(gt_axis_w, init_theta)
                        @ (np.asarray(handle_rest_w) - gt_pivot_w))

    out = {}
    for tag, ea, ep, use_state_err in (("A", est_axis_A, est_pivot_A, True),
                                       ("B", est_axis_B, est_pivot_B, True),
                                       ("B2", est_axis_B2, est_pivot_B2, True),
                                       ("C", est_axis_C, est_pivot_C, False)):
        rng = np.random.default_rng(exec_seed)             # fresh, identical stream each mode
        # state_err_signed is in gen_perseed.py's native per-kind unit: degrees for
        # revolute, CENTIMETRES for prismatic (its header note) -- init_state_deg is
        # already metres for prismatic (see the draw above), so the prismatic branch
        # must divide by 100 before adding, exactly mirroring e4e5_object.py::_one()'s
        # `iw_m + spec_args["eps_deg"] / 100.0`. Omitting this scaled a ~cm error as if
        # it were ~metres (caught by a local pilot: max_track_err blew up to 1000s of cm).
        est_state_native = init_state_deg + ((state_err_signed / 100.0 if prismatic
                                              else state_err_signed) if use_state_err else 0.0)
        if prismatic:
            plan_total = sgn * spec.target_m - sgn * est_state_native
            n_steps = max(1, int(abs(plan_total) / PO.STEP_M))
            step_sign = np.sign(plan_total) if plan_total != 0 else 1.0
            rel_disp = step_sign * np.arange(1, n_steps + 1) * PO.STEP_M
            e_geo_cm = PB.geometric_path_deviation_prismatic_cm(gt_axis_w, ea, rel_disp)
            e_norm_cm = PB.normal_path_deviation_prismatic_cm(gt_axis_w, ea, rel_disp)
            r = PO.run_trial_prismatic(spec, ea, est_state_native, rng, sgn=sgn,
                                       mass_scale=mass_scale, damping_scale=damping_scale,
                                       grasp_jitter_cm=1.0, init_state_m=init_state_deg,
                                       ctrl_noise_mm=1.0, f_max=f_max)
        else:
            plan_total = sgn * np.radians(spec.target_deg) - sgn * np.radians(est_state_native)
            n_steps = max(1, int(abs(np.degrees(plan_total)) / PO.STEP_DEG))
            step_sign = np.sign(plan_total) if plan_total != 0 else 1.0
            rel_angles = step_sign * np.arange(1, n_steps + 1) * np.radians(PO.STEP_DEG)
            e_geo_cm = PB.geometric_path_deviation_cm(
                gt_axis_w, gt_pivot_w, ea, ep, handle_start, rel_angles)
            e_norm_cm = PB.normal_path_deviation_cm(
                gt_axis_w, gt_pivot_w, ea, ep, handle_start, rel_angles)
            r = PO.run_trial(spec, ea, ep, est_state_native, rng, sgn=sgn,
                             mass_scale=mass_scale, damping_scale=damping_scale,
                             grasp_jitter_cm=1.0, init_state_deg=init_state_deg,
                             ctrl_noise_mm=1.0, f_max=f_max)
        r["task_done"] = int(r["progress"] >= 0.80)
        for name, fn in CRITERIA.items():
            r[name] = int(fn(r))
        r["e_geo_nominal_cm"] = e_geo_cm
        r["e_norm_nominal_cm"] = e_norm_cm
        out[tag] = r
    out["_phys"] = dict(mass_scale=mass_scale, damping_scale=damping_scale,
                        init_state_deg=init_state_deg, mode_a_frame_ok=True,
                        trial_seed=trial_seed, exec_seed=exec_seed,
                        direction_seed=trial_seed + 3_000_000,
                        b2_seed=trial_seed + 5_000_000, b2_phi=phi)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="35059")
    ap.add_argument("--joint", default="joint_0")
    ap.add_argument("--preset", default="flow_gaussian")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--selector", choices=["heuristic", "crossfit"], default="heuristic")
    ap.add_argument("--gen-seeds", type=int, default=8, help="seeds for gen_perseed.py")
    ap.add_argument("--reps", type=int, default=8, help="physical-perturbation reps per estimate")
    ap.add_argument("--out-suffix", default="", help="insert before .csv; useful for non-destructive reruns")
    args = ap.parse_args(argv)

    tag = f"_p0_{args.selector}_se3"
    print(f"[1/4] gen_perseed.py --obj {args.obj} --seeds {args.gen_seeds} "
          f"--presets {args.preset}:{args.level} --selector {args.selector} --tag {tag}")
    gen_perseed.main(["--obj", args.obj, "--seeds", str(args.gen_seeds),
                      "--presets", f"{args.preset}:{args.level}",
                      "--selector", args.selector, "--tag", tag])
    src_rows = list(csv.DictReader((OUT / f"perseed_errors_{args.obj}{tag}.csv").open()))

    print(f"[2/4] building PyBullet scene + R_frame for {args.obj}/{args.joint}")
    spec = PO.ObjSpec(args.obj, args.joint)
    # selfcheck() opens/closes its own DIRECT connection (sets spec.radius/f_max/
    # substeps/target_deg as a side effect) -- must NOT be nested inside another
    # open connection (e4e5_object.py's main() does these two steps sequentially,
    # not nested, for the same reason: pybullet's default-client pointer is global).
    devmax, sgn = PO.selfcheck(spec, save=False)
    # spec.radius stays None for prismatic (only the revolute selfcheck() branch
    # calls spec._calibrate(); a radius/arm-length has no meaning for a slider) --
    # print spec.target_m instead of spec.radius*100 for that branch.
    extra = (f"target={spec.target_m*100:.1f}cm" if spec.kind == "prismatic"
             else f"radius={spec.radius*100:.1f}cm")
    print(f"   selfcheck devmax={devmax*1000:.3f}mm sgn={sgn:+.0f} "
          f"{extra} f_max={spec.f_max:.1f}N")
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
    from prime.data.dataset import load_clip                            # noqa: E402
    rd = PRIME / "outputs" / "renders" / f"{args.obj}_circular_1dof"
    if not (rd / "geometry.npz").exists():
        rd = PRIME / "outputs" / "renders" / f"{args.obj}_circular"
    clip = load_clip(str(rd), n_points=4096)
    parent_r = clip.link_poses[spec.parent][0].numpy().astype(float)
    R_frame, translation_frame = PB.frame_transform(
        parent_r[:3, :3], parent_r[:3, 3], R_parent_w, t_parent_w)

    print(f"[3/4] replaying {len(src_rows)} estimates x {args.reps} reps x 4 modes")
    out_rows = []
    for ri, row in enumerate(src_rows):
        if row["failed"] == "1" or not row.get("est_axis"):
            continue
        est_axis_r = PB.parse_vec(row["est_axis"])
        est_pivot_r = PB.parse_vec(row["est_pivot"])
        gt_axis_r = PB.parse_vec(row["gt_axis"])
        gt_pivot_r = PB.parse_vec(row["gt_pivot"])
        axis_err_deg = float(row["axis_err"])
        pivot_err_cm = float(row["pivot_err"]) if row["pivot_err"] not in ("", "nan") else 0.0
        state_err_signed = float(row.get("state_err_signed", 0.0) or 0.0)
        for t in range(args.reps):
            trial_seed = 500_000 + 1_009 * ri + t
            res = one_trial_all_modes(spec, gt_axis_w, gt_pivot_w, handle_rest_w, R_frame,
                                      est_axis_r, est_pivot_r, gt_axis_r, gt_pivot_r,
                                      axis_err_deg, pivot_err_cm, state_err_signed,
                                      trial_seed, spec.f_max, sgn=sgn, translation_frame=translation_frame)
            for tag_m in ("A", "B", "B2", "C"):
                r = res[tag_m]
                out_rows.append(dict(
                    obj=args.obj, estimator=row["estimator"], preset=args.preset,
                    level=args.level, src_row=ri, rep=t, mode=tag_m,
                    axis_err=axis_err_deg, pivot_err=pivot_err_cm,
                    state_err_signed=state_err_signed,
                    protocol="parent_se3_v1", b_policy="legacy_scalar_at_urdf_pivot",
                    trial_seed=res["_phys"]["trial_seed"], exec_seed=res["_phys"]["exec_seed"],
                    direction_seed=res["_phys"]["direction_seed"],
                    b2_seed=res["_phys"]["b2_seed"], b2_phi=res["_phys"]["b2_phi"],
                    mass_scale=res["_phys"]["mass_scale"],
                    damping_scale=res["_phys"]["damping_scale"],
                    init_state_deg=res["_phys"]["init_state_deg"],
                    e_geo_nominal_cm=r["e_geo_nominal_cm"],
                    e_norm_nominal_cm=r["e_norm_nominal_cm"],
                    progress=r["progress"], max_track_err=r["max_track_err"],
                    mean_track_err=r["mean_track_err"], sat_frac=r["sat_frac"],
                    peak_force=r["peak_force"], n_steps=r["n_steps"],
                    firm=r["firm"], rpmart85=r["rpmart85"], task_done=r["task_done"]))
        if (ri + 1) % 4 == 0:
            print(f"   ... {ri+1}/{len(src_rows)} estimates done")

    out_p = OUT / f"p0_{args.obj}_{args.preset}_L{args.level}_{args.selector}_se3{args.out_suffix}.csv"
    with out_p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f"[4/4] -> {out_p}  ({len(out_rows)} rows)")

    print(f"\n{'estimator':14s}{'mode':>6}{'firm':>8}{'rpmart85':>10}{'task_done':>11}"
         f"{'max_track_err(cm)':>20}")
    for est in sorted(set(r["estimator"] for r in out_rows)):
        for m in ("A", "B", "B2", "C"):
            sub = [r for r in out_rows if r["estimator"] == est and r["mode"] == m]
            if not sub:
                continue
            firm = np.mean([r["firm"] for r in sub])
            rp85 = np.mean([r["rpmart85"] for r in sub])
            td = np.mean([r["task_done"] for r in sub])
            mte = np.mean([r["max_track_err"] for r in sub]) * 100
            print(f"{est:14s}{m:>6}{firm:8.3f}{rp85:10.3f}{td:11.3f}{mte:20.2f}")


if __name__ == "__main__":
    main()
