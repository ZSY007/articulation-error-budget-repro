"""P0 (EXPERIMENTS.md SS P-X): carry an estimator's ACTUAL axis/pivot deviation from the
render/GT-flow frame (where gen_perseed.py / real_tracker_replay.py compute axis_err /
pivot_err) into the PyBullet world frame (where pilot_object.run_trial executes it) --
without re-randomizing the direction, which is what e4e5_object.py's `_one()` and
run_budget.py's `one_trial()` currently do (see EXPERIMENTS.md SS P-X, P0 motivation).

Method: the two frames differ by a full rigid transform (R_frame, translation_frame).
Both components come from corresponding parent-link frames. Do NOT derive the
translation from GT pivots: the renderer uses a canonical point on the axis,
whereas PyBullet uses the URDF joint origin. Aligning those different points
translates a tilted estimated axis to a different physical line.

The scalar axis_err/pivot_err self-check is necessary but insufficient: translating
the estimated line along the GT axis leaves both metrics unchanged. Physical
invariants (pivot gauge and full motion under SE(3)) are tested separately in
tests/test_p0_frames.py. The historical GT-pivot matching bug and paired replays
are documented in docs/P0_10905_手持重放核查_2026-09-12.md.
"""
from __future__ import annotations

import numpy as np


def _ax_deg(a, b):
    """Same metric as scripts/hybrid_eval.py::_ax -- headless axis angle, degrees."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(
        abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9), 0, 1))))


def _piv_cm(p, ax, pg):
    """Same metric as scripts/hybrid_eval.py::_piv -- perpendicular pivot offset, cm."""
    d = np.asarray(p, float) - np.asarray(pg, float)
    a = np.asarray(ax, float) / (np.linalg.norm(ax) + 1e-9)
    return float(np.linalg.norm(d - (d @ a) * a) * 100)


def parse_vec(s: str) -> np.ndarray:
    """Inverse of gen_perseed.py's `_v()` / real_tracker_replay.py's `_v()` packing."""
    return np.array([float(x) for x in s.split(";")], dtype=float)


def frame_rotation(R_parent_render: np.ndarray, R_parent_world: np.ndarray) -> np.ndarray:
    """R_frame such that R_frame @ v_render ~= v_world for any object-rigid vector v.

    Derived from the parent link's rest-pose rotation in each frame (a full 3x3
    orthonormal frame, not a single vector -- one vector under-determines a
    rotation about that vector's own axis).
    """
    R_frame = R_parent_world @ R_parent_render.T
    # R_frame must be a proper rotation (orthogonal, det=+1). If it isn't, the
    # single-fixed-rotation model is already wrong before we even get to self_check.
    det = np.linalg.det(R_frame)
    if (not np.isfinite(R_frame).all() or abs(det - 1.0) > 1e-3
            or not np.allclose(R_frame.T @ R_frame, np.eye(3), atol=1e-5)):
        raise ValueError(f"frame_rotation: det={det:.4f}, not a proper rotation -- "
                          "render/world frames are not related by a single rigid "
                          "rotation as assumed; do not proceed for this object.")
    return R_frame


def frame_transform(R_parent_render, t_parent_render, R_parent_world, t_parent_world):
    """Full SE(3) from corresponding parent-link frames, with positions in metres."""
    rotation = frame_rotation(np.asarray(R_parent_render), np.asarray(R_parent_world))
    translation = np.asarray(t_parent_world, float) - rotation @ np.asarray(t_parent_render, float)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("frame_transform: invalid parent-frame translation")
    return rotation, translation


def apply_relative_deviation(R_frame: np.ndarray,
                              gt_axis_render: np.ndarray, gt_pivot_render: np.ndarray,
                              est_axis_render: np.ndarray, est_pivot_render: np.ndarray,
                              gt_axis_world: np.ndarray, gt_pivot_world: np.ndarray,
                              *, translation_frame: np.ndarray):
    """Transport the complete estimated axis line using the parent-frame SE(3).

    GT axis lines validate the transform, but their arbitrary pivot representatives
    must never determine its translation. Axis sign is aligned to the target GT
    hemisphere; callers use a signed opening-magnitude error, not raw signed theta.
    """
    translation_frame = np.asarray(translation_frame, float)
    ga_r = np.asarray(gt_axis_render, float)
    ga_w = np.asarray(gt_axis_world, float)
    ga_w = ga_w / np.linalg.norm(ga_w)
    mapped_gt = R_frame @ ga_r
    if abs(mapped_gt @ ga_w) / np.linalg.norm(mapped_gt) < 1 - 1e-5:
        raise ValueError("apply_relative_deviation: GT axes do not agree under parent SE(3)")
    gt_offset = R_frame @ np.asarray(gt_pivot_render, float) + translation_frame - gt_pivot_world
    if np.linalg.norm(gt_offset - (gt_offset @ ga_w) * ga_w) > 1e-3:
        raise ValueError("apply_relative_deviation: GT axis lines do not agree under parent SE(3)")
    est_axis_world = R_frame @ np.asarray(est_axis_render, float)
    n = np.linalg.norm(est_axis_world)
    if not np.isfinite(n) or n < 1e-9:
        raise ValueError("apply_relative_deviation: degenerate estimated axis (norm ~0)")
    est_axis_world = est_axis_world / n
    # axis estimators are headless (sign-ambiguous); align to the GT hemisphere so the
    # transported deviation matches the same (headless) axis_err, not its 180 deg twin.
    if est_axis_world @ np.asarray(gt_axis_world, float) < 0:
        est_axis_world = -est_axis_world
    est_pivot_world = R_frame @ np.asarray(est_pivot_render, float) + translation_frame
    if not np.isfinite(est_pivot_world).all():
        raise ValueError("apply_relative_deviation: non-finite estimated pivot")
    return est_axis_world, est_pivot_world


def apply_axis_deviation(R_frame: np.ndarray,
                         gt_axis_render: np.ndarray,
                         est_axis_render: np.ndarray,
                         gt_axis_world: np.ndarray) -> np.ndarray:
    """Transport a headless prismatic direction through a rigid-frame rotation.

    A prismatic joint is a direction, not an axis *line*: translation and pivot
    representatives have no physical role.  The mapped GT direction is checked
    before the estimate is returned, and the estimate is sign-aligned to the GT
    hemisphere so signed opening-state errors keep their original convention.
    """
    R_frame = np.asarray(R_frame, float)
    ga_r = np.asarray(gt_axis_render, float)
    ga_w = np.asarray(gt_axis_world, float)
    ga_w = ga_w / np.linalg.norm(ga_w)
    mapped_gt = R_frame @ ga_r
    if abs(mapped_gt @ ga_w) / (np.linalg.norm(mapped_gt) + 1e-12) < 1 - 1e-5:
        raise ValueError("apply_axis_deviation: GT axes do not agree under parent rotation")
    est_axis_world = R_frame @ np.asarray(est_axis_render, float)
    n = np.linalg.norm(est_axis_world)
    if not np.isfinite(n) or n < 1e-9:
        raise ValueError("apply_axis_deviation: degenerate estimated axis (norm ~0)")
    est_axis_world = est_axis_world / n
    if est_axis_world @ ga_w < 0:
        est_axis_world = -est_axis_world
    return est_axis_world


def rotate_axis_line_about_true_axis(gt_axis_world, gt_gauge_point,
                                      line_dir, line_point, phi: float):
    """Rigidly rotate an estimated (direction, point-on-line) pair about the TRUE
    axis by angle phi -- the geometry-preserving randomization proposed in
    docs/P0_10905_手持重放核查_2026-09-12.md SS5 as mode B2, replacing the legacy
    independently-randomized B for a strict direction-only intervention.

    Q = Rot(gt_axis, phi); dir' = Q @ line_dir; point' = gauge + Q @ (point - gauge).

    Exactly preserves, for ANY choice of gauge point on the true axis (the result
    does not depend on which point along the true axis is used -- Q fixes every
    point on the true axis, so shifting the gauge point by k*gt_axis leaves point'
    unchanged; see test_p0_frames.py):
      * axis_err  = angle(line_dir, gt_axis)   -- Q fixes gt_axis, rotations preserve angles.
      * pivot_err = perpendicular distance from point to the true axis line -- Q
        rotates the perpendicular component's direction but not its length, and
        leaves the component parallel to gt_axis untouched.
    Unlike the legacy B (independently redraws the axis-tilt direction and the
    pivot-offset direction via two separate calls), this keeps the joint tilt/offset
    geometry of ONE coherent tilted line -- the only thing randomized is where
    around the true axis that whole line sits, not its internal shape.
    """
    a = np.asarray(gt_axis_world, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + np.sin(phi) * K + (1 - np.cos(phi)) * (K @ K)
    new_dir = R @ np.asarray(line_dir, float)
    n = np.linalg.norm(new_dir)
    if not np.isfinite(n) or n < 1e-9:
        raise ValueError("rotate_axis_line_about_true_axis: degenerate direction (norm ~0)")
    new_dir = new_dir / n
    gauge = np.asarray(gt_gauge_point, float)
    new_point = gauge + R @ (np.asarray(line_point, float) - gauge)
    if not np.isfinite(new_point).all():
        raise ValueError("rotate_axis_line_about_true_axis: non-finite rotated point")
    return new_dir, new_point


def geometric_path_deviation_cm(gt_axis, gt_pivot, est_axis, est_pivot,
                                  handle_start, relative_angles) -> float:
    """Maximum nominal handle-path deviation induced by an estimated axis line.

    Both paths start from the same handle point and use the same relative joint
    angles.  The value therefore isolates pre-execution joint geometry from
    controller noise, contact dynamics, mass/damping draws, and grasp jitter.
    Inputs use metres/radians; the returned maximum is in centimetres.
    """
    gt_axis = np.asarray(gt_axis, float)
    est_axis = np.asarray(est_axis, float)
    gt_axis = gt_axis / np.linalg.norm(gt_axis)
    est_axis = est_axis / np.linalg.norm(est_axis)
    gt_pivot = np.asarray(gt_pivot, float)
    est_pivot = np.asarray(est_pivot, float)
    handle_start = np.asarray(handle_start, float)
    angles = np.asarray(relative_angles, float).reshape(-1)
    if not all(np.isfinite(x).all() for x in
               (gt_axis, est_axis, gt_pivot, est_pivot, handle_start, angles)):
        raise ValueError("geometric_path_deviation_cm: non-finite input")
    if angles.size == 0:
        return 0.0

    def _rot(axis, theta):
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    deviations = []
    for theta in angles:
        true_point = gt_pivot + _rot(gt_axis, theta) @ (handle_start - gt_pivot)
        estimated_point = est_pivot + _rot(est_axis, theta) @ (handle_start - est_pivot)
        deviations.append(np.linalg.norm(estimated_point - true_point))
    return float(max(deviations) * 100.0)


def normal_path_deviation_cm(gt_axis, gt_pivot, est_axis, est_pivot,
                              handle_start, relative_angles) -> float:
    """Maximum distance from the commanded (estimated-axis) path to the TRUE handle
    circle -- the part of geometric_path_deviation_cm a 1-DOF revolute joint cannot
    absorb by reaching a different joint angle.

    geometric_path_deviation_cm compares two points at the SAME relative angle, which
    mixes a normal component (radial/axial displacement off the true circle) with a
    tangential component (the commanded point still lies near the true circle, just at
    a different arc-length -- something the joint trivially cancels by running ahead of
    or behind the nominal schedule). This function isolates the normal component only,
    by projecting onto the true circle (centre = axial projection of the handle onto
    gt_axis through gt_pivot, radius = perpendicular distance of the handle from that
    axis) instead of onto the true angle-matched point.

    Inputs use metres/radians; the returned maximum is in centimetres.  See
    docs/条件误差预算_P3c-P4-E5_方案_2026-09-13.md SS0 for the derivation and the
    post-hoc evidence that this predicts executed tracking error (e_exec) far better
    than geometric_path_deviation_cm (pooled Spearman 0.983 vs 0.826 on the 6656-trial
    P3(a) holdout).
    """
    gt_axis = np.asarray(gt_axis, float)
    est_axis = np.asarray(est_axis, float)
    gt_axis = gt_axis / np.linalg.norm(gt_axis)
    est_axis = est_axis / np.linalg.norm(est_axis)
    gt_pivot = np.asarray(gt_pivot, float)
    est_pivot = np.asarray(est_pivot, float)
    handle_start = np.asarray(handle_start, float)
    angles = np.asarray(relative_angles, float).reshape(-1)
    if not all(np.isfinite(x).all() for x in
               (gt_axis, est_axis, gt_pivot, est_pivot, handle_start, angles)):
        raise ValueError("normal_path_deviation_cm: non-finite input")
    if angles.size == 0:
        return 0.0

    def _rot(axis, theta):
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    v0 = handle_start - gt_pivot
    axial0 = v0 @ gt_axis
    radius = np.linalg.norm(v0 - axial0 * gt_axis)
    deviations = []
    for theta in angles:
        commanded = est_pivot + _rot(est_axis, theta) @ (handle_start - est_pivot)
        w = commanded - gt_pivot
        axial = w @ gt_axis
        radial = np.linalg.norm(w - axial * gt_axis)
        deviations.append(float(np.hypot(axial - axial0, radial - radius)))
    return float(max(deviations) * 100.0)


def geometric_path_deviation_prismatic_cm(gt_axis, est_axis,
                                           relative_displacements) -> float:
    """Maximum straight-path deviation for a prismatic joint, in centimetres.

    The true and estimated paths start at the same handle point and are evaluated
    at identical signed relative displacements (metres).  The start point cancels,
    so only the transported direction error contributes.
    """
    gt_axis = np.asarray(gt_axis, float)
    est_axis = np.asarray(est_axis, float)
    displacements = np.asarray(relative_displacements, float).reshape(-1)
    ng, ne = np.linalg.norm(gt_axis), np.linalg.norm(est_axis)
    if (not np.isfinite(ng) or not np.isfinite(ne) or ng < 1e-9 or ne < 1e-9
            or not np.isfinite(displacements).all()):
        raise ValueError("geometric_path_deviation_prismatic_cm: invalid input")
    if displacements.size == 0:
        return 0.0
    gt_axis, est_axis = gt_axis / ng, est_axis / ne
    deviations = np.linalg.norm(
        displacements[:, None] * (est_axis - gt_axis)[None, :], axis=1)
    return float(deviations.max() * 100.0)


def normal_path_deviation_prismatic_cm(gt_axis, est_axis, relative_displacements) -> float:
    """Prismatic counterpart to normal_path_deviation_cm: max distance from the commanded
    (straight-line, direction est_axis) path to the TRUE straight line (direction gt_axis,
    through the same start point). Isolates the component of a direction error a 1-DOF
    slider cannot absorb by reaching a different displacement; geometric_path_deviation_prismatic_cm
    always upper-bounds it (see docs/条件误差预算_P3c-P4-E5_方案_2026-09-13.md SS5).

    Inputs use metres; the returned maximum is in centimetres.
    """
    gt_axis = np.asarray(gt_axis, float)
    est_axis = np.asarray(est_axis, float)
    displacements = np.asarray(relative_displacements, float).reshape(-1)
    ng, ne = np.linalg.norm(gt_axis), np.linalg.norm(est_axis)
    if (not np.isfinite(ng) or not np.isfinite(ne) or ng < 1e-9 or ne < 1e-9
            or not np.isfinite(displacements).all()):
        raise ValueError("normal_path_deviation_prismatic_cm: invalid input")
    if displacements.size == 0:
        return 0.0
    gt_axis, est_axis = gt_axis / ng, est_axis / ne
    perp = est_axis - (est_axis @ gt_axis) * gt_axis
    return float(np.abs(displacements).max() * np.linalg.norm(perp) * 100.0)


def tangent_similarity_revolute(gt_axis, gt_pivot, est_axis, est_pivot, handle_start,
                                 q_min: float, q_max: float, n: int = 64) -> float:
    """Heppert et al. (IROS 2022), Eq.16: mean cosine similarity between the true
    and predicted grasp-point VELOCITY direction, averaged over the joint's own
    range [q_min, q_max] (radians) -- NOT the shorter per-trial commanded sweep
    (docs/方向调整与补实验审阅_2026-09-16.md SS3.1: the original integrates over the
    whole joint range, "single-point/finite-swing" was a wrong critique of it).

    True position x(q) = gt_pivot + Rot(gt_axis,q)@(handle_start-gt_pivot) (same
    parametrization as geometric_path_deviation_cm); velocity is its q-derivative,
    v(q) = gt_axis x Rot(gt_axis,q)@(handle_start-gt_pivot). Predicted x_hat(q) uses
    (est_axis, est_pivot) at the SAME q -- Heppert compares configurations, not
    matched 3D positions. J in [-1, 1]; 1 = identical direction at every q.
    """
    gt_axis = np.asarray(gt_axis, float); gt_axis = gt_axis / np.linalg.norm(gt_axis)
    est_axis = np.asarray(est_axis, float); est_axis = est_axis / np.linalg.norm(est_axis)
    gt_pivot = np.asarray(gt_pivot, float)
    est_pivot = np.asarray(est_pivot, float)
    handle_start = np.asarray(handle_start, float)
    if not all(np.isfinite(x).all() for x in (gt_axis, est_axis, gt_pivot, est_pivot, handle_start)):
        raise ValueError("tangent_similarity_revolute: non-finite input")
    if q_max <= q_min:
        raise ValueError("tangent_similarity_revolute: q_max must be > q_min")

    def _rot(axis, theta):
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    qs = np.linspace(q_min, q_max, n)
    cosines = []
    for q in qs:
        v = np.cross(gt_axis, _rot(gt_axis, q) @ (handle_start - gt_pivot))
        v_hat = np.cross(est_axis, _rot(est_axis, q) @ (handle_start - est_pivot))
        nv, nvh = np.linalg.norm(v), np.linalg.norm(v_hat)
        if nv < 1e-9 or nvh < 1e-9:
            continue
        cosines.append(float((v @ v_hat) / (nv * nvh)))
    if not cosines:
        return float("nan")
    return float(np.mean(cosines))


def tangent_similarity_revolute_eq16(gt_axis, gt_pivot, est_axis, est_pivot, handle_start,
                                     q_min: float, q_max: float, n: int = 64) -> float:
    """CORRECTED per Heppert et al. (IROS 2022) Eq. 15-16, verified against the paper's own
    text (arxiv.org/html/2205.03721v2#S6, fetched and checked 2026-09-18): both the true and
    estimated linear velocity are evaluated at the SAME point -- the TRUE path x(q) = Exp(q*nu)*x0
    -- via f(nu, q) = v + [w]_x @ x(q). For a pure rotation (w = q_dot * axis, v = -w x pivot),
    f(nu,q) = axis x (x(q) - pivot).

    `tangent_similarity_revolute` (above, no suffix) does NOT do this: it evaluates the
    estimated velocity at an ESTIMATED path point x_hat(q) = est_pivot + Rot(est_axis,q)@
    (handle_start-est_pivot), i.e. est_axis x (x_hat(q) - est_pivot) -- comparing two twists
    at two DIFFERENT points, not the same point Eq.16 specifies. That function is kept
    (unmodified, not deleted) because it is what actually produced every `J`/`J_full`/`j_full`
    number already in this paper's tables as of 2026-09-17 -- those are `J_cfg`, a locally-
    configured approximation, not a reproduction of Heppert et al.'s own formula. This function
    is the paper's honest attempt at the actual Eq.16; Section V-A reports both, does not retro-
    actively relabel old `J_cfg` numbers as `J_eq16`, and states which is which everywhere
    "beats/loses to Heppert's own metric" is claimed (2026-09-18 EDITORIAL_NOTES.md finding).
    """
    gt_axis = np.asarray(gt_axis, float); gt_axis = gt_axis / np.linalg.norm(gt_axis)
    est_axis = np.asarray(est_axis, float); est_axis = est_axis / np.linalg.norm(est_axis)
    gt_pivot = np.asarray(gt_pivot, float)
    est_pivot = np.asarray(est_pivot, float)
    handle_start = np.asarray(handle_start, float)
    if not all(np.isfinite(x).all() for x in (gt_axis, est_axis, gt_pivot, est_pivot, handle_start)):
        raise ValueError("tangent_similarity_revolute_eq16: non-finite input")
    if q_max <= q_min:
        raise ValueError("tangent_similarity_revolute_eq16: q_max must be > q_min")

    def _rot(axis, theta):
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    qs = np.linspace(q_min, q_max, n)
    cosines = []
    for q in qs:
        x_q = gt_pivot + _rot(gt_axis, q) @ (handle_start - gt_pivot)   # TRUE path point ONLY
        v = np.cross(gt_axis, x_q - gt_pivot)
        v_hat = np.cross(est_axis, x_q - est_pivot)                    # estimated twist, SAME x_q
        nv, nvh = np.linalg.norm(v), np.linalg.norm(v_hat)
        if nv < 1e-9 or nvh < 1e-9:
            continue
        cosines.append(float((v @ v_hat) / (nv * nvh)))
    if not cosines:
        return float("nan")
    return float(np.mean(cosines))


def tangent_similarity_prismatic(gt_axis, est_axis) -> float:
    """Prismatic counterpart: velocity direction is the (headless) slide axis
    itself, constant along the whole travel -- J degenerates to a single cosine,
    cos(e_axis), with no q-dependence (no pivot, no integration needed).
    """
    gt_axis = np.asarray(gt_axis, float)
    est_axis = np.asarray(est_axis, float)
    ng, ne = np.linalg.norm(gt_axis), np.linalg.norm(est_axis)
    if not np.isfinite(ng) or not np.isfinite(ne) or ng < 1e-9 or ne < 1e-9:
        raise ValueError("tangent_similarity_prismatic: degenerate axis")
    return float(np.abs((gt_axis / ng) @ (est_axis / ne)))


def self_check(est_axis_world, est_pivot_world, gt_axis_world, gt_pivot_world,
               expect_axis_err_deg: float, expect_pivot_err_cm: float,
               tol_axis_deg: float = 1e-2, tol_pivot_cm: float = 1e-2):
    """Recompute axis_err/pivot_err in the WORLD frame from the transported estimate
    and compare against the RENDER-frame values gen_perseed.py already logged.

    A proper rigid transform is an isometry, so these must match to numerical
    precision. Matching scalar metrics alone does not validate transport of the
    physical estimated line; see the module docstring and geometric tests.
    Returns (ok: bool, axis_err_world, pivot_err_world) -- always returns the
    computed values even when ok=False, so a failing object can be inspected
    rather than silently skipped.
    """
    ae_w = _ax_deg(est_axis_world, gt_axis_world)
    pe_w = _piv_cm(est_pivot_world, gt_axis_world, gt_pivot_world) \
        if np.all(np.isfinite(gt_pivot_world)) else float("nan")
    ok = abs(ae_w - expect_axis_err_deg) < tol_axis_deg
    if np.isfinite(expect_pivot_err_cm) and np.isfinite(pe_w):
        ok = ok and abs(pe_w - expect_pivot_err_cm) < tol_pivot_cm
    return ok, ae_w, pe_w
