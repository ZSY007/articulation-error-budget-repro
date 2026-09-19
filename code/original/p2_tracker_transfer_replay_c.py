"""方案2 §7, E2-C: unified-execution-policy replay, for error-source attribution
(is a transfer gap, if any, caused by noise STRUCTURE or by replay/state-POLICY
differences between the two domains?). Per the doc, triggered here by explicit
user authorization (2026-09-18) even though E2-A/E2-B's own trigger condition
("A/B差异明显") was not met -- both already showed a clear, consistent signal.
Run anyway as requested; report whatever it shows.

Reuses `pilot_object.py::run_trial` UNCHANGED (the same function
p4e_feedback_noise_stress.py and the whole Table V-1 pipeline already use) with
one specific parameter policy that unifies source and target execution, per
docs/实验方案2_结构化跟踪误差与预算迁移_2026-09-18.md §7:

  init_state_deg=0.0   (true q0=0, not the per-rep random draw the existing
                        source pipeline uses -- see p4e_feedback_noise_stress.py's
                        `phys_rng.uniform(0,5)`, NOT used here)
  est_state_deg=0.0    (the controller's belief about q0 is also 0 -- matches
                        what the target/RGB pipeline already implicitly assumes)
  horizon=None         (open-loop, fixed-start point-constraint -- same
                        executor as Table V-1)
  obs_delay_steps=0, obs_noise_mm=0.0   (no horizon-mechanism feedback in play
                        at horizon=None anyway; kept explicit/default)
  same physical-perturbation distribution as the project's other replay
  scripts: mass_scale, damping_scale ~ U(0.5, 2.0); grasp_jitter_cm=1.0;
  ctrl_noise_mm=1.0 (fixed, not swept -- these are NOT the axis under test)

Three cohorts, 19,088 total physical CPU executions (no GPU, no new tracking):
  source global/crossfit: 976 estimates (61 objects, excl. the 12 RGB objects)
                          x 16 reps = 15,616
  target RGB:             144 (obj,mode,level) cells x 16 reps = 2,304
  GT control:              73 objects (all of Table V-1's library) x 16 reps
                          = 1,168

    conda activate cv   (local) / ee900-prime (JARVIS, see jarvis_p2c_replay.sh)
    python error_budget/p2_tracker_transfer_replay_c.py --procs 16
    -> out/p2_tracker_transfer_20260918/e2c_replay_{source,target,gt_control}.csv
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_object as PO                                                     # noqa: E402
from p1_tangent_similarity_offline import (                                   # noqa: E402
    _scene_geometry as _source_scene_geometry, _reconstruct_axis_pivot,
    JOINT_OVERRIDE,
)
from p0_direction_preserving_replay import CRITERIA                           # noqa: E402
from p2_tracker_transfer_manifest import (                                    # noqa: E402
    TARGET_OBJECTS, EVAL_FILES, target_scene_geometry,
)
import pybullet as p                                                          # noqa: E402


def gt_control_scene_geometry(obj):
    """Like p2_tracker_transfer_manifest.target_scene_geometry, but applies
    JOINT_OVERRIDE -- needed here because the GT-control cohort spans all 73
    Table V-1 objects (not just the 12 RGB objects target_scene_geometry was
    written for), and several of the 61 source objects are NOT on joint_0."""
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
    return dict(spec=spec, sgn=sgn, gt_axis_w=np.asarray(gt_axis_w, float),
               gt_pivot_w=np.asarray(gt_pivot_w, float))

OUT = Path(__file__).resolve().parent / "out"
RUN_DIR = OUT / "p2_tracker_transfer_20260918"
SEED_BASE = 3_200_000  # distinct block from p3/p4's 3_100_000, same SHA256 scheme
REPS = 16


def stable_seed(namespace, obj, rep, extra="", seed_base=SEED_BASE):
    payload = f"p2c|{namespace}|{obj}|{rep}|{extra}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 1_000_000
    return seed_base + offset


def _run_one(spec_kwargs, ea, ep, sgn, rep, phys_seed):
    spec = PO.ObjSpec(spec_kwargs["obj"], spec_kwargs["joint"])
    spec.target_deg, spec.radius, spec.f_max, spec.substeps = (
        spec_kwargs["target_deg"], spec_kwargs["radius"], spec_kwargs["f_max"], spec_kwargs["substeps"])
    phys_rng = np.random.default_rng(phys_seed)
    mass_scale = float(phys_rng.uniform(0.5, 2.0))
    damping_scale = float(phys_rng.uniform(0.5, 2.0))
    exec_seed = phys_seed + 7_000_000
    result = PO.run_trial(
        spec, np.asarray(ea), np.asarray(ep), 0.0, np.random.default_rng(exec_seed), sgn=sgn,
        mass_scale=mass_scale, damping_scale=damping_scale, grasp_jitter_cm=1.0,
        init_state_deg=0.0, ctrl_noise_mm=1.0, f_max=spec.f_max,
        horizon=None, obs_delay_steps=0, obs_noise_mm=0.0)
    for name, fn in CRITERIA.items():
        result[name] = int(fn(result))
    return result


# ---------------------------------------------------------------- workers --
_SOURCE_GEO = None
_TARGET_GEO = None
_GT_GEO = None


def _init_source(geo_cache_serialized, perseed_cache_serialized):
    global _SOURCE_GEO
    _SOURCE_GEO = (geo_cache_serialized, perseed_cache_serialized)


def _one_source(task):
    obj, selector, src_row, rep = task
    geo_by_obj, perseed_by_key = _SOURCE_GEO
    geo = geo_by_obj[obj]
    src_rows = perseed_by_key[(obj, selector)]
    src_row_data = src_rows[int(src_row)]
    ea, ep = _reconstruct_axis_pivot(geo, src_row_data, "A", 0, 0.0)
    spec_kwargs = dict(obj=obj, joint=geo["spec"].joint, target_deg=geo["spec"].target_deg,
                       radius=geo["spec"].radius, f_max=geo["spec"].f_max, substeps=geo["spec"].substeps)
    seed = stable_seed("source_global_crossfit", obj, rep, extra=str(src_row))
    r = _run_one(spec_kwargs, ea, ep, geo["sgn"], rep, seed)
    return dict(cohort="source_global_crossfit", object_id=obj, selector=selector, src_row=src_row,
               rep=rep, **{k: r[k] for k in ("firm", "rpmart85", "progress", "max_track_err", "sat_frac")})


def _init_target(geo_by_obj):
    global _TARGET_GEO
    _TARGET_GEO = geo_by_obj


def _one_target(task):
    obj, mode, level, est_axis, est_pivot, rep = task
    geo = _TARGET_GEO[obj]
    spec_kwargs = dict(obj=obj, joint=geo["spec"].joint, target_deg=geo["spec"].target_deg,
                       radius=geo["spec"].radius, f_max=geo["spec"].f_max, substeps=geo["spec"].substeps)
    seed = stable_seed("target_rgb", obj, rep, extra=f"{mode}_{level}")
    r = _run_one(spec_kwargs, est_axis, est_pivot, geo["sgn"], rep, seed)
    return dict(cohort="target_rgb", object_id=obj, corruption=mode, level=level, rep=rep,
               **{k: r[k] for k in ("firm", "rpmart85", "progress", "max_track_err", "sat_frac")})


def _init_gt(geo_by_obj):
    global _GT_GEO
    _GT_GEO = geo_by_obj


def _one_gt(task):
    obj, rep = task
    geo = _GT_GEO[obj]
    spec_kwargs = dict(obj=obj, joint=geo["spec"].joint, target_deg=geo["spec"].target_deg,
                       radius=geo["spec"].radius, f_max=geo["spec"].f_max, substeps=geo["spec"].substeps)
    seed = stable_seed("gt_control", obj, rep)
    r = _run_one(spec_kwargs, geo["gt_axis_w"], geo["gt_pivot_w"], geo["sgn"], rep, seed)
    return dict(cohort="gt_control", object_id=obj, rep=rep,
               **{k: r[k] for k in ("firm", "rpmart85", "progress", "max_track_err", "sat_frac")})


# ---------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--cohort", choices=["source", "target", "gt", "all"], default="all")
    args = ap.parse_args()

    RUN_DIR.mkdir(exist_ok=True)
    ctx = mp.get_context("spawn")

    # ---------------- source: 976 global/crossfit estimates (61 objects) ----------------
    if args.cohort in ("source", "all"):
        p1_path = OUT / "p1_tangent_all.csv"
        src_keys = []
        with open(p1_path) as f:
            for row in csv.DictReader(f):
                if (row["kind"] == "revolute" and row["mode"] == "A"
                        and row["obj"] not in TARGET_OBJECTS
                        and row["estimator"] == "global" and row["selector"] == "crossfit"):
                    src_keys.append((row["obj"], row["selector"], row["src_row"]))
        src_keys = sorted(set(src_keys))
        print(f"[source] {len(src_keys)} global/crossfit estimates "
              f"(want 976); {len(set(k[0] for k in src_keys))} objects (want 61)")

        objs = sorted(set(k[0] for k in src_keys))
        geo_by_obj = {}
        perseed_by_key = {}
        for obj in objs:
            geo_by_obj[obj] = _source_scene_geometry(obj)
            key = (obj, "crossfit")
            perseed_p = OUT / f"perseed_errors_{obj}_p0_crossfit_se3.csv"
            perseed_by_key[key] = list(csv.DictReader(perseed_p.open()))
        print(f"[source] geometry+perseed cache built for {len(objs)} objects")

        tasks = [(obj, sel, src_row, rep) for (obj, sel, src_row) in src_keys for rep in range(REPS)]
        print(f"[source] {len(tasks)} total executions (want 15,616)")
        with ctx.Pool(args.procs, initializer=_init_source, initargs=(geo_by_obj, perseed_by_key)) as pool:
            rows = pool.map(_one_source, tasks, chunksize=8)
        out_p = RUN_DIR / "e2c_replay_source.csv"
        with open(out_p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"-> {out_p} ({len(rows)} rows)")

    # ---------------- target: 144 RGB cells (12 objects) ----------------
    if args.cohort in ("target", "all"):
        cells = []
        for ef in EVAL_FILES:
            for c in json.loads((OUT / ef).read_text()):
                if c["type"] == "revolute" and c["obj"] in TARGET_OBJECTS:
                    cells.append(c)
        print(f"[target] {len(cells)} cells (want 144)")
        objs = sorted(set(c["obj"] for c in cells))
        geo_by_obj = {obj: target_scene_geometry(obj) for obj in objs}
        tasks = [(c["obj"], c["mode"], c["level"], c["est_axis"], c["est_pivot"], rep)
                for c in cells for rep in range(REPS)]
        print(f"[target] {len(tasks)} total executions (want 2,304)")
        with ctx.Pool(args.procs, initializer=_init_target, initargs=(geo_by_obj,)) as pool:
            rows = pool.map(_one_target, tasks, chunksize=8)
        out_p = RUN_DIR / "e2c_replay_target.csv"
        with open(out_p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"-> {out_p} ({len(rows)} rows)")

    # ---------------- GT control: all 73 objects ----------------
    if args.cohort in ("gt", "all"):
        p1_path = OUT / "p1_tangent_all.csv"
        all_objs = set()
        with open(p1_path) as f:
            for row in csv.DictReader(f):
                if row["kind"] == "revolute" and row["mode"] == "A":
                    all_objs.add(row["obj"])
        all_objs = sorted(all_objs)
        print(f"[gt_control] {len(all_objs)} objects (want 73)")
        geo_by_obj = {obj: gt_control_scene_geometry(obj) for obj in all_objs}
        tasks = [(obj, rep) for obj in all_objs for rep in range(REPS)]
        print(f"[gt_control] {len(tasks)} total executions (want 1,168)")
        with ctx.Pool(args.procs, initializer=_init_gt, initargs=(geo_by_obj,)) as pool:
            rows = pool.map(_one_gt, tasks, chunksize=8)
        out_p = RUN_DIR / "e2c_replay_gt_control.csv"
        with open(out_p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"-> {out_p} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
