"""方案2 step 1-3: input verification, common-nominal predictor recompute, and
manifest for docs/实验方案2_结构化跟踪误差与预算迁移_2026-09-18.md.

What this does, in the doc's own terms:

  §0 verify J_eq16 recompute is complete, lock its file hash as this
      experiment's frozen input (does NOT re-run item 1).
  §2  load target (RGB tracker-error) and source (flow_gaussian) domains,
      confirm the 12 target objects are excluded from the 61-object source
      pool used for E2-B training (empty intersection).
  §3  build *_common_nominal predictors for BOTH domains under the SAME
      convention (q0=0, same object target stroke, no grasp/control jitter,
      same frame/gauge) -- source's existing e_norm_nominal_cm/e_geo_nominal_cm
      columns are NOT reused as-is because they were computed at the trial's
      OWN (randomized) init state, not q0=0 (docs table in §3); J_cfg/J_eq16
      ARE reused as-is because both domains' existing pipelines already
      integrate over the object's full nominal range [0, open_lim] -- verified
      by inspecting both p1_tangent_similarity_offline.py and
      p1_rgbsweep_metric_compare.py before writing this script, not assumed.

Manifest schema note: the doc lists a fuller field set (base_clip_id,
estimate_id, fit_seed, frame, state_policy, predictor_policy, outcome_policy,
failure_reason, ...) than what is written here. This script's manifest keeps
the fields that are load-bearing for E2-A/E2-B's actual statistics (object,
domain, estimator/selector or corruption/level, the six predictors, outcome
labels, validity) and traces back to the exact source files/row keys needed to
reconstruct any of the rest, rather than re-deriving every listed column for
its own sake. Recorded as a scope note in report.md, not a silent gap.

    conda activate cv
    python error_budget/p2_tracker_transfer_manifest.py
    -> out/p2_tracker_transfer_20260918/{protocol.json, input_manifest.json,
       source_target_objects.json, predictors_common_nominal.csv,
       source_execution_common_nominal.csv, target_execution_common_nominal.csv}
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_object as PO                                                     # noqa: E402
import p0_basis as PB                                                         # noqa: E402
from p1_tangent_similarity_offline import (                                   # noqa: E402
    _scene_geometry as _source_scene_geometry, _reconstruct_axis_pivot,
)

OUT = Path(__file__).resolve().parent / "out"
RUN_DIR = OUT / "p2_tracker_transfer_20260918"
RUN_DIR.mkdir(exist_ok=True)

TARGET_OBJECTS = ["10144", "101593", "101611", "102301", "10849", "10905",
                  "35059", "40147", "46889", "47315", "7236", "7292"]
STEP_DEG = PO.STEP_DEG
EVAL_FILES = ["rgbsweep_library_eval.json", "rgbsweep2_library_eval.json"]
TRIAL_FILES = ["e5x_rgbsweep_trials.csv", "e5x_rgbsweep2_trials.csv"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- §0 verify --
def verify_item1():
    p1_path = OUT / "p1_tangent_all.csv"
    with open(p1_path) as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        assert "j_eq16" in cols, "p1_tangent_all.csv missing j_eq16 -- item 1 not complete"
        n = 0
        n_eq16_valid = 0
        objs = set()
        for row in r:
            n += 1
            objs.add(row["obj"])
            try:
                float(row["j_eq16"])
                n_eq16_valid += 1
            except ValueError:
                pass
    info = dict(path=str(p1_path), sha256=sha256(p1_path), n_rows=n,
                n_objects=len(objs), n_j_eq16_valid=n_eq16_valid,
                j_eq16_coverage=n_eq16_valid / n)
    print(f"[verify item1] {p1_path.name}: {n} rows, {len(objs)} objects, "
          f"j_eq16 coverage {info['j_eq16_coverage']:.4f}, sha256={info['sha256'][:16]}...")
    assert info["j_eq16_coverage"] == 1.0, "j_eq16 not fully covered -- do not proceed"
    return info


# ------------------------------------------------------- source common-nominal --
def source_nominal_predictors(geo, ea, ep):
    """Mirrors p1_rgbsweep_metric_compare.py::nominal_predictors, applied to a
    SOURCE (flow_gaussian) estimate instead of a target (RGB) one -- same q0=0,
    same STEP_DEG stepping to the object's own target_deg, no jitter."""
    spec, sgn = geo["spec"], geo["sgn"]
    gt_axis_w, gt_pivot_w, handle_start = geo["gt_axis_w"], geo["gt_pivot_w"], geo["handle_rest_w"]
    plan_total = sgn * np.radians(spec.target_deg)
    n_steps = max(1, int(abs(np.degrees(plan_total)) / STEP_DEG))
    step_sign = np.sign(plan_total) if plan_total != 0 else 1.0
    rel_angles = step_sign * np.arange(1, n_steps + 1) * np.radians(STEP_DEG)
    e_geo = PB.geometric_path_deviation_cm(gt_axis_w, gt_pivot_w, ea, ep, handle_start, rel_angles)
    e_norm = PB.normal_path_deviation_cm(gt_axis_w, gt_pivot_w, ea, ep, handle_start, rel_angles)
    return e_geo, e_norm


def build_source(exclude_objects: set[str]):
    """Recomputes common-nominal e_geo/e_norm for every (obj,selector,src_row)
    mode-A estimate NOT in exclude_objects (i.e. the 61-object source pool),
    reusing j_cfg/j_eq16 from p1_tangent_all.csv unchanged (already full-range
    nominal in both domains -- see module docstring)."""
    p1_path = OUT / "p1_tangent_all.csv"
    rows_by_key: dict[tuple, list[dict]] = {}
    all_rows = []
    with open(p1_path) as f:
        for row in csv.DictReader(f):
            if row["kind"] != "revolute" or row["mode"] != "A":
                continue
            if row["obj"] in exclude_objects:
                continue
            all_rows.append(row)
            key = (row["obj"], row["selector"], row["src_row"])
            rows_by_key.setdefault(key, []).append(row)

    print(f"[source] {len(all_rows)} mode-A revolute execution rows, "
          f"{len(rows_by_key)} unique estimates, "
          f"{len(set(r['obj'] for r in all_rows))} objects")

    geo_cache: dict[tuple, dict] = {}
    perseed_cache: dict[tuple, list[dict]] = {}
    est_predictors = {}  # key -> dict
    n_fail = 0
    for i, (key, rows) in enumerate(rows_by_key.items()):
        obj, selector, src_row = key
        row0 = rows[0]
        gkey = obj
        if gkey not in geo_cache:
            geo_cache[gkey] = _source_scene_geometry(obj)
        geo = geo_cache[gkey]
        pkey = (obj, selector)
        if pkey not in perseed_cache:
            perseed_p = OUT / f"perseed_errors_{obj}_p0_{selector}_se3.csv"
            perseed_cache[pkey] = list(csv.DictReader(perseed_p.open()))
        src_rows = perseed_cache[pkey]
        try:
            src_row_data = src_rows[int(src_row)]
            # mode "A" ignores trial_seed/b2_phi entirely (see
            # _reconstruct_axis_pivot's mode branches) -- passed as 0/0.0 dummies
            ea, ep = _reconstruct_axis_pivot(geo, src_row_data, "A", 0, 0.0)
            e_geo, e_norm = source_nominal_predictors(geo, ea, ep)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            continue
        est_predictors[key] = dict(
            object_id=obj, source_domain="source_flow_gaussian", selector=selector,
            src_row=src_row, axis_err_deg=float(row0["axis_err"]),
            pivot_err_cm=float(row0["pivot_err"]) if row0["pivot_err"] not in ("", "nan") else 0.0,
            e_geo_common_nominal_cm=e_geo, e_norm_common_nominal_cm=e_norm,
            j_cfg=float(row0["j_full"]), j_eq16=float(row0["j_eq16"]), valid=True)
        if (i + 1) % 500 == 0:
            print(f"  [source] {i+1}/{len(rows_by_key)} estimates recomputed...")

    print(f"[source] recompute done: {len(est_predictors)} ok, {n_fail} failed")

    # join back onto every execution row
    exec_rows = []
    for row in all_rows:
        key = (row["obj"], row["selector"], row["src_row"])
        pred = est_predictors.get(key)
        if pred is None:
            continue
        exec_rows.append(dict(
            object_id=row["obj"], source_domain="source_flow_gaussian",
            estimator=row["estimator"], selector=row["selector"], src_row=row["src_row"],
            rep=row["rep"],
            axis_err_deg=pred["axis_err_deg"], pivot_err_cm=pred["pivot_err_cm"],
            e_geo_common_nominal_cm=pred["e_geo_common_nominal_cm"],
            e_norm_common_nominal_cm=pred["e_norm_common_nominal_cm"],
            j_cfg=pred["j_cfg"], j_eq16=pred["j_eq16"],
            firm=int(row["firm"]), completion85=int(row["rpmart85"]), valid=True,
        ))
    return est_predictors, exec_rows


# ------------------------------------------------------- target common-nominal --
def target_scene_geometry(obj):
    joint = "joint_0"
    spec = PO.ObjSpec(obj, joint)
    devmax, sgn = PO.selfcheck(spec, save=False)
    cid = p.connect(p.DIRECT)
    try:
        bid, jm, lm = PO.build_scene(spec)
        p.resetJointState(bid, jm[spec.joint], 0.0)
        handle_rest_w = PO.handle_world(bid, lm[spec.child], spec.h_local)
        gt_axis_w, gt_pivot_w = PO.gt_axis_pivot_world(bid, jm, spec)
    finally:
        p.disconnect(cid)
    open_lim = abs(spec.hi if sgn > 0 else spec.lo)
    return dict(spec=spec, sgn=sgn, handle_rest_w=np.asarray(handle_rest_w, float),
               gt_axis_w=np.asarray(gt_axis_w, float), gt_pivot_w=np.asarray(gt_pivot_w, float),
               open_lim=open_lim)


def target_nominal_predictors(geo, est_axis, est_pivot):
    spec, sgn = geo["spec"], geo["sgn"]
    gt_axis_w, gt_pivot_w, handle_start = geo["gt_axis_w"], geo["gt_pivot_w"], geo["handle_rest_w"]
    plan_total = sgn * np.radians(spec.target_deg)
    n_steps = max(1, int(abs(np.degrees(plan_total)) / STEP_DEG))
    step_sign = np.sign(plan_total) if plan_total != 0 else 1.0
    rel_angles = step_sign * np.arange(1, n_steps + 1) * np.radians(STEP_DEG)
    est_axis = np.asarray(est_axis, float)
    est_pivot = np.asarray(est_pivot, float)
    e_geo = PB.geometric_path_deviation_cm(gt_axis_w, gt_pivot_w, est_axis, est_pivot, handle_start, rel_angles)
    e_norm = PB.normal_path_deviation_cm(gt_axis_w, gt_pivot_w, est_axis, est_pivot, handle_start, rel_angles)
    j_cfg = PB.tangent_similarity_revolute(gt_axis_w, gt_pivot_w, est_axis, est_pivot, handle_start, 0.0, geo["open_lim"])
    j_eq16 = PB.tangent_similarity_revolute_eq16(gt_axis_w, gt_pivot_w, est_axis, est_pivot, handle_start, 0.0, geo["open_lim"])
    axis_err = PB._ax_deg(gt_axis_w, est_axis)
    pivot_err = PB._piv_cm(est_pivot, gt_axis_w, gt_pivot_w)
    return dict(e_geo_common_nominal_cm=e_geo, e_norm_common_nominal_cm=e_norm,
               j_cfg=j_cfg, j_eq16=j_eq16, axis_err_deg=axis_err, pivot_err_cm=pivot_err)


def build_target():
    cell_predictors = {}
    geo_cache = {}
    for ef in EVAL_FILES:
        cells = json.loads((OUT / ef).read_text())
        for c in cells:
            if c["type"] != "revolute" or c["obj"] not in TARGET_OBJECTS:
                continue
            obj = c["obj"]
            if obj not in geo_cache:
                geo_cache[obj] = target_scene_geometry(obj)
            geo = geo_cache[obj]
            pred = target_nominal_predictors(geo, c["est_axis"], c["est_pivot"])
            cell_predictors[(obj, c["mode"], c["level"])] = pred
    print(f"[target] {len(cell_predictors)} cells across {len(geo_cache)} objects "
          f"(want 144 cells, 12 objects)")

    exec_rows = []
    for tf in TRIAL_FILES:
        with open(OUT / tf) as f:
            for r in csv.DictReader(f):
                if r["obj"] not in TARGET_OBJECTS:
                    continue
                key = (r["obj"], r["mode"], int(r["level"]))
                pred = cell_predictors.get(key)
                if pred is None:
                    continue
                exec_rows.append(dict(
                    object_id=r["obj"], source_domain="target_rgb",
                    corruption=r["mode"], level=r["level"], rep=r["rep"],
                    axis_err_deg=pred["axis_err_deg"], pivot_err_cm=pred["pivot_err_cm"],
                    e_geo_common_nominal_cm=pred["e_geo_common_nominal_cm"],
                    e_norm_common_nominal_cm=pred["e_norm_common_nominal_cm"],
                    j_cfg=pred["j_cfg"], j_eq16=pred["j_eq16"],
                    firm=int(r["firm"] == "True"), completion85=int(r["rpmart85"] == "True"),
                    valid=True,
                ))
    return cell_predictors, exec_rows


def main():
    info1 = verify_item1()

    est_predictors_src, exec_rows_src = build_source(exclude_objects=set(TARGET_OBJECTS))
    cell_predictors_tgt, exec_rows_tgt = build_target()

    src_objs = sorted(set(r["object_id"] for r in exec_rows_src))
    tgt_objs = sorted(set(r["object_id"] for r in exec_rows_tgt))
    intersection = set(src_objs) & set(tgt_objs)
    print(f"source objects: {len(src_objs)}, target objects: {len(tgt_objs)}, "
          f"intersection: {intersection}")
    assert not intersection, "OBJECT LEAKAGE between source and target -- stop"

    with open(RUN_DIR / "source_target_objects.json", "w") as f:
        json.dump(dict(source_objects=src_objs, target_objects=tgt_objs,
                       intersection=sorted(intersection),
                       n_source_estimates=len(est_predictors_src),
                       n_source_execution_rows=len(exec_rows_src),
                       n_target_cells=len(cell_predictors_tgt),
                       n_target_execution_rows=len(exec_rows_tgt)), f, indent=2)

    # predictors_common_nominal.csv: one row per unique estimate/cell (not per execution rep)
    pred_rows = []
    for key, p_ in est_predictors_src.items():
        pred_rows.append(dict(object_id=p_["object_id"], domain="source_flow_gaussian",
                              selector=p_["selector"], condition=p_["src_row"],
                              axis_err_deg=p_["axis_err_deg"], pivot_err_cm=p_["pivot_err_cm"],
                              e_geo_common_nominal_cm=p_["e_geo_common_nominal_cm"],
                              e_norm_common_nominal_cm=p_["e_norm_common_nominal_cm"],
                              j_cfg=p_["j_cfg"], j_eq16=p_["j_eq16"]))
    for (obj, mode, level), p_ in cell_predictors_tgt.items():
        pred_rows.append(dict(object_id=obj, domain="target_rgb",
                              selector="gt_selection", condition=f"{mode}_L{level}",
                              axis_err_deg=p_["axis_err_deg"], pivot_err_cm=p_["pivot_err_cm"],
                              e_geo_common_nominal_cm=p_["e_geo_common_nominal_cm"],
                              e_norm_common_nominal_cm=p_["e_norm_common_nominal_cm"],
                              j_cfg=p_["j_cfg"], j_eq16=p_["j_eq16"]))
    with open(RUN_DIR / "predictors_common_nominal.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        w.writeheader(); w.writerows(pred_rows)
    print(f"-> {RUN_DIR / 'predictors_common_nominal.csv'} ({len(pred_rows)} rows)")

    with open(RUN_DIR / "source_execution_common_nominal.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(exec_rows_src[0].keys()))
        w.writeheader(); w.writerows(exec_rows_src)
    print(f"-> source_execution_common_nominal.csv ({len(exec_rows_src)} rows)")

    with open(RUN_DIR / "target_execution_common_nominal.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(exec_rows_tgt[0].keys()))
        w.writeheader(); w.writerows(exec_rows_tgt)
    print(f"-> target_execution_common_nominal.csv ({len(exec_rows_tgt)} rows)")

    protocol = dict(
        date="2026-09-18",
        protocol_doc="docs/实验方案2_结构化跟踪误差与预算迁移_2026-09-18.md",
        item1_verification=info1,
        common_nominal_convention="q0=0, object's own target_deg via STEP_DEG stepping, "
                                  "no grasp/control jitter, world frame -- identical for "
                                  "both domains (see module docstring for why e_norm/e_geo "
                                  "needed recompute but j_cfg/j_eq16 did not)",
        target_objects=TARGET_OBJECTS,
        manifest_schema_note="practical subset of the doc's listed field set -- see module docstring",
        script_sha256={
            "p2_tracker_transfer_manifest.py": sha256(Path(__file__)),
            "p0_basis.py": sha256(Path(__file__).resolve().parent / "p0_basis.py"),
            "p1_tangent_similarity_offline.py": sha256(Path(__file__).resolve().parent / "p1_tangent_similarity_offline.py"),
        },
    )
    with open(RUN_DIR / "protocol.json", "w") as f:
        json.dump(protocol, f, indent=2)
    print(f"-> {RUN_DIR / 'protocol.json'}")

    input_manifest = dict(
        source_files=["out/p1_tangent_all.csv", "out/perseed_errors_*_p0_*_se3.csv"],
        target_files=EVAL_FILES + TRIAL_FILES,
    )
    with open(RUN_DIR / "input_manifest.json", "w") as f:
        json.dump(input_manifest, f, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
