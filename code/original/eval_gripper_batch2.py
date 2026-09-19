"""Gripper sample expansion, batch 2 (2026-09-18, user-requested review-response
item #4: "gripper实验扩样"). Same generic grasp-geometry heuristic, same fixed
gate/success/bug-fix machinery as `eval_gripper_10obj.py` -- reuses its helper
functions directly rather than re-deriving them -- applied to 36 NEW objects
(scanned from the full 73-object revolute library, excluding the 10 already
tested there and the original single-object 35059 pilot) that have a real
handle/knob/bar visual mesh (not the door-edge fallback, which a real
parallel-jaw gripper cannot grasp). Writes to its OWN output files
(gripper_batch2_gate.json / gripper_batch2_sweep.csv) -- does not touch or
overwrite the original 10-object results.

    conda activate cv
    python error_budget/eval_gripper_batch2.py --gate
    python error_budget/eval_gripper_batch2.py --sweep
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import numpy as np

import eval_gripper_10obj as G
import pilot_object as PO

OUT = PO.OUT
OBJS = ["10068", "10489", "10620", "10685", "11178", "11231", "11622", "12036", "12252",
       "40147", "41003", "44781", "45087", "45162", "45189", "45244", "46019", "46033",
       "46037", "46084", "46108", "46120", "46145", "46856", "46859", "47529", "47570",
       "47944", "48063", "48379", "48452", "49042", "49133", "7167", "7263", "7310"]
JOINT_OVERRIDE = {"10068": "joint_1", "10489": "joint_1", "10620": "joint_1",
                  "10685": "joint_1", "11231": "joint_1", "45087": "joint_1"}

N_GATE = G.N_GATE
GATE_THRESH = G.GATE_THRESH


def prep_spec(obj):
    joint = JOINT_OVERRIDE.get(obj, "joint_0")
    spec = PO.ObjSpec(obj, joint)
    dev, sgn = PO.selfcheck(spec, save=False)
    spec._sgn = sgn
    return spec, dev


def gate(procs=6):
    for obj in OBJS:
        spec, dev = prep_spec(obj)
        status = "PASS" if dev < 1e-3 else "FAIL"
        print(f"[{obj}] selfcheck dev={dev*1000:.3f}mm sgn={spec._sgn:+.0f} "
             f"target={spec.target_deg:.1f}deg  [{status}]")

    jobs = []
    for obj in OBJS:
        joint = JOINT_OVERRIDE.get(obj, "joint_0")
        for t in range(N_GATE):
            seed = G.stable_seed("gate_b2", obj, t)
            init_s = float(np.random.default_rng(seed + 1).uniform(0, 5))
            mass_s = float(np.random.default_rng(seed + 2).uniform(0.7, 1.5))
            jobs.append((obj, joint, 0.0, seed, init_s, mass_s, None))

    with mp.Pool(procs) as pool:
        rows = pool.map(G._gripper_worker, jobs)

    by_obj = {}
    for r in rows:
        by_obj.setdefault(r["obj"], []).append(r["task_done"])
    result = {}
    print("\n" + "=" * 60)
    print(f"BATCH-2 GATE results (delta=0, ideal axis, n={N_GATE}/object, "
         f"threshold >={GATE_THRESH}):")
    for obj in OBJS:
        rate = float(np.mean(by_obj[obj]))
        passed = rate >= GATE_THRESH
        result[obj] = dict(gate_rate=rate, passed=passed)
        print(f"  {obj}: task_done rate = {rate:.3f}  [{'PASS' if passed else 'FAIL -- excluded'}]")

    (OUT / "gripper_batch2_gate.json").write_text(json.dumps(result, indent=2))
    print(f"\n-> {OUT / 'gripper_batch2_gate.json'}")
    n_pass = sum(1 for v in result.values() if v["passed"])
    print(f"\n{n_pass}/{len(OBJS)} batch-2 objects passed the gate.")


def sweep(procs=6):
    gate_path = OUT / "gripper_batch2_gate.json"
    if not gate_path.exists():
        print("run --gate first"); return
    gate_result = json.loads(gate_path.read_text())
    passed_objs = [o for o in OBJS if gate_result.get(o, {}).get("passed")]
    print(f"Sweeping {len(passed_objs)} gate-passing batch-2 objects: {passed_objs}")

    jobs_g, jobs_g_generous, jobs_f = [], [], []
    for obj in passed_objs:
        joint = JOINT_OVERRIDE.get(obj, "joint_0")
        for i, delta in enumerate(G.DELTAS):
            for t in range(G.N_SWEEP):
                seed = G.stable_seed(f"sweep_b2_d{i}", obj, t)
                init_s = float(np.random.default_rng(seed + 1).uniform(0, 5))
                mass_s = float(np.random.default_rng(seed + 2).uniform(0.7, 1.5))
                jobs_g.append((obj, joint, float(delta), seed, init_s, mass_s, None))
                jobs_g_generous.append((obj, joint, float(delta), seed, init_s, mass_s,
                                       G.WRIST_FMAX_GENEROUS))
                jobs_f.append((obj, joint, float(delta), seed, init_s, mass_s))

    with mp.Pool(procs) as pool:
        rows_g = pool.map(G._gripper_worker, jobs_g)
    with mp.Pool(procs) as pool:
        rows_gg = pool.map(G._gripper_worker, jobs_g_generous)
    with mp.Pool(procs) as pool:
        rows_f = pool.map(G._firm_worker, jobs_f)

    out_csv = OUT / "gripper_batch2_sweep.csv"
    fieldnames = ["obj", "delta_deg", "seed", "condition", "progress", "max_track_err",
                 "quality_frac", "peak_force", "f_max_used"]
    import csv
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
        return float(xs[-1])

    def curves_and_crossing(rows_by_cond, obj, track_budget_cm):
        out = {}
        for cond, rows in rows_by_cond.items():
            by_delta = {}
            for r in rows:
                if r["obj"] != obj:
                    continue
                succ = G.firm_like(r["progress"], r["max_track_err"], r["quality_frac"], track_budget_cm)
                by_delta.setdefault(r["delta_deg"], []).append(int(succ))
            xs = sorted(by_delta)
            ys = [np.mean(by_delta[x]) for x in xs]
            out[cond] = crossing(xs, ys)
        return out

    print("\n" + "=" * 78)
    print(f"batch-2 per-object 50%-crossing delta_deg, firm_like() threshold sensitivity "
         f"{G.TRACK_BUDGETS_CM} sim-cm:")
    all_rows = dict(gripper_matched_fmax=[dict(obj=r["obj"], delta_deg=r["delta_deg"],
                                                progress=r["progress"], max_track_err=r["max_track_err"],
                                                quality_frac=r["contact_lost_frac"]) for r in rows_g],
                    firm_point=[dict(obj=r["obj"], delta_deg=r["delta_deg"],
                                     progress=r["progress"], max_track_err=r["max_track_err"],
                                     quality_frac=r["sat_frac"]) for r in rows_f])
    widenings_by_budget = {b: [] for b in G.TRACK_BUDGETS_CM}
    for obj in passed_objs:
        line = f"  {obj}: "
        for b in G.TRACK_BUDGETS_CM:
            cross = curves_and_crossing(all_rows, obj, b)
            ratio = cross["gripper_matched_fmax"] / cross["firm_point"] if cross["firm_point"] > 0 else float("nan")
            widenings_by_budget[b].append(ratio)
            line += f"[{b:.0f}cm: gripper={cross['gripper_matched_fmax']:.1f} firm={cross['firm_point']:.1f} ratio={ratio:.2f}x]  "
        print(line)

    print("\nbatch-2 median widening ratio by tracking-error budget (matched f_max):")
    for b in G.TRACK_BUDGETS_CM:
        valid = [w for w in widenings_by_budget[b] if not np.isnan(w)]
        if valid:
            print(f"  {b:.0f}cm budget: median={np.median(valid):.2f}x  (n={len(valid)} objects)")
        else:
            print(f"  {b:.0f}cm budget: no valid crossings")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    if args.gate:
        gate(args.procs)
    elif args.sweep:
        sweep(args.procs)
    else:
        print("pass --gate or --sweep")


if __name__ == "__main__":
    main()
