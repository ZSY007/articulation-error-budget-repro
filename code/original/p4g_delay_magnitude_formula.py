"""Formula-fitting follow-up to p4f_joint_hd_budget_validate.py's interaction finding
(2026-09-18): that run showed the delay-caused firm-rate gap grows with perturbation
magnitude at ONE d/H ratio (0.1875) for two horizons. To fit an actual functional form
for gap(magnitude, d/H) rather than describe one curve qualitatively, this holds H=16
fixed and sweeps delay across FIVE values (d=0..5, d/H=0..0.3125) x the full 13-point
magnitude grid -- a proper 2D (magnitude x delay) response surface on one horizon,
self-contained, reusing p4f's exact seed scheme/object pool/method so it is directly
comparable.

    conda activate cv   (local) / ee900-prime (JARVIS)
    python error_budget/p4g_delay_magnitude_formula.py --procs 16 --reps 8
    -> out/p4g_delay_magnitude_trials.csv, out/p4g_delay_magnitude_fit.md
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4f_joint_hd_budget_validate import (                             # noqa: E402
    PRED_PATH, PRED_SHA, run_config, stable_seed,
)
import hashlib                                                         # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
H = 16
DELAYS = [0, 1, 2, 3, 4, 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--reps", type=int, default=8)
    args = ap.parse_args()

    current_sha = hashlib.sha256(PRED_PATH.read_bytes()).hexdigest().upper()
    if current_sha != PRED_SHA:
        raise ValueError(f"prediction CSV changed: {current_sha} != {PRED_SHA}")
    predictions = list(csv.DictReader(PRED_PATH.open(encoding="utf-8")))

    all_rows = []
    for d in DELAYS:
        name = f"H{H}_d{d}"
        print(f"=== {name}: H={H} delay={d} d/H={d/H:.4f} ===")
        rows = run_config(predictions, name, H, d, args.reps, args.procs)
        all_rows.extend(rows)

    OUT.mkdir(exist_ok=True)
    trials_p = OUT / "p4g_delay_magnitude_trials.csv"
    with trials_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in all_rows for k in r}))
        w.writeheader(); w.writerows(all_rows)
    print(f"-> {trials_p} ({len(all_rows)} trials)")

    # pooled firm-rate surface F(multiplier, delay), and gap(multiplier, delay) = F(m,0)-F(m,d)
    from collections import defaultdict
    by_md = defaultdict(list)
    for r in all_rows:
        by_md[(float(r["multiplier"]), int(r["delay"]))].append(int(r["firm"]))
    mults = sorted({m for (m, d) in by_md})
    surface = {(m, d): float(np.mean(by_md[(m, d)])) for (m, d) in by_md}

    lines = ["# Delay x magnitude formula fit (H=16, 2026-09-18)", "",
            "## Pooled firm-rate surface F(multiplier, delay)", "",
            "| multiplier | " + " | ".join(f"d={d}" for d in DELAYS) + " |",
            "|---|" + "---:|" * len(DELAYS)]
    for m in mults:
        cells = [f"{surface[(m, d)]:.4f}" for d in DELAYS]
        lines.append(f"| {m:.2f} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Gap(multiplier, delay) = F(multiplier,0) - F(multiplier,delay)")
    lines.append("")
    lines.append("| multiplier | " + " | ".join(f"d={d}" for d in DELAYS[1:]) + " |")
    lines.append("|---|" + "---:|" * (len(DELAYS) - 1))
    gap_rows = []
    for m in mults:
        base = surface[(m, 0)]
        cells = []
        for d in DELAYS[1:]:
            gap = base - surface[(m, d)]
            cells.append(f"{gap:.4f}")
            gap_rows.append((m, d, d / H, gap))
        lines.append(f"| {m:.2f} | " + " | ".join(cells) + " |")
    lines.append("")

    # candidate functional forms: gap ~ a*(d/H) + b*(d/H)*m  (bilinear, interaction term explicit)
    X = np.array([[rho, rho * m] for (m, d, rho, gap) in gap_rows])
    y = np.array([gap for (m, d, rho, gap) in gap_rows])
    coef, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    lines.append(f"## Fitted bilinear model: gap = a*(d/H) + b*(d/H)*multiplier")
    lines.append(f"")
    lines.append(f"- a (pure-delay term, magnitude-independent) = {coef[0]:.4f}")
    lines.append(f"- b (interaction term, per unit multiplier) = {coef[1]:.4f}")
    lines.append(f"- R^2 (no-intercept fit, {len(y)} (multiplier,delay) points) = {r2:.4f}")
    lines.append(f"- interpretation: gap(m, rho) = rho * ({coef[0]:.3f} + {coef[1]:.3f}*m) -- "
                f"a linear-in-rho, linear-in-magnitude interaction, if R^2 is high; if low, this "
                f"functional form is rejected and the true relationship is not bilinear.")

    # also fit power-law-in-magnitude at fixed rho, for comparison
    lines.append("")
    lines.append("## Alternative: fit gap = a*rho + b*rho*m^p (magnitude exponent free)")
    from scipy.optimize import curve_fit
    def model(X, a, b, p):
        rho, m = X
        return a * rho + b * rho * np.power(np.maximum(m, 1e-6), p)
    rho_arr = np.array([rho for (m, d, rho, gap) in gap_rows])
    m_arr = np.array([m for (m, d, rho, gap) in gap_rows])
    try:
        popt, pcov = curve_fit(model, (rho_arr, m_arr), y, p0=[coef[0], coef[1], 1.0], maxfev=10000)
        pred2 = model((rho_arr, m_arr), *popt)
        ss_res2 = float(np.sum((y - pred2) ** 2))
        r2b = 1 - ss_res2 / ss_tot if ss_tot > 0 else float("nan")
        lines.append(f"- a={popt[0]:.4f}, b={popt[1]:.4f}, p(magnitude exponent)={popt[2]:.4f}, R^2={r2b:.4f}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"- fit failed: {e}")

    md_p = OUT / "p4g_delay_magnitude_fit.md"
    md_p.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {md_p}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
