# E3-C compensation vs. E3-B baseline, paired (2026-09-18)

Matched 1024 compensated trials to their exact E3-B baseline (same object/condition/H/rho/rep, identical physical realization).

**Overall**: progress +0.2160 (95% CI [0.1931,0.2320]), tracking-gate diff -0.0117 (95% CI [-0.0352,0.0000]), stall-gate diff +0.0010 (95% CI [0.0000,0.0029]).

| condition | H | rho | n_obj | Δprogress | 95% CI | ΔT | 95% CI | ΔQ | 95% CI |
|---|---|---|---|---:|---|---:|---|---:|---|
| axis | 20 | 0.2 | 8 | +0.1530 | [0.1234,0.1712] | +0.0000 | [0.0000,0.0000] | +0.0000 | [0.0000,0.0000] |
| axis | 20 | 0.4 | 8 | +0.3046 | [0.2440,0.3414] | +0.0000 | [0.0000,0.0000] | +0.0000 | [0.0000,0.0000] |
| axis | 40 | 0.2 | 8 | +0.1227 | [0.1014,0.1358] | -0.0312 | [-0.0938,0.0000] | +0.0078 | [0.0000,0.0234] |
| axis | 40 | 0.4 | 8 | +0.2463 | [0.2028,0.2734] | -0.0625 | [-0.1875,0.0000] | +0.0000 | [0.0000,0.0000] |
| pivot | 20 | 0.2 | 8 | +0.1676 | [0.1621,0.1751] | +0.0000 | [0.0000,0.0000] | +0.0000 | [0.0000,0.0000] |
| pivot | 20 | 0.4 | 8 | +0.3351 | [0.3254,0.3462] | +0.0000 | [0.0000,0.0000] | +0.0000 | [0.0000,0.0000] |
| pivot | 40 | 0.2 | 8 | +0.1318 | [0.1274,0.1359] | +0.0000 | [0.0000,0.0000] | +0.0000 | [0.0000,0.0000] |
| pivot | 40 | 0.4 | 8 | +0.2666 | [0.2566,0.2789] | +0.0000 | [0.0000,0.0000] | +0.0000 | [0.0000,0.0000] |

**Verdict**: pooled overall, compensation looks like a cost-free fix by this study's own +-0.05 working threshold (progress CI excludes 0, gate diffs within +-0.05). But per-cell, 2/8 cells have a tracking or stall gate-diff CI that does NOT stay within +-0.05: axis/H=40/rho=0.2, axis/H=40/rho=0.4. These are the wrong-axis-estimate cells at the longer horizon, consistent with the doc's own warning that compensation can fail under a wrong axis estimate. Per-cell is the more honest unit of claim than the pooled number here.
