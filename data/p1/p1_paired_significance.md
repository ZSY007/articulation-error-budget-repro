# Paired object-level bootstrap significance test (e_norm vs J_full, e_norm vs J_eq16)

n_boot=2000, same folds/seed as Table V-1 (`p1_metric_comparison_cv.py`, RNG_SEED=12345). Positive AUROC diff / positive Brier diff both favor e_norm.
`J_full` = `J_cfg` (this paper's original implementation); `J_eq16` = the formula-corrected implementation (Section III-B' correction note, 2026-09-18).

# e_norm vs J_full

## firm

- point estimate: AUROC(e_norm) - AUROC(J_full) = 0.0222
- paired bootstrap 95% CI of the difference: [0.0120, 0.0345]
- approx two-sided p-value bound (bootstrap sign test, 2000 draws -- a value of exactly 0 means NO reversal was observed, bounding p<1/2000, not measuring it exactly): 0
- fraction of bootstrap draws favoring e_norm: 1.000
- Brier(J_full) - Brier(e_norm) median [95% CI]: 0.0421 [0.0280, 0.0589]

## rpmart85

- point estimate: AUROC(e_norm) - AUROC(J_full) = -0.0424
- paired bootstrap 95% CI of the difference: [-0.0694, -0.0182]
- approx two-sided p-value bound (bootstrap sign test, 2000 draws -- a value of exactly 0 means NO reversal was observed, bounding p<1/2000, not measuring it exactly): 0.001
- fraction of bootstrap draws favoring e_norm: 0.001
- Brier(J_full) - Brier(e_norm) median [95% CI]: -0.0546 [-0.0796, -0.0331]

# e_norm vs J_eq16

## firm

- point estimate: AUROC(e_norm) - AUROC(J_eq16) = 0.0208
- paired bootstrap 95% CI of the difference: [0.0127, 0.0304]
- approx two-sided p-value bound (bootstrap sign test, 2000 draws -- a value of exactly 0 means NO reversal was observed, bounding p<1/2000, not measuring it exactly): 0
- fraction of bootstrap draws favoring e_norm: 1.000
- Brier(J_eq16) - Brier(e_norm) median [95% CI]: 0.0453 [0.0327, 0.0601]

## rpmart85

- point estimate: AUROC(e_norm) - AUROC(J_eq16) = -0.0401
- paired bootstrap 95% CI of the difference: [-0.0639, -0.0213]
- approx two-sided p-value bound (bootstrap sign test, 2000 draws -- a value of exactly 0 means NO reversal was observed, bounding p<1/2000, not measuring it exactly): 0
- fraction of bootstrap draws favoring e_norm: 0.000
- Brier(J_eq16) - Brier(e_norm) median [95% CI]: -0.0358 [-0.0565, -0.0177]

