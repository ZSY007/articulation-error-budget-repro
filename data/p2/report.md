# 方案2 E2-A/E2-B results (2026-09-18, common-support fix applied same evening)

**Common-support correction**: the 3 primary paired comparisons below use only the 6/12 objects valid across ALL 6 predictors x 2 endpoints simultaneously (1728 rows, 50.0% coverage) -- an earlier version of this script let each predictor drop its own abstained objects independently before pairing, which is not a valid paired comparison when abstention patterns differ by predictor (caught by an independent audit). Excluded objects: [np.str_('101593'), np.str_('102301'), np.str_('40147'), np.str_('47315'), np.str_('7236'), np.str_('7292')].

## Primary comparisons (Bonferroni 98.333% CI, family-wise 5% over 3 tests)

| comparison | point | 95% CI | Bonferroni 98.333% CI |
|---|---:|---|---|
| e_norm_minus_J_eq16__firm__AUROC_diff | 0.0789 | [0.0097,0.1568] | [0.0059,0.1656] |
| e_norm_minus_J_eq16__completion85__AUROC_diff | -0.0571 | [-0.2000,0.0000] | [-0.2676,0.0000] |
| diff_of_diffs__firm_minus_completion85 | 0.1360 | [0.0129,0.3101] | [0.0051,0.3846] |

## E2-B primary test: e_norm -> firm, r=0.90

- accepted: 1104 executions, 6 objects, 46 estimates
- mean predicted - actual success rate: 95% CI [-0.0140,-0.0021] (pass needs both bounds within +-0.05, n_objects>=6, n_estimates>=24)
- threshold sub-check (CI within +-0.05 and coverage): PASS
- Brier(frozen) - Brier(source const baseline) 95% CI upper bound < 0: PASS
- **operational verdict: PASS** (5-percentage-point threshold is this study's own working standard, not a field-standard safety guarantee)

See metrics.csv / paired_ci.csv / coverage.csv / acceptance.csv / predictions_rgb_looo.csv / predictions_source_frozen.csv / calibration.png for full numeric detail.
