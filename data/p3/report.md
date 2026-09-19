# 方案3 E3-B analysis (2026-09-18)

## Primary result 1: progress model comparison (GT condition)

- finite-block ideal-recursion model: object-macro MAE = 0.0046, 95% object-cluster bootstrap CI [0.0025,0.0071]
- asymptotic 1-d/H model: object-macro MAE = 0.0566
- constant-progress (null) baseline: object-macro MAE = 0.1819
- pre-specified threshold (finite-block MAE 95% upper bound <= 0.05): PASS (<=5 percentage points, this study's own working threshold)

## Primary result 2: paired progress/tracking/stall diffs, d=0 -> rho

| condition | H | rho | n_obj | Δprogress | 95% CI | ΔT (tracking gate) | 95% CI | ΔQ (stall gate) | 95% CI |
|---|---|---|---|---:|---|---:|---|---:|---|
| gt | 20 | 0.2 | 8 | -0.1689 | [-0.1716,-0.1666] | 0.0000 | [0.0000,0.0000] | 0.0000 | [0.0000,0.0000] |
| gt | 20 | 0.4 | 8 | -0.3372 | [-0.3425,-0.3335] | 0.0000 | [0.0000,0.0000] | 0.0000 | [0.0000,0.0000] |
| gt | 40 | 0.2 | 8 | -0.1353 | [-0.1380,-0.1331] | 0.0000 | [0.0000,0.0000] | 0.0000 | [0.0000,0.0000] |
| gt | 40 | 0.4 | 8 | -0.2697 | [-0.2741,-0.2669] | 0.0000 | [0.0000,0.0000] | 0.0000 | [0.0000,0.0000] |
| axis | 20 | 0.2 | 8 | -0.1544 | [-0.1707,-0.1280] | 0.0000 | [0.0000,0.0000] | -0.0156 | [-0.0469,0.0000] |
| axis | 20 | 0.4 | 8 | -0.3057 | [-0.3400,-0.2491] | 0.0000 | [0.0000,0.0000] | -0.0156 | [-0.0469,0.0000] |
| axis | 40 | 0.2 | 8 | -0.1232 | [-0.1357,-0.1029] | 0.0000 | [0.0000,0.0000] | 0.0000 | [0.0000,0.0000] |
| axis | 40 | 0.4 | 8 | -0.2451 | [-0.2729,-0.1994] | 0.0000 | [0.0000,0.0000] | 0.0000 | [0.0000,0.0000] |
| pivot | 20 | 0.2 | 8 | -0.1685 | [-0.1743,-0.1637] | 0.0000 | [0.0000,0.0000] | 0.0000 | [0.0000,0.0000] |
| pivot | 20 | 0.4 | 8 | -0.3361 | [-0.3455,-0.3281] | 0.0000 | [0.0000,0.0000] | 0.0000 | [0.0000,0.0000] |
| pivot | 40 | 0.2 | 8 | -0.1323 | [-0.1347,-0.1297] | 0.0000 | [0.0000,0.0000] | 0.0000 | [0.0000,0.0000] |
| pivot | 40 | 0.4 | 8 | -0.2668 | [-0.2758,-0.2588] | 0.0000 | [0.0000,0.0000] | 0.0000 | [0.0000,0.0000] |

## Allowed conclusion, per doc §6

- GT condition at rho=0.40: progress drops, tracking/stall gate diffs both stay within +-0.05 -> supports 'primarily progress loss, these two gates roughly preserved.'
