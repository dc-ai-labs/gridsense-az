# Metrics — what we measure and why

This document explains every number surfaced by GridSense-AZ — the ML metrics
computed during training/eval, the risk score rendered in the UI, and how each
maps onto decisions a distribution-operations engineer actually cares about.

## Forecast accuracy metrics

Computed over the held-out test split (1,725 windows, chronologically last
~15% of the 2022-06 → 2023-10 corpus). Code paths:

- Training-time: `scripts/train.py` → writes `data/models/metrics.json`
- Eval-time: [`src/gridsense/eval.py`](../src/gridsense/eval.py)
- Stress-window report: `data/models/eval_report.json`

### MAE — Mean Absolute Error

```
MAE = mean( |y_true − y_p50| )
```

| Split | MAE (kW) | MAE (MW) |
|---|---:|---:|
| train | 2,370.12 | 2.37 |
| val | 3,525.18 | 3.53 |
| **test** | **4,574.04** | **4.57** |
| persistence (test) | 5,603.75 | 5.60 |

Gives an average magnitude of error in physical units. The ~1 MW val→test gap
is mild seasonal drift: val ends mid-summer 2023, test tails into Sep 2023 as
Phoenix leaves peak cooling.

### RMSE — Root Mean Squared Error

```
RMSE = sqrt( mean( (y_true − y_p50)^2 ) )
```

Penalises large errors more than MAE. RMSE is not frozen into metrics.json for
the v0 run but is computed inline by the eval harness for the reliability plot.
Typical ratio RMSE/MAE on this dataset is ~1.35, implying a roughly Gaussian
residual with heavy-ish tails on stress hours.

### MAPE — Mean Absolute Percentage Error

```
MAPE = mean( |y_true − y_p50| / |y_true| ) · 100
```

Useful for normalising across buses of very different base loads (a 500 kW
error on a 50 MW feeder is ~1%; on a 2 MW lateral it's 25%). Per-bus MAPE is
the basis for the risk-leaderboard sort tiebreaker.

## Quantile calibration

A well-calibrated quantile output means the empirical coverage matches the
nominal level.

| Quantile | Expected coverage | What wrong coverage means |
|---:|---:|---|
| p10 | ~10% of actuals below p10 | Too many below → model overconfident on low end; too few → too pessimistic |
| p50 | ~50% of actuals below p50 | Median bias |
| p90 | ~90% of actuals below p90 | Too few below → model overconfident on high end (dangerous — missed peaks) |

Calibration is encouraged by the pinball loss but not guaranteed;
post-deployment we recommend a monthly reliability plot against observed AMI
(advanced-metering-infrastructure) data to re-calibrate if coverage drifts.

## Risk score definition

Rendered on every bus card in the dashboard. Source:
[`scripts/precompute_forecasts.py`](../scripts/precompute_forecasts.py),
function `_compute_per_bus_metrics` (line 410–460). Exact formula:

```python
# per bus i:
rating_kw   = max(peak_load_kw * 1.5, 50.0)             # nameplate rating proxy
peak_share  = peak_load_kw / rating_kw                  # 0..1 utilisation
worst_loading = max line-loading % incident to bus i    # from OpenDSS snapshot
overload_prob = clip( (worst_loading / 150.0) * peak_share, 0, 1 )
vdev_i      = | bus_voltage_pu − 1.0 |                   # ANSI C84.1 deviation

risk_score = clip( 0.6 * overload_prob + 0.4 * (vdev_i / 0.05), 0, 1 )
```

**Weighting rationale:**
- **60% thermal overload probability** — a conductor running >100% of its rating
  is the most urgent failure mode (insulation degradation, fault).
- **40% voltage deviation** — normalised to the ±5% ANSI C84.1 envelope. A bus
  at 0.95 p.u. or 1.05 p.u. scores a full 1.0 on that component.

Thresholds rendered in the UI:

| Range | Tier | UI colour |
|---|---|---|
| `risk_score ≥ 0.7` | primary (hot) | accent teal |
| `0.4 ≤ risk_score < 0.7` | secondary (warm) | amber |
| `risk_score < 0.4` | low | neutral |

See `web/lib/validate.ts::riskTier`.

## Confidence band width

```
uncertainty_mw = p90 − p10   (MW, per hour)
```

- **Narrow band** → high confidence. Typical Tuesday-afternoon baseline runs
  show feeder-total p90−p10 of ~100–150 MW on a ~4,500 MW peak.
- **Wide band** → low confidence. Heat-scenario peaks push uncertainty past
  300 MW because temperatures near / outside the training distribution amplify
  quantile spread.

Operators should treat the band as a dispatch window, not a prediction
interval to ignore.

## Stress-window MAE (honest disclosure)

From `data/models/eval_report.json`:

| Window | MAE (kW) |
|---|---:|
| Overall (model) | 4,573.70 |
| Overall (persistence) | 5,603.75 |
| Overall improvement | **+18.38%** |
| Stress window (model) — summer evenings 17:00–21:00 Jun–Sep | **6,530.27** |
| Stress window (persistence) | 4,038.77 |
| Stress window improvement | **−61.69%** (model loses) |
| Stress hours / total hours | 2,154 / 10,350 |

**Why disclose this?** Because the stress window is exactly when the utility
needs the forecast most. The current model is a genuine improvement
day-in-day-out but the v2 roadmap explicitly targets this regime:

1. Weighted pinball loss (upweight 17:00–21:00 Jun–Sep samples 3–5×).
2. Dedicated high-quantile head (p95/p99) with asymmetric penalty for misses.
3. Retrain on a 5-year corpus that includes the 2023 and 2024 heat-dome
   weeks, not just two half-summers.

## What operators care about — ML metric → operator value

| Operator concern | Measured by |
|---|---|
| "Will I hit the peak at the right hour?" | **Peak-timing error** = \|peak_hour_pred − peak_hour_actual\| in hours. Derived from the Forecast Ribbon; no separate metric, but visually obvious in the UI. |
| "How bad is the peak?" | **Peak-magnitude error** = \|peak_mw_pred − peak_mw_actual\|. Falls out of MAE evaluated only at the peak hour. |
| "How early do I know a bus will overload?" | **Lead time to overload** = hours between forecast generation and first bus with `risk_score ≥ 0.7`. The dashboard's Risk Leaderboard exposes this. |
| "How confident should I be?" | **Confidence band width** `p90 − p10`. Wider → widen dispatch reserves. |
| "Which buses should I watch?" | **Top-K risk buses** by `risk_score`. Rendered on the TacticalMap heat layer and in RiskLeaderboard (K = 10). |

## Regressions we guard against

- **Heat scenario sanity:** `heat.peak_mw ≥ 1.3 × baseline.peak_mw` — asserted
  by [`web/lib/validate.ts`](../web/lib/validate.ts).
- **EV scenario sanity:** `ev.peak_hour ∈ [17, 22]` local — asserted by the
  same validator. If the precompute run produces values outside these bounds,
  the dashboard red-lines and does not render.

Both are cheap property-based tests that catch silent scenario-transform bugs
before judges (or operators) see nonsense.
