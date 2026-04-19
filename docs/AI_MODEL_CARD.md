# AI Model Card — GridSense-AZ / GWNet v0

## Model summary

GridSense-AZ uses a compact **Graph WaveNet** (Wu et al., IJCAI 2019) augmented
with a 3-quantile head that emits p10, p50, and p90 simultaneously under a
pinball loss. The network has **59,890 parameters** and was trained for
**200 epochs** on a single **NVIDIA A100 (40 GB, Colab Pro)**, wall clock
**547.9 s** (2.74 s/epoch).

The hero checkpoint lives at `data/models/gwnet_v0.pt` (263 KB, tracked in git).
Training telemetry and final scalars are in
[`data/models/metrics.json`](../data/models/metrics.json) and per-epoch history
in [`data/models/history.json`](../data/models/history.json). The long-form
evaluation write-up is [`reports/gwnet_v1.md`](../reports/gwnet_v1.md).

## Why Graph WaveNet

We evaluated four architectural families against the IEEE 123-bus forecasting
problem:

| Option | Why rejected |
|---|---|
| LSTM / GRU per-bus | 132 independent recurrent nets ignore spatial coupling; O(N) parameter blow-up; slow on sequences > 24 h |
| Plain GCN + temporal MLP | Requires a hand-crafted adjacency; the "true" electrical adjacency of IEEE 123 is switch-dependent and not directly available in the DSS master file |
| Vanilla Transformer | ~3–5× more params for a 132-node feeder; attention is overkill when temporal receptive field of 48 h is covered by a dilated conv stack with ~60 k params |
| **Graph WaveNet** | **Learns adjacency from data** via the `softmax(ReLU(E1·E2ᵀ))` trick; dilated causal convolutions cover long horizons without recurrence; compact enough (< 60 k params) to ship inside a CPU-friendly inference container |

The **IEEE 123 feeder has 132 buses** in our ingested topology (the extra nine
come from regulator / switch taps the reference master file exposes). Graph
WaveNet's adaptive adjacency lets the model discover non-obvious correlations
(e.g. two buses downstream of the same regulator behave as a unit even though
they're topologically distant).

## Input features

Lookback **48 h** (input history) → horizon **24 h** (four rolled 6-h
predictions). Per time step:

- **11 exogenous features** (shared across all buses):
  `temp_c, dewpoint_c, wind_mps, slp_hpa, hour_sin, hour_cos, dow_sin, dow_cos,
  is_weekend, month_sin, month_cos`
- **1 node feature per bus**: last-hour load, z-scored using train-set
  mean/std (`μ = 30,560.74 kW`, `σ = 43,357.15 kW`).

Cyclical calendar features use sin/cos pairs so the network sees Dec→Jan and
23→00 as continuous transitions. Weekend flag is kept as a plain binary
because the weekly cycle in Phoenix load is not purely periodic.

## Output

Per hour, per bus, three quantiles:

- **132 buses × 24 hours × 3 quantiles = 9,504 scalars per forecast run**
- Feeder-level p50 = `sum_over_buses(p50)` per hour (the dashboard's
  "Forecast Ribbon" chart).
- Quantiles are sorted along the Q axis at predict time (not inside the
  training loss) so the network can freely learn order during training.

## Training data

| Field | Value |
|---|---|
| Source | Real (NOAA ISD KPHX weather + EIA-930 AZPS demand) |
| Window | 2022-06-01 → 2023-10-01 |
| T (hourly steps) | 11,688 |
| N (buses) | 132 |
| Split (chronological) | 8,152 train / 1,724 val / 1,725 test windows |
| Target | Per-bus kW, disaggregated from AZPS system demand by each bus's nominal nameplate kW share |
| Imputation | 38 / 11,688 missing exogenous hours forward-filled |

## Loss — pinball (quantile regression)

For quantile `q ∈ {0.1, 0.5, 0.9}`:

```
L_q(y, ŷ_q) = max( q · (y − ŷ_q), (q − 1) · (y − ŷ_q) )
```

The training objective is the sum over all three quantiles. At `q = 0.5` the
pinball loss is exactly half-MAE, giving the p50 its own useful gradient.

**Why pinball instead of MSE:** grid operators do not act on a point forecast
alone — they act on the confidence interval. Pinball loss produces calibrated
quantiles directly, so the resulting p10/p90 band can be read as "with ~80%
probability, actual load will fall in this range." MSE would only give a mean
and require a separate post-hoc uncertainty model (Gaussian residuals,
bootstrapped ensembles, etc.).

## Metrics (from `data/models/metrics.json`)

All values on the raw kW scale at the p50 quantile.

| Split | MAE (kW) | MAE (MW) |
|---|---:|---:|
| train | 2,370.12 | 2.37 |
| val | 3,525.18 | 3.53 |
| **test** | **4,574.04** | **4.57** |
| persistence baseline (test) | 5,603.75 | 5.60 |

**Improvement vs persistence: +18.38%** (see
[`data/models/metrics.json`](../data/models/metrics.json) field
`improvement_pct`).

### Stress-window disclosure (from `data/models/eval_report.json`)

Honest numbers, not hidden:

| Window | MAE (kW) |
|---|---:|
| Overall model | 4,573.70 |
| Overall persistence | 5,603.75 |
| **Stress window (model)** — summer evenings 17:00–21:00 local, Jun–Sep | **6,530.27** |
| **Stress window (persistence)** | **4,038.77** |
| Stress-window improvement | **−61.69%** (model is *worse* than persistence) |

**Interpretation:** GWNet v0 beats persistence in aggregate (+18.4%) but
**underperforms persistence on the specific summer-evening ramp** (17:00–21:00,
Jun–Sep — the moment when Phoenix AC load peaks and the grid actually cares).
This is a known limitation of a two-summer corpus trained without explicit
extreme-event weighting. We flag it rather than hide it. Remediation path:
weighted sampling of stress windows, explicit extreme-quantile head, longer
corpus covering the 2020 & 2024 heat domes.

### Quantile calibration (expected)

A well-calibrated quantile model should cover:

- p10 → actual below 10% of the time
- p50 → actual below 50% of the time
- p90 → actual below 90% of the time (i.e. p10–p90 band catches ~80%)

Empirical coverage checks were not frozen into metrics.json; running the
predictor over test windows via `scripts/precompute_forecasts.py` shows
p10–p90 widths that scale with weekend/weekday and temperature swings,
consistent with the pinball objective.

## Confidence interpretation

```
uncertainty_mw = p90 − p10
```

- **Narrow band (small uncertainty_mw):** model is confident — typical weekday,
  familiar weather.
- **Wide band:** model is uncertain — weekend, holiday, unusual weather, hours
  outside the training distribution.

The dashboard renders p10 / p50 / p90 as three layered curves in the Forecast
Ribbon and surfaces `peak_mw ± (p90−p10)/2` in the top nav.

## Baselines

1. **Hourly persistence** — `ŷ[t + h] = y[t − 1]` for every horizon step
   `h ∈ [1, 24]`. Hard to beat on hourly power data because autocorrelation at
   lag 1 is ~0.95 on AZPS demand.
2. **Seasonal naïve** — `ŷ[t + h] = y[t + h − 168]` (same hour last week).
   Implicitly baked into the val/test split because the test window inherits
   the previous week's pattern. Not separately reported in metrics.json, but
   shown as a dashed line on the Forecast Ribbon where relevant.

## Limitations

1. **Synthetic topology.** The IEEE 123-bus test feeder is a standard research
   feeder; it is **not** APS's real distribution network. Per-bus values are
   plausible but not ground-truth.
2. **Stress-window underperformance.** The model is worse than persistence
   during summer-evening ramp hours. Already flagged above.
3. **No feeder-switching state input.** Reconfiguration events (e.g. tie
   switches closing) would invalidate the learned adjacency. The current
   model has no feature indicating switch state.
4. **No NSRDB irradiance or EV-Pro features.** Data pullers 404'd during the
   hackathon window; the feature bundle falls back to temp/dewpoint/wind.
5. **6-hour native horizon, 24-hour via rolling.** The model was trained with
   `T_out = 6` and is rolled 4× for day-ahead. A native 24-hour horizon would
   need retraining.

## Reproduction

One command on A100:

```bash
python scripts/train.py \
  --start 2022-06-01 --end 2023-10-01 \
  --epochs 200 --batch-size 128 --lr 2e-3 \
  --scheduler cosine --warmup-epochs 10 \
  --t-in 24 --t-out 6 --seed 1337 \
  --device cuda --out-dir data/models
```

Seed `1337` is pinned throughout the pipeline (torch, numpy, python).
