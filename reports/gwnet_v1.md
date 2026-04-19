# GWNet v1 — hourly load forecast evaluation

Hero model for GridSense-AZ: a compact Graph WaveNet with an adaptive adjacency and a 3-quantile head (p10 / p50 / p90). This report documents the real 200-epoch A100 run whose artefacts live under [`data/models/`](../data/models/).

Raw numbers this write-up pulls from:

- [`data/models/metrics.json`](../data/models/metrics.json) — final scalar metrics + full config.
- [`data/models/history.json`](../data/models/history.json) — per-epoch train/val pinball loss and LR schedule.
- [`data/models/train.log`](../data/models/train.log) — training log.
- [`data/models/gwnet_v0.pt`](../data/models/gwnet_v0.pt) — serialised checkpoint.

---

## Dataset

| field | value |
|---|---|
| source              | real (NOAA ISD KPHX weather + EIA-930 AZPS demand) |
| window              | 2022-06-01 → 2023-10-01 |
| T (hourly steps)    | 11 688 |
| N (buses)           | 132 (IEEE 123-bus loaded onto `data/raw/ieee123/`) |
| F_exog              | 11 (weather + calendar cyclical features) |
| F_node              | 1 (z-scored per-bus load) |
| target              | raw kW per bus, disaggregated from AZPS system demand via each bus's nominal-kW share of the feeder's total nameplate load |
| target scaler (μ, σ) | (30 560.74, 43 357.15) kW |
| split               | chronological 70 / 15 / 15 → 8 152 train / 1 724 val / 1 725 test windows |
| imputation          | 38 of 11 688 hours had missing exogenous values; forward-filled |

Features come out of `gridsense.features.build_hourly_features`. The split is leakage-safe by construction — each subset's window-start indices live inside disjoint time intervals, so a train window's target horizon never overlaps a val window's input history.

---

## Model

Compact Graph WaveNet. Per-block stack: GLU-gated dilated causal conv → adaptive adjacency mixing across the node axis → residual + BatchNorm. Dilation doubles per block.

| hyperparameter | value |
|---|---|
| architecture            | Graph WaveNet (compact) with learned adaptive adjacency |
| d_hidden                | 32 |
| blocks × layers/block   | 4 × 2 |
| dilations (per block)   | 1, 2, 4, 8 |
| kernel size             | 2 |
| dropout                 | 0.1 |
| adjacency               | learned `softmax(ReLU(E1 · E2ᵀ))` with `emb_dim = d_hidden // 2 = 16` |
| T_in (input history)    | 24 hours |
| T_out (horizon)         | 6 hours |
| quantile head           | Conv2d → `[B, T_out, N, 3]` with quantiles = (0.1, 0.5, 0.9) |
| loss                    | **pinball loss over all 3 quantiles** (see `gridsense.model.pinball_loss`) — pure quantile regression, no separate MAE term |
| monotonic guard         | quantiles sorted along the Q axis at *predict* time (kept out of the training loss so the network can freely learn ordering) |
| total parameters        | **59 890** |

---

## Training

| setting | value |
|---|---|
| epochs           | 200 |
| optimiser        | Adam |
| lr               | 2e-3 |
| schedule         | cosine (PyTorch `CosineAnnealingLR`) with 10-epoch linear warmup from 1% → 100% of target lr, via `SequentialLR` |
| warmup final lr  | 1.802e-3 at epoch 10 (before the cosine phase clicks in at epoch 11 with lr = 2e-3) |
| final lr         | 1.367e-7 at epoch 200 |
| batch size       | 128 |
| seed             | 1337 |
| device           | cuda (Nvidia A100 40 GB, Colab Pro) |
| wall clock       | **547.9 s total** (2.74 s/epoch) |

Loss trajectory (from `history.json`):

| epoch | train pinball | val pinball |
|---:|---:|---:|
|   1 | 14 058.4 | 16 860.3 |
|  10 | 12 744.3 | 15 260.2 |
|  50 |  2 224.5 |  3 039.8 |
| 100 |    936.5 |  1 520.0 |
| 200 |    766.8 |  1 238.8 |

The curve is monotonically decreasing through epoch 100 and plateaus from ~epoch 150 onward — cosine anneal does its job, and the remaining val-loss slack is model capacity, not optimisation.

---

## Results

All MAE numbers below are on the **raw kW scale at the p50 quantile**, evaluated over the held-out test windows.

| split | MAE (kW) | MAE (MW) |
|---|---:|---:|
| train | 2 370.12 | 2.37 |
| val   | 3 525.18 | 3.53 |
| **test** | **4 574.04** | **4.57** |
| persistence baseline (test) | 5 603.75 | 5.60 |

**Improvement vs persistence on test: +18.38%.**

The persistence baseline is the honest "just repeat the last observed value" forecast: `ŷ[t + h] = y[t − 1]` for every horizon step `h ∈ [1, 6]`. Anything below that line is learning something real.

### Interpretation

- +18% MAE reduction over persistence is **meaningful for a 6-hour horizon on an aggregate grid feed**. Persistence is surprisingly hard to beat on hourly power data because short-horizon autocorrelation is extremely high; breaking it by ~18% means the model is picking up real weather, diurnal, and weekly signal beyond "yesterday was like today."
- There is a **val → test gap of ~1 MW (3.53 → 4.57 MW MAE)**. That is mild overfit / distributional drift: the val window is late summer 2023, the test window is the tail of our series (Sep 2023) and shows a slightly different load mixture as Phoenix rolls out of peak cooling season. We are not claiming the model generalises perfectly across seasons — it is specifically tuned on a two-summer corpus.
- End-to-end sanity: running the saved checkpoint through the predictor on a 2023-08-01 → 2023-08-08 bundle yields **system-total p50 forecasts in the 4 987 – 5 082 MW band across the 6-hour horizon**, which matches AZPS's real summer-afternoon load envelope.

---

## Limitations

1. **Per-bus target is synthetic disaggregation, not metering.** We take AZPS system demand from EIA-930 and split it across the 132 buses by each bus's nominal nameplate kW share. This is a sensible-but-not-physical prior; we do not have real AMI at feeder level. The point forecast therefore predicts a physically plausible but statistically generated per-bus load, and the aggregate MAE should be read as "system-shape MAE," not "feeder-shape MAE."
2. **NSRDB solar irradiance and EVI-Pro EV load are unavailable.** Both endpoints are currently 404/400-ing against our credentials (NSRDB PSM3 returns 400 on the v2 points endpoint; EVI-Pro Lite's county-profile endpoint returns 404 for Maricopa County under our key). The feature bundle falls back to NOAA temp/dewpoint/humidity/wind + calendar features — the demand-driving signals. Adding irradiance and EV profiles is expected to help the val→test gap but is not blocking this release.
3. **6-hour horizon cap.** The model is configured for `T_out = 6`. Longer-horizon claims are not supported by this run; extending would require retraining (cheap — 2.74 s/epoch on A100) and re-tuning the cosine schedule.
4. **Loss is pure pinball.** We intentionally did not add an MAE or Huber term on the p50 slot; the pinball loss at q=0.5 reduces to half-MAE, which gives the p50 its own gradient contribution. If coverage calibration slips in v2 we will revisit.

---

## Reproduce

### Full pipeline, A100 / Colab Pro

1. Open [`notebooks/colab_train_gwnet.ipynb`](../notebooks/colab_train_gwnet.ipynb) in Colab Pro, select the A100 runtime.
2. The notebook clones the repo, installs deps, runs `scripts/train.py`, and zips `data/models/` back.
3. Expect ~9 minutes wall clock end-to-end.

### Just the trainer, locally or on your own GPU

```bash
python scripts/train.py \
    --start 2022-06-01 \
    --end 2023-10-01 \
    --epochs 200 \
    --batch-size 128 \
    --lr 2e-3 \
    --scheduler cosine \
    --warmup-epochs 10 \
    --t-in 24 \
    --t-out 6 \
    --seed 1337 \
    --device cuda \
    --out-dir data/models
```

The run writes `gwnet_v0.pt`, `metrics.json`, and `history.json` into `--out-dir`. On an A100 this takes 547.9 s; on CPU expect roughly 2 – 3 hours.

### Point-check a trained checkpoint

```bash
python -c "
import torch
from gridsense.model import GWNet, GWNetConfig
ckpt = torch.load('data/models/gwnet_v0.pt', map_location='cpu', weights_only=False)
m = GWNet(GWNetConfig(**ckpt['config']))
m.load_state_dict(ckpt['state_dict'])
print('loaded', sum(p.numel() for p in m.parameters()), 'params')
"
```

---

## Pointers for v2

- Restore NSRDB PSM3 and EVI-Pro pullers once the upstream endpoints are reachable — adds irradiance + EV load features.
- Larger `d_hidden` (64) + 6 blocks to see if the val→test gap closes with more capacity.
- Coverage diagnostics: compute p10/p90 empirical coverage on test and reliability plot.
- Swap synthetic disaggregation for ResStock-derived per-feeder profiles once the ResStock puller is wired into features.
