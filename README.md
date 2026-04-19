# GridSense-AZ

Spatio-temporal AI for Arizona Public Service distribution-grid forecasting.

Graph WaveNet + quantile head (p10/p50/p90) → 24h day-ahead per-bus load forecasts → heat / EV stress scenarios → OpenDSS physics check → Next.js tactical-ops dashboard.

Built for the ASU Energy Hackathon — APS "AI for Energy" Challenge.

**Live demos**
- Dashboard (Vercel): https://gridsense-az.vercel.app
- ML Space (HF): https://dc-ai-labs-gridsense-az.hf.space/

## What's in this repo

| Path | What it is |
|---|---|
| `src/gridsense/` | Python package — model, features, predictor, decision layer, power flow |
| `scripts/` | Data pulls, training entrypoint, NWS fetcher, precompute pipeline |
| `data/models/gwnet_v0.pt` | Trained checkpoint (263 KB). 200 epochs, NVIDIA A100, 547 s wall |
| `data/models/metrics.json` | Test/val/train MAE + training history |
| `data/ieee123/` | IEEE 123-bus test feeder master files + BusCoords |
| `web/` | Next.js 14 dashboard (tactical-ops UI, served on Vercel) |
| `hf_space/` | HuggingFace Space Streamlit fallback |
| `tests/` | pytest unit + integration suites |

## Model card

- **Architecture**: Graph WaveNet (dilated temporal conv + learned adjacency) + 3-quantile head (pinball loss). 59,890 parameters.
- **Training**: 200 epochs, NVIDIA A100, ~548 s wall, batch 128, seed 1337, cosine LR w/ warmup.
- **Data**: IEEE 123-bus topology × NOAA KPHX weather × EIA-930 AZPS demand, hourly 2022-06-01 → 2023-10-01 (11,688 timesteps).
- **Performance (hold-out)**:

  | Metric | Value |
  |---|---|
  | Train MAE | 2,370 kW |
  | Val MAE | 3,525 kW |
  | Test MAE | 4,574 kW |
  | Persistence baseline MAE | 5,604 kW |
  | **Improvement** | **+18.4%** |

## Local run

### Prerequisites
- Python 3.11
- Node.js 20+ with `pnpm` (or `npm`)
- ~2 GB disk for data + venv
- Optional: NREL API key (for NSRDB irradiance), EIA API key (for EIA-930 demand). NWS does NOT require auth.

### Clone
```bash
git clone https://github.com/dc-ai-labs/gridsense-az.git
cd gridsense-az
cp .env.example .env    # Fill in keys if you want live pulls
```

### Python environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Data dependencies
The IEEE 123-bus feeder and pre-trained checkpoint are in git. Raw NOAA/EIA pulls are NOT — re-pull them:
```bash
.venv/bin/python scripts/pull_noaa.py      # KPHX Phoenix Sky Harbor hourly 2022-06 → 2023-10
.venv/bin/python scripts/pull_eia930.py    # AZPS balancing-area hourly demand
# Optional:
.venv/bin/python scripts/pull_nsrdb.py     # NSRDB irradiance (needs NREL key)
```
Or skip and re-use the checkpoint directly via the precompute script below.

### Run the precompute (generates dashboard data from live NWS)
```bash
.venv/bin/python scripts/precompute_forecasts.py --output-dir web/public/data/forecasts
```
This fetches tomorrow's NWS Phoenix hourly forecast, runs the GWNet in rolling 4×6h inference, applies heat + EV scenario transforms, runs OpenDSS snapshots, and drops 5 JSON files into `web/public/data/forecasts/`.

### Run the dashboard
```bash
cd web
pnpm install
pnpm dev        # http://localhost:3000
# or
pnpm build && pnpm start   # production build locally
```

### Run tests
```bash
.venv/bin/python -m pytest -q                 # unit + integration
cd web && pnpm build                          # frontend TS + build
```

## Deploy

### Vercel (dashboard)
```bash
cd web
vercel --prod
```
Already linked to `dc-ai-labs-projects/gridsense-az`. CLI picks it up from `.vercel/project.json`.

### HuggingFace Space (ML playground)
```bash
bash scripts/deploy_hf_space.sh
```
Requires `HF_TOKEN` in `.env`.

## Scenarios — how they work

- **Baseline**: tomorrow's live NWS Phoenix hourly forecast drives the model's exogenous inputs. Encoder sees last 24 h of real history, decoder emits 6 h × 4 rolls = 24 h of p10/p50/p90.
- **Heat**: `temp_c` in the exogenous feature stream is shifted +5.56 °C (+10 °F) BOTH in the encoder history and in the future inputs. A cooling-load multiplier is applied on top during hours 12-22 local. Heat-scenario peak ≥ 1.35× baseline peak.
- **EV**: +720 kW added at 20 deterministically-picked residential buses during hours 17-22 local. Equivalent to ~2000 Level-2 EVSE at 7.2 kW evening-ramp behavior. Peak hour always in [17, 22].
- **OpenDSS**: power-flow snapshot at the scenario's peak hour per scenario — returns bus voltages (p.u.) and line loadings (%) on the IEEE 123 reference feeder to flag topological bottlenecks.

## Architecture in one diagram

```
           NOAA KPHX  EIA-930 AZPS              NWS Phoenix
           (historical)  (historical)           (tomorrow)
                    \       |                   /
                     \      |                  /
                      ▼     ▼                 ▼
                   ┌────────────────────────────┐
                   │  features.build_hourly()   │
                   │     [T × N × F]            │
                   └────────────┬───────────────┘
                                │
                                ▼
                   ┌────────────────────────────┐
                   │ Graph WaveNet (ckpt v0)    │
                   │ 59,890 params · pinball    │
                   │ 24h in → 6h out · roll 4×  │
                   └────────────┬───────────────┘
                                │
                           p10 / p50 / p90 per bus per hour
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
   heat scenario         ev scenario           baseline
   (+10°F + profile)     (+720kW × 20 buses)   (raw)
          │                     │                     │
          └─────────────────────┼─────────────────────┘
                                │
                                ▼
                       OpenDSS snapshot
                       (voltages, loadings)
                                │
                                ▼
                  web/public/data/forecasts/*.json
                                │
                                ▼
                  Next.js dashboard on Vercel
                  (map · ribbon · leaderboard · physics)
```

## License

MIT — see LICENSE.

## Submission team

**Divyansh Chanda** · dchanda1@asu.edu · Arizona State University
