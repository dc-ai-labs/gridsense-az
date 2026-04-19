# AI for Energy — APS Challenge: Zero-Ambiguity Build Plan (v2)

> **Owner:** Divyansh Chandarana (ASU) · **Manager:** JARVIS · **Sponsor:** Arizona Public Service (APS)
> **Brief file:** `./APS_AI_for_Energy_Technical_Problem_Statement.docx`
> **Plan v2 locked:** 2026-04-18 (v1 archived at `PLAN.v1.md.bak`)
> **Project root:** `/home/divyansh/Downloads/hackathon/energy/` (this folder) — monorepo
> **Product codename:** `GridSense-AZ`

---

## 0. TL;DR for the judges (one paragraph)

> **GridSense-AZ is a spatio-temporal AI system for Arizona Public Service's distribution grid.** It forecasts hourly load for every feeder of an IEEE-123-bus distribution network parameterised with real Phoenix weather (NOAA KPHX) and real AZPS balancing-authority demand (EIA-930), stress-tests the forecast under user-controllable extreme-heat and EV-evening-peak scenarios, and converts predictions into operator-ready interventions via an interactive map-based decision layer. Under the hood: a **Graph WaveNet** with a **3-quantile output head** (p10/p50/p90) trained on Colab Pro L4, a **physics-consistency check** via OpenDSS power flow, and an interactive **Streamlit + pydeck** dashboard deployed on **HuggingFace Spaces** that a utility planner could use the day we ship it.

---

## 0.5 Locked decisions — 2026-04-18 (owner-confirmed)

| Decision | Locked value | Rationale |
|---|---|---|
| Hackathon window | **24 h** solo | owner-confirmed |
| Submission | GitHub repo + deck + reel → mandatory. **Deployed URL → nice-to-have (we ship it).** | brief §Required submission package |
| Deploy target | **HuggingFace Spaces (Streamlit SDK)** · weights on HF Hub under `dchanda/gridsense-az` | Streamlit-native (Vercel serverless can't host long-running OpenDSS calls); 16 GB CPU free; weights co-located |
| Optional: marketing landing | **Vercel (Next.js)** linking to the HF Space | Owner preference; 30-min build, not on critical path |
| Owner responsibilities | Deck + reel + marketing — owner. Code + training + deploy — JARVIS. | owner-confirmed |
| Compute | **Colab Pro (plain, ASU-free tier)** → L4 GPU. Keep training ≤45 min per run, ≤2 runs per model. | owner-confirmed |
| Hero model | **Graph WaveNet with 3-quantile head** (p10/p50/p90) | Graph + quantile in one model; ~300 LOC; ~25 min train on L4; no ensemble needed |
| Baseline (stretch) | DCRNN point forecast | Only if primary run finishes with budget to spare |
| Time resolution | **Hourly** (not 15-min) | Planners use hourly; cuts training cost 4x |
| Forecast horizon | **24 h** (24 steps) | Matches brief "useful for planning or operations" |
| Topology | **IEEE 123-bus** (primary); cite IEEE 34-bus as AZ-native context | 123 = industry DER/EV stress standard; 34 = authentically Arizona |
| Dataset strategy | NOAA KPHX + NREL NSRDB Phoenix + EIA-930 AZPS + NREL ResStock Maricopa + NREL EVI-Pro Lite + IEEE 123 (.dss) + HF EnergyBench fallback | All standard, industry-grade, zero-approval |
| Pecan Street | **Skipped** (3-day approval incompatible with 24h window) | Substituted with HF EnergyBench + ResStock |
| Interpretability | **Integrated Gradients** (Captum) on Graph WaveNet + GNNExplainer (stretch) | No extra training cost; works on any torch model |

---

## 1. Sponsor-intent decoder — what APS *actually* cares about

Decoded from the brief + public APS/SRP filings (Oct 2024 IRP, 2025 ISP) + April 2026 $7 M heat-shutoff settlement context:

| The brief says… | What the sponsor *means*… | How we respond |
|---|---|---|
| "Realistic APS-like distribution network" | Don't hand us a synthetic toy with zero Arizona flavour. | Base topology on IEEE 123-bus; overlay Phoenix weather + AZPS demand curves; cite real APS peak (8,631 MW, Aug 7 2025) in the dashboard; call out that IEEE 34-bus is literally an AZ feeder from 1991 as a nod to authenticity. |
| "Spatio-temporal AI model with learned parameters" | LLM wrappers are out. Show us weights. | Graph WaveNet with quantile head, trained on Colab Pro L4; weights hosted on HF Hub. |
| "At least one stress scenario using extreme heat and/or EV evening peak growth" | The scenario engine is the killer feature. | Build a **what-if simulator** with sliders: ΔT (+0 → +15 °F), EV penetration (0 → 50 %), PV penetration (0 → 60 %). Precompute 75 scenario lookups for instant slider response. |
| "Decision-ready outputs for planners or operators" | If an engineer can't act on it, it doesn't count. | Map-based feeder-level risk score (prob. peak > rating in next 24 h) + top-3 Integrated-Gradient drivers + ranked intervention list ("dispatch battery at bus 812", "switch load to feeder 22 (3.1 MW spare)"). |
| "Clear error metrics" + "good performance during stress periods" | Don't report a single MAPE. | Report MAE, RMSE, MAPE overall and **separately for heat-wave days vs normal days** and for the 4-7 pm evening ramp. Include **quantile reliability diagram** (90 % intervals cover 90 %?). |
| "Scoring implication: Arizona-oriented, stronger heat and EV treatment" | Judges reward AZ-specific storytelling. | Phoenix 2023 heat wave (31 consecutive days ≥110 °F) as validation scenario; 500+ MW/hr duck-curve evening ramp as case study; cite APS EV Overnight TOU + Smart EV Rewards as deployment targets. |
| "GitHub repo + 5-slide deck + 60-90 s reel" | Three-artifact submission. | Owner ships deck + reel; JARVIS ships repo + deploy. |

**What the brief does not say but sponsor clearly wants** (decoded from APS 2023 IRP): alignment with their stated 2.4 % annual peak growth and the 3,000 MW solar + 2,800 MW battery additions by 2027. We frame the decision layer as **capacity-deferral screening** — identifying *which* feeders most need the 2027 upgrades, helping prioritise APS capex.

---

## 2. Product scope — MVP vs stretch

### MVP (must ship, ~14 h of coding + 1-2 training runs)
- IEEE 123-bus distribution feeder model in OpenDSS, parameterised with Phoenix weather-driven load shapes from ResStock AZ.
- Graph WaveNet with 3-quantile head, trained on 123 buses × hourly demand, 2 years of data, 24-h horizon.
- Streamlit + pydeck dashboard with:
  - Feeder map coloured by 24-h peak-load risk
  - Per-feeder p10/p50/p90 forecast ribbon
  - Top-3 Integrated-Gradient drivers per flagged feeder
  - ΔT stress scenario toggle (+10 °F heat wave)
  - Physics-consistency check (OpenDSS power flow → voltage/thermal flags)
  - Metrics tab: heat-wave vs normal MAE + quantile reliability
- Deployed to **HuggingFace Spaces** (public URL in submission).
- GitHub repo with model card, data manifest, README.

### Stretch (if time allows)
- DCRNN point-forecast baseline for "we beat the simpler GNN" plot.
- Full what-if sliders (ΔT × EV% × PV% with precomputed 75-scenario lookup).
- GNNExplainer neighbour-influence plots.
- Zero-shot transferability test (train on 80 buses, eval on 43).
- Foundation-model zero-shot baseline (Chronos-Bolt — CPU-runnable, no extra Colab time).
- Export "feeder upgrade ranking CSV" for APS planners.
- Vercel landing page linking to the HF Space.

### Explicitly out of scope
- AMI-level (customer) forecasts — feeder-level only.
- Real SCADA ingestion — public synthetic inputs only.
- Multi-feeder dispatch optimisation — screening + recommendation only.

---

## 3. Technical architecture

### 3.1 Data flow (end-to-end)

```
┌────────────────────────────────────────────────────────────────────────┐
│ LAYER 1 — PUBLIC DATA (pulled once, cached locally)                    │
│   • NOAA ISD KPHX hourly temp 2019-2025   (no auth)                    │
│   • NREL NSRDB irradiance Phoenix (lat 33.45, lon -112.07) 2019-2023   │
│   • EIA-930 hourly demand for AZPS balancing authority                  │
│   • NREL ResStock AZ residential load profiles (Maricopa, hourly agg)  │
│   • NREL EVI-Pro Lite fleet load profiles (Phoenix climate zone)       │
│   • IEEE 123-bus feeder (.dss from github.com/tshort/OpenDSS)          │
│   • HuggingFace EnergyBench (residential AMI — Pecan Street substitute)│
└─────────────────┬──────────────────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ LAYER 2 — FEATURE PIPELINE (ETL, CPU, pandas/polars)                   │
│   Outputs parquet at data/features/:                                   │
│   • feeder_load_timeseries.parquet  (bus × hour × kW/kVAR)             │
│   • weather_features.parquet        (T, T_lag1..24, GHI, humidity)     │
│   • calendar_features.parquet       (DoW, hour, holidays, heat-wave    │
│                                      flag: ≥3 consecutive days ≥110°) │
│   • topology.pt                     (PyG graph w/ line impedances,     │
│                                      adjacency from .dss)              │
└─────────────────┬──────────────────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ LAYER 3 — MODEL (Colab Pro L4, driven via ml-trainer agent)            │
│   ┌──────────────────────────────────────────────────────────┐         │
│   │  Graph WaveNet + 3-quantile head                         │         │
│   │    — learned adjacency (adaptive) + fixed topology       │         │
│   │    — temporal conv with dilated gated TCN                │         │
│   │    — output: 3 channels (p10, p50, p90), 24 h horizon    │         │
│   │    — ~1.5 M params; ~25 min on L4; pinball + MAE loss    │         │
│   │    — weights: models/gwnet_v1.pt → HF Hub                │         │
│   │                                                          │         │
│   │  [stretch] DCRNN baseline for comparison plot            │         │
│   └──────────────────────────────────────────────────────────┘         │
└─────────────────┬──────────────────────────────────────────────────────┘
                  │  per-feeder p10/p50/p90 kW forecast
                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ LAYER 4 — PHYSICS CHECK (OpenDSS, local CPU, ~sec/scenario)             │
│   For each horizon step t:                                              │
│     • write predicted kW/kVAR into OpenDSS Load objects                 │
│     • Solution.Solve() → per-bus voltage, per-line loading              │
│     • flag voltage violations (ANSI C84.1 ±5 %)                         │
│     • flag thermal violations (loading > 100 % of line rating)          │
│   Outputs: risk_flags.parquet                                            │
└─────────────────┬──────────────────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ LAYER 5 — DECISION ENGINE (Python rules + explainers)                   │
│   For each flagged feeder/time:                                         │
│     • Integrated Gradients (Captum) → top-3 input drivers               │
│     • [stretch] GNNExplainer → neighbour influence map                  │
│     • rule-book lookup: load-transfer options (feeders ≤2 hops),        │
│       DR levers, battery dispatch candidates                            │
│     • emit ranked intervention list                                     │
└─────────────────┬──────────────────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ LAYER 6 — DASHBOARD (Streamlit + pydeck, deployed on HF Spaces)         │
│   • Map: pydeck ScatterplotLayer on feeder centroids, coloured by risk  │
│   • Click feeder → p10/p50/p90 ribbon, drivers, interventions           │
│   • Scenario panel: ΔT slider (+0/+5/+10/+15 °F); EV% + PV% stretch     │
│   • Metrics tab: reliability diagram, heat-wave vs normal split         │
│   • Export tab: CSV of feeder-upgrade ranking                           │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Why Graph WaveNet with quantile head (vs. alternatives)

| Model | Pros | Cons | Decision |
|---|---|---|---|
| **Graph WaveNet + quantile head** | Graph + temporal + uncertainty **in one model**; ~300 LOC; ~25 min on L4; proven on traffic/load; learned adjacency works perfectly when topology is known but influence patterns aren't. | Research-grade code, needs light patching. | **Hero.** |
| MTGNN | Strong on multi-variate TS. | ~2× training cost vs GWNet; no native quantile head. | Replaced by GWNet. |
| TFT | Native quantile, variable importance. | Per-series (not graph-native); heavy. | Replaced — quantile head on GWNet + Captum covers the same ground cheaper. |
| DCRNN | Canonical baseline; simple to explain. | Static adjacency; weaker. | **Stretch baseline.** |
| PatchTST | SOTA accuracy on many TS benchmarks. | No covariates, no quantiles, no graph. | Skip. |
| Foundation models (Chronos-Bolt) | Zero-shot baseline = great "we beat it" plot. | Univariate. | **Stretch** — CPU-runnable, no Colab cost. |
| PINN (full) | Physics-embedded. | Slow, finicky. | No — OpenDSS used instead. |

**Training budget envelope:** 1 full run (≤45 min on L4) + 1 contingency re-run = ~1.5 h of Colab compute. Well within ASU-free Pro quota.

### 3.3 Stack — exact versions

| Purpose | Package | Version | Why |
|---|---|---|---|
| DL framework | `torch` | 2.4+ | Colab-native. |
| Graph NN | `torch-geometric` | 2.6+ | Graph primitives for adjacency matrix ops. |
| Graph WaveNet impl | vendored from `nnzhan/Graph-WaveNet` | pinned commit | ~400 LOC, MIT-licensed, minor patch for quantile head. |
| Trainer | `pytorch-lightning` | 2.4+ | Clean checkpoint callbacks, Drive-safe. |
| Forecasting utils | `pytorch-forecasting` | 1.4+ | (stretch) DCRNN-ish + metrics. |
| Power flow | `OpenDSSDirect.py` | 0.9.4+ | Prebuilt Ubuntu 24.04 wheels. |
| Dashboard | `streamlit` + `pydeck` + `plotly` | latest | HF-Space-native; 4–6 h build. |
| Data wrangling | `pandas`, `polars`, `duckdb` | latest | DuckDB for S3 parquet filtering (ResStock). |
| Geospatial | `geopandas`, `shapely` | latest | Feeder centroids + map. |
| Explainability | `captum` | latest | Integrated Gradients on GWNet outputs. |
| Conformal (stretch) | `mapie` | latest | Distribution-free coverage. |
| Weights hosting | **HuggingFace Hub** (`huggingface_hub`) | latest | Weights at `dchanda/gridsense-az`. |
| Deploy runtime | **HuggingFace Spaces** | Streamlit SDK | CPU basic (16 GB), free. |
| Training runtime | **Colab Pro (ASU)** via `colab-mcp` | — | L4 GPU, ~45 min per run. |
| Optional marketing | `next` | 15+ | Tiny Vercel landing (stretch). |

---

## 4. Dataset inventory — exact access, auth, licensing

Run from repo root in a uv-managed venv: `uv venv && source .venv/bin/activate && uv pip sync requirements.txt`.

### 4.1 NOAA ISD — hourly Phoenix temperature (no auth)
```bash
mkdir -p data/raw/noaa
for Y in 2019 2020 2021 2022 2023 2024 2025; do
  curl -sSf -o "data/raw/noaa/KPHX_$Y.csv" \
    "https://www.ncei.noaa.gov/data/global-hourly/access/$Y/72278023183.csv"
done
```
- Station: KPHX (Phoenix Sky Harbor), USAF 722780, WBAN 23183. Licence: public domain.

### 4.2 NREL NSRDB — Phoenix irradiance (free API key)
- Signup: `https://developer.nrel.gov/signup/` → instant email.
- Store in `.env` as `NREL_API_KEY`, owner email `dchanda1@asu.edu`.
```bash
curl -sSf "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?\
wkt=POINT(-112.07%2033.45)&names=2020,2021,2022,2023&interval=60&\
attributes=ghi,dhi,dni,air_temperature,relative_humidity&\
api_key=${NREL_API_KEY}&email=dchanda1@asu.edu" \
  -o data/raw/nsrdb/phoenix_2020_2023.csv
```

### 4.3 EIA-930 — AZPS balancing-authority demand (free API key)
- Signup: `https://www.eia.gov/opendata/register.php` → instant email.
- Store in `.env` as `EIA_API_KEY`.
```python
# scripts/pull_eia930.py
import os, requests, pandas as pd
key = os.environ['EIA_API_KEY']
url = 'https://api.eia.gov/v2/electricity/rto/region-data/data/'
params = dict(api_key=key, frequency='hourly', data='value',
              **{'facets[respondent][]': 'AZPS', 'facets[type][]': 'D'},
              start='2019-01-01T00', end='2025-12-31T23', length=5000)
rows = []; offset = 0
while True:
    r = requests.get(url, params={**params, 'offset': offset}).json()
    batch = r['response']['data']
    if not batch: break
    rows.extend(batch); offset += 5000
pd.DataFrame(rows).to_parquet('data/raw/eia930/azps_demand.parquet')
```

### 4.4 NREL ResStock — Arizona Maricopa County residential profiles
```bash
python -c "
import duckdb
duckdb.sql('''
  INSTALL httpfs; LOAD httpfs;
  COPY (
    SELECT * FROM read_parquet(
      's3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/resstock_amy2018_release_2/timeseries_individual_buildings/by_county/state=AZ/county=G04013/*.parquet'
    ) LIMIT 500
  ) TO 'data/raw/resstock/maricopa_sample.parquet'
''')
"
```

### 4.5 NREL EVI-Pro Lite — EV charging fleet profiles
```bash
# Parameter sweep: fleet size × charging strategy
for FLEET in 100 500 1000 2000; do
  for STRAT in night afternoon; do
    curl -sSf "https://developer.nrel.gov/api/evi-pro-lite/v1/daily-load-profile?\
api_key=${NREL_API_KEY}&fleet_size=${FLEET}&climate_zone=very-hot-dry&\
home_access_dist=REAL_ESTATE&home_power_dist=MOSTLY_L2&\
work_power_dist=MOSTLY_L2&res_charging=${STRAT}&work_charging=min_delay" \
      -o "data/raw/evi_pro/phx_${FLEET}ev_${STRAT}.json"
  done
done
```

### 4.6 IEEE 123-bus feeder
```bash
git clone --depth 1 https://github.com/tshort/OpenDSS.git data/raw/opendss_repo
cp -r data/raw/opendss_repo/Distrib/IEEETestCases/123Bus data/raw/ieee123
```

### 4.7 HuggingFace EnergyBench — Pecan Street substitute
```python
# scripts/pull_energybench.py
from datasets import load_dataset
ds = load_dataset('ai-iot/EnergyBench', split='train', streaming=True)
# Filter to residential US households for feature engineering
```

### 4.8 BLOCKED — Pecan Street Dataport
- 3-business-day approval. **Substituted** by EnergyBench + ResStock.
- Submit application anyway for post-hackathon work.

### 4.9 Data manifest (commit `data/MANIFEST.yaml`)
```yaml
sources:
  - name: noaa_isd_kphx
    url: https://www.ncei.noaa.gov/data/global-hourly/access/
    licence: public-domain
    rows_expected: ~60000  # 7 years × ~8760 hourly obs
  - name: nsrdb_phoenix
    url: https://developer.nrel.gov/docs/solar/nsrdb/
    licence: CC-BY-NC
    rows_expected: ~35000  # 4 years × 8760
  - name: eia930_azps
    url: https://www.eia.gov/opendata/
    licence: CC0
    rows_expected: ~60000  # 7 years × 8760
  - name: resstock_maricopa
    url: s3://oedi-data-lake/nrel-pds-building-stock/.../state=AZ/county=G04013
    licence: public-domain
  - name: evi_pro_lite_phx
    url: https://developer.nrel.gov/docs/transportation/evi-pro-lite-v1/
    licence: public-domain
  - name: ieee_123
    url: https://github.com/tshort/OpenDSS
    licence: public-domain
  - name: hf_energybench
    url: https://huggingface.co/datasets/ai-iot/EnergyBench
    licence: (per dataset card)
```

---

## 5. Agent roster — who does what

| Agent | Purpose | Tools |
|---|---|---|
| **manager** (JARVIS) | Orchestration, task ledger, reviews, checkpoints. | Read, Bash, Grep, Glob, WebFetch, WebSearch, Agent, colab-mcp, context7-mcp |
| **sde** | Code: data pipelines, features, OpenDSS wrapper, decision engine, dashboard, tests, HF-Spaces deploy, CI. | All code + Playwright MCP |
| **ml-trainer** (NEW) | Compose + run Colab notebooks; train Graph WaveNet; checkpoint to Drive; emit metrics reports; push weights to HF Hub. | Read/Write/Edit, Bash, colab-mcp, context7-mcp |
| **reviewer** | Pre-commit review; flags bugs, security, scope creep. | Read, Grep, Glob, Bash, context7-mcp |
| **ui-verifier** | Drives Playwright against the Streamlit app (locally and on HF Space). | Playwright MCP |

**Parallelization contract:** ≤4 sub-agents concurrently. Typical fan-out at T+1 — `sde(data pulls)` ‖ `sde(OpenDSS wrapper)` ‖ `sde(dashboard scaffold)` ‖ `ml-trainer(smoke train)`.

---

## 6. Repo structure

```
energy/                          ← this folder = repo root
├── APS_AI_for_Energy_Technical_Problem_Statement.docx
├── PLAN.md                      ← this document (v2)
├── PLAN.v1.md.bak               ← archived v1
├── TASKS.md                     ← live task ledger
├── README.md                    ← written late, for judges + HF Space front-page
├── LICENSE                      ← MIT
├── pyproject.toml               ← uv-managed
├── requirements.txt             ← pinned deps for HF Space
├── .env.example                 ← NREL_API_KEY, EIA_API_KEY, HF_TOKEN
├── .gitignore
├── .gitattributes               ← git-lfs rules for *.pt
├── scripts/
│   ├── pull_all.sh
│   ├── pull_noaa.py
│   ├── pull_nsrdb.py
│   ├── pull_eia930.py
│   ├── pull_resstock.py
│   ├── pull_evi_pro.py
│   └── pull_energybench.py
├── data/
│   ├── MANIFEST.yaml
│   ├── raw/                     ← .gitignored
│   ├── features/                ← committed small; LFS for big
│   └── splits/                  ← committed
├── src/gridsense/
│   ├── __init__.py
│   ├── topology.py              ← IEEE 123 .dss → PyG Data
│   ├── features.py              ← weather × load × calendar join
│   ├── models/
│   │   ├── gwnet.py             ← Graph WaveNet + quantile head (hero)
│   │   └── dcrnn.py             ← stretch baseline
│   ├── power_flow.py            ← OpenDSSDirect.py wrapper
│   ├── decision.py              ← rule engine + IG explainer
│   └── eval.py                  ← MAE/RMSE/MAPE + reliability + heat-split
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_gwnet_train.ipynb     ← Colab-ready, self-contained
│   └── 03_evaluation.ipynb
├── app/
│   ├── streamlit_app.py         ← HF-Space entry point
│   ├── components/
│   │   ├── feeder_map.py
│   │   ├── forecast_chart.py
│   │   ├── scenario_panel.py
│   │   └── metrics_panel.py
│   └── assets/
├── models/
│   ├── gwnet_v1.pt              ← via git-lfs or HF Hub (≤200 MB)
│   └── model_card.md
├── reports/
│   └── gwnet_v1.md
├── tests/
│   ├── test_topology.py
│   ├── test_features.py
│   ├── test_power_flow.py
│   ├── test_model_smoke.py
│   └── test_dashboard.py        ← Playwright
├── .github/workflows/
│   ├── ci.yml                   ← pytest on push
│   └── deploy.yml               ← sync to HF Space on main-merge
└── hf_space/                    ← files copied into HF Space repo by deploy.yml
    ├── app.py                   ← symlink-ish of app/streamlit_app.py
    ├── requirements.txt
    └── README.md                ← HF Space front-matter
```

---

## 7. Hour-by-hour execution schedule (24 h)

| Hour | Parallel tracks | Deliverable |
|---|---|---|
| **T+0 → T+1** | sde: `create-next-app`-style scaffold of Python repo; git + GitHub via `gh`; register NREL/EIA keys; stamp `.env`; CI skeleton. manager: commit PLAN v2 + TASKS.md. | Green repo on GitHub; keys in `.env`. |
| **T+1 → T+3** | sde(A): dataset pullers (NOAA, NSRDB, EIA-930, ResStock, EVI-Pro, EnergyBench) run to completion. sde(B): OpenDSSDirect wrapper + IEEE 123 loads/solves. sde(C): feature pipeline (join weather × demand × calendar). ml-trainer: compose smoke-train notebook. | `data/features/*.parquet` populated; physics baseline solves; TFT-free GWNet skeleton compiles. |
| **T+3 → T+5** | ml-trainer: **full GWNet+quantile training on Colab L4** (~45 min) → `models/gwnet_v1.pt` + `reports/gwnet_v1.md`. sde: decision-engine + metrics module + IG explainer. | Trained weights; reliability diagram saved. |
| **T+5 → T+9** | sde: dashboard — map + forecast chart + scenario panel + metrics panel. ui-verifier: smoke pass on localhost:8501. reviewer: sweep. | `streamlit run app/streamlit_app.py` fully functional locally. |
| **T+9 → T+11** | sde: HF Space deploy workflow (`.github/workflows/deploy.yml` syncs `hf_space/` → `huggingface.co/spaces/dchanda/gridsense-az`); push weights to HF Hub. ui-verifier: end-to-end on the live HF Space URL. | Live HF Space URL. |
| **T+11 → T+15** | **BUFFER / SLEEP.** manager runs reviewer + ui-verifier in background; queues polish. | Clean CI, stable deploy. |
| **T+15 → T+19** | Stretch: ΔT × EV% × PV% precomputed lookup for instant slider. DCRNN baseline training (stretch, ~25 min). Chronos-Bolt zero-shot baseline (CPU). GNNExplainer (stretch). | Stretch features behind flags. |
| **T+19 → T+21** | sde: README + model card + submission.md. Optional: Vercel landing page linking to HF Space. | Publication-ready README. |
| **T+21 → T+23** | Owner: deck + reel. JARVIS: submission package, final CI green, tag `v1.0-submission`. | Submission artifacts. |
| **T+23 → T+24** | Buffer for last-minute fixes + final demo smoke run. | Ready to present. |

---

## 8. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Colab session dies mid-training | MED | Medium | Checkpoint every 2 min to Drive; ml-trainer resumes. |
| ASU Google SSO expired in Chrome Profile 2 | MED | Blocks Colab | Ping owner for DUO 2FA re-auth. |
| Graph WaveNet NaN-loss / diverges | MED | High | Gradient clip 5.0; LR 1e-4; LayerNorm; fallback to DCRNN. |
| OpenDSSDirect install fails on 24.04 | LOW | High | Fallback: dockerised EPRI OpenDSS + subprocess shim. |
| Dashboard inference slow on HF Spaces CPU | MED | Low | Precompute baseline + 75 scenarios; slider reads lookup, not live inference. |
| HF Space build fails (dep conflict) | MED | Medium | Pin all versions in `requirements.txt`; test build locally in Docker first. |
| Git-LFS bandwidth cap | LOW | Medium | Host weights on HF Hub (primary); no LFS dependency. |
| 4-core / 8 GB RAM laptop can't run dashboard + OpenDSS | MED | Medium | Precompute; dashboard only reads parquet; OpenDSS only runs on-demand for "what-if" with lightweight scenario shim. |
| Judges question "why synthetic not real APS data?" | MED | Medium | Slide 5 pre-answer: real APS AMI is confidential; ours is calibrated to public AZPS macro (EIA-930) + Phoenix climate (NOAA) + ResStock Maricopa; bias disclosed. |
| EIA-930 API throttles mid-pull | LOW | Low | Chunk by year; retry with exponential backoff. |

---

## 9. Judge scorecard — what we optimise for

| Dimension | Our hook |
|---|---|
| Arizona-orientation | IEEE 123 + mention of 34's AZ provenance; Phoenix NOAA; AZPS EIA-930; Maricopa ResStock; dashboard cites 8,631 MW peak + 2023 heat wave + $7 M settlement. |
| Heat treatment | Heat-wave feature (≥3 consecutive days ≥110 °F); separate MAE on heat days; +10 °F and +15 °F stress scenarios. |
| EV treatment | EVI-Pro Lite Phoenix-climate fleet profiles; EV% slider; evening-peak alignment with APS EV Overnight TOU. |
| Decision-layer quality | Feeder-level risk score + top-3 IG drivers + ranked interventions + export CSV. |
| Model rigour | GWNet + quantile head; pinball + MAE loss; quantile reliability ≥85 % coverage on held-out week; physics-consistency via OpenDSS; IG explainability. |
| Polished demo | Live HF Spaces URL; Streamlit+pydeck map; owner-produced deck + reel. |

---

## 10. Definition of done

- [ ] `pytest -q` → all green on `tests/`.
- [ ] `streamlit run app/streamlit_app.py` → loads <5 s, all tabs render, no console errors.
- [ ] Live HF Space URL accessible; ui-verifier passes end-to-end on the deployed URL.
- [ ] `reports/gwnet_v1.md` exists with MAE, RMSE, MAPE (overall + heat-wave split), pinball loss, p90 coverage ≥85 %, training time, GPU used.
- [ ] `models/gwnet_v1.pt` hosted on HF Hub; `models/model_card.md` complete.
- [ ] OpenDSS physics check flags ≥1 voltage or thermal violation in the +10 °F scenario.
- [ ] `README.md` documents local setup in ≤10 commands + HF-Space URL.
- [ ] `submission.md` drafted (paste-into-form text).
- [ ] GitHub repo tagged `v1.0-submission`; public; LICENSE (MIT).
- [ ] Owner has deck + reel ready.

---

## 11. Pitch deck — 5 slides (storyboard for owner)

| # | Title | Content |
|---|---|---|
| 1 | **The grid problem Arizona is already failing at** | 54 days ≥110 °F in Phoenix 2024 · $7 M heat-shutoff settlement · 500+ MW/hr duck-curve evening ramp. *Visual: Phoenix skyline + thermal overlay.* |
| 2 | **GridSense-AZ in one diagram** | Data-flow figure from §3.1; call out GWNet + OpenDSS + decision layer. |
| 3 | **What it does that others don't** | Baseline vs +10 °F scenario on the feeder map; quantile reliability plot; top-3 drivers for feeder 17. |
| 4 | **Numbers** | MAE, RMSE, MAPE, heat-day MAE, quantile coverage, inference latency. One chart: error during heat wave vs normal. |
| 5 | **Why APS should care + what's next** | Screening for 2027 IRP capex; integration path with APS ADMS; next: real AMI fine-tune + DERMS orchestration. QR to GitHub + HF Space. |

---

## 12. Reel — 75-s storyboard (for owner)

| 0–5 s | Phoenix skyline + "It's 118 °F. The grid is thinking." |
| 5–20 s | HF Space dashboard map zoom; heatmap turns red on a cluster of feeders. |
| 20–35 s | Click feeder 17 → p10/p50/p90 forecast ribbon + "top driver: temperature lag-24 (0.42)". |
| 35–55 s | Drag +10 °F slider; map redraws; violations appear; "interventions: dispatch battery bus 812, transfer 1.8 MW to feeder 22". |
| 55–70 s | Physics-check panel showing OpenDSS voltage heatmap. |
| 70–75 s | Logo + GitHub + HF Space URL. |

---

## 13. Post-hackathon extensibility (for "what's next" slide)

- Real APS AMI partner integration — fine-tune on confidential substation AMI.
- APS ADMS / SRP DERMS integration (both deployed Jan 2025).
- DER dispatch optimisation (today = screening only).
- Customer-level DR enrollment targeting (APS Smart EV Rewards).
- SRP joint-operation model with 1,000 MW pumped hydro by 2035.

---

## 14. Kickoff checklist (first 30 min after green-light)

```
[ ] 1. Restart Claude Code → verify mcp__colab__*, mcp__context7__*, mcp__playwright__* tools active
[ ] 2. Launch ASU-profile Chrome + open Colab bridge URL → verify L4 runtime
[ ] 3. sde: create repo + GitHub remote + .env.example + requirements.txt + CI skeleton
[ ] 4. Copy PLAN.md + write TASKS.md from ~/hackathon-kit/TASKS.md.template
[ ] 5. Register free keys: NREL + EIA (auto-scrape confirmation emails if needed)
[ ] 6. Install uv env: `uv venv && uv pip sync requirements.txt`
[ ] 7. Smoke-run `scripts/pull_all.sh` (~10-15 min)
[ ] 8. ml-trainer: smoke-train GWNet on 1% slice (20-min ceiling)
[ ] 9. Commit + tag `v0.1-scaffolded`
[ ] 10. Begin parallel SDE fan-out per §7
```

---

*End of plan v2. Living document — iterate as we build. Commit every edit.*
