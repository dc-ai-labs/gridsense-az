# GridSense-AZ

**Spatio-temporal AI that forecasts the Arizona power grid 24 hours ahead, stress tests it against heat and EV demand, and validates every prediction against real power flow physics.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Graph%20WaveNet-EE4C2C?logo=pytorch&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=nextdotjs&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)
![OpenDSS](https://img.shields.io/badge/OpenDSS-Power%20Flow-2C7FB8)
![Vercel](https://img.shields.io/badge/Deploy-Vercel-000000?logo=vercel&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> Built at the ASU Energy Hackathon for the Arizona Public Service "AI for Energy" Challenge.

**TLDR:** GridSense-AZ predicts electricity demand for every node on a Phoenix distribution feeder a full day ahead, then runs a physics simulation to flag which nodes will overload during a heat wave or an evening EV charging surge. A Python backend turns raw weather and demand data into validated forecasts. A web dashboard lets an operator explore any scenario instantly, with no server in the loop.

<img width="800" height="520" alt="grid" src="https://github.com/user-attachments/assets/4236f3da-e0c8-4866-9f0d-7a87615cb83e" />


**Live Link:** [link](https://gridsense-az.vercel.app)

**ML playground:** [link](https://dc-ai-labs-gridsense-az.hf.space)

---


## The problem
 
Grid operators need to see trouble before it happens. On a brutal Phoenix afternoon, or when thousands of EVs plug in at 6 pm, some feeder nodes cross voltage and thermal limits and fail. GridSense-AZ forecasts that stress 24 hours out, runs it through a real power flow engine, and surfaces the exact nodes at risk along with a recommended operator action for each.
 
---
 
## Highlights
 
### Backend and systems
 
* **Two stage design, zero runtime inference.** All model inference and physics solves run offline in a precompute step that emits static JSON. The browser never calls a Python server, so the frontend is trivially cacheable and scenario switches happen in memory with no network round trip.
* **Concurrency handled correctly around unsafe code.** OpenDSS wraps a process global singleton that corrupts its state under concurrent access. Every solve is serialized behind a module level lock, so the app stays correct under multiple simultaneous users instead of silently returning garbage.
* **Crash safe writes.** The precompute step writes each artifact to a temp file and then atomically replaces the target, so an interrupted run can never leave the frontend serving a half written forecast.
* **A real contract between Python and TypeScript.** The TypeScript types are the single source of truth for the data shape, and the frontend asserts invariants on every payload at load time (heat peak at least 1.3x baseline, EV peak hour within 17 to 22). Bad data fails loud with an error banner rather than rendering a wrong map. Backend and frontend cannot silently drift.
* **Clean module boundaries.** Training, inference, the scenario engine, the physics solver, and serialization are independent units with narrow interfaces.
### AI and ML native, not bolted on
 
* **A graph neural network built and trained from scratch.** A Graph WaveNet model (stacked dilated causal temporal convolutions plus a data learned adaptive adjacency) forecasts demand for all 132 nodes in a single forward pass. 59,890 parameters, trained in under 10 minutes on one GPU.
* **Calibrated uncertainty, no ensemble.** A quantile head trained with pinball loss emits p10/p50/p90 directly, so every forecast ships with a confidence band.
* **Rolling inference to extend the horizon.** The model natively predicts 6 hours. The pipeline splices its own median predictions back into the input buffer and rolls four times to reach a full 24 hours.
* **The ML closes a loop with physics.** Forecasts drive per node load overrides into OpenDSS, which returns ANSI C84.1 voltage violations and line loading, which the dashboard turns into ranked risk and concrete actions.

---
 
## Architecture
 
```
STAGE 1   Offline precompute (Python)
 
  NOAA ISD weather + EIA 930 demand + calendar
      → features.py builds FeatureBundle [T, 132, 12]
      → GWNet (model.py, 59,890 params) produces p10/p50/p90 over 6 hours
      → predictor.py rolls inference 4x to reach a 24 hour horizon
      → decision.py applies baseline / heat / EV / combined transforms
      → power_flow.py runs an OpenDSS snapshot solve per scenario
      → precompute_forecasts.py writes 6 JSON files atomically
 
STAGE 2   Runtime in the browser (zero server side inference)
 
  web/public/data/forecasts/*.json
      → ScenarioProvider fetches all 6 in parallel
      → validate.ts asserts heat peak ≥ 1.3x baseline and EV peak hour ∈ [17, 22]
      → components render: TacticalMap, ForecastRibbon, RiskLeaderboard, PhysicsCheck
      → scenario switches swap from memory with no refetch
```
 
---
 
## Key engineering decisions
 
* **Static precompute over a live inference API.** A forward pass plus three OpenDSS solves take a few seconds, and the upstream demand feed publishes with a multi day lag. A live API would add a server, latency, and failure modes for no benefit. Precomputed JSON gives instant page loads and free static hosting.
* **Raw kW as the model output, not standardized.** An earlier version applied the input scaler to predictions and inflated outputs by roughly 43,000x. I changed the target so the model maps standardized input straight to raw kW, removing the denormalization step where the bug lived. The scaler is retained only for diagnostics.
* **A process lock around OpenDSS.** Because the engine is a global singleton, the only correct way to expose it to a multi user app is to serialize access. I chose correctness over throughput here deliberately.
* **Atomic JSON writes.** Cheap insurance against serving a partially written file if precompute is killed mid run.
* **Snapshot power flow at the peak hour, not a full time series.** Each scenario maps to its forecast peak hour, which is the moment violations actually occur, so a single snapshot answers the question at a fraction of the cost.

---

## Tech stack
 
| Layer | Tools |
|---|---|
| Backend and data | Python 3.11, NetworkX (parses the IEEE 123 feeder into a graph), pandas, NumPy, SciPy, scikit-learn. Data from NOAA ISD weather and the U.S. EIA 930 demand API. |
| ML | PyTorch, a custom Graph WaveNet with an adaptive adjacency and a 3 quantile head, pinball loss, torch-geometric for topology. |
| Physics | OpenDSSDirect.py driving snapshot power flow on the canonical IEEE 123 bus feeder. |
| Frontend | Next.js 14 (App Router, static generation), React 18, TypeScript 5.5, Tailwind, Recharts, Leaflet. A parallel Streamlit operator app uses pydeck and plotly with live solves. |
| Infra | Static frontend on Vercel, Streamlit app on HuggingFace Spaces, model trained on Colab. |
 
---
 
## Results
 
* **18.4% lower mean absolute error** than a persistence baseline on 1,725 held out test windows from Arizona summers 2022 and 2023 (4,574 kW versus 5,604 kW).
* **59,890 parameters.** 200 epochs in 547 seconds on one GPU, with cosine learning rate decay and a 10 epoch warmup.
* **Leakage free split** on the time axis, so no test window overlaps any training input.
---
 
## Run it locally
 
Backend and the Streamlit app:
 
```bash
git clone https://github.com/dc-ai-labs/gridsense-az.git
cd gridsense-az
pip install .
streamlit run app/streamlit_app.py
```
 
Regenerate the forecast JSON the frontend reads:
 
```bash
python scripts/precompute_forecasts.py
```
 
Frontend:
 
```bash
cd web
pnpm install
pnpm dev
```
 
A trained checkpoint ships in the repo, so inference works immediately on a fresh clone.
 
---
 
## Roadmap
 
* Wire Captum integrated gradients into the inference path for real per forecast feature attribution.
* Add a scheduled refresh that pulls new demand and weather and regenerates forecasts automatically.
* Add authentication and request limiting ahead of any shared deployment.
* Add formal quantile calibration reporting (coverage and reliability).

---
 
## Credits
 
Built on the Kersting and EPRI IEEE 123 bus test feeder. Weather from NOAA ISD, demand from the U.S. EIA 930 API. Graph WaveNet architecture adapted from Wu et al., 2019.
