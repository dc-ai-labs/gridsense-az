# TASKS — GridSense-AZ

Brief: Spatio-temporal AI system for Arizona Public Service distribution-grid feeder-load forecasting. IEEE-123-bus topology, Graph WaveNet + 3-quantile head trained on Colab L4, OpenDSS physics-consistency check, Streamlit + pydeck operator dashboard deployed to HuggingFace Spaces. 24-h solo hackathon.

Stack: Python 3.11 · uv · torch / torch-geometric · pytorch-lightning · OpenDSSDirect.py · Streamlit + pydeck + plotly · duckdb · captum · HF Hub + Spaces.
Repo: https://github.com/dc-ai-labs/gridsense-az
Deploy: https://huggingface.co/spaces/dchanda/gridsense-az (pending T+9)

## Legend
- ⬜ todo | 🟡 in-progress | ♻️ review | ✅ done | 🔴 blocked

## Ledger

| # | Phase / Hour | Task | Assignee | Status | Commit | Notes |
|---|---|------|----------|--------|--------|-------|
| 1 | Kickoff (T+0) | Scaffold GridSense-AZ monorepo + public GitHub repo + CI + tag v0.1-scaffolded | sde-scaffold | ✅ | (this commit) | Directory tree, pyproject, requirements (train + hf_space split), .env.example, LICENSE (MIT 2026), README starter, data/MANIFEST.yaml, empty module stubs, ci.yml (min-deps), deploy.yml stub, scripts/pull_all.sh stub. CI intentionally installs only pytest + pyyaml to keep smoke runs fast; full requirements are validated by Colab + HF Space builds. |
| 2 | Phase 1 (T+0 → T+1) | Register NREL + EIA API keys; write .env | manager | ⬜ | — | Owner email dchanda1@asu.edu. Signup links in .env.example. |
| 3 | Phase 1 (T+1 → T+3) | scripts/pull_noaa.py — KPHX 2019-2025 hourly | sde-data | ⬜ | — | Seven-year loop; public-domain, no auth. |
| 4 | Phase 1 (T+1 → T+3) | scripts/pull_nsrdb.py — Phoenix 2020-2023 | sde-data | ⬜ | — | Requires NREL_API_KEY. |
| 5 | Phase 1 (T+1 → T+3) | scripts/pull_eia930.py — AZPS demand 2019-2025 | sde-data | ⬜ | — | Requires EIA_API_KEY; paginate @ 5000/batch. |
| 6 | Phase 1 (T+1 → T+3) | scripts/pull_resstock.py — Maricopa AZ sample | sde-data | ⬜ | — | DuckDB httpfs against OEDI S3. |
| 7 | Phase 1 (T+1 → T+3) | scripts/pull_evi_pro.py — Phoenix fleet sweep | sde-data | ⬜ | — | Fleet x strategy grid. |
| 8 | Phase 1 (T+1 → T+3) | scripts/pull_energybench.py — HF dataset | sde-data | ⬜ | — | Pecan Street substitute. |
| 9 | Phase 1 (T+1 → T+3) | src/gridsense/power_flow.py — OpenDSS wrapper + IEEE 123 | sde-physics | ⬜ | — | Load, solve, flag voltage/thermal violations. |
| 10 | Phase 1 (T+1 → T+3) | src/gridsense/topology.py — .dss -> PyG Data | sde-physics | ⬜ | — | Adjacency + line impedances. |
| 11 | Phase 1 (T+1 → T+3) | src/gridsense/features.py — weather × demand × calendar ETL | sde-data | ⬜ | — | Emit data/features/*.parquet. |
| 12 | Phase 1 (T+1 → T+3) | notebooks/02_gwnet_train.ipynb — smoke-train skeleton | ml-trainer | ⬜ | — | Colab-ready, self-contained. |
| 13 | Phase 2 (T+3 → T+5) | Full Graph WaveNet + quantile head train (L4, ≤45 min) | ml-trainer | ⬜ | — | Pinball + MAE loss; checkpoint to Drive; push to HF Hub. |
| 14 | Phase 2 (T+3 → T+5) | src/gridsense/eval.py — MAE/RMSE/MAPE + reliability + heat-split | sde-ml | ⬜ | — | Reports to reports/gwnet_v1.md. |
| 15 | Phase 2 (T+3 → T+5) | src/gridsense/decision.py — IG drivers + rule-book | sde-ml | ⬜ | — | Captum Integrated Gradients on GWNet. |
| 16 | Phase 3 (T+5 → T+9) | app/components/feeder_map.py — pydeck risk map | sde-dashboard | ⬜ | — | Centroid scatter coloured by risk. |
| 17 | Phase 3 (T+5 → T+9) | app/components/forecast_chart.py — p10/p50/p90 ribbon | sde-dashboard | ⬜ | — | plotly. |
| 18 | Phase 3 (T+5 → T+9) | app/components/scenario_panel.py — ΔT / EV% / PV% sliders | sde-dashboard | ⬜ | — | Precompute lookup for instant response. |
| 19 | Phase 3 (T+5 → T+9) | app/components/metrics_panel.py — reliability diagram + heat split | sde-dashboard | ⬜ | — | plotly. |

## Blockers
<!-- Append here when stuck; clear when resolved. -->

## Decisions log
- 2026-04-18: CI installs only `pytest + pyyaml` (not full requirements.txt). Torch + torch-geometric wheel resolution is slow and adds no signal at the smoke-import stage; full deps are validated by the Colab training job and the HF Space build. Revisit at T+9 once we have lightweight runtime tests worth running against real deps.
- 2026-04-18: `tests/conftest.py` adds `src/` AND the repo root to `sys.path` so `from app.components import ...` works without an editable install. Trade-off: keeps `pip install -e .` optional.
- 2026-04-18: HF Space requirements are split from main `requirements.txt` — HF Space ships `torch + streamlit + pydeck + plotly + pandas + duckdb + opendssdirect.py + hf_hub + captum` only. Keeps the Space build under the 10-min free-tier ceiling.
- 2026-04-18: GitHub account used is `dc-ai-labs` (active `gh auth` account); HF_USERNAME is `dchanda` per PLAN §0.5.
- 2026-04-18: Model weights shipped via HF Hub, not git-LFS (PLAN §8 risk register). `.gitattributes` still declares LFS filters for *.pt/*.ckpt/*.bin in case of local caches.


## Reviewer log

- **2026-04-18 T+0 · Scaffold (SHA 7fc782e, tag v0.1-scaffolded)** — VERDICT: approve. Blockers: none.
  - Nit (deferred to T+20 README pass): shields.io python-version badge has  escape glitch.
  - Nit (deferred / stretch-only): requirements.txt omits pytorch-forecasting + mapie (both §3.3 stretch).
  - Nit (monitoring): hf_space torch CPU wheel ~700 MB; watch HF Space 10-min build budget.
  - Nit (non-blocking): brief .docx + PLAN.v1.md.bak committed at repo root; intentional, bloats clone.
