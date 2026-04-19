# GridSense-AZ

> Spatio-temporal AI for Arizona Public Service distribution-grid feeder-load forecasting, with OpenDSS physics-consistency and a Streamlit + pydeck operator dashboard.

![ci](https://img.shields.io/badge/ci-pending-lightgrey)
![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![status](https://img.shields.io/badge/status-scaffolded-yellow)

GridSense-AZ forecasts hourly load for every feeder of an IEEE-123-bus distribution network parameterised with real Phoenix weather (NOAA KPHX) and real AZPS balancing-authority demand (EIA-930). It stress-tests forecasts under extreme-heat and EV-evening-peak scenarios, runs an OpenDSS power-flow sanity check for voltage/thermal violations, and turns the output into ranked operator interventions on an interactive map. Hero model: a Graph WaveNet with a 3-quantile head (p10/p50/p90). Deployed to HuggingFace Spaces.

Full architectural plan: see [`PLAN.md`](./PLAN.md).

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/dc-ai-labs/gridsense-az.git && cd gridsense-az

# 2. Install uv (once, skip if present)
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"

# 3. Create venv (Python 3.11)
uv venv --python 3.11 && source .venv/bin/activate

# 4. Sync dependencies
uv pip sync requirements.txt

# 5. Configure secrets
cp .env.example .env   # then fill in NREL_API_KEY, EIA_API_KEY, HF_TOKEN

# 6. Pull public datasets (NOAA, NSRDB, EIA-930, ResStock, EVI-Pro, IEEE 123)
bash scripts/pull_all.sh

# 7. Smoke-test the install
pytest -q

# 8. Launch the dashboard locally
make run-app        # or: streamlit run app/streamlit_app.py

# 9. (Training) open the Colab notebook and point it at a GPU runtime
#    notebooks/02_gwnet_train.ipynb

# 10. (Deploy) push to main — .github/workflows/deploy.yml syncs hf_space/ to HuggingFace
```

---

## Stack

- **Model:** Graph WaveNet + 3-quantile head (pinball + MAE loss), trained on Colab Pro L4
- **Graph / DL:** `torch` 2.4+, `torch-geometric` 2.6+, `pytorch-lightning` 2.4+
- **Power flow:** `OpenDSSDirect.py` against an IEEE-123-bus .dss
- **Dashboard:** `streamlit` + `pydeck` + `plotly`
- **Data wrangling:** `pandas`, `polars`, `duckdb`, `pyarrow`
- **Geospatial:** `geopandas`, `shapely`
- **Explainability:** Integrated Gradients via `captum`
- **Weights hosting:** HuggingFace Hub (`dchanda/gridsense-az`)
- **Deploy:** HuggingFace Spaces (Streamlit SDK)

Full rationale and alternatives considered in [`PLAN.md` §3](./PLAN.md).

---

## Data

All source licences, URLs, and expected row counts are declared in [`data/MANIFEST.yaml`](./data/MANIFEST.yaml). Raw files land in `data/raw/` (gitignored), cleaned features in `data/features/`, and train/val/test splits in `data/splits/`.

Sources: NOAA ISD KPHX, NREL NSRDB (Phoenix), EIA-930 (AZPS), NREL ResStock (Maricopa County, AZ), NREL EVI-Pro Lite, IEEE 123-bus feeder, HuggingFace EnergyBench.

---

## Layout

See [`PLAN.md` §6](./PLAN.md) for the authoritative repo structure.

## Licence

[MIT](./LICENSE) © 2026 Divyansh Chandarana.
