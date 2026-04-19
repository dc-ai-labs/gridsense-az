# Data — sources, licensing, freshness, reproducibility

Every number GridSense-AZ renders ultimately traces back to one of three data
families: weather, demand, or topology. This document is the audit trail for
each.

## Weather

### Training corpus — NOAA ISD KPHX hourly
- **Station**: KPHX — Phoenix Sky Harbor International Airport.
- **Source**: NOAA Integrated Surface Database (ISD).
- **Window shipped with this repo**: 2022-06-01 → 2023-10-01 (11,688 hours).
- **Fields used**: `temp_c`, `dewpoint_c`, `wind_mps`, `slp_hpa`.
- **Puller**: [`scripts/pull_noaa.py`](../scripts/pull_noaa.py) → writes
  `data/raw/noaa_kphx_2022-06_2023-10.csv`.
- **Attribution**: NOAA is U.S. federal public domain — no licence constraints.
- **Gaps**: 38 of 11,688 hours had missing exogenous values (see
  `reports/gwnet_v1.md`). Imputed via forward-fill during feature build.

### Live inference — NWS api.weather.gov
- **Endpoint**: `https://api.weather.gov/gridpoints/PSR/<x>,<y>/forecast/hourly`
  for Phoenix, lat/lon `33.4484, −112.0740`.
- **Auth**: none; a `User-Agent` header is required per NWS policy.
- **Units**: NWS delivers temperature in °C; the dashboard converts to °F for
  North American human readability.
- **Puller**: [`scripts/nws_fetch.py`](../scripts/nws_fetch.py) via
  `fetch_phoenix_hourly()`.
- **Attribution**: NWS data is U.S. federal public domain.
- **Cadence**: precompute runs fetch on demand. NWS updates the gridpoint
  forecast roughly hourly.

## Demand

### Training corpus — EIA-930 AZPS hourly demand
- **Source**: U.S. Energy Information Administration (EIA) Form 930, Balancing
  Authority hourly demand, operating-area code **AZPS** (Arizona Public
  Service).
- **Endpoint**: `https://api.eia.gov/v2/electricity/rto/region-data/data/`.
- **Auth**: free EIA API key (not required at request time for the cached
  CSV we ship, but required to re-pull).
- **Window shipped**: 2022-06-01 → 2023-10-01 (11,688 hours), aligning with
  the weather corpus.
- **Puller**: [`scripts/pull_eia930.py`](../scripts/pull_eia930.py) → writes
  `data/raw/eia930_azps_2022-06_2023-10.csv`.
- **Attribution**: EIA data is U.S. federal public domain. See the EIA
  open-data terms at https://www.eia.gov/opendata/.

### Disaggregation — system demand → per-bus kW

EIA-930 gives AZPS **system-total** demand. The IEEE 123-bus feeder needs
**per-bus** values. Our disaggregation is a fixed share assignment:

```
  y_kw[t, bus_i] = system_demand[t] · (nominal_kw[bus_i] / Σ_j nominal_kw[bus_j])
```

where `nominal_kw` comes from the IEEE 123 reference loads in the DSS master
file. This yields a physically plausible per-bus time series that matches
the system total exactly, but does **not** reflect real per-bus AMI
behaviour (see `reports/gwnet_v1.md` for the disclaimer).

## Topology

### IEEE 123-bus test feeder
- **Source**: IEEE PES Distribution System Analysis Subcommittee (DSASC).
- **Resource page**:
  https://cmte.ieee.org/pes-testfeeders/resources/
- **Files shipped**: `data/ieee123/` — master DSS file, BusCoords, loads,
  lines, capacitors, regulators.
- **Licence**: IEEE test feeders are published for research and benchmarking
  and are routinely redistributed with academic and commercial software; we
  redistribute verbatim under that permissive convention. No commercial
  restrictions in practice.
- **Puller**: [`scripts/pull_ieee123.sh`](../scripts/pull_ieee123.sh)
  (idempotent; skips if already present).

### Map projection disclaimer
The IEEE 123 feeder is a **synthetic test feeder** with no geographic
coordinates of its own. The tactical map projects the feeder onto the
Phoenix area (lat/lon `33.4484, −112.0740`) as a visualisation convenience.
This is **not a claim about APS's real distribution network**. APS's actual
feeders have different topologies, different conductor mixes, different
switching states. Any judge or reviewer should read the map as "here is
a realistic feeder-scale grid problem, shown geographically for context"
and not "here is APS feeder X."

## Live-data freshness

The dashboard renders a provenance pill in the top nav:

```
DATA: nws · FORECAST 2026-04-19 · GEN 14 minutes ago
```

Fields:

| Field | Meaning |
|---|---|
| `DATA: nws` | Upstream weather source used this run (`nws` for live NWS). |
| `FORECAST YYYY-MM-DD` | Target forecast date ("tomorrow" in local Phoenix time). |
| `GEN <relative>` | How long ago the precompute pipeline ran. |

Backing file: [`web/public/data/forecasts/generated_at.json`](../web/public/data/forecasts/generated_at.json)
is emitted by `scripts/precompute_forecasts.py` at the end of every run.

## File layout

```
data/
├── ieee123/              # test-feeder topology (committed)
│   ├── IEEE123Master.dss
│   ├── BusCoords.dat
│   └── ...
├── raw/                  # external pulls (gitignored — re-pull on demand)
│   ├── noaa_kphx_*.csv
│   └── eia930_azps_*.csv
├── processed/            # feature-bundle parquet caches
└── models/
    ├── gwnet_v0.pt       # trained checkpoint (263 KB, committed)
    ├── metrics.json      # final MAE + config (committed)
    ├── eval_report.json  # stress-window breakdown (committed)
    ├── history.json      # per-epoch loss (committed)
    └── train.log         # gitignored runtime log
```

`data/raw/**` is excluded from git by `.gitignore` — raw pulls are
intentionally not redistributed; you re-pull them with the scripts above
(takes < 3 minutes on a home connection).

## Reproducibility

- **Seed**: `1337`, pinned in [`scripts/train.py`](../scripts/train.py)
  across `torch.manual_seed`, `numpy.random.seed`, and Python's `random.seed`.
- **Config persisted**: the full training configuration is embedded in
  `data/models/metrics.json` (`config` key) and in the checkpoint itself
  under `ckpt['config']`, so loading the weights also locks in architecture.
- **Idempotency**:
  - NWS fetches are idempotent per (lat, lon, hour) tuple — calling the
    fetcher twice in the same hour returns the same payload because NWS
    publishes hourly.
  - NOAA and EIA CSV pulls are date-windowed; re-running writes the same
    file contents byte-for-byte for a frozen window.
- **Versioning**: data pullers accept `--start`/`--end` flags, so a reviewer
  can expand or shift the corpus without touching code.

## Disclaimers (summary)

1. The **IEEE 123 feeder is synthetic** — not APS's network.
2. **Per-bus load is disaggregated**, not metered.
3. Live weather is **one station** (KPHX); the feeder projection is a single
   lat/lon point, not a real geographic footprint.
4. **NSRDB solar irradiance** and **EVI-Pro EV profiles** were unavailable
   during the hackathon window (credentials / endpoints failing); the model
   was trained without them. See `reports/gwnet_v1.md`.
5. Data licences: all training / inference sources are U.S. public domain
   (NOAA, NWS, EIA) or IEEE-research-permissive (123-bus feeder). Nothing
   in this repo requires a commercial licence to re-run.
