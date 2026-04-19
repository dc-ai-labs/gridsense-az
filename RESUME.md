# RESUME — pause checkpoint 2026-04-18T18:27

## Where we are

**4 modules green + reviewed on main:**
1. `feat(data)` — 6 pullers + orchestrator (8abf5c0, reviewed 379d804)
2. `fix(data)` — canonical IEEE 123 LineCodes (a0ef187)
3. `feat(topology)` — IEEE 123 .dss parser + PyG converter (de2e0c9, reviewed 7616742) — 132 buses, 131 edges, 85 loads @ 3490 kW canonical ✓
4. `feat(power-flow)` — OpenDSSDirect snapshot wrapper + overrides (09e603a, reviewed ac108f0) — converges in 32 iters, cold 27ms / warm 39ms, L115=157.7% matches Kersting ✓

**Test state:** 41 tests green (8 topology + 7 power_flow + 19 pullers + 4 model smoke + 1 each features/power_flow/dashboard stub)

## Raw data on disk (not committed; gitignored under data/raw/)
- ✅ NOAA ISD KPHX: 41 MB, 2019-2025 (full 7 years)
- ✅ EIA-930 AZPS hourly demand: 520 KB, 63,961 rows, 2019-01-01 → 2026-04-19
- ✅ ResStock Maricopa baseline: 324 KB
- ✅ IEEE 123 feeder: vendored + spec (committed)
- 🟡 EnergyBench (HF): 1.3 GB partial download — killed at ~80%; can resume with `python scripts/pull_energybench.py`
- ❌ NSRDB PSM3: endpoint returned 404 (NREL deprecated v2); not blocking — see BLOCKERS.md
- ❌ EVI-Pro Lite: 400 Bad Request on all sweeps (param enum change); not blocking — see BLOCKERS.md

## Known issues (see BLOCKERS.md)
- NSRDB endpoint dead — workaround: skip irradiance, use NOAA only for demand model
- EIA-930 has source-data outliers (max 101 GW, min -22 GW — AZPS typical is 2-8 GW); feature pipeline needs winsorise to [500, 10000] MW
- EVI-Pro API changed — synthesize EV load overlay from first principles instead

## SDEs in flight when paused
1. **features SDE** (async, was writing src/gridsense/features.py — 727 lines of work reverted when paused; will need re-dispatch)
2. **model SDE** (async agent a57389c42230cdf84 — may have completed on Claude's side while we paused; check for notification on resume)

## Next 4 dispatches queued (priority order)

1. **Re-dispatch features SDE** — spec in manager transcript. Writes `src/gridsense/features.py` + `tests/test_features.py`. Produces a `FeatureBundle` dataclass that the training script consumes. Uses NOAA+EIA disaggregated to 132 buses via nominal-kW share. Synthetic fallback if raw missing.
2. **model SDE (if not already green)** — Graph WaveNet + pinball loss + dataloader factory. Already dispatched; may be done on Claude's side.
3. **train SDE** — wires features + model → `scripts/train.py` that runs `fit()`, saves checkpoint to `data/models/gwnet_v0.pt`, writes `data/models/metrics.json`. CPU-runnable.
4. **Reviewer + checkpoint** each commit per the manager loop.

## Training plan once features + model land
- CPU baseline: 4-year hourly, 3h validation split, ~10 epochs @ ~5-8 min each ≈ 60-90 min total
- If Colab MCP handshake is available: move to L4 GPU for 50 epochs in ~20 min
- Target: p50 MAE < persistence baseline × 0.95 on the val split

## How to resume
- `cd ~/Downloads/hackathon/energy && source .venv/bin/activate`
- `git log --oneline -5` → confirm last commit is `docs(tasks): reviewer verdict — 09e603a power_flow` or later
- `pytest -v` → confirm 41 tests green
- Then pick up with feature-pipeline re-dispatch

## Directory state
- Working directory: `/home/divyansh/Downloads/hackathon/energy/`
- Branch: `main` (clean after revert of features.py)
- Untracked: `BLOCKERS.md`, `RESUME.md`
- `.env` has NREL + EIA + HF keys verified live
