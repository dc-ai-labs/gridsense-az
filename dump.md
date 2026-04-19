# GridSense-AZ — Context Dump

_Generated 2026-04-18 22:22 for future-agent handoff. This is a live snapshot; cross-check against `git log --oneline -20` and `git status` before acting._

## 1. What this project is

GridSense-AZ is our entry to the APS "AI for Energy" track at the ASU Energy Hackathon. The challenge asks teams to take an existing utility data stream (we use EIA-930 AZPS balancing-area hourly demand and NOAA KPHX weather as proxies for distribution-level telemetry) and turn it into an operator-facing decision product.

The three rubric dimensions we're shooting for:

1. **Forecasting quality** — per-bus hourly load forecast with a real probabilistic (p10/p50/p90) output, measured against a persistence baseline. We land at test MAE 4,574 kW (+18.4% over persistence).
2. **Scenario / stress reasoning** — "what happens tomorrow if a heat dome arrives" and "what happens if EV charging ramps" both modeled as exogenous-input shifts into the same model, producing side-by-side comparisons.
3. **Decision support** — a tactical-ops dashboard that surfaces top-risk buses, an OpenDSS physics check, and recommended operator actions. A human operator can make a call in <30 s.

Our winning angle is the *integration*: a real trained spatio-temporal model (GWNet, not LightGBM), a live weather hook (NWS Phoenix, no auth), and a dashboard that judges can literally click through in a browser. Most other teams will ship slides — we ship a URL.

## 2. What's built and working

| Layer | Status | Notes |
|---|---|---|
| Data pulls | working | NOAA KPHX + EIA-930 AZPS, hourly 2022-06 → 2023-10, 11,688 timesteps |
| IEEE 123 feeder | checked in | `data/ieee123/` with master + BusCoords |
| Feature builder | working | `src/gridsense/features.py::build_hourly_features()`, 11 exogenous features |
| Model | trained | `data/models/gwnet_v0.pt`, 59,890 params, commit SHA of training recipe `12271d1` |
| Metrics | logged | `data/models/metrics.json` (train 2,370 · val 3,525 · test 4,574 kW MAE) |
| Predictor | working | `src/gridsense/predictor.py`; denormalisation bug fixed in commit `c31f18d` |
| Scenarios | working | heat + EV transforms in `src/gridsense/decision.py` |
| Power flow | working | `src/gridsense/power_flow.py` wraps OpenDSS |
| Precompute | working | `scripts/precompute_forecasts.py` — fetches NWS, runs model, writes JSONs. peak_hour local-vs-UTC fix landed in `d378b80`. Reviewer flagged capacity_mw stability + heat encoder-history temp shift; slice B follow-up agent is fixing those right now. |
| Dashboard | live | Next.js 14, 7 components, Vercel prod at https://gridsense-az.vercel.app. Base shipped in `482d0c1`. Slice C (side-by-side overlay + keyboard shortcuts) currently being written by a parallel agent. |
| HF Space | live | Streamlit fallback at https://dc-ai-labs-gridsense-az.hf.space/ |

Recent relevant commits (bottom-up):

- `06c0103` — predictor wired into dashboard forecast tab
- `c31f18d` — **fix**: predictor denormalisation dropped (targets already raw kW)
- `12271d1` — 200-epoch A100 training run committed (README + report)
- `482d0c1` — slice A: tactical-ops dashboard scaffold + stub data
- `58fc84c` — slice B: precompute + scenario pipeline end-to-end
- `d378b80` — **fix**: peak_hour now reports local hour; peak_temp_f reflects heat shift

Check `git log --oneline -10` when you read this — by the time you do, slices B-follow-up and C will have landed.

## 3. Deployment topology

| Component | Identifier | URL |
|---|---|---|
| GitHub repo | `dc-ai-labs/gridsense-az` | https://github.com/dc-ai-labs/gridsense-az |
| Vercel project | `dc-ai-labs-projects/gridsense-az` | https://gridsense-az.vercel.app |
| HuggingFace Space | `dc-ai-labs/gridsense-az` | https://dc-ai-labs-gridsense-az.hf.space/ |

**Important:** Vercel is NOT hooked to GitHub. Pushes to `main` do NOT auto-deploy. Redeploys are manual:

```bash
cd web
vercel --prod
```

The `web/.vercel/project.json` is gitignored but present locally — the CLI reads it. If you clone fresh and need to redeploy, run `vercel link` once.

HF Space deploys via `bash scripts/deploy_hf_space.sh`, which pushes the `hf_space/` subtree. Needs `HF_TOKEN` in `.env`.

## 4. Known issues / recent fixes

### Recently fixed
- **Predictor denormalisation bug** (commit `c31f18d`): forecaster was double-inverse-scaling outputs. Fixed — outputs are now raw kW, matching the `y_kw` contract of `FeatureBundle`.
- **peak_hour label mismatch** (commit `d378b80`): precompute was writing UTC hour into a field the dashboard rendered as "local hour of peak." Now consistently local Phoenix time. Also `peak_temp_f` now reflects the +10 °F shift in the heat scenario, not baseline temp.

### In flight (slice B follow-up agent, same repo, right now)
- **capacity_mw stability**: reviewer noted that the value should be stable across reruns (it's a topology constant, not a forecast artifact). Agent is pinning it in the precompute output.
- **Heat encoder-history temp shift**: currently the +10 °F shift is only on the decoder future inputs; reviewer noted it should also apply to the 24 h encoder history for self-consistency. Agent is threading the shift through both.

### In flight (slice C agent)
- Side-by-side baseline vs stressed overlay in `ForecastRibbon` + `TacticalMap`.
- B/H/E/C keyboard shortcuts plumbed through `ScenarioProvider` in `web/lib/context.tsx`.

### Open, not in flight
- **OpenDSS scale mismatch** (3.5 MW feeder nameplate vs 3.5 GW EIA-930 forecast): IEEE 123 is a toy distribution feeder. We use it for topology + voltage-drop narrative, not as a literal 1:1 grid. Needs an explicit README caveat in the architecture section (P1 on TODO.md).
- **Captum Integrated-Gradients scores**: `top_drivers` in the dashboard uses heuristic placeholders ("temp_c", "hour_sin") instead of real IG attributions. Placeholder now; real IG is P1.
- **REPL_LAG: 42ms** hardcoded in `BottomStatus`: reviewer called it out. Cosmetic but should be live-driven or removed.
- **`href="#"` placeholders in TopNav**: same reviewer note.

## 5. Agent operating notes

- **Chrome profiles**: NEVER use the ASU profile. Default to the dcgaming Chrome profile for deployment CLIs, or use a fresh Playwright context for scripted sign-ins.
- **Third-party signups**: use `dcgaming0711@gmail.com`, not `dchanda1@asu.edu`.
- **Vercel CLI auth**: do NOT rely on the `VERCEL_TOKEN` env var — it's stale. Unset it and use `~/.config/vercel/auth.json` (CLI reads this automatically after `vercel login`).
- **NWS API**: auth-free, but the `User-Agent` header is required. `scripts/nws_fetch.py` sets it.
- **Python**: 3.11 only. `.venv/` at project root. Always `source .venv/bin/activate` before `pytest` or any script.
- **Tests**: live at `tests/`. Run with `.venv/bin/python -m pytest -q` from repo root.
- **Secrets**: `.env` at project root is source of truth. Gitignored. Never print values.
- **Raw data**: `data/raw/` is gitignored — never commit pulls.
- **Model checkpoint**: `data/models/gwnet_v0.pt` (263 KB) is tracked directly in git with an explicit `.gitignore` negation. Future `.pt` files remain ignored by default. No git-LFS.
- **Never push to origin** without the manager's explicit go-ahead. The manager batches pushes.

## 6. How to continue

1. `cd /home/divyansh/Downloads/hackathon/energy` (or wherever the repo lives).
2. `git log --oneline -10` — see what just landed.
3. `git status` — check for in-progress work.
4. Read `TASKS.md` for the hackathon-wide task registry, then `TODO.md` (in this repo root) for the prioritized list.
5. Pick the top-most P0 item not marked in flight. If all P0s are in flight, pick a P1.
6. Read `ARCHITECTURE.md` if you need the system mental model before touching code.
7. Work in the Manager + SDE pattern: Manager (you, if you're the orchestrator) breaks the task into one or more focused prompts and spawns SDE agents via the Agent tool. Each SDE returns a structured STATUS / CHANGES / TESTS / NOTES report. Manager aggregates.
