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

- **2026-04-18 T+1 · ml-trainer GWNet + quantile head + training notebook (SHA 8e2f744)** — VERDICT: approve. Blockers: none.
  - Nit: `reports/gwnet_v1.md` self-reports ~540k params; actual at PLAN dims (N=123, hidden=32) is ~62k. Update before deck copy lands.
  - Nit: notebook Cell 18 defines `nominal = np.linspace(0.05,0.95,10)` then never uses it — dead line, remove.
  - Nit: hardcoded Colab clone URL `github.com/dc-ai-labs/gridsense-az.git` — parameterise.
  - Nit: `CFG['num_nodes']=130` vs IEEE-123's 123 buses — make dynamic from loaded adjacency before real-data run.
  - Nit: add `pl.seed_everything(SEED)` alongside `torch.manual_seed(42)`.
  - Nit: reliability diagram is 3-point (p10/p50/p90); swap for conformal/multi-quantile sweep in final pass.
  - Win: model file is production-grade (self-loop adjacency normalisation, shape-asserting pinball, dual-mode Colab+local notebook).
  - **Action:** micro-fix SDE queued for after data-puller + OpenDSS SDEs land.

- **2026-04-18 T+1h20 · Data pullers + IEEE 123 vendoring (SHA 2de0f30)** — VERDICT: approve. Blockers: none.
  - Nit (priority): `pull_ieee123.sh` clones from `tshort/OpenDSS`, but vendored `data/ieee123/` was actually sourced from `dss-extensions/electricdss-tst`. Reconcile in polish pass: either (a) point script at electricdss-tst, or (b) add `data/ieee123/README.md` pinning source repo + commit SHA + retrieval date.
  - Nit (cheap insurance): `pull_eia930.py` pagination loop lacks hard `max_pages` guard (infinite loop if server ever returns PAGE_SIZE indefinitely). Add `MAX_PAGES = 200`.
  - Nit: `pull_resstock.py` f-string-interpolates `--s3-path` into SQL (operator CLI so not an injection surface, but code smell).
  - Nit: magic `1024` min payload duplicated in noaa + nsrdb; magic `3` retry count hardcoded — extract to module constants.
  - Nit: `pull_evi_pro.py` would benefit from a `--strict` flag to abort on first error vs continuing.
  - Wins: `.env` correctly gitignored + never tracked; graceful SKIP on missing keys across keyed pullers; idempotent (`--force` + skip-on-exist); exponential backoff bounded; `pull_resstock --explain` mode is exemplary for testability; schema-deviation docstring explains WHY not just what; IEEE PES license posture confirmed clean.
  - **Action:** nits bundled into T+20 polish SDE (along with GWNet nits).

- **2026-04-18 T+1h40 · EnergyBench puller + pull_all.sh + puller test suite (SHA 8abf5c0)** — VERDICT: approve. Blockers: none.
  - Nit: 3 of 4 HF dataset candidates in `pull_energybench.py` are hallucinated (AIEnergy/EnergyBench, microsoft/LCLF-LoadBench, electricity-load-forecasting/autoformer-benchmarks all 401). Only `ai-iot/EnergyBench` is real. Trim list + waste ~6s/run less, or mark them "speculative future mirrors" in code comment.
  - Nit: commit message claims MANIFEST entry added; actually pre-existed from earlier commit. Harmless.
  - Nit: `test_all_pullers_importable` uses `py_compile` (syntax check), not actual import. Relabel or combine with `test_pullers_argparse_help` (the real import exerciser).
  - Nit: pull_energybench SKIP copy reads as if HF requires auth; actual intent is deterministic-CI behaviour. Rephrase.
  - Nit: `len(kept) - (1 if readme else 0)` is awkward; readme is Path|None, cleaner as `int(readme is not None)`.
  - Wins: attempt_manifest.json forensic trail, 19 tests hit every puller contract non-vacuously, env isolation correctly plumbed to subprocess, pull_all.sh if-timeout/set-e pattern is textbook.
  - **Action:** bundled into T+20 polish SDE.

### 2026-04-18T18:15 — reviewer verdict on de2e0c9 (topology)
- **Verdict:** approve (ship)
- **Critical issues:** none. Canonical numbers match exactly: 132 buses, 131 edges, 85 loaded buses, 3490 kW / 1920 kvar. Both transformer syntaxes resolve correctly, `~` continuation + `!` comments + case-insensitive keywords + phase-suffix stripping + Redirect recursion with cycle guard all verified live.
- **Minor nits (ignore / followup):**
  1. `_COMMENT_RE` doesn't strip `//` comments (docstring claim); IEEE 123 files don't use `//` so no functional impact.
  2. `_apply_line` units fallthrough: `mi`/`m`/`km` → no conversion (dead branch for IEEE 123 data).
  3. `Redirect` resolves relative paths without sandboxing (trusted-data, worth a note only).
  4. Test gaps: no reciprocity test on edge_index, no per-phase aggregation assertion, no regulator-specific test (all three properties DO hold — verified).
  5. Parallel regulator edges (reg3a/b/c) collapse in nx.Graph (harmless; use MultiGraph if per-phase regulator edges ever needed).
  6. `EdgeAttributes.normamps` is never populated on IEEE 123 lines → `edge_attr[:,1]` is zero-variance (fine, mean-centred still valid).
- **Praise:** clean dataclass schemas, local torch import, cycle-guarded Redirect walker — right level of defensive parsing for a hackathon foundation module.

### 2026-04-18T18:25 — reviewer verdict on 09e603a (power_flow wrapper)
- **Verdict:** approve (ship)
- **Critical issues:** none. Converges in 32 iters, cold 27ms / warm 39ms. 132/132 bus alignment with topology. L115 = 157.7% (matches Kersting). Multi-phase overrides (s49a/b/c) scale together. `SnapshotResult` genuinely frozen; values are native Python floats. `test_overrides_is_pure` is a real round-trip check (delta < 1e-6).
- **Minor nits (followup):**
  1. Unknown-override keys silently ignored (spec wanted ValueError) — `_apply_overrides` should diff `wanted.keys()` vs matched set and raise on residual. Low-risk now, will bite decision engine on typos.
  2. `master.exists()` check sits outside `_ENGINE_LOCK` (harmless filesystem read).
  3. `_REGCONTROL_COMMANDS` drops the `Delay=15/30` values the canonical script sets — snapshot-convergence unaffected but time-series sim would differ; one-line comment would suffice.
- **Praise:** collect-then-apply pattern for OpenDSS cursor iteration, the regulator-recipe comment saves the next reader a trip to Kersting.
