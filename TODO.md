# TODO — prioritized by hackathon scoring leverage

## P0 — must have before submission
- [ ] **Slice C land + deploy** — side-by-side baseline vs stressed overlay in `ForecastRibbon` + `TacticalMap`, B/H/E/C keyboard shortcuts via `ScenarioProvider`. Judges' rubric explicitly calls for this ("strong teams will show side-by-side"). Currently in flight by a parallel SDE agent.
- [ ] **Slice B follow-up reviewer fixes** — stable `capacity_mw` across reruns, heat-encoder temp shift threaded through encoder history (not just decoder future inputs). In flight by a parallel SDE agent.
- [ ] Pitch deck (5 slides): problem / data / model / scenarios / why APS.
- [ ] 60-90 s product reel — screen recording of the dashboard demonstrating the scenario toggle end-to-end (B → H → E → C).
- [ ] `ui-verifier` pass against https://gridsense-az.vercel.app — click every button, smoke-test the scenario toggle, grab screenshots for the deck.
- [ ] Final `git push origin main` once slices B/C have landed (manager only).

## P1 — strong but not blocking
- [ ] Stress-period MAE breakout — report summer-peak MAE (June-August 17:00-21:00 local) separately from overall hold-out MAE. Juices the forecasting scoring pillar because our win is largest in the stressed window.
- [ ] Beat persistence on summer-evening stress window (currently -61.7% vs persistence on 2154 stress hours). Likely needs: (a) richer temp exogenous, (b) longer training, (c) separate stress-window loss weight.
- [ ] Real Captum Integrated-Gradients attributions for `top_drivers` in the dashboard (currently heuristic placeholders like "temp_c", "hour_sin").
- [ ] OpenDSS scale-mismatch narrative note in README and ARCHITECTURE — explicitly call out that IEEE 123 is a 3.5 MW nameplate feeder while EIA-930 AZPS is a 3.5 GW balancing area. The feeder is a topology scaffold for voltage-drop narrative, not a literal 1:1 grid.
- [ ] Replay mode — hardcode a real summer heat-wave day (e.g., 2023-07-15 when Phoenix hit 118 F) as a demo preset, alongside "tomorrow live NWS."
- [ ] UI polish:
  - [ ] Replace hardcoded `REPL_LAG: 42ms` in `BottomStatus` (reviewer flag) with a real or removed metric
  - [ ] Resolve `href="#"` placeholders in `TopNav` (reviewer flag)

## P2 — nice-to-have
- [ ] Vercel <-> GitHub auto-deploy integration (currently Vercel is unlinked; redeploys are manual).
- [ ] Dark/light theme toggle (currently dark only).
- [ ] Export forecast JSON as CSV for operators.
- [ ] SSE/websocket live ticker on `BottomStatus` (replace the fake REPL_LAG with a real per-fetch round-trip).

## Backlog / deferred
- [ ] Train a v1 checkpoint on more data (include 2021 + 2024 summers).
- [ ] Integrate NREL EVI-Pro DES trajectories for EV-scenario realism (previously blocked — the module isn't publicly downloadable; revisit via NREL collab if we're past submission).
- [ ] Integrate NSRDB irradiance as an exogenous feature (not in current model; could help on clear-sky cooling days).
