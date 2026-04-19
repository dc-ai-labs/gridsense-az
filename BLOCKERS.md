# BLOCKERS.md — what needs human attention

## ✅ Resolved
- (none right now)

## ⚠️ Known but working-around

### NSRDB PSM3 endpoint returns 404 (2026-04-18T18:20)
- URL tried: `https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?names=2020..2023&...`
- NREL appears to have deprecated this endpoint (PSM3 v2). Newer endpoint may be `v3` or the Himawari/MTS2 variant.
- **Impact:** no solar irradiance (GHI/DNI/DHI) in the feature bundle. NOAA still provides temp/dewpoint/humidity/wind — the demand-driving signals. Solar irradiance was a nice-to-have for future solar-generation modelling.
- **Workaround:** feature pipeline treats irradiance as optional; model trains on NOAA+EIA+calendar. If time at polish, switch to PSM3 v3 or fall back to ERA5 reanalysis.
- **Not blocking training.**

### EIA-930 AZPS has outliers (max 101 GW, min -22 GW)
- Typical AZPS demand: 2-8 GW. Outliers are clearly source-data glitches in the EIA v2 raw stream.
- **Workaround:** feature pipeline winsorises `value` to [500, 10000] MW before use. Document in features.py.

### EVI-Pro Lite API: 400 Bad Request (2026-04-18T18:20)
- NREL endpoint rejects our parameter combination. Likely the `home_access_dist=REAL_ESTATE` enum value is outdated — their API spec may have changed.
- **Impact:** no EV daily-load profiles for stress-test scenarios.
- **Workaround:** synthesize EV load overlay from first principles (typical L2 charging 7-9 pm, fleet-size sweep) — simpler and controllable.
- Not blocking demand forecast; only the stress-test scenario.

## 🛑 Actually blocking (need user)
- (none right now)
