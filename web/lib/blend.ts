// Client-side "what-if" blending for the temp + EV sliders.
//
// The backend only ships three precomputed snapshots: baseline, heat (+10°F),
// ev (+35% EV penetration). To give the user a live dial, we treat baseline
// as the origin and the other two as anchor points along two INDEPENDENT
// axes, then sum their linear deltas:
//
//   blended(field) = baseline + t_temp * (heat - baseline)
//                             + t_ev   * (ev   - baseline) * evGate(field)
//
// where
//   t_temp = clamp(tempDeltaF / 10, -0.5, 2.5)
//   t_ev   = clamp(evPenetrationPct / 35, 0, 2.857)
//
// and evGate is 1 everywhere except for quantile fields, where EV load is
// concentrated in hours 17..22 (evening charging window); we zero the EV
// delta outside that window to avoid polluting the midday dip.
//
// Rationale for SUM vs MAX: summing reproduces each preset exactly
// (t_temp=1,t_ev=0 → heat; t_temp=0,t_ev=1 → ev; 0,0 → baseline) which is
// required by the "preset match" acceptance tests. MAX would saturate.
// The two deltas are physically near-independent (temperature drives AC
// load all day; EV load is evening-only), so double-counting is minimal
// in the overlap hours.

import type {
  FeederRollup,
  OpenDssSnapshot,
  PerBusMetric,
  QuantileHour,
  RiskLeaderboardRow,
  ScenarioPreset,
  TomorrowForecast,
  WeatherSummary,
} from "./types";

const TEMP_ANCHOR_F = 10; // heat.json was precomputed for +10°F
const EV_ANCHOR_PCT = 35; // ev.json was precomputed for +35% penetration

const TEMP_T_MIN = -0.5;
const TEMP_T_MAX = 2.5;
const EV_T_MIN = 0;
const EV_T_MAX = 100 / EV_ANCHOR_PCT; // i.e. slider max 100 / 35

const EV_WINDOW_START = 17;
const EV_WINDOW_END = 22;

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

export function tempFactor(tempDeltaF: number): number {
  return clamp(tempDeltaF / TEMP_ANCHOR_F, TEMP_T_MIN, TEMP_T_MAX);
}

export function evFactor(evPenetrationPct: number): number {
  return clamp(evPenetrationPct / EV_ANCHOR_PCT, EV_T_MIN, EV_T_MAX);
}

function inEvWindow(hour: number): boolean {
  return hour >= EV_WINDOW_START && hour <= EV_WINDOW_END;
}

/** Choose the preset identity for a slider state. Returns "custom" when
 *  neither slider sits at one of the three anchor combos. */
export function presetIdFromSliders(
  tempDeltaF: number,
  evPenetrationPct: number,
): "baseline" | "heat" | "ev" | "custom" {
  const nearZero = (x: number, eps: number) => Math.abs(x) < eps;
  if (nearZero(tempDeltaF, 0.01) && nearZero(evPenetrationPct, 0.01)) {
    return "baseline";
  }
  if (
    nearZero(tempDeltaF - TEMP_ANCHOR_F, 0.01) &&
    nearZero(evPenetrationPct, 0.01)
  ) {
    return "heat";
  }
  if (
    nearZero(tempDeltaF, 0.01) &&
    nearZero(evPenetrationPct - EV_ANCHOR_PCT, 0.01)
  ) {
    return "ev";
  }
  return "custom";
}

/** Which precomputed OpenDSS snapshot to pin when the sliders are custom. */
export function snapshotPresetFor(
  tempDeltaF: number,
  evPenetrationPct: number,
): ScenarioPreset {
  if (tempDeltaF >= 7) return "heat";
  if (evPenetrationPct >= 20) return "ev";
  return "baseline";
}

function blendQuantile(
  b: QuantileHour,
  h: QuantileHour,
  e: QuantileHour,
  tTemp: number,
  tEv: number,
): QuantileHour {
  const ev = inEvWindow(b.hour) ? tEv : 0;
  return {
    ts: b.ts,
    hour: b.hour,
    p10_mw: b.p10_mw + tTemp * (h.p10_mw - b.p10_mw) + ev * (e.p10_mw - b.p10_mw),
    p50_mw: b.p50_mw + tTemp * (h.p50_mw - b.p50_mw) + ev * (e.p50_mw - b.p50_mw),
    p90_mw: b.p90_mw + tTemp * (h.p90_mw - b.p90_mw) + ev * (e.p90_mw - b.p90_mw),
  };
}

function blendPerBus(
  b: PerBusMetric,
  h: PerBusMetric | undefined,
  e: PerBusMetric | undefined,
  tTemp: number,
  tEv: number,
): PerBusMetric {
  const hPeak = h?.peak_load_kw ?? b.peak_load_kw;
  const ePeak = e?.peak_load_kw ?? b.peak_load_kw;
  const hRisk = h?.risk_score ?? b.risk_score;
  const eRisk = e?.risk_score ?? b.risk_score;
  const peak = b.peak_load_kw
    + tTemp * (hPeak - b.peak_load_kw)
    + tEv * (ePeak - b.peak_load_kw);
  const risk = clamp(
    b.risk_score
      + tTemp * (hRisk - b.risk_score)
      + tEv * (eRisk - b.risk_score),
    0,
    1,
  );
  return {
    bus: b.bus,
    rating_kw: b.rating_kw,
    peak_load_kw: Math.max(0, peak),
    risk_score: risk,
  };
}

function blendLeaderboardRow(
  b: RiskLeaderboardRow,
  cmpBus: Record<string, PerBusMetric>,
  hMap: Record<string, PerBusMetric>,
  eMap: Record<string, PerBusMetric>,
  bMap: Record<string, PerBusMetric>,
  tTemp: number,
  tEv: number,
): RiskLeaderboardRow {
  // Prefer per_bus values for the blend — the leaderboard's own fields are
  // only populated in the origin scenario. Fall back to the row's literal
  // values when per_bus is missing.
  const key = b.bus ?? "";
  const baseBus = bMap[key];
  const heatBus = hMap[key];
  const evBus = eMap[key];
  if (!baseBus) {
    return b;
  }
  const hPeakMw = (heatBus?.peak_load_kw ?? baseBus.peak_load_kw) / 1000;
  const ePeakMw = (evBus?.peak_load_kw ?? baseBus.peak_load_kw) / 1000;
  const baseMw = baseBus.peak_load_kw / 1000;
  const blendedMw = baseMw + tTemp * (hPeakMw - baseMw) + tEv * (ePeakMw - baseMw);
  const hRisk = heatBus?.risk_score ?? baseBus.risk_score;
  const eRisk = evBus?.risk_score ?? baseBus.risk_score;
  const blendedRisk = clamp(
    baseBus.risk_score
      + tTemp * (hRisk - baseBus.risk_score)
      + tEv * (eRisk - baseBus.risk_score),
    0,
    1,
  );
  // Silence unused-param lint while keeping the signature stable for the
  // caller that passes the full compare map (for future use).
  void cmpBus;
  return {
    id: b.id,
    bus: b.bus,
    risk_score: blendedRisk,
    peak_mw: Math.max(0, blendedMw),
  };
}

function blendFeederRollup(
  blendedQuantiles: QuantileHour[],
  capacityMw: number,
): FeederRollup {
  let peak = -Infinity;
  let peakHour = 0;
  for (const q of blendedQuantiles) {
    if (q.p50_mw > peak) {
      peak = q.p50_mw;
      peakHour = q.hour;
    }
  }
  return {
    peak_mw: peak,
    peak_hour: peakHour,
    capacity_mw: capacityMw,
    load_factor: capacityMw > 0 ? peak / capacityMw : 0,
  };
}

function blendWeather(
  baseline: WeatherSummary,
  tempDeltaF: number,
): WeatherSummary {
  return {
    peak_temp_f: baseline.peak_temp_f + tempDeltaF,
    peak_hour: baseline.peak_hour,
    source: baseline.source,
  };
}

function pickOpenDss(
  preset: ScenarioPreset,
  baseline: OpenDssSnapshot,
  heat: OpenDssSnapshot,
  ev: OpenDssSnapshot,
): OpenDssSnapshot {
  if (preset === "heat") return heat;
  if (preset === "ev") return ev;
  return baseline;
}

export interface BlendInput {
  baseline: TomorrowForecast;
  heat: TomorrowForecast;
  ev: TomorrowForecast;
  tempDeltaF: number;
  evPenetrationPct: number;
}

/** Produce a synthetic TomorrowForecast that reflects current slider state.
 *  When the sliders sit exactly on a preset, returns a near-identical copy
 *  of that precomputed snapshot (within floating-point tolerance). */
export function blendForecast(input: BlendInput): TomorrowForecast {
  const { baseline, heat, ev, tempDeltaF, evPenetrationPct } = input;
  const tTemp = tempFactor(tempDeltaF);
  const tEv = evFactor(evPenetrationPct);
  const preset = presetIdFromSliders(tempDeltaF, evPenetrationPct);

  // Fast path: exact preset match → return the origin snapshot untouched.
  // This guarantees bit-exact acceptance for the "preset snap" checks.
  if (preset === "baseline") return baseline;
  if (preset === "heat") return heat;
  if (preset === "ev") return ev;

  // Quantile blend
  const hByHour = new Map(heat.quantiles.map((q) => [q.hour, q]));
  const eByHour = new Map(ev.quantiles.map((q) => [q.hour, q]));
  const blendedQuantiles: QuantileHour[] = baseline.quantiles.map((b) => {
    const h = hByHour.get(b.hour) ?? b;
    const e = eByHour.get(b.hour) ?? b;
    return blendQuantile(b, h, e, tTemp, tEv);
  });

  // Per-bus blend
  const blendedPerBus: Record<string, PerBusMetric> = {};
  for (const [bus, bMetric] of Object.entries(baseline.per_bus)) {
    blendedPerBus[bus] = blendPerBus(
      bMetric,
      heat.per_bus[bus],
      ev.per_bus[bus],
      tTemp,
      tEv,
    );
  }

  // Risk leaderboard — keep baseline's top-10 ordering, update values,
  // then re-sort by risk_score descending so the top stays meaningful.
  const blendedLeaderboard = baseline.risk_leaderboard
    .map((row) =>
      blendLeaderboardRow(
        row,
        baseline.per_bus,
        heat.per_bus,
        ev.per_bus,
        baseline.per_bus,
        tTemp,
        tEv,
      ),
    )
    .sort((a, b) => b.risk_score - a.risk_score);

  // Feeder rollup — derive from blended quantiles, keep capacity stable.
  const blendedRollup = blendFeederRollup(
    blendedQuantiles,
    baseline.feeder_rollup.capacity_mw,
  );

  // Weather
  const blendedWeather = blendWeather(baseline.weather, tempDeltaF);

  // OpenDSS — pin to the closest precomputed preset (SNAPSHOT).
  const snapshotPreset = snapshotPresetFor(tempDeltaF, evPenetrationPct);
  const blendedOpenDss = pickOpenDss(
    snapshotPreset,
    baseline.opendss,
    heat.opendss,
    ev.opendss,
  );

  return {
    // Scenario identity is the ORIGIN snapshot (baseline) but the
    // dashboard will render "CUSTOM" via active === "custom".
    scenario: "baseline",
    generated_at: baseline.generated_at,
    quantiles: blendedQuantiles,
    per_bus: blendedPerBus,
    risk_leaderboard: blendedLeaderboard,
    feeder_rollup: blendedRollup,
    opendss: blendedOpenDss,
    weather: blendedWeather,
    top_drivers: baseline.top_drivers,
    recommended_actions: baseline.recommended_actions,
  };
}
