import { describe, it, expect } from "vitest";
import {
  tempFactor,
  evFactor,
  presetIdFromSliders,
  blendForecast,
  type BlendInput,
} from "./blend";
import type {
  OpenDssSnapshot,
  PerBusMetric,
  QuantileHour,
  RecommendedAction,
  RiskLeaderboardRow,
  ScenarioPreset,
  TomorrowForecast,
  TopDriver,
  WeatherSummary,
} from "./types";

// ---------------------------------------------------------------------------
// Scalar factor tests
// ---------------------------------------------------------------------------

describe("tempFactor", () => {
  it("returns 0 at zero delta", () => {
    expect(tempFactor(0)).toBe(0);
  });

  it("snaps to 1 at the +10F heat anchor", () => {
    expect(tempFactor(10)).toBe(1);
  });

  it("clamps extreme positive delta to the 2.5 ceiling", () => {
    expect(tempFactor(25)).toBe(2.5);
  });

  it("applies asymmetric HALF sensitivity on the cool side — -5F gives power-1.5 result", () => {
    // linearN = 5/20 = 0.25 → -Math.pow(0.25, 1.5) ≈ -0.125
    expect(tempFactor(-5)).toBeCloseTo(-Math.pow(0.25, 1.5), 10);
  });

  it("applies asymmetric HALF sensitivity for a mild cool delta", () => {
    // linearN = 2/20 = 0.1 → -Math.pow(0.1, 1.5) ≈ -0.031623
    expect(tempFactor(-2)).toBeCloseTo(-Math.pow(0.1, 1.5), 10);
  });

  it("clamps extreme negative delta to the -0.25 floor", () => {
    expect(tempFactor(-20)).toBe(-0.25);
  });
});

describe("evFactor", () => {
  it("returns 0 at zero penetration", () => {
    expect(evFactor(0)).toBe(0);
  });

  it("snaps to 1 at the 35% EV anchor", () => {
    expect(evFactor(35)).toBe(1);
  });

  it("linearly extrapolates past the anchor", () => {
    expect(evFactor(100)).toBeCloseTo(100 / 35, 12);
  });

  it("clamps negative penetration to zero", () => {
    expect(evFactor(-10)).toBe(0);
  });
});

describe("presetIdFromSliders", () => {
  it("maps (0, 0) to baseline", () => {
    expect(presetIdFromSliders(0, 0)).toBe("baseline");
  });

  it("maps (10, 0) to heat", () => {
    expect(presetIdFromSliders(10, 0)).toBe("heat");
  });

  it("maps (0, 35) to ev", () => {
    expect(presetIdFromSliders(0, 35)).toBe("ev");
  });

  it("maps (5, 35) to custom", () => {
    expect(presetIdFromSliders(5, 35)).toBe("custom");
  });

  it("maps (-5, 100) — the user's reported case — to custom", () => {
    expect(presetIdFromSliders(-5, 100)).toBe("custom");
  });
});

// ---------------------------------------------------------------------------
// Fixture builders for permutation matrix
// ---------------------------------------------------------------------------

function makeQuantiles(p50ByHour: (hour: number) => number): QuantileHour[] {
  const rows: QuantileHour[] = [];
  for (let h = 0; h < 24; h += 1) {
    const p50 = p50ByHour(h);
    rows.push({
      ts: `2026-04-20T${String(h).padStart(2, "0")}:00:00Z`,
      hour: h,
      p10_mw: p50 * 0.9,
      p50_mw: p50,
      p90_mw: p50 * 1.1,
    });
  }
  return rows;
}

function makePerBus(peakKw: number, risk: number): Record<string, PerBusMetric> {
  return {
    "1": {
      bus: "1",
      rating_kw: 500,
      peak_load_kw: peakKw,
      risk_score: risk,
    },
    "150": {
      bus: "150",
      rating_kw: 500,
      peak_load_kw: peakKw * 0.8,
      risk_score: risk * 0.8,
    },
  };
}

function makeLeaderboard(peakMw: number, risk: number): RiskLeaderboardRow[] {
  return [
    { id: "F-1", bus: "1", risk_score: risk, peak_mw: peakMw },
    { id: "F-150", bus: "150", risk_score: risk * 0.8, peak_mw: peakMw * 0.8 },
  ];
}

function makeOpendss(scenario: ScenarioPreset): OpenDssSnapshot {
  return {
    converged: true,
    scenario,
    top_bus_deviations: [],
    overloads: [],
  };
}

function makeWeather(peakF: number): WeatherSummary {
  return { peak_temp_f: peakF, peak_hour: 17, source: "synthetic" };
}

const TOP_DRIVERS: TopDriver[] = [{ name: "temp_air_f", ig: 0.5 }];
const ACTIONS: RecommendedAction[] = [
  { label: "Pre-cool critical buses", severity: "primary" },
];

function makeForecast(
  scenario: ScenarioPreset,
  quantiles: QuantileHour[],
  per_bus: Record<string, PerBusMetric>,
  leaderboard: RiskLeaderboardRow[],
  peakMw: number,
  peakHour: number,
  capacityMw: number,
  weatherF: number,
): TomorrowForecast {
  return {
    scenario,
    generated_at: "2026-04-19T12:00:00Z",
    quantiles,
    per_bus,
    risk_leaderboard: leaderboard,
    feeder_rollup: {
      peak_mw: peakMw,
      peak_hour: peakHour,
      capacity_mw: capacityMw,
      load_factor: capacityMw > 0 ? peakMw / capacityMw : 0,
    },
    opendss: makeOpendss(scenario),
    weather: makeWeather(weatherF),
    top_drivers: TOP_DRIVERS,
    recommended_actions: ACTIONS,
  };
}

// ---------------------------------------------------------------------------
// Permutation matrix — simple flat-load fixtures
// ---------------------------------------------------------------------------

describe("blendForecast permutations", () => {
  // Baseline: flat 100 MW every hour.
  // Heat: flat 150 MW every hour (+50 across the board).
  // EV: 100 MW everywhere except hours 17..22 which are 130 (+30 only in EV window).
  const baseline = makeForecast(
    "baseline",
    makeQuantiles(() => 100),
    makePerBus(100_000, 0.1),
    makeLeaderboard(100, 0.1),
    100,
    12,
    3600,
    95,
  );
  const heat = makeForecast(
    "heat",
    makeQuantiles(() => 150),
    makePerBus(150_000, 0.3),
    makeLeaderboard(150, 0.3),
    150,
    13,
    3600,
    105,
  );
  const ev = makeForecast(
    "ev",
    makeQuantiles((h) => (h >= 17 && h <= 22 ? 130 : 100)),
    makePerBus(130_000, 0.2),
    makeLeaderboard(130, 0.2),
    130,
    19,
    3600,
    95,
  );

  function run(t: number, e: number): TomorrowForecast {
    const input: BlendInput = {
      baseline,
      heat,
      ev,
      tempDeltaF: t,
      evPenetrationPct: e,
    };
    return blendForecast(input);
  }

  interface Case {
    t: number;
    e: number;
    expectBaseline?: boolean;
    expectPeak?: number;
    expectMin?: number;
    label: string;
    preset?: "baseline" | "heat" | "ev" | "custom";
  }

  const cases: Case[] = [
    { t: 0, e: 0, expectBaseline: true, label: "baseline preset", preset: "baseline" },
    { t: 10, e: 0, expectPeak: 150, label: "heat preset", preset: "heat" },
    { t: 0, e: 35, expectPeak: 130, label: "ev preset", preset: "ev" },
    { t: -5, e: 0, expectPeak: 100 + -Math.pow(0.25, 1.5) * 50, label: "mild cool relief", preset: "custom" },
    { t: 0, e: 100, expectMin: 100, label: "100% EV must be >= baseline", preset: "custom" },
    { t: -5, e: 100, expectMin: 101, label: "CRITICAL user case (-5F, 100% EV) must beat baseline", preset: "custom" },
    {
      // Peak occurs in the EV evening window. Heat-delta applies full at
      // tTemp=1 (+50), EV-delta applies at tEv=100/35 (+30 * 100/35) only
      // in hours 17..22. Full-blend per-hour = 100 + 50 + (100/35)*30.
      t: 10,
      e: 100,
      expectPeak: 100 + 50 + (100 / 35) * 30,
      label: "heat + 100% EV",
      preset: "custom",
    },
    { t: 25, e: 100, expectMin: 150, label: "extreme heat + EV", preset: "custom" },
    { t: 5, e: 17, expectMin: 100, label: "mid-range custom", preset: "custom" },
  ];

  for (const c of cases) {
    it(`case ${c.label} (t=${c.t}, e=${c.e})`, () => {
      const out = run(c.t, c.e);
      if (c.preset !== undefined) {
        expect(presetIdFromSliders(c.t, c.e)).toBe(c.preset);
      }
      if (c.expectBaseline) {
        expect(out.feeder_rollup.peak_mw).toBeCloseTo(baseline.feeder_rollup.peak_mw, 12);
      }
      if (c.expectPeak !== undefined) {
        expect(out.feeder_rollup.peak_mw).toBeCloseTo(c.expectPeak, 6);
      }
      if (c.expectMin !== undefined) {
        expect(out.feeder_rollup.peak_mw).toBeGreaterThan(c.expectMin);
      }
    });
  }
});

// ---------------------------------------------------------------------------
// CRITICAL regression test — exact reported user case
// ---------------------------------------------------------------------------

describe("blendForecast — user regression: (-5F, 100% EV) must not drop below baseline", () => {
  // Use production-equivalent numbers from the freshly regenerated snapshots:
  //   BASELINE peak = 3675 MW @ h19
  //   HEAT peak     = 5139 MW @ h13 (midday AC spike)  — but we model it as
  //                   a uniform +~450 MW lift at h19 too (heat presses evening)
  //   EV peak       = 3963 MW @ h19 (+288 MW over baseline at h19)
  //
  // Reproduce with synthetic-but-ratio-accurate fixtures. Baseline is flat
  // 3675. Heat hits 4410 at h19 (+735 i.e. same shape). EV hits 3963 at h19.
  const BASELINE_MW = 3675;
  const HEAT_MW_AT_19 = 4410;
  const EV_MW_AT_19 = 3963;

  const baseline = makeForecast(
    "baseline",
    makeQuantiles(() => BASELINE_MW),
    makePerBus(BASELINE_MW * 1000 / 132, 0.1),
    makeLeaderboard(BASELINE_MW, 0.1),
    BASELINE_MW,
    19,
    3600,
    95,
  );
  const heat = makeForecast(
    "heat",
    makeQuantiles((h) => (h === 19 ? HEAT_MW_AT_19 : BASELINE_MW * 1.15)),
    makePerBus((HEAT_MW_AT_19 * 1000) / 132, 0.3),
    makeLeaderboard(HEAT_MW_AT_19, 0.3),
    HEAT_MW_AT_19,
    19,
    3600,
    110,
  );
  const ev = makeForecast(
    "ev",
    makeQuantiles((h) => (h >= 17 && h <= 22 ? EV_MW_AT_19 : BASELINE_MW)),
    makePerBus((EV_MW_AT_19 * 1000) / 132, 0.2),
    makeLeaderboard(EV_MW_AT_19, 0.2),
    EV_MW_AT_19,
    19,
    3600,
    95,
  );

  it("blend at (-5, 100) produces peak_mw strictly greater than baseline peak", () => {
    const blended = blendForecast({
      baseline,
      heat,
      ev,
      tempDeltaF: -5,
      evPenetrationPct: 100,
    });
    // More EVs on a cool day must still push peak UP vs the baseline —
    // never DOWN. This is the direct regression test for the user-reported
    // bug where the pipeline showed 91% of baseline.
    expect(blended.feeder_rollup.peak_mw).toBeGreaterThan(
      baseline.feeder_rollup.peak_mw,
    );
  });

  it("blend at (-5, 100) outperforms the naive fully-mirrored cool model", () => {
    // With power-1.5 ramp: linearN = 5/20 = 0.25 → factor = -Math.pow(0.25,1.5) ≈ -0.125.
    // This is gentler than the old -0.25 linear factor.
    const blended = blendForecast({
      baseline,
      heat,
      ev,
      tempDeltaF: -5,
      evPenetrationPct: 100,
    });
    // Peak is NOT at h19 but at the other EV-window hours (17,18,20,21,22)
    // where heat is at its 1.15× floor (4226.25) rather than the h19 spike.
    // tTemp = -Math.pow(0.25, 1.5), tEv = 100/35
    //   non-h19 EV-window hour: heat=3675*1.15=4226.25, ev=3963.
    //   blended = 3675 + tTemp*(4226.25-3675) + (100/35)*(3963-3675)
    // At h19: heat=4410, ev=3963 → 3675 + tTemp*735 + (100/35)*288
    // Non-h19 shoulder is the peak. Assert it.
    const tTemp = -Math.pow(0.25, 1.5);
    const expectedPeak =
      3675 + tTemp * (3675 * 1.15 - 3675) + (100 / 35) * (3963 - 3675);
    expect(blended.feeder_rollup.peak_mw).toBeCloseTo(expectedPeak, 2);
  });
});

// ---------------------------------------------------------------------------
// Action-blending tests — verify recommended_actions tracks slider state
// ---------------------------------------------------------------------------

describe("blendForecast recommended_actions selection", () => {
  const BASE_ACTIONS: RecommendedAction[] = [
    { label: "BASE1", severity: "primary" },
    { label: "BASE2", severity: "primary" },
  ];
  const HEAT_ACTIONS: RecommendedAction[] = [
    { label: "HEAT1", severity: "secondary" },
    { label: "HEAT2", severity: "secondary" },
    { label: "HEAT3", severity: "secondary" },
  ];
  const EV_ACTIONS: RecommendedAction[] = [
    { label: "EV1", severity: "tertiary" },
    { label: "EV2", severity: "tertiary" },
    { label: "EV3", severity: "tertiary" },
  ];

  function makeFcWithActions(
    scenario: ScenarioPreset,
    actions: RecommendedAction[],
  ): TomorrowForecast {
    return {
      scenario,
      generated_at: "2026-04-19T12:00:00Z",
      quantiles: makeQuantiles(() => 100),
      per_bus: makePerBus(100_000, 0.1),
      risk_leaderboard: makeLeaderboard(100, 0.1),
      feeder_rollup: {
        peak_mw: 100,
        peak_hour: 12,
        capacity_mw: 3600,
        load_factor: 100 / 3600,
      },
      opendss: makeOpendss(scenario),
      weather: makeWeather(95),
      top_drivers: TOP_DRIVERS,
      recommended_actions: actions,
    };
  }

  const baseline = makeFcWithActions("baseline", BASE_ACTIONS);
  const heat = makeFcWithActions("heat", HEAT_ACTIONS);
  const ev = makeFcWithActions("ev", EV_ACTIONS);

  function run(t: number, e: number): TomorrowForecast {
    return blendForecast({
      baseline,
      heat,
      ev,
      tempDeltaF: t,
      evPenetrationPct: e,
    });
  }

  it("returns baseline actions when neither slider is engaged (0, 0)", () => {
    const out = run(0, 0);
    expect(out.recommended_actions.map((a) => a.label)).toEqual(["BASE1", "BASE2"]);
  });

  it("returns heat actions when only temp slider is engaged (10, 0)", () => {
    const out = run(10, 0);
    expect(out.recommended_actions.map((a) => a.label)).toEqual([
      "HEAT1",
      "HEAT2",
      "HEAT3",
    ]);
  });

  it("returns ev actions when only EV slider is engaged (0, 50)", () => {
    const out = run(0, 50);
    expect(out.recommended_actions.map((a) => a.label)).toEqual([
      "EV1",
      "EV2",
      "EV3",
    ]);
  });

  it("returns mixed heat+ev actions when both sliders are engaged (10, 50)", () => {
    const out = run(10, 50);
    expect(out.recommended_actions.map((a) => a.label)).toEqual([
      "HEAT1",
      "HEAT2",
      "EV1",
      "EV2",
    ]);
  });
});

