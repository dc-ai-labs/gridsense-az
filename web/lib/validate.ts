import type { TomorrowForecast } from "./types";

/**
 * Data-integrity assertions that guard against Slice B regressions.
 *
 *  - heat peak must be >= 1.3x baseline peak
 *  - EV peak hour must fall in [17, 22]
 *
 * Logs a console.warn (only in dev) instead of throwing so production is
 * resilient to upstream data surprises.
 */
export function validateForecasts(
  baseline: TomorrowForecast,
  heat: TomorrowForecast,
  ev: TomorrowForecast,
): string[] {
  const warnings: string[] = [];

  const basePeak = baseline.feeder_rollup.peak_mw;
  const heatPeak = heat.feeder_rollup.peak_mw;
  if (heatPeak < basePeak * 1.3) {
    warnings.push(
      `heat peak (${heatPeak.toFixed(1)} MW) is not >= 1.3 x baseline peak (${basePeak.toFixed(1)} MW)`,
    );
  }

  const evPeakHour = ev.feeder_rollup.peak_hour;
  if (evPeakHour < 17 || evPeakHour > 22) {
    warnings.push(
      `ev peak hour (${evPeakHour}) is outside expected [17, 22] range`,
    );
  }

  if (warnings.length && process.env.NODE_ENV !== "production") {
    for (const w of warnings) {
      // eslint-disable-next-line no-console
      console.warn(`[gridsense:validate] ${w}`);
    }
  }

  return warnings;
}

/** Tier color for a 0..1 risk score, matching MissionControl/Leaderboard styling. */
export function riskTier(score: number): "primary" | "secondary" | "error" {
  if (score > 0.7) return "error";
  if (score > 0.4) return "secondary";
  return "primary";
}
