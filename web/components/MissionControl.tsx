"use client";

import { useScenario } from "@/lib/context";
import ScenarioSliders from "./ScenarioSliders";

const SEVERITY_BORDER: Record<string, string> = {
  error: "border-l-error",
  secondary: "border-l-secondary",
  tertiary: "border-l-tertiary",
  primary: "border-l-primary",
};

const SEVERITY_TEXT: Record<string, string> = {
  error: "text-error",
  secondary: "text-secondary",
  tertiary: "text-tertiary",
  primary: "text-primary",
};

function hourLabel(h: number): string {
  return `${String(h).padStart(2, "0")}:00`;
}

export default function MissionControl() {
  const { current } = useScenario();
  const { weather, feeder_rollup, recommended_actions } = current;

  const loadFactorPct = Math.max(0, feeder_rollup.load_factor * 100);
  const deltaPct = loadFactorPct - 100;
  // Normalise bar to a 150% scale so baseline (100%) sits at ~67% fill
  // and heat (139%) is visually distinct rather than both maxing out.
  const BAR_SCALE = 150;
  const barFillPct = Math.min(100, (loadFactorPct / BAR_SCALE) * 100);
  const refTickPct = (100 / BAR_SCALE) * 100; // position of the 100% reference mark
  const peakDisplay = (feeder_rollup.peak_mw ?? 0).toFixed(0);
  const capacityDisplay = (feeder_rollup.capacity_mw ?? 0).toFixed(0);

  return (
    <div className="flex flex-col font-mono text-xs uppercase tracking-widest">
      <ScenarioSliders />

      {/* WEATHER_DRIVER */}
      <div className="p-6 etched-b space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-primary tracking-[0.2em]">
            WEATHER_DRIVER
          </span>
          <span className="text-[9px] border border-outline-variant px-1.5 py-0.5 text-on-surface-variant">
            {weather.source.toUpperCase()}
          </span>
        </div>
        <div className="flex items-end justify-between">
          <div className="flex flex-col">
            <span className="text-[10px] text-on-surface-variant">
              PHOENIX_PEAK_TEMP
            </span>
            <span className="text-3xl font-bold text-secondary font-headline leading-none">
              {Math.round(weather.peak_temp_f)}°F
            </span>
          </div>
          <span
            className="material-symbols-outlined text-secondary text-4xl"
            style={{ fontVariationSettings: "'FILL' 1" }}
          >
            local_fire_department
          </span>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-on-surface-variant">
          <span className="material-symbols-outlined text-[12px]">schedule</span>
          <span>PEAK @ {hourLabel(weather.peak_hour)} MST</span>
        </div>
      </div>

      {/* FEEDER_ROLLUP */}
      <div className="p-6 etched-b space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-primary tracking-[0.2em]">
            FEEDER_ROLLUP
          </span>
          <span className="text-[9px] text-on-surface-variant">
            PEAK @ {hourLabel(feeder_rollup.peak_hour)}
          </span>
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-4xl font-bold text-primary font-headline leading-none">
            {peakDisplay}
          </span>
          <span className="text-[10px] text-on-surface-variant">MW</span>
        </div>
        <div className="space-y-1.5">
          <div className="flex justify-between items-center text-[10px]">
            <span>PEAK_vs_REF</span>
            <div className="flex items-center gap-2">
              {/* Delta badge — only shown when not baseline */}
              {Math.abs(deltaPct) >= 0.1 && (
                <span
                  className={`text-[9px] font-mono px-1 border ${
                    deltaPct > 0
                      ? "text-error border-error/50 bg-error/10"
                      : "text-primary border-primary/50 bg-primary/10"
                  }`}
                >
                  {deltaPct > 0 ? "▲" : "▼"}
                  {deltaPct > 0 ? "+" : ""}
                  {deltaPct.toFixed(1)}%
                </span>
              )}
              <span
                className={
                  loadFactorPct >= 100
                    ? "text-error"
                    : loadFactorPct >= 85
                      ? "text-secondary"
                      : "text-primary"
                }
              >
                {loadFactorPct.toFixed(1)}%
              </span>
            </div>
          </div>
          <div className="h-1.5 bg-surface-container-lowest w-full relative">
            <div
              className={
                loadFactorPct >= 100
                  ? "absolute top-0 left-0 h-full bg-error"
                  : loadFactorPct >= 85
                    ? "absolute top-0 left-0 h-full bg-secondary"
                    : "absolute top-0 left-0 h-full bg-primary"
              }
              style={{ width: `${barFillPct}%` }}
            />
            {/* Reference tick at the 100% / baseline mark */}
            <div
              className="absolute top-0 h-full w-px bg-on-surface-variant opacity-40"
              style={{ left: `${refTickPct}%` }}
            />
          </div>
          <div className="flex justify-between text-[9px] text-on-surface-variant">
            <span>BASELINE_PEAK</span>
            <span>{capacityDisplay} MW</span>
          </div>
        </div>
      </div>

      {/* RECOMMENDED_ACTIONS */}
      <div className="p-6 space-y-3">
        <span className="text-[10px] text-primary tracking-[0.2em] block">
          RECOMMENDED_ACTIONS
        </span>
        {recommended_actions.length === 0 ? (
          <div className="border-l-2 border-l-primary pl-3 py-1.5 bg-surface-container-low text-[10px] leading-tight">
            <span className="text-primary">NO IMMEDIATE ACTION REQUIRED — GRID OPERATING WITHIN NORMAL PARAMETERS</span>
          </div>
        ) : (
          <ul className="space-y-2">
            {recommended_actions.slice(0, 5).map((a, i) => (
              <li
                key={i}
                className={[
                  "border-l-2 pl-3 py-1.5 bg-surface-container-low text-[10px] leading-tight",
                  SEVERITY_BORDER[a.severity] ?? "border-l-primary",
                ].join(" ")}
              >
                <span className={SEVERITY_TEXT[a.severity] ?? "text-primary"}>
                  {a.label}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
