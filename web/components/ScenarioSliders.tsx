"use client";

import { useScenario } from "@/lib/context";
import type { ScenarioKind, ScenarioPreset } from "@/lib/types";

interface Option {
  key: ScenarioPreset;
  label: string;
  shortcut: "B" | "H" | "E";
  sublabel?: string;
}

const OPTIONS: Option[] = [
  { key: "baseline", label: "BASELINE", shortcut: "B", sublabel: "NOAA" },
  { key: "heat", label: "HEAT +10°F", shortcut: "H", sublabel: "STRESS" },
  { key: "ev", label: "EV SURGE", shortcut: "E", sublabel: "+35%" },
];

type ComparePill = {
  key: ScenarioPreset | null;
  label: string;
};

const COMPARE_PILLS: ComparePill[] = [
  { key: null, label: "OFF" },
  { key: "baseline", label: "BASELINE" },
  { key: "heat", label: "HEAT" },
  { key: "ev", label: "EV" },
];

const TEMP_MIN = -5;
const TEMP_MAX = 25;
const EV_MIN = 0;
const EV_MAX = 100;

function formatTemp(v: number): string {
  const rounded = Math.round(v);
  const sign = rounded > 0 ? "+" : rounded < 0 ? "" : "±";
  return `${sign}${rounded}°F`;
}

function formatEv(v: number): string {
  return `${Math.round(v)}%`;
}

/**
 * Three-way scenario toggle + live temp/EV sliders.
 *
 *  - Clicking B/H/E snaps both sliders to their preset combo.
 *  - Dragging either slider moves the state to "custom" (no preset is hot).
 *  - [R] or the RESET pill returns to baseline (0/0).
 *  - Slider state persists in sessionStorage across reloads.
 */
export default function ScenarioSliders() {
  const {
    active,
    setActive,
    compareWith,
    setCompareWith,
    tempDeltaF,
    setTempDeltaF,
    evPenetrationPct,
    setEvPenetrationPct,
    resetSliders,
    current,
  } = useScenario();

  const isCustom = active === "custom";
  const isBaseline = active === "baseline";

  return (
    <div className="p-6 etched-b space-y-5">
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-primary tracking-widest">
          SCENARIO_CONTROLS
          {isCustom && (
            <span className="ml-2 text-secondary">· CUSTOM</span>
          )}
        </span>
        <span className="text-[9px] text-on-surface-variant">
          {current.weather.source.toUpperCase()}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-1.5">
        {OPTIONS.map((opt) => {
          const isActive = active === opt.key;
          return (
            <button
              key={opt.key}
              type="button"
              onClick={() => setActive(opt.key as ScenarioKind)}
              data-active={isActive}
              className={[
                "group flex flex-col items-center justify-center py-2 px-1 border transition-colors",
                isActive
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-outline-variant text-on-surface-variant hover:border-primary/40 hover:text-primary",
              ].join(" ")}
              aria-pressed={isActive}
              title={`Press ${opt.shortcut} to activate ${opt.label}`}
            >
              <span className="text-[9px] font-bold tracking-widest leading-none">
                {opt.label}
              </span>
              <span className="text-[8px] opacity-70 mt-1">
                [{opt.shortcut}] {opt.sublabel}
              </span>
            </button>
          );
        })}
      </div>

      {/* COMPARE_WITH */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <span className="text-[9px] text-on-surface-variant tracking-widest uppercase">
            COMPARE_WITH
          </span>
          <span className="text-[8px] text-on-surface-variant opacity-70">
            [C] CYCLE
          </span>
        </div>
        <div className="grid grid-cols-4 gap-1">
          {COMPARE_PILLS.map((pill) => {
            const isActive = compareWith === pill.key;
            const isSelfCompare =
              pill.key !== null && pill.key === active;
            const disabled = isSelfCompare;
            return (
              <button
                key={pill.key ?? "off"}
                type="button"
                disabled={disabled}
                onClick={() => setCompareWith(pill.key)}
                data-active={isActive}
                className={[
                  "py-1 px-1 border text-[9px] font-bold uppercase tracking-widest transition-colors",
                  disabled
                    ? "border-outline-variant/30 text-on-surface-variant/30 cursor-not-allowed"
                    : isActive
                      ? "border-primary bg-primary text-on-primary"
                      : "border-outline-variant text-on-surface-variant hover:border-primary/40 hover:text-primary",
                ].join(" ")}
                aria-pressed={isActive}
                title={
                  disabled
                    ? "Cannot compare a scenario with itself"
                    : pill.key === null
                      ? "Clear comparison overlay"
                      : `Overlay ${pill.label} on top of active scenario`
                }
              >
                {pill.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* WHAT-IF SLIDERS */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-[9px] text-on-surface-variant tracking-widest uppercase">
            WHAT_IF
          </span>
          <button
            type="button"
            onClick={resetSliders}
            disabled={isBaseline}
            className={[
              "px-2 py-0.5 text-[8px] font-bold tracking-widest uppercase border transition-colors",
              isBaseline
                ? "border-outline-variant/30 text-on-surface-variant/30 cursor-not-allowed"
                : "border-outline-variant text-on-surface-variant hover:border-primary hover:text-primary",
            ].join(" ")}
            title="Reset both sliders to 0 (baseline) — [R]"
          >
            [R] RESET
          </button>
        </div>

        {/* TEMP OFFSET SLIDER */}
        <div className="space-y-1.5">
          <div className="flex justify-between text-[10px]">
            <span>HEAT OFFSET</span>
            <span
              className={
                Math.abs(tempDeltaF) > 0.1
                  ? "text-secondary font-mono tabular-nums"
                  : "text-on-surface-variant font-mono tabular-nums"
              }
            >
              {formatTemp(tempDeltaF)}
            </span>
          </div>
          <div className="relative">
            <input
              type="range"
              min={TEMP_MIN}
              max={TEMP_MAX}
              step={1}
              value={Math.round(tempDeltaF)}
              onChange={(e) => setTempDeltaF(Number(e.target.value))}
              className="gridsense-slider gridsense-slider--secondary"
              aria-label="Temperature offset in Fahrenheit"
              title={`Heat offset: ${formatTemp(tempDeltaF)}`}
            />
          </div>
          <div className="flex justify-between text-[8px] text-on-surface-variant opacity-70 font-mono tabular-nums">
            <span>{TEMP_MIN}°F</span>
            <span>0</span>
            <span>+10 (HEAT)</span>
            <span>+{TEMP_MAX}°F</span>
          </div>
        </div>

        {/* EV PENETRATION SLIDER */}
        <div className="space-y-1.5">
          <div className="flex justify-between text-[10px]">
            <span>EV PENETRATION</span>
            <span
              className={
                evPenetrationPct > 0.1
                  ? "text-primary font-mono tabular-nums"
                  : "text-on-surface-variant font-mono tabular-nums"
              }
            >
              {formatEv(evPenetrationPct)}
            </span>
          </div>
          <div className="relative">
            <input
              type="range"
              min={EV_MIN}
              max={EV_MAX}
              step={1}
              value={Math.round(evPenetrationPct)}
              onChange={(e) => setEvPenetrationPct(Number(e.target.value))}
              className="gridsense-slider gridsense-slider--primary"
              aria-label="EV penetration percent"
              title={`EV penetration: ${formatEv(evPenetrationPct)}`}
            />
          </div>
          <div className="flex justify-between text-[8px] text-on-surface-variant opacity-70 font-mono tabular-nums">
            <span>0%</span>
            <span>35% (EV)</span>
            <span>100%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
