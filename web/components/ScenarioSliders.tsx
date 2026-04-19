"use client";

import { useScenario } from "@/lib/context";
import type { ScenarioKind } from "@/lib/types";

interface Option {
  key: ScenarioKind;
  label: string;
  shortcut: "B" | "H" | "E";
  sublabel?: string;
}

const OPTIONS: Option[] = [
  { key: "baseline", label: "BASELINE", shortcut: "B", sublabel: "NOAA" },
  { key: "heat", label: "HEAT +10°F", shortcut: "H", sublabel: "STRESS" },
  { key: "ev", label: "EV SURGE", shortcut: "E", sublabel: "+35%" },
];

/**
 * Three-way scenario toggle. Active button takes border-primary + bg-primary/10.
 * Keyboard B/H/E are wired globally in ScenarioProvider; clicking here fires
 * the same `setActive`, so all dependent components repaint from pre-loaded
 * data (no fetch).
 */
export default function ScenarioSliders() {
  const { active, setActive, current } = useScenario();

  const heatOffset = active === "heat" ? "+10°F" : "+0°F";
  const heatOffsetPct = active === "heat" ? 70 : 0;
  const evPct = active === "ev" ? 35 : 0;
  const evLabel = `${evPct}%`;

  return (
    <div className="p-6 etched-b space-y-6">
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-primary tracking-widest">
          SCENARIO_CONTROLS
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
              onClick={() => setActive(opt.key)}
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

      <div className="space-y-4">
        <div className="space-y-2">
          <div className="flex justify-between text-[10px]">
            <span>HEAT OFFSET</span>
            <span
              className={
                heatOffsetPct > 0 ? "text-secondary" : "text-on-surface-variant"
              }
            >
              {heatOffset}
            </span>
          </div>
          <div className="h-1.5 bg-surface-container-lowest w-full relative">
            <div
              className="absolute top-0 left-0 h-full bg-secondary transition-[width] duration-150"
              style={{ width: `${heatOffsetPct}%` }}
            />
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-[10px]">
            <span>EV PENETRATION</span>
            <span
              className={
                evPct > 0 ? "text-primary" : "text-on-surface-variant"
              }
            >
              {evLabel}
            </span>
          </div>
          <div className="h-1.5 bg-surface-container-lowest w-full relative">
            <div
              className="absolute top-0 left-0 h-full bg-primary transition-[width] duration-150"
              style={{ width: `${evPct}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
