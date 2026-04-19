"use client";

import { useScenario } from "@/lib/context";
import type { TomorrowForecast } from "@/lib/types";

function isFocused(
  current: { bus?: string; element?: string } | null | undefined,
  match: { bus?: string; element?: string },
): boolean {
  if (!current) return false;
  if (match.bus && current.bus === match.bus) return true;
  if (match.element && current.element === match.element) return true;
  return false;
}

function formatVdev(v: number): string {
  const s = v >= 0 ? "+" : "-";
  return `${s}${Math.abs(v).toFixed(3)} p.u.`;
}

function topOverloadLabel(forecast: TomorrowForecast): string {
  const o = forecast.opendss.overloads[0];
  if (!o) return "—";
  return `${o.element.slice(0, 14)} ${o.loading_pct.toFixed(0)}%`;
}

export default function PhysicsCheck() {
  const { active, compareWith, current, baseline, heat, ev, focus, setFocus } =
    useScenario();
  const { opendss } = current;

  const onPick = (next: { bus?: string; element?: string; source: string }) => {
    if (isFocused(focus, next)) {
      setFocus(null);
    } else {
      setFocus(next);
    }
  };
  const hasOverloads = opendss.overloads.length > 0;
  const showViolationBanner =
    (active === "heat" || active === "ev") && hasOverloads;

  const compareScenario: TomorrowForecast | null =
    compareWith === null
      ? null
      : compareWith === "baseline"
        ? baseline
        : compareWith === "heat"
          ? heat
          : ev;

  return (
    <div className="p-4 flex flex-col font-mono">
      <div className="flex items-center justify-between mb-3">
        <span className="text-[10px] text-primary uppercase tracking-[0.2em]">
          OPENDSS_PHYSICS_CHECK
        </span>
        <span
          className={`text-[9px] px-1.5 py-0.5 border uppercase tracking-widest ${
            opendss.converged
              ? "border-primary text-primary"
              : "border-error text-error"
          }`}
        >
          {opendss.converged ? "CONVERGED ✓" : "NON-CONV ✗"}
        </span>
      </div>

      {showViolationBanner && (
        <div className="mb-3 border border-error bg-error/10 px-2 py-1.5 text-[10px] uppercase tracking-widest text-error flex items-center gap-2">
          <span
            className="material-symbols-outlined text-[14px]"
            style={{ fontVariationSettings: "'FILL' 1" }}
          >
            warning
          </span>
          VIOLATIONS DETECTED
        </div>
      )}

      {compareScenario && (() => {
        const curCount = opendss.overloads.length;
        const cmpCount = compareScenario.opendss.overloads.length;
        const delta = curCount - cmpCount;
        return (
          <div className="mb-3 border border-outline-variant bg-surface-container-lowest p-2 space-y-2">
            <div className="flex justify-between items-center text-[9px] uppercase tracking-widest text-on-surface-variant">
              <span>SCENARIO_DIFF</span>
              {delta > 0 ? (
                <span className="text-error font-bold">
                  ▲ +{delta} NEW VIOLATIONS
                </span>
              ) : delta < 0 ? (
                <span className="text-primary font-bold">
                  ▼ {delta} FEWER
                </span>
              ) : (
                <span className="text-on-surface-variant">EQUAL</span>
              )}
            </div>
            <div className="grid grid-cols-2 gap-2 text-[10px]">
              <div className="border-l-2 border-l-primary pl-2">
                <div className="text-[9px] text-primary uppercase tracking-widest">
                  ACTIVE · {current.scenario}
                </div>
                <div className="text-on-surface">
                  {curCount} overload{curCount === 1 ? "" : "s"}
                </div>
                <div className="text-[9px] text-on-surface-variant truncate">
                  {topOverloadLabel(current)}
                </div>
              </div>
              <div className="border-l-2 border-l-outline-variant pl-2">
                <div className="text-[9px] text-on-surface-variant uppercase tracking-widest">
                  COMPARE · {compareScenario.scenario}
                </div>
                <div className="text-on-surface-variant">
                  {cmpCount} overload{cmpCount === 1 ? "" : "s"}
                </div>
                <div className="text-[9px] text-on-surface-variant truncate">
                  {topOverloadLabel(compareScenario)}
                </div>
              </div>
            </div>
          </div>
        );
      })()}

      <div className="space-y-3">
        <div>
          <span className="text-[9px] text-on-surface-variant uppercase tracking-widest block mb-1">
            TOP_BUS_DEVIATIONS
          </span>
          {opendss.top_bus_deviations.length === 0 ? (
            <div className="text-[10px] text-on-surface-variant">
              — no deviations reported —
            </div>
          ) : (
            <ul className="space-y-1">
              {opendss.top_bus_deviations.slice(0, 5).map((d) => {
                const mag = Math.abs(d.vdev_pu);
                const color =
                  mag > 0.05
                    ? "text-error"
                    : mag > 0.03
                      ? "text-secondary"
                      : "text-on-surface";
                const sel = isFocused(focus, { bus: d.bus });
                return (
                  <li key={d.bus}>
                    <button
                      type="button"
                      onClick={() =>
                        onPick({ bus: d.bus, source: "BUS_DEV" })
                      }
                      aria-pressed={sel}
                      title={
                        sel
                          ? "Click to clear focus"
                          : `Highlight bus ${d.bus} on map`
                      }
                      className={`w-full flex justify-between text-[10px] border-b border-outline-variant/30 py-1 px-1 -mx-1 cursor-pointer transition-colors ${
                        sel
                          ? "bg-primary/15 border-l-2 border-l-primary"
                          : "border-l-2 border-l-transparent hover:bg-surface-container-low"
                      }`}
                    >
                      <span className="text-on-surface flex items-center gap-1.5">
                        {sel && (
                          <span
                            className="material-symbols-outlined text-primary text-[12px] leading-none"
                            aria-hidden
                          >
                            my_location
                          </span>
                        )}
                        BUS_{d.bus}
                      </span>
                      <span className={color}>{formatVdev(d.vdev_pu)}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        <div>
          <span className="text-[9px] text-on-surface-variant uppercase tracking-widest block mb-1">
            OVERLOADS
          </span>
          {opendss.overloads.length === 0 ? (
            <div className="text-[10px] text-primary">— all within limits —</div>
          ) : (
            <ul className="space-y-1">
              {opendss.overloads.slice(0, 5).map((o) => {
                const color =
                  o.loading_pct >= 110
                    ? "text-error"
                    : o.loading_pct >= 100
                      ? "text-secondary"
                      : "text-tertiary";
                const elementKey = o.element.trim().toLowerCase();
                const sel = isFocused(focus, { element: elementKey });
                return (
                  <li key={o.element}>
                    <button
                      type="button"
                      onClick={() =>
                        onPick({ element: elementKey, source: "OVERLOAD" })
                      }
                      aria-pressed={sel}
                      title={
                        sel
                          ? "Click to clear focus"
                          : `Highlight line ${o.element} on map`
                      }
                      className={`w-full flex justify-between text-[10px] border-b border-outline-variant/30 py-1 gap-2 px-1 -mx-1 cursor-pointer transition-colors ${
                        sel
                          ? "bg-error/15 border-l-2 border-l-error"
                          : "border-l-2 border-l-transparent hover:bg-surface-container-low"
                      }`}
                    >
                      <span className="text-on-surface truncate flex items-center gap-1.5">
                        {sel && (
                          <span
                            className="material-symbols-outlined text-error text-[12px] leading-none"
                            aria-hidden
                          >
                            my_location
                          </span>
                        )}
                        {o.element}
                      </span>
                      <span className={`${color} whitespace-nowrap`}>
                        {o.loading_pct.toFixed(1)}% / {o.limit_mva.toFixed(1)} MVA
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
