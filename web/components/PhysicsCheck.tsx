"use client";

import { useScenario } from "@/lib/context";

function formatVdev(v: number): string {
  const s = v >= 0 ? "+" : "-";
  return `${s}${Math.abs(v).toFixed(3)} p.u.`;
}

export default function PhysicsCheck() {
  const { active, current } = useScenario();
  const { opendss } = current;
  const hasOverloads = opendss.overloads.length > 0;
  const showViolationBanner =
    (active === "heat" || active === "ev") && hasOverloads;

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
                return (
                  <li
                    key={d.bus}
                    className="flex justify-between text-[10px] border-b border-outline-variant/30 py-1"
                  >
                    <span className="text-on-surface">BUS_{d.bus}</span>
                    <span className={color}>{formatVdev(d.vdev_pu)}</span>
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
                return (
                  <li
                    key={o.element}
                    className="flex justify-between text-[10px] border-b border-outline-variant/30 py-1 gap-2"
                  >
                    <span className="text-on-surface truncate">
                      {o.element}
                    </span>
                    <span className={`${color} whitespace-nowrap`}>
                      {o.loading_pct.toFixed(1)}% / {o.limit_mva.toFixed(1)} MVA
                    </span>
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
