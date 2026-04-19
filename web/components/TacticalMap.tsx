"use client";

import { useMemo, useState } from "react";
import { useScenario } from "@/lib/context";
import { riskTier } from "@/lib/validate";
import type { PerBusMetric, ScenarioKind, TomorrowForecast } from "@/lib/types";

const SCENARIO_ACCENT: Record<ScenarioKind, string> = {
  baseline: "#4fdbc8",
  heat: "#ffb95f",
  ev: "#ffb3ad",
};

const SCENARIO_LABEL_MAP: Record<ScenarioKind, string> = {
  baseline: "BASELINE",
  heat: "HEAT +10°F",
  ev: "EV SURGE",
};

const TIER_COLOR = {
  error: "#ffb4ab",
  secondary: "#ffb95f",
  primary: "#4fdbc8",
} as const;

const VIEW_W = 1000;
const VIEW_H = 600;

interface HoverState {
  bus: string;
  cx: number;
  cy: number;
  metric: PerBusMetric;
}

export default function TacticalMap() {
  const { active, compareWith, current, baseline, heat, ev, topology } =
    useScenario();
  const [hover, setHover] = useState<HoverState | null>(null);

  const compareScenario: TomorrowForecast | null =
    compareWith === null
      ? null
      : compareWith === "baseline"
        ? baseline
        : compareWith === "heat"
          ? heat
          : ev;
  const compareAccent =
    compareWith !== null ? SCENARIO_ACCENT[compareWith] : null;

  // Project nodes into viewBox + index by bus name.
  const nodeXY = useMemo(() => {
    const m = new Map<string, { cx: number; cy: number }>();
    for (const n of topology.nodes) {
      m.set(n.bus, {
        cx: n.x_norm * VIEW_W,
        cy: n.y_norm * VIEW_H,
      });
    }
    return m;
  }, [topology.nodes]);

  const edgeLines = useMemo(
    () =>
      topology.edges
        .map((e, i) => {
          const a = nodeXY.get(e.from);
          const b = nodeXY.get(e.to);
          if (!a || !b) return null;
          return { key: i, x1: a.cx, y1: a.cy, x2: b.cx, y2: b.cy, kind: e.kind };
        })
        .filter((x): x is NonNullable<typeof x> => x !== null),
    [topology.edges, nodeXY],
  );

  // Per-bus metric lookup for current scenario.
  const perBus = current.per_bus;
  const busNames = topology.nodes.map((n) => n.bus);

  // Top-10 at-risk for pulsing markers.
  const topRiskSet = useMemo(() => {
    const withMetric = busNames
      .map((b) => ({ bus: b, m: perBus[b] }))
      .filter(
        (x): x is { bus: string; m: PerBusMetric } => x.m !== undefined,
      );
    withMetric.sort((a, b) => b.m.risk_score - a.m.risk_score);
    return new Set(withMetric.slice(0, 10).map((x) => x.bus));
  }, [busNames, perBus]);

  // Top-10 at-risk for compareWith scenario (dashed ghost rings).
  const compareTopRiskSet = useMemo(() => {
    if (!compareScenario) return new Set<string>();
    const cmpPerBus = compareScenario.per_bus;
    const withMetric = busNames
      .map((b) => ({ bus: b, m: cmpPerBus[b] }))
      .filter(
        (x): x is { bus: string; m: PerBusMetric } => x.m !== undefined,
      );
    withMetric.sort((a, b) => b.m.risk_score - a.m.risk_score);
    return new Set(withMetric.slice(0, 10).map((x) => x.bus));
  }, [busNames, compareScenario]);

  // Top-20 highest peak_load_kw → residential proxy for EV overlay.
  const evBusSet = useMemo(() => {
    if (active !== "ev") return new Set<string>();
    const withMetric = busNames
      .map((b) => ({ bus: b, m: perBus[b] }))
      .filter(
        (x): x is { bus: string; m: PerBusMetric } => x.m !== undefined,
      );
    withMetric.sort((a, b) => b.m.peak_load_kw - a.m.peak_load_kw);
    return new Set(withMetric.slice(0, 20).map((x) => x.bus));
  }, [busNames, perBus, active]);

  // Heat tint for background
  const bgGradient =
    active === "heat"
      ? "radial-gradient(ellipse at center, rgba(255, 185, 95, 0.18) 0%, rgba(17, 19, 25, 0) 70%)"
      : active === "ev"
        ? "radial-gradient(ellipse at 60% 70%, rgba(255, 179, 173, 0.12) 0%, rgba(17, 19, 25, 0) 75%)"
        : "none";

  return (
    <div
      className="relative flex-1 overflow-hidden bg-surface-container-lowest"
      style={{ backgroundImage: bgGradient }}
    >
      {/* Layer label */}
      <div className="absolute top-3 left-3 bg-surface/80 p-2 font-mono text-[9px] uppercase border border-outline-variant z-10 tracking-widest">
        LAYER: IEEE_123_BUS_NETWORK // PROJECTION: AZ_COORD_S01 · SCEN:{" "}
        <span className="text-primary">{active.toUpperCase()}</span>
      </div>

      {/* Ghost-layer legend when comparing */}
      {compareWith && compareAccent && (
        <div
          className="absolute top-3 right-3 bg-surface/85 p-2 font-mono text-[9px] uppercase border z-10 tracking-widest space-y-1"
          style={{ borderColor: compareAccent }}
        >
          <div style={{ color: compareAccent }}>GHOST_LAYER</div>
          <div className="flex items-center gap-2 text-on-surface-variant">
            <svg width="14" height="6">
              <circle
                cx="7"
                cy="3"
                r="2.5"
                fill="none"
                stroke={compareAccent}
                strokeWidth="1"
                strokeDasharray="3 2"
              />
            </svg>
            <span>
              {SCENARIO_LABEL_MAP[compareWith]} TOP-10
            </span>
          </div>
          <div className="text-[8px] text-on-surface-variant opacity-70">
            SOLID = ACTIVE · DASHED = COMPARE
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="absolute bottom-3 left-3 bg-surface/80 border border-outline-variant p-2 font-mono text-[9px] uppercase z-10 space-y-1 tracking-widest">
        <div className="text-on-surface-variant">RISK_TIER</div>
        <div className="flex items-center gap-2">
          <span
            className="inline-block w-2 h-2"
            style={{ background: TIER_COLOR.primary }}
          />
          <span>LOW (&lt; 0.40)</span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className="inline-block w-2 h-2"
            style={{ background: TIER_COLOR.secondary }}
          />
          <span>MED (0.40-0.70)</span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className="inline-block w-2 h-2"
            style={{ background: TIER_COLOR.error }}
          />
          <span>HIGH (&gt; 0.70)</span>
        </div>
        {active === "ev" && (
          <div className="flex items-center gap-2 pt-1 mt-1 border-t border-outline-variant">
            <span
              className="inline-block w-2 h-2"
              style={{ background: "#ffb95f", outline: "1px solid #ffb95f" }}
            />
            <span>EV_HOTSPOT</span>
          </div>
        )}
      </div>

      <svg
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        preserveAspectRatio="xMidYMid meet"
        className="w-full h-full"
      >
        {/* Edges */}
        <g>
          {edgeLines.map((e) => (
            <line
              key={e.key}
              x1={e.x1}
              y1={e.y1}
              x2={e.x2}
              y2={e.y2}
              stroke="#3c4947"
              strokeOpacity={e.kind === "switch" ? 0.5 : 0.9}
              strokeWidth={1}
              strokeDasharray={e.kind === "switch" ? "3 3" : undefined}
            />
          ))}
        </g>

        {/* Nodes */}
        <g>
          {topology.nodes.map((n) => {
            const xy = nodeXY.get(n.bus);
            if (!xy) return null;
            const metric = perBus[n.bus];
            const risk = metric?.risk_score ?? 0;
            const tier = riskTier(risk);
            const fill = TIER_COLOR[tier];
            const isTop = topRiskSet.has(n.bus);
            const r = isTop ? 7 : 4;
            return (
              <g key={n.bus}>
                {isTop && (
                  <circle
                    cx={xy.cx}
                    cy={xy.cy}
                    r={r}
                    fill="none"
                    stroke={fill}
                    strokeWidth={2}
                    className="pulse-violation-svg"
                  />
                )}
                <circle
                  cx={xy.cx}
                  cy={xy.cy}
                  r={r}
                  fill={fill}
                  fillOpacity={isTop ? 1 : 0.85}
                  onMouseEnter={() =>
                    metric &&
                    setHover({
                      bus: n.bus,
                      cx: xy.cx,
                      cy: xy.cy,
                      metric,
                    })
                  }
                  onMouseLeave={() => setHover(null)}
                  style={{ cursor: "pointer" }}
                />
              </g>
            );
          })}
        </g>

        {/* Compare-with ghost rings */}
        {compareAccent && (
          <g>
            {Array.from(compareTopRiskSet).map((bus) => {
              const xy = nodeXY.get(bus);
              if (!xy) return null;
              return (
                <circle
                  key={`cmp-${bus}`}
                  cx={xy.cx}
                  cy={xy.cy}
                  r={10}
                  fill="none"
                  stroke={compareAccent}
                  strokeOpacity={0.85}
                  strokeWidth={1}
                  strokeDasharray="3 2"
                />
              );
            })}
          </g>
        )}

        {/* EV overlay */}
        {active === "ev" && (
          <g>
            {Array.from(evBusSet).map((bus) => {
              const xy = nodeXY.get(bus);
              if (!xy) return null;
              return (
                <circle
                  key={`ev-${bus}`}
                  cx={xy.cx}
                  cy={xy.cy}
                  r={10}
                  fill="none"
                  stroke="#ffb95f"
                  strokeOpacity={0.7}
                  strokeWidth={1}
                  strokeDasharray="2 2"
                />
              );
            })}
          </g>
        )}
      </svg>

      {/* Hover tooltip */}
      {hover && (
        <div
          className="absolute pointer-events-none bg-surface-container-high border border-primary p-2 font-mono text-[10px] uppercase shadow-2xl z-20"
          style={{
            left: `${(hover.cx / VIEW_W) * 100}%`,
            top: `${(hover.cy / VIEW_H) * 100}%`,
            transform: "translate(12px, 12px)",
            minWidth: 180,
          }}
        >
          <div className="flex items-center gap-2 mb-1">
            <div
              className="w-1.5 h-1.5"
              style={{
                background: TIER_COLOR[riskTier(hover.metric.risk_score)],
              }}
            />
            <span className="font-bold text-primary tracking-widest">
              BUS_{hover.bus}
            </span>
          </div>
          <div className="space-y-0.5 text-[10px]">
            <p className="flex justify-between gap-4">
              <span className="text-on-surface-variant">RISK_SCORE</span>
              <span
                style={{
                  color: TIER_COLOR[riskTier(hover.metric.risk_score)],
                }}
              >
                {hover.metric.risk_score.toFixed(2)}
              </span>
            </p>
            <p className="flex justify-between gap-4">
              <span className="text-on-surface-variant">PEAK_LOAD</span>
              <span>{(hover.metric.peak_load_kw / 1000).toFixed(2)} MW</span>
            </p>
            <p className="flex justify-between gap-4">
              <span className="text-on-surface-variant">RATING</span>
              <span>{(hover.metric.rating_kw / 1000).toFixed(2)} MW</span>
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
