"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Area,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";
import { useScenario } from "@/lib/context";
import type { ScenarioKind, TomorrowForecast } from "@/lib/types";

const SCENARIO_COLOR: Record<ScenarioKind, string> = {
  baseline: "#4fdbc8",
  heat: "#ffb95f",
  ev: "#ffb3ad",
};

const SCENARIO_LABEL: Record<ScenarioKind, string> = {
  baseline: "BASELINE",
  heat: "HEAT +10°F",
  ev: "EV SURGE",
};

export default function ForecastRibbon() {
  const { active, compareWith, current, baseline, heat, ev, generatedAt } =
    useScenario();

  const color = SCENARIO_COLOR[active];

  const compareScenario: TomorrowForecast | null =
    compareWith === null
      ? null
      : compareWith === "baseline"
        ? baseline
        : compareWith === "heat"
          ? heat
          : ev;
  const compareColor =
    compareWith !== null ? SCENARIO_COLOR[compareWith] : null;

  const data = useMemo(() => {
    const compareByHour = new Map<number, { p10: number; p90: number; p50: number }>();
    if (compareScenario) {
      for (const q of compareScenario.quantiles) {
        compareByHour.set(q.hour, {
          p10: q.p10_mw,
          p50: q.p50_mw,
          p90: q.p90_mw,
        });
      }
    }
    return current.quantiles.map((q) => {
      const c = compareByHour.get(q.hour);
      return {
        hour: q.hour,
        p10: q.p10_mw,
        p50: q.p50_mw,
        p90: q.p90_mw,
        // Stacked bands for recharts: base + (p90-p10) area
        bandBase: q.p10_mw,
        bandDelta: q.p90_mw - q.p10_mw,
        cmpBandBase: c ? c.p10 : null,
        cmpBandDelta: c ? c.p90 - c.p10 : null,
        cmpP50: c ? c.p50 : null,
      };
    });
  }, [current.quantiles, compareScenario]);

  // NOW reference line: only render after client mount to avoid SSR/hydration
  // mismatch (server's hour != browser's local hour).
  const [nowHour, setNowHour] = useState<number | null>(null);
  useEffect(() => {
    setNowHour(new Date().getHours());
  }, []);

  // Generated-at display
  const genStamp = (() => {
    try {
      const d = new Date(generatedAt.iso);
      return d.toISOString().replace("T", " ").slice(0, 16) + " UTC";
    } catch {
      return generatedAt.iso;
    }
  })();

  // Delta badge: (current.peak - compare.peak) / compare.peak * 100
  let deltaBadge: { text: string; color: string } | null = null;
  if (compareScenario) {
    const curPeak = current.feeder_rollup.peak_mw;
    const cmpPeak = compareScenario.feeder_rollup.peak_mw;
    const deltaPct = cmpPeak !== 0 ? ((curPeak - cmpPeak) / cmpPeak) * 100 : 0;
    const sign = deltaPct >= 0 ? "+" : "";
    const text = `${sign}${deltaPct.toFixed(1)}% PEAK`;
    // Badge reflects compare's own accent when current is higher, else primary.
    const badgeColor =
      deltaPct > 0
        ? compareWith === "heat"
          ? SCENARIO_COLOR.heat
          : compareWith === "ev"
            ? SCENARIO_COLOR.ev
            : "#ffb4ab"
        : SCENARIO_COLOR.baseline;
    deltaBadge = { text, color: badgeColor };
  }

  return (
    <div className="etched-b p-4 flex flex-col gap-2" style={{ height: 180 }}>
      <div className="flex justify-between items-center">
        <h3 className="font-mono text-[11px] uppercase tracking-widest text-primary flex items-center gap-2">
          <span className="material-symbols-outlined text-sm">query_stats</span>
          24H AHEAD — TOMORROW&apos;S DISPATCH
          <span
            className="ml-2 px-1.5 py-0.5 text-[9px] border"
            style={{ color, borderColor: color }}
          >
            {SCENARIO_LABEL[active]}
          </span>
          {deltaBadge && (
            <span
              className="px-1.5 py-0.5 text-[9px] border font-bold"
              style={{
                color: deltaBadge.color,
                borderColor: deltaBadge.color,
              }}
            >
              Δ {deltaBadge.text}
            </span>
          )}
        </h3>
        <span className="font-mono text-[10px] text-on-surface-variant">
          GENERATED: {genStamp}
        </span>
      </div>

      <div className="flex-1 relative">
        {/* Legend chip: active + compareWith */}
        {compareWith && compareColor && (
          <div className="absolute top-1 right-2 z-10 flex items-center gap-3 bg-surface/80 border border-outline-variant px-2 py-1 font-mono text-[9px] uppercase tracking-widest">
            <span className="flex items-center gap-1" style={{ color }}>
              <span
                className="inline-block"
                style={{
                  width: 10,
                  height: 2,
                  background: color,
                }}
              />
              ACTIVE: {SCENARIO_LABEL[active]}
            </span>
            <span
              className="flex items-center gap-1"
              style={{ color: compareColor }}
            >
              <span
                className="inline-block"
                style={{
                  width: 10,
                  height: 0,
                  borderTop: `2px dashed ${compareColor}`,
                }}
              />
              COMPARE: {SCENARIO_LABEL[compareWith]}
            </span>
          </div>
        )}
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={data}
            margin={{ top: 6, right: 8, bottom: 4, left: 8 }}
          >
            <XAxis
              dataKey="hour"
              tick={{
                fill: "#bbcac6",
                fontSize: 9,
                fontFamily: "var(--font-ibm-plex-mono), monospace",
              }}
              tickLine={false}
              axisLine={{ stroke: "#3c4947" }}
              interval={2}
              tickFormatter={(h: number) => `${String(h).padStart(2, "0")}:00`}
            />
            <YAxis
              tick={{
                fill: "#bbcac6",
                fontSize: 9,
                fontFamily: "var(--font-ibm-plex-mono), monospace",
              }}
              tickLine={false}
              axisLine={{ stroke: "#3c4947" }}
              width={38}
              tickFormatter={(v: number) => `${Math.round(v)}`}
            />
            <defs>
              <linearGradient id="bandFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.35} />
                <stop offset="100%" stopColor={color} stopOpacity={0.15} />
              </linearGradient>
              {compareColor && (
                <linearGradient
                  id="cmpBandFill"
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="1"
                >
                  <stop
                    offset="0%"
                    stopColor={compareColor}
                    stopOpacity={0.25}
                  />
                  <stop
                    offset="100%"
                    stopColor={compareColor}
                    stopOpacity={0.1}
                  />
                </linearGradient>
              )}
            </defs>
            {/* Invisible base to stack the band on */}
            <Area
              dataKey="bandBase"
              stackId="band"
              stroke="none"
              fill="transparent"
              fillOpacity={0}
              isAnimationActive={false}
            />
            <Area
              dataKey="bandDelta"
              stackId="band"
              stroke={color}
              strokeOpacity={0.3}
              strokeWidth={1}
              fill="url(#bandFill)"
              fillOpacity={1}
              isAnimationActive={false}
            />
            {/* Ghost overlay for compareWith */}
            {compareColor && (
              <>
                <Area
                  dataKey="cmpBandBase"
                  stackId="cmp"
                  stroke="none"
                  fill="transparent"
                  fillOpacity={0}
                  isAnimationActive={false}
                />
                <Area
                  dataKey="cmpBandDelta"
                  stackId="cmp"
                  stroke={compareColor}
                  strokeOpacity={0.7}
                  strokeDasharray="3 3"
                  strokeWidth={1}
                  fill="url(#cmpBandFill)"
                  fillOpacity={1}
                  isAnimationActive={false}
                />
                <Line
                  dataKey="cmpP50"
                  stroke={compareColor}
                  strokeOpacity={0.75}
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                  dot={false}
                  isAnimationActive={false}
                />
              </>
            )}
            <Line
              dataKey="p50"
              stroke={color}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            {nowHour !== null && (
              <ReferenceLine
                x={nowHour}
                stroke="#e1e2ea"
                strokeOpacity={0.6}
                strokeDasharray="4 4"
                label={{
                  value: "NOW",
                  position: "top",
                  fill: "#e1e2ea",
                  fontSize: 9,
                  fontFamily:
                    "var(--font-ibm-plex-mono), monospace",
                }}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="flex justify-between text-[9px] text-on-surface-variant font-mono uppercase tracking-widest">
        <span>X: HOUR (MST)</span>
        <span>Y: SYSTEM_MW · P10-P90 BAND</span>
      </div>
    </div>
  );
}
