"use client";

import { useMemo } from "react";
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

const SCENARIO_COLOR = {
  baseline: "#4fdbc8",
  heat: "#ffb95f",
  ev: "#ffb3ad",
} as const;

const SCENARIO_LABEL = {
  baseline: "BASELINE",
  heat: "HEAT +10°F",
  ev: "EV SURGE",
} as const;

export default function ForecastRibbon() {
  const { active, current, generatedAt } = useScenario();

  const color = SCENARIO_COLOR[active];

  const data = useMemo(
    () =>
      current.quantiles.map((q) => ({
        hour: q.hour,
        p10: q.p10_mw,
        p50: q.p50_mw,
        p90: q.p90_mw,
        // Stacked bands for recharts: base + (p90-p10) area
        bandBase: q.p10_mw,
        bandDelta: q.p90_mw - q.p10_mw,
      })),
    [current.quantiles],
  );

  const currentHour = new Date().getHours();

  // Generated-at display
  const genStamp = (() => {
    try {
      const d = new Date(generatedAt.iso);
      return d.toISOString().replace("T", " ").slice(0, 16) + " UTC";
    } catch {
      return generatedAt.iso;
    }
  })();

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
        </h3>
        <span className="font-mono text-[10px] text-on-surface-variant">
          GENERATED: {genStamp}
        </span>
      </div>

      <div className="flex-1 relative">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={data}
            margin={{ top: 6, right: 8, bottom: 4, left: 8 }}
          >
            <defs>
              <linearGradient id="bandFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.32} />
                <stop offset="100%" stopColor={color} stopOpacity={0.12} />
              </linearGradient>
            </defs>
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
            {/* Invisible base to stack the band on */}
            <Area
              dataKey="bandBase"
              stackId="band"
              stroke="none"
              fill="transparent"
              isAnimationActive={false}
            />
            <Area
              dataKey="bandDelta"
              stackId="band"
              stroke="none"
              fill="url(#bandFill)"
              isAnimationActive={false}
            />
            <Line
              dataKey="p50"
              stroke={color}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <ReferenceLine
              x={currentHour}
              stroke="#e1e2ea"
              strokeOpacity={0.5}
              strokeDasharray="3 3"
              label={{
                value: "NOW",
                position: "top",
                fill: "#e1e2ea",
                fontSize: 9,
                fontFamily: "var(--font-ibm-plex-mono), monospace",
              }}
            />
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
