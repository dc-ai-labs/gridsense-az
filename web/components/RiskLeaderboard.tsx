"use client";

import { useScenario } from "@/lib/context";
import { riskTier } from "@/lib/validate";

const TIER_BAR: Record<string, string> = {
  error: "bg-error",
  secondary: "bg-secondary",
  primary: "bg-primary",
};

const TIER_TEXT: Record<string, string> = {
  error: "text-error",
  secondary: "text-secondary",
  primary: "text-primary",
};

export default function RiskLeaderboard() {
  const { current } = useScenario();
  const rows = current.risk_leaderboard.slice(0, 10);

  return (
    <div className="p-4 etched-b flex flex-col overflow-hidden font-mono">
      <div className="flex items-center justify-between mb-3">
        <span className="text-[10px] text-primary uppercase tracking-[0.2em]">
          RISK_LEADERBOARD — TOP 10
        </span>
        <span className="text-[9px] text-on-surface-variant uppercase">
          SCEN: {current.scenario}
        </span>
      </div>
      <div className="overflow-y-auto pr-1" style={{ maxHeight: 280 }}>
        <table className="w-full text-left text-[10px]">
          <thead className="text-on-surface-variant border-b border-outline-variant uppercase tracking-widest">
            <tr>
              <th className="pb-2 pr-2 font-medium w-6">#</th>
              <th className="pb-2 pr-2 font-medium">ID</th>
              <th className="pb-2 pr-2 font-medium">RISK</th>
              <th className="pb-2 font-medium text-right">PEAK</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const tier = riskTier(r.risk_score);
              const bar = TIER_BAR[tier];
              const text = TIER_TEXT[tier];
              return (
                <tr
                  key={r.id}
                  className={`border-b border-outline-variant/30 ${
                    tier === "error"
                      ? "bg-error/10 border-l-2 border-l-error"
                      : tier === "secondary"
                        ? "border-l-2 border-l-secondary/50"
                        : "border-l-2 border-l-transparent"
                  }`}
                >
                  <td className="py-2 pl-2 text-on-surface-variant">
                    {String(i + 1).padStart(2, "0")}
                  </td>
                  <td className="py-2 pr-2 text-on-surface">{r.id}</td>
                  <td className="py-2 pr-2">
                    <div className="flex items-center gap-2">
                      <div className="h-1 bg-surface-container-highest w-14 relative">
                        <div
                          className={`h-full ${bar}`}
                          style={{
                            width: `${Math.min(100, Math.max(3, r.risk_score * 100))}%`,
                          }}
                        />
                      </div>
                      <span className={`${text} text-[9px]`}>
                        {r.risk_score.toFixed(2)}
                      </span>
                    </div>
                  </td>
                  <td className={`py-2 text-right ${text}`}>
                    {r.peak_mw.toFixed(2)} MW
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
