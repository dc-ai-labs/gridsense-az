"use client";

import { useScenario } from "@/lib/context";

function fmt(n: number, digits = 0): string {
  if (!Number.isFinite(n)) return "—";
  if (digits === 0) return Math.round(n).toLocaleString();
  return n.toFixed(digits);
}

export default function ModelMetrics() {
  const { metrics } = useScenario();

  const cards = [
    { label: "TRAIN_MAE", value: fmt(metrics.train_mae_kw), unit: "kW" },
    { label: "VAL_MAE", value: fmt(metrics.val_mae_kw), unit: "kW" },
    { label: "TEST_MAE", value: fmt(metrics.test_mae_kw), unit: "kW" },
    {
      label: "PERSIST_MAE",
      value: fmt(metrics.persistence_mae_kw),
      unit: "kW",
    },
    {
      label: "IMPROVE",
      value: `${fmt(metrics.improvement_pct, 1)}`,
      unit: "%",
      accent: true,
    },
    {
      label: "PARAMS",
      value: fmt(metrics.n_params),
      unit: "n",
    },
  ];

  const maxIg = Math.max(...metrics.top_drivers.map((d) => d.ig), 1e-6);

  return (
    <div className="p-4 etched-b flex flex-col font-mono">
      <span className="text-[10px] text-primary uppercase tracking-[0.2em] mb-3">
        MODEL_METRICS
      </span>
      <div className="grid grid-cols-3 gap-1.5">
        {cards.map((c) => (
          <div
            key={c.label}
            className={`border border-outline-variant/60 bg-surface-container-low px-2 py-2 flex flex-col ${
              c.accent ? "border-primary/60" : ""
            }`}
          >
            <span className="text-[9px] text-on-surface-variant uppercase tracking-widest">
              {c.label}
            </span>
            <span
              className={`font-headline text-lg font-bold leading-none mt-1 ${
                c.accent ? "text-primary" : "text-on-surface"
              }`}
            >
              {c.value}
            </span>
            <span className="text-[9px] text-on-surface-variant mt-0.5 font-mono">
              {c.unit}
            </span>
          </div>
        ))}
      </div>

      <div className="mt-4 space-y-2">
        <span className="text-[10px] text-on-surface-variant uppercase tracking-[0.2em]">
          TOP_DRIVERS · INTEGRATED_GRADIENTS
        </span>
        <div className="space-y-1.5">
          {metrics.top_drivers.slice(0, 5).map((d) => (
            <div key={d.name} className="flex items-center gap-2 text-[10px]">
              <span className="w-24 truncate text-on-surface">{d.name}</span>
              <div className="flex-1 h-1 bg-surface-container-highest relative">
                <div
                  className="absolute top-0 left-0 h-full bg-primary"
                  style={{ width: `${(d.ig / maxIg) * 100}%` }}
                />
              </div>
              <span className="w-10 text-right text-on-surface-variant">
                {d.ig.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
