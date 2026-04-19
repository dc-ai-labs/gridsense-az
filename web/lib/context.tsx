"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  loadForecast,
  loadGeneratedAt,
  loadMetrics,
  loadTopology,
} from "./data";
import type {
  FeederTopology,
  GeneratedAt,
  ModelMetrics,
  ScenarioKind,
  TomorrowForecast,
} from "./types";
import { validateForecasts } from "./validate";

/** Cross-component focus broadcast. PhysicsCheck rows + (future) leaderboard
 *  rows set this; TacticalMap reads it to highlight the matching node/edge. */
export interface MapFocus {
  /** Lowercase bus name (e.g. "150r"). */
  bus?: string;
  /** Lowercase OpenDSS line/switch element name (e.g. "l115", "sw1"). */
  element?: string;
  /** Optional UI label / origin ("BUS_DEV" | "OVERLOAD" | …). */
  source?: string;
}

interface ScenarioState {
  active: ScenarioKind;
  setActive: (s: ScenarioKind) => void;
  /** Reserved for Slice C ghost overlay; null = no overlay. */
  compareWith: ScenarioKind | null;
  setCompareWith: (s: ScenarioKind | null) => void;
  baseline: TomorrowForecast;
  heat: TomorrowForecast;
  ev: TomorrowForecast;
  topology: FeederTopology;
  metrics: ModelMetrics;
  generatedAt: GeneratedAt;
  /** Convenience: the currently-active forecast record. */
  current: TomorrowForecast;
  /** Cross-component focus (clicking PhysicsCheck rows etc.). */
  focus: MapFocus | null;
  setFocus: (f: MapFocus | null) => void;
}

const ScenarioContext = createContext<ScenarioState | null>(null);

type LoadState =
  | { kind: "loading" }
  | { kind: "error"; error: Error }
  | {
      kind: "ready";
      baseline: TomorrowForecast;
      heat: TomorrowForecast;
      ev: TomorrowForecast;
      topology: FeederTopology;
      metrics: ModelMetrics;
      generatedAt: GeneratedAt;
    };

export function ScenarioProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<LoadState>({ kind: "loading" });
  const [active, setActive] = useState<ScenarioKind>("baseline");
  const [compareWith, setCompareWith] = useState<ScenarioKind | null>(null);
  const [focus, setFocusState] = useState<MapFocus | null>(null);
  const activeRef = useRef(active);
  activeRef.current = active;

  // Drop the focus when the active scenario changes — overload "l115" in heat
  // may not exist in baseline, so a stale focus would highlight a now-clean line.
  useEffect(() => {
    setFocusState(null);
  }, [active]);

  // Auto-clear compareWith if user activates the same scenario.
  useEffect(() => {
    if (compareWith !== null && compareWith === active) {
      setCompareWith(null);
    }
  }, [active, compareWith]);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      loadForecast("baseline"),
      loadForecast("heat"),
      loadForecast("ev"),
      loadTopology(),
      loadMetrics(),
      loadGeneratedAt(),
    ])
      .then(([baseline, heat, ev, topology, metrics, generatedAt]) => {
        if (cancelled) return;
        validateForecasts(baseline, heat, ev);
        setState({
          kind: "ready",
          baseline,
          heat,
          ev,
          topology,
          metrics,
          generatedAt,
        });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const error = err instanceof Error ? err : new Error(String(err));
        // eslint-disable-next-line no-console
        console.error("[gridsense:context] load failed", error);
        setState({ kind: "error", error });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Keyboard shortcuts B / H / E for scenario toggle, C for compare-cycle.
  // Skip when a form input is focused.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (document.activeElement?.tagName ?? "").toUpperCase();
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const key = e.key.toLowerCase();
      if (key === "b") {
        e.preventDefault();
        setActive("baseline");
      } else if (key === "h") {
        e.preventDefault();
        setActive("heat");
      } else if (key === "e") {
        e.preventDefault();
        setActive("ev");
      } else if (key === "escape") {
        e.preventDefault();
        setFocusState(null);
      } else if (key === "c") {
        e.preventDefault();
        // Cycle compareWith through null -> next non-active scenario -> null.
        setCompareWith((curCompare) => {
          const curActive = activeRef.current;
          const order: ScenarioKind[] = ["baseline", "heat", "ev"];
          const candidates = order.filter((s) => s !== curActive);
          if (curCompare === null) return candidates[0] ?? null;
          const idx = candidates.indexOf(curCompare);
          if (idx === -1) return candidates[0] ?? null;
          if (idx === candidates.length - 1) return null;
          return candidates[idx + 1];
        });
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const setActiveCb = useCallback((s: ScenarioKind) => setActive(s), []);
  const setCompareCb = useCallback(
    (s: ScenarioKind | null) => setCompareWith(s),
    [],
  );
  const setFocusCb = useCallback(
    (f: MapFocus | null) => setFocusState(f),
    [],
  );

  const value = useMemo<ScenarioState | null>(() => {
    if (state.kind !== "ready") return null;
    const current =
      active === "baseline"
        ? state.baseline
        : active === "heat"
          ? state.heat
          : state.ev;
    return {
      active,
      setActive: setActiveCb,
      compareWith,
      setCompareWith: setCompareCb,
      baseline: state.baseline,
      heat: state.heat,
      ev: state.ev,
      topology: state.topology,
      metrics: state.metrics,
      generatedAt: state.generatedAt,
      current,
      focus,
      setFocus: setFocusCb,
    };
  }, [state, active, compareWith, focus, setActiveCb, setCompareCb, setFocusCb]);

  if (state.kind === "loading") {
    return <LoadingSkeleton />;
  }
  if (state.kind === "error") {
    return <LoadError error={state.error} />;
  }

  return (
    <ScenarioContext.Provider value={value}>{children}</ScenarioContext.Provider>
  );
}

export function useScenario(): ScenarioState {
  const ctx = useContext(ScenarioContext);
  if (!ctx) {
    throw new Error("useScenario must be called inside <ScenarioProvider>");
  }
  return ctx;
}

function LoadingSkeleton() {
  return (
    <div className="flex h-screen w-screen flex-col bg-surface text-on-surface font-mono">
      <div className="h-14 border-b border-outline-variant flex items-center px-6 text-primary text-xl font-headline tracking-tighter font-bold">
        GRIDSENSE-AZ
      </div>
      <div className="flex-1 flex items-center justify-center">
        <div className="flex items-center gap-3 text-xs uppercase tracking-widest text-on-surface-variant">
          <span className="inline-block w-2 h-2 bg-primary blinking" />
          LOADING_FORECAST_DATA...
        </div>
      </div>
    </div>
  );
}

function LoadError({ error }: { error: Error }) {
  return (
    <div className="flex h-screen w-screen flex-col bg-surface text-on-surface font-mono">
      <div className="h-14 border-b border-outline-variant flex items-center px-6 text-primary text-xl font-headline tracking-tighter font-bold">
        GRIDSENSE-AZ
      </div>
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="max-w-md border border-error bg-error/10 p-6 space-y-2">
          <div className="text-error text-xs uppercase tracking-widest font-bold">
            DATA_LOAD_FAILURE
          </div>
          <div className="text-xs text-on-error-container break-words">
            {error.message}
          </div>
          <div className="text-[10px] text-on-surface-variant uppercase">
            expected files under /public/data/forecasts/
          </div>
        </div>
      </div>
    </div>
  );
}
