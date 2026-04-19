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
  ScenarioPreset,
  TomorrowForecast,
} from "./types";
import { validateForecasts } from "./validate";
import { blendForecast, presetIdFromSliders } from "./blend";

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
  /** Derived: which preset the sliders match ("baseline"|"heat"|"ev"|"custom"). */
  active: ScenarioKind;
  /** Snap sliders to a preset. Passing "custom" is a no-op. */
  setActive: (s: ScenarioKind) => void;
  /** Compare overlay preset; must be a concrete preset (not "custom"). */
  compareWith: ScenarioPreset | null;
  setCompareWith: (s: ScenarioPreset | null) => void;
  /** Raw slider values. */
  tempDeltaF: number;
  setTempDeltaF: (v: number) => void;
  evPenetrationPct: number;
  setEvPenetrationPct: (v: number) => void;
  /** Reset both sliders to zero (= baseline). */
  resetSliders: () => void;
  baseline: TomorrowForecast;
  heat: TomorrowForecast;
  ev: TomorrowForecast;
  topology: FeederTopology;
  metrics: ModelMetrics;
  generatedAt: GeneratedAt;
  /** Convenience: the currently-active (possibly blended) forecast record. */
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

const SLIDER_STORAGE_KEY = "gridsense:sliders:v1";

interface PersistedSliders {
  tempDeltaF: number;
  evPenetrationPct: number;
}

const TEMP_MIN = -5;
const TEMP_MAX = 25;
const EV_MIN = 0;
const EV_MAX = 100;

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function loadPersistedSliders(): PersistedSliders {
  if (typeof window === "undefined") {
    return { tempDeltaF: 0, evPenetrationPct: 0 };
  }
  try {
    const raw = window.sessionStorage.getItem(SLIDER_STORAGE_KEY);
    if (!raw) return { tempDeltaF: 0, evPenetrationPct: 0 };
    const parsed = JSON.parse(raw) as Partial<PersistedSliders>;
    return {
      tempDeltaF: clamp(Number(parsed.tempDeltaF ?? 0), TEMP_MIN, TEMP_MAX),
      evPenetrationPct: clamp(
        Number(parsed.evPenetrationPct ?? 0),
        EV_MIN,
        EV_MAX,
      ),
    };
  } catch {
    return { tempDeltaF: 0, evPenetrationPct: 0 };
  }
}

export function ScenarioProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<LoadState>({ kind: "loading" });
  const [tempDeltaF, setTempDeltaFState] = useState<number>(0);
  const [evPenetrationPct, setEvPenetrationPctState] = useState<number>(0);
  const [compareWith, setCompareWith] = useState<ScenarioPreset | null>(null);
  const [focus, setFocusState] = useState<MapFocus | null>(null);

  // Hydrate slider state from sessionStorage after mount (SSR-safe).
  useEffect(() => {
    const persisted = loadPersistedSliders();
    setTempDeltaFState(persisted.tempDeltaF);
    setEvPenetrationPctState(persisted.evPenetrationPct);
  }, []);

  // Persist slider state.
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.sessionStorage.setItem(
        SLIDER_STORAGE_KEY,
        JSON.stringify({ tempDeltaF, evPenetrationPct }),
      );
    } catch {
      // quota exceeded / private mode — ignore.
    }
  }, [tempDeltaF, evPenetrationPct]);

  // Derive the "active" identity from sliders.
  const active: ScenarioKind = useMemo(
    () => presetIdFromSliders(tempDeltaF, evPenetrationPct),
    [tempDeltaF, evPenetrationPct],
  );
  const activeRef = useRef(active);
  activeRef.current = active;

  const setTempDeltaF = useCallback((v: number) => {
    setTempDeltaFState(clamp(v, TEMP_MIN, TEMP_MAX));
  }, []);
  const setEvPenetrationPct = useCallback((v: number) => {
    setEvPenetrationPctState(clamp(v, EV_MIN, EV_MAX));
  }, []);
  const resetSliders = useCallback(() => {
    setTempDeltaFState(0);
    setEvPenetrationPctState(0);
  }, []);

  const setActiveCb = useCallback((s: ScenarioKind) => {
    if (s === "baseline") {
      setTempDeltaFState(0);
      setEvPenetrationPctState(0);
    } else if (s === "heat") {
      setTempDeltaFState(10);
      setEvPenetrationPctState(0);
    } else if (s === "ev") {
      setTempDeltaFState(0);
      setEvPenetrationPctState(35);
    }
    // "custom" → no-op (you get custom by dragging a slider)
  }, []);

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

  // Keyboard shortcuts: B/H/E snap to presets, R resets, C cycles compare.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (document.activeElement?.tagName ?? "").toUpperCase();
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const key = e.key.toLowerCase();
      if (key === "b") {
        e.preventDefault();
        setActiveCb("baseline");
      } else if (key === "h") {
        e.preventDefault();
        setActiveCb("heat");
      } else if (key === "e") {
        e.preventDefault();
        setActiveCb("ev");
      } else if (key === "r") {
        e.preventDefault();
        setTempDeltaFState(0);
        setEvPenetrationPctState(0);
      } else if (key === "escape") {
        e.preventDefault();
        setFocusState(null);
      } else if (key === "c") {
        e.preventDefault();
        // Cycle compareWith through null -> next non-active preset -> null.
        setCompareWith((curCompare) => {
          const curActive = activeRef.current;
          const order: ScenarioPreset[] = ["baseline", "heat", "ev"];
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
  }, [setActiveCb]);

  const setCompareCb = useCallback(
    (s: ScenarioPreset | null) => setCompareWith(s),
    [],
  );
  const setFocusCb = useCallback(
    (f: MapFocus | null) => setFocusState(f),
    [],
  );

  const value = useMemo<ScenarioState | null>(() => {
    if (state.kind !== "ready") return null;
    const current = blendForecast({
      baseline: state.baseline,
      heat: state.heat,
      ev: state.ev,
      tempDeltaF,
      evPenetrationPct,
    });
    return {
      active,
      setActive: setActiveCb,
      compareWith,
      setCompareWith: setCompareCb,
      tempDeltaF,
      setTempDeltaF,
      evPenetrationPct,
      setEvPenetrationPct,
      resetSliders,
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
  }, [
    state,
    active,
    tempDeltaF,
    evPenetrationPct,
    compareWith,
    focus,
    setActiveCb,
    setCompareCb,
    setFocusCb,
    setTempDeltaF,
    setEvPenetrationPct,
    resetSliders,
  ]);

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
