import type {
  FeederTopology,
  GeneratedAt,
  ModelMetrics,
  ScenarioKind,
  TomorrowForecast,
} from "./types";

const cache = new Map<string, unknown>();

async function fetchJson<T>(url: string): Promise<T> {
  if (cache.has(url)) return cache.get(url) as T;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to load ${url}: HTTP ${res.status}`);
  }
  const data = (await res.json()) as T;
  cache.set(url, data);
  return data;
}

const SCENARIO_FILE: Record<ScenarioKind, string> = {
  baseline: "/data/forecasts/tomorrow_baseline.json",
  heat: "/data/forecasts/tomorrow_heat.json",
  ev: "/data/forecasts/tomorrow_ev.json",
};

export async function loadForecast(kind: ScenarioKind): Promise<TomorrowForecast> {
  return fetchJson<TomorrowForecast>(SCENARIO_FILE[kind]);
}

export async function loadTopology(): Promise<FeederTopology> {
  return fetchJson<FeederTopology>("/data/forecasts/feeder_topology.json");
}

export async function loadMetrics(): Promise<ModelMetrics> {
  return fetchJson<ModelMetrics>("/data/forecasts/model_metrics.json");
}

export async function loadGeneratedAt(): Promise<GeneratedAt> {
  return fetchJson<GeneratedAt>("/data/forecasts/generated_at.json");
}
