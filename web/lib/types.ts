// Type contracts shared with scripts/precompute_forecasts.py output.
// Keep in sync with REBUILD_PLAN.md §3.

export type ScenarioKind = "baseline" | "heat" | "ev";

export interface QuantileHour {
  /** ISO-8601 UTC timestamp aligned to hour. */
  ts: string;
  /** Hour index 0-23 relative to generated_at. */
  hour: number;
  /** 10th-percentile system total MW. */
  p10_mw: number;
  /** Median system total MW. */
  p50_mw: number;
  /** 90th-percentile system total MW. */
  p90_mw: number;
}

export interface FeederRollup {
  /** System peak MW across the 24h horizon. */
  peak_mw: number;
  /** Hour index where peak occurs (0..23). */
  peak_hour: number;
  /** Nameplate capacity on the system-level aggregation, MW. */
  capacity_mw: number;
  /** Fraction of capacity used at peak (0..1+). */
  load_factor: number;
}

export interface PerBusMetric {
  /** 132-node IEEE 123 bus name. */
  bus: string;
  /** 0..1 risk score (weighted of overload probability + voltage deviation). */
  risk_score: number;
  /** Forecast peak load in kW over the horizon. */
  peak_load_kw: number;
  /** Rating / ampacity-derived cap in kW. */
  rating_kw: number;
}

export interface RiskLeaderboardRow {
  /** Short display ID, e.g. F-17_PHX. */
  id: string;
  /** 0..1. */
  risk_score: number;
  /** MW, displayed alongside the bar. */
  peak_mw: number;
  /** Underlying bus reference (for deep-link). */
  bus?: string;
}

export interface OpenDssBusDeviation {
  bus: string;
  /** Deviation from 1.0 p.u. (signed). Positive = overvoltage. */
  vdev_pu: number;
}

export interface OpenDssOverload {
  element: string;
  loading_pct: number;
  limit_mva: number;
}

export interface OpenDssSnapshot {
  converged: boolean;
  scenario: ScenarioKind;
  top_bus_deviations: OpenDssBusDeviation[];
  overloads: OpenDssOverload[];
}

export interface WeatherSummary {
  /** Peak air temperature over the horizon, Fahrenheit. */
  peak_temp_f: number;
  /** Peak temperature local hour 0..23. */
  peak_hour: number;
  /** Source tag: "nws" | "replay" | "synthetic". */
  source: string;
}

export interface TopDriver {
  name: string;
  /** Integrated-gradient importance score (unit-free). */
  ig: number;
}

export interface RecommendedAction {
  label: string;
  severity: "error" | "secondary" | "tertiary" | "primary";
}

export interface TomorrowForecast {
  scenario: ScenarioKind;
  generated_at: string;
  /** 24 hourly system-total quantile rows. */
  quantiles: QuantileHour[];
  /** Per-bus metrics keyed by bus name. */
  per_bus: Record<string, PerBusMetric>;
  risk_leaderboard: RiskLeaderboardRow[];
  feeder_rollup: FeederRollup;
  opendss: OpenDssSnapshot;
  weather: WeatherSummary;
  top_drivers: TopDriver[];
  recommended_actions: RecommendedAction[];
}

export interface ModelMetrics {
  train_mae_kw: number;
  val_mae_kw: number;
  test_mae_kw: number;
  persistence_mae_kw: number;
  improvement_pct: number;
  n_params: number;
  rmse_kw?: number;
  mape_pct?: number;
  bias_kw?: number;
  top_drivers: TopDriver[];
}

export interface TopologyNode {
  bus: string;
  /** Normalized layout coordinate in [0, 1]. */
  x_norm: number;
  y_norm: number;
  /** Optional feeder group label for coloring. */
  feeder?: string;
}

export interface TopologyEdge {
  from: string;
  to: string;
  kind?: "line" | "xfmr" | "switch";
}

export interface FeederTopology {
  n_nodes: number;
  nodes: TopologyNode[];
  edges: TopologyEdge[];
}

export interface GeneratedAt {
  iso: string;
  nws_source: "live" | "replay" | "synthetic";
  hours_forecast: number;
  git_sha: string;
}
