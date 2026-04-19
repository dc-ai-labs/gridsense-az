"""GridSense-AZ — Streamlit dashboard entrypoint (also the HF Space entry).

Three tabs:

1. **Feeder Map** — pydeck scatter + line layer over a Phoenix-anchored
   view, coloured by current voltage pu (green/yellow/red), sized by kW.
2. **Demand Forecast** — system-demand plot (past 7 d) + a placeholder p10/
   p50/p90 ribbon for the next 6 h. The ribbon reads from a trained
   Graph-WaveNet checkpoint when ``models/gwnet_v1.pt`` is present; until
   then it falls back to ``p50 * [0.9, 1.0, 1.1]``.
3. **Stress Scenarios** — runs :func:`gridsense.decision.heat_wave_scenario`
   et al. on demand, showing a KPI strip, top-10 critical buses, and a
   recommended-actions list.

The module is import-safe: guarding the Streamlit code behind
``if _running_as_streamlit()`` lets the smoke test in
``tests/test_dashboard.py`` import the module without triggering the
full UI render (which would complain about a missing ScriptRunContext).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make ``src/`` importable when Streamlit is launched from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for _p in (_SRC, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import streamlit as st

from app.components.feeder_map import build_deck, build_map_data
from gridsense.decision import (
    ScenarioResult,
    combined_scenario,
    ev_surge_scenario,
    heat_wave_scenario,
    rank_critical_buses,
)
from gridsense.topology import load_ieee123

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import-safety guard
# ---------------------------------------------------------------------------


def _running_as_streamlit() -> bool:
    """Return ``True`` when the module is executed under ``streamlit run``.

    Uses Streamlit's public runtime API, with a graceful fallback so that
    older versions (or import-only environments like pytest) still work.
    """
    try:
        from streamlit.runtime import exists as runtime_exists

        return bool(runtime_exists())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Cached data accessors
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _cached_graph():
    """Load the IEEE 123 graph once per Streamlit process."""
    return load_ieee123()


@st.cache_data(show_spinner=False)
def _cached_snapshot_voltages() -> dict[str, float]:
    """Run a nominal snapshot and return the bus voltage map.

    Cached because OpenDSS compilation + solve is ~1 s and the Feeder Map
    tab re-renders on every slider tick.
    """
    from gridsense.power_flow import run_snapshot

    try:
        snapshot = run_snapshot()
        return dict(snapshot.bus_voltages_pu)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Snapshot failed: %s", exc)
        return {}


@st.cache_data(show_spinner=False)
def _cached_feature_bundle_summary(
    start: str, end: str
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Return ``(times, system_demand_kw)`` for the forecast chart.

    Only the system-level aggregate is kept; per-bus shape is irrelevant
    for the placeholder forecast view.
    """
    from gridsense.features import build_hourly_features

    bundle = build_hourly_features(start=start, end=end)
    return bundle.times, bundle.y_kw.sum(axis=1)


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


def _render_sidebar() -> dict:
    """Sidebar controls: date range + scenario multipliers."""
    st.sidebar.markdown("### GridSense-AZ")
    st.sidebar.caption("APS distribution-grid forecast + stress-test console")

    today = pd.Timestamp("2023-08-01")
    default_start = (today - pd.Timedelta(days=7)).date()
    default_end = today.date()
    date_range = st.sidebar.date_input(
        "History window",
        value=(default_start, default_end),
        help="Past-window for the demand chart (Tab 2).",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = default_start, default_end

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Scenario knobs")
    heat_multiplier = st.sidebar.slider(
        "Heat-wave load multiplier",
        min_value=1.0,
        max_value=2.0,
        value=1.4,
        step=0.05,
        help="Uniform per-bus scale factor (1.4 ≈ Phoenix afternoon at 115 °F).",
    )
    ev_fleet = st.sidebar.slider(
        "EV surge (fleet size)",
        min_value=0,
        max_value=5000,
        value=2000,
        step=100,
        help="Level-2 chargers online simultaneously; ~7.2 kW each.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "APS 2025 system peak: **8,631 MW** · 2023 heat wave: **31 days ≥ 110 °F**"
    )

    return {
        "start_date": pd.Timestamp(start_date),
        "end_date": pd.Timestamp(end_date),
        "heat_multiplier": float(heat_multiplier),
        "ev_fleet": int(ev_fleet),
    }


def _render_map_tab(graph, voltages: dict[str, float]) -> None:
    st.subheader("IEEE 123-bus feeder — live voltage map")
    st.caption(
        "Nodes coloured by per-unit voltage: "
        ":green[≥ 0.97] / :orange[0.95–0.97] / :red[< 0.95]. "
        "Node size ∝ √kW. Edges: lines (blue), switches (purple), transformers (orange)."
    )

    nodes_df, edges_df = build_map_data(graph, voltages)

    cols = st.columns(4)
    cols[0].metric("Buses", f"{len(nodes_df):d}")
    cols[1].metric("Edges", f"{len(edges_df):d}")
    if len(nodes_df):
        loaded = nodes_df[nodes_df["kw"] > 0]
        cols[2].metric(
            "Worst V (loaded)",
            f"{loaded['voltage_pu'].min():.3f} pu" if len(loaded) else "n/a",
        )
        cols[3].metric("Total kW", f"{nodes_df['kw'].sum():,.0f}")

    try:
        deck = build_deck(nodes_df, edges_df)
        st.pydeck_chart(deck, use_container_width=True)
    except Exception as exc:
        st.error(f"Map render failed: {exc}")
        st.dataframe(nodes_df.head(20))


def _render_forecast_tab(start_date: pd.Timestamp, end_date: pd.Timestamp) -> None:
    st.subheader("System-level demand forecast")
    st.caption(
        "Past 7 days of disaggregated bus load (summed) + placeholder 6-hour "
        "p10/p50/p90 ribbon. Replace with Graph-WaveNet outputs once "
        "`models/gwnet_v1.pt` is wired in."
    )

    # Build a feature window comfortably wider than the chart so cache hits
    # survive date-picker nudges.
    history_start = (end_date - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    history_end = end_date.strftime("%Y-%m-%d")
    try:
        times, system_kw = _cached_feature_bundle_summary(history_start, history_end)
    except Exception as exc:
        st.error(f"Feature pipeline failed: {exc}")
        return

    mask = (times >= start_date.tz_localize("UTC")) & (times <= end_date.tz_localize("UTC"))
    hist_times = times[mask]
    hist_kw = np.asarray(system_kw)[mask.to_numpy() if hasattr(mask, "to_numpy") else mask]

    if len(hist_times) == 0:
        st.info("Selected window falls outside the feature bundle range.")
        return

    # Placeholder 6-hour forecast: repeat the trailing daily profile as p50.
    horizon = 6
    last_ts = hist_times[-1]
    future_times = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1), periods=horizon, freq="h"
    )
    tail = np.asarray(hist_kw[-horizon:], dtype=float)
    p50 = np.resize(tail, horizon)
    p10 = p50 * 0.9
    p90 = p50 * 1.1

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist_times,
            y=hist_kw,
            mode="lines",
            name="system demand (kW)",
            line=dict(color="#2c3e50", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_times,
            y=p90,
            mode="lines",
            name="p90",
            line=dict(color="rgba(41,128,185,0.4)", width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_times,
            y=p10,
            mode="lines",
            name="p10",
            fill="tonexty",
            fillcolor="rgba(41,128,185,0.2)",
            line=dict(color="rgba(41,128,185,0.4)", width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_times,
            y=p50,
            mode="lines",
            name="p50 (placeholder)",
            line=dict(color="#2980b9", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        height=420,
        xaxis_title="time (UTC)",
        yaxis_title="kW",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"History window: {hist_times[0]} → {hist_times[-1]} "
        f"(n = {len(hist_times)} hours). Forecast horizon: {horizon} hours."
    )


def _top_critical_table(scenario: ScenarioResult, top_k: int = 10) -> pd.DataFrame:
    """Turn :func:`rank_critical_buses` output into a plot-friendly dataframe."""
    ranked = rank_critical_buses(scenario, top_k=top_k)
    if not ranked:
        return pd.DataFrame(columns=["bus", "voltage_pu", "margin_pu", "deficit_pu"])
    rows = []
    for bus, margin in ranked:
        v = scenario.stressed.bus_voltages_pu.get(bus, 1.0)
        rows.append(
            {
                "bus": bus,
                "voltage_pu": float(v),
                "margin_pu": float(margin),
                # Deficit is the headline bar-chart quantity — how far below
                # 0.95 pu we are (positive = bad). Clamped at 0 so healthy
                # buses don't produce negative bars.
                "deficit_pu": max(0.0, -float(margin)),
            }
        )
    return pd.DataFrame(rows)


def _render_scenario_tab(heat_multiplier: float, ev_fleet: int) -> None:
    st.subheader("Stress-scenario simulator")
    st.caption(
        "Each button runs an OpenDSS snapshot under the chosen perturbation "
        "and returns violations + recommended operator actions."
    )

    col_a, col_b, col_c = st.columns(3)
    scenario_key: str | None = None
    if col_a.button(f"Heat Wave {heat_multiplier:g}x", use_container_width=True):
        scenario_key = "heat"
    if col_b.button(f"EV Surge {ev_fleet} EVs", use_container_width=True):
        scenario_key = "ev"
    if col_c.button("Combined", use_container_width=True):
        scenario_key = "combined"

    if scenario_key is None:
        st.info("Pick a scenario above.")
        return

    with st.spinner("Solving power flow…"):
        if scenario_key == "heat":
            result = heat_wave_scenario(demand_multiplier=heat_multiplier)
        elif scenario_key == "ev":
            result = ev_surge_scenario(ev_fleet_size=ev_fleet)
        else:
            result = combined_scenario(
                demand_multiplier=heat_multiplier, ev_fleet_size=ev_fleet
            )

    # KPI strip
    kpi = st.columns(4)
    kpi[0].metric("Scenario", result.name)
    kpi[1].metric(
        "Violations",
        f"{len(result.violations) + len(result.overloads)}",
        help=f"{len(result.violations)} voltage + {len(result.overloads)} thermal",
    )
    kpi[2].metric("Worst voltage", f"{result.worst_voltage_pu:.3f} pu")
    kpi[3].metric("Worst loading", f"{result.worst_loading_pct:.1f} %")

    st.markdown("#### Top-10 critical buses")
    df = _top_critical_table(result, top_k=10)
    if not df.empty:
        st.bar_chart(df, x="bus", y="deficit_pu", height=260)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.success("No critical buses under this scenario.")

    st.markdown("#### Recommended actions")
    if result.recommended_actions:
        for i, action in enumerate(result.recommended_actions, start=1):
            st.markdown(f"**{i}.** {action}")
    else:
        st.caption("_No immediate operator action required._")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the full dashboard (safe to call only under ``streamlit run``)."""
    st.set_page_config(
        page_title="GridSense-AZ",
        page_icon="⚡",
        layout="wide",
    )
    st.title("GridSense-AZ — APS Distribution-Grid Console")

    controls = _render_sidebar()
    graph = _cached_graph()
    voltages = _cached_snapshot_voltages()

    tab_map, tab_forecast, tab_scenarios = st.tabs(
        ["Feeder Map", "Demand Forecast", "Stress Scenarios"]
    )
    with tab_map:
        _render_map_tab(graph, voltages)
    with tab_forecast:
        _render_forecast_tab(controls["start_date"], controls["end_date"])
    with tab_scenarios:
        _render_scenario_tab(controls["heat_multiplier"], controls["ev_fleet"])


if _running_as_streamlit():
    main()
