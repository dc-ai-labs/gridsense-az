"""Per-feeder p10/p50/p90 forecast ribbon (plotly).

Two building blocks used by :mod:`app.streamlit_app`:

* :func:`get_cached_predictor` — cached :class:`LoadedPredictor` so the
  (small but non-trivial) checkpoint load happens once per Streamlit
  process, not once per rerun. Returns ``None`` when the checkpoint is
  missing so the dashboard can fall back to the legacy placeholder ribbon.
* :func:`build_forecast_figure` — the plotly rendering. Accepts either a
  trained :class:`Forecast` payload (from
  :func:`gridsense.predictor.forecast_from_bundle`) or the legacy
  ``p10/p50/p90`` scalar arrays, so the dashboard can call the same
  renderer in both the "model-wired" and "placeholder" paths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gridsense.features import FeatureBundle
from gridsense.predictor import (
    DEFAULT_CKPT_PATH,
    DEFAULT_METRICS_PATH,
    Forecast,
    LoadedPredictor,
    forecast_from_bundle,
    load_predictor,
)

logger = logging.getLogger(__name__)

__all__ = [
    "get_cached_predictor",
    "run_forecast",
    "build_forecast_figure",
    "aggregate_system_forecast",
    "select_bus_forecast",
]


# ---------------------------------------------------------------------------
# Cached loader
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_cached_predictor(
    ckpt_path: str = str(DEFAULT_CKPT_PATH),
    metrics_path: str = str(DEFAULT_METRICS_PATH),
) -> LoadedPredictor | None:
    """Load and cache a :class:`LoadedPredictor` for the lifetime of the app.

    Returns ``None`` when the checkpoint is absent — callers are expected
    to fall back to the legacy placeholder ribbon in that case so the
    dashboard still renders something useful on a fresh clone.

    String-valued arguments (instead of :class:`Path`) are intentional:
    Streamlit's ``cache_resource`` hashes arguments, and ``Path`` objects
    are not hashable across Windows / POSIX variants in some Streamlit
    versions.
    """
    path = Path(ckpt_path)
    if not path.exists():
        logger.info("Forecast checkpoint missing at %s — using placeholder ribbon.", path)
        return None
    try:
        return load_predictor(ckpt_path=path, metrics_path=Path(metrics_path))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("load_predictor failed (%s) — using placeholder ribbon.", exc)
        return None


def run_forecast(
    predictor: LoadedPredictor,
    bundle: FeatureBundle,
) -> Forecast | None:
    """Thin wrapper so the dashboard can trap prediction errors without crashing."""
    try:
        return forecast_from_bundle(
            predictor,
            bundle,
            t_in=predictor.config.t_in,
            t_out=predictor.config.t_out,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("forecast_from_bundle failed (%s) — using placeholder ribbon.", exc)
        return None


# ---------------------------------------------------------------------------
# Forecast shaping helpers
# ---------------------------------------------------------------------------


def aggregate_system_forecast(forecast: Forecast) -> dict[str, np.ndarray]:
    """Sum the per-bus quantile arrays into system-level p10/p50/p90.

    Summing the quantile arrays is a conservative approximation: it
    over-estimates the true system-level interval width (since per-bus
    errors are positively correlated), but it's the simplest band that
    keeps p10 <= p50 <= p90 at the system level.
    """
    return {
        "p10": forecast.p10.sum(axis=1),
        "p50": forecast.p50.sum(axis=1),
        "p90": forecast.p90.sum(axis=1),
    }


def select_bus_forecast(forecast: Forecast, bus: str) -> dict[str, np.ndarray]:
    """Slice the forecast payload to a single bus."""
    if bus not in forecast.bus_names:
        raise KeyError(f"bus {bus!r} not in forecast (have {len(forecast.bus_names)}).")
    j = forecast.bus_names.index(bus)
    return {
        "p10": forecast.p10[:, j],
        "p50": forecast.p50[:, j],
        "p90": forecast.p90[:, j],
    }


# ---------------------------------------------------------------------------
# Plotly rendering
# ---------------------------------------------------------------------------


def build_forecast_figure(
    hist_times: Iterable[pd.Timestamp],
    hist_values: np.ndarray,
    future_times: Iterable[pd.Timestamp],
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    *,
    y_label: str = "kW",
    hist_label: str = "observed",
    p50_label: str = "p50",
    placeholder: bool = False,
) -> go.Figure:
    """Render the classic history + p10/p50/p90 ribbon plot.

    Parameters:
        hist_times, hist_values: Past-window series to plot as a solid line.
        future_times, p10, p50, p90: Forecast horizon + three quantile arrays.
        y_label: Y-axis label (e.g. ``"kW"`` for system totals,
            ``"kW (bus X)"`` for per-bus view).
        hist_label: Legend label for the historical line.
        p50_label: Legend label for the median forecast line.
        placeholder: If True, the p50 line is drawn with a dashed stroke and
            tagged "(placeholder)" — matches the original fallback style.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(hist_times),
            y=np.asarray(hist_values),
            mode="lines",
            name=hist_label,
            line=dict(color="#2c3e50", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(future_times),
            y=np.asarray(p90),
            mode="lines",
            name="p90",
            line=dict(color="rgba(41,128,185,0.4)", width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(future_times),
            y=np.asarray(p10),
            mode="lines",
            name="p10",
            fill="tonexty",
            fillcolor="rgba(41,128,185,0.2)",
            line=dict(color="rgba(41,128,185,0.4)", width=0),
            showlegend=False,
        )
    )
    median_dash = "dash" if placeholder else "solid"
    median_label = f"{p50_label} (placeholder)" if placeholder else p50_label
    fig.add_trace(
        go.Scatter(
            x=list(future_times),
            y=np.asarray(p50),
            mode="lines",
            name=median_label,
            line=dict(color="#2980b9", width=2, dash=median_dash),
        )
    )
    fig.update_layout(
        height=420,
        xaxis_title="time (UTC)",
        yaxis_title=y_label,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def render() -> None:
    """Deprecated no-op kept for backwards compatibility with older imports."""
    return None
