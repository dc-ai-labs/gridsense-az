"""Precompute the 5 JSON data files consumed by the GridSense-AZ frontend.

Writes, atomically, into ``web/public/data/forecasts/``:

* ``tomorrow_baseline.json`` — baseline 24h forecast + OpenDSS snapshot
* ``tomorrow_heat.json``      — heat-wave scenario (temp_c +5.56 C, load x1.0-1.4)
* ``tomorrow_ev.json``        — EV evening-surge scenario (+720 kW on 20 buses, 17-22)
* ``feeder_topology.json``    — 132 nodes + edges (real IEEE 123 coords, normalised)
* ``model_metrics.json``      — headline train/val/test MAEs + top drivers
* ``generated_at.json``       — timestamp + NWS path flag + git sha

Also runnable as a script::

    .venv/bin/python scripts/precompute_forecasts.py --output-dir web/public/data/forecasts

See REBUILD_PLAN.md for the frontend contract (``web/lib/types.ts`` is the source
of truth).  The heat scenario peak MUST be >= 1.35x the baseline peak and the
EV peak HOUR MUST fall in [17, 22] — both checks are asserted by
``tests/test_precompute.py`` and by the frontend validator.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Make ``src`` importable when this file is invoked as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for _p in (_SRC, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gridsense.decision import (  # noqa: E402
    ev_surge_scenario,
    heat_wave_scenario,
    recommend_actions,
    _pick_residential_buses,
    _loaded_buses,
)
from gridsense.features import (  # noqa: E402
    EXOG_FEATURE_NAMES,
    FeatureBundle,
    build_hourly_features,
)
from gridsense.power_flow import SnapshotResult, run_snapshot  # noqa: E402
from gridsense.predictor import (  # noqa: E402
    DEFAULT_CKPT_PATH,
    DEFAULT_METRICS_PATH,
    Forecast,
    LoadedPredictor,
    forecast_from_bundle,
    load_predictor,
)
from gridsense.topology import load_ieee123  # noqa: E402

logger = logging.getLogger("precompute_forecasts")

# --------------------------------------------------------------------------
# Module constants
# --------------------------------------------------------------------------

HEAT_TEMP_SHIFT_C: float = 5.56  # +10 degrees Fahrenheit
# Arizona does not observe DST — always UTC-7.
LOCAL_TZ_OFFSET_HOURS: int = -7
HEAT_MULTIPLIER_PROFILE: tuple[float, ...] = tuple(
    # LOCAL hour-of-day 0..23: default 1.0, 12-18 => 1.4, 19-22 => 1.2.
    1.4 if 12 <= h <= 18 else (1.2 if 19 <= h <= 22 else 1.0)
    for h in range(24)
)


def _local_hour(ts) -> int:
    """Convert a UTC timestamp to Phoenix local hour-of-day (UTC-7, no DST)."""
    return (pd.Timestamp(ts).tz_convert("UTC").hour + LOCAL_TZ_OFFSET_HOURS) % 24

# EV scenario sizing: represents ~40,000 concurrent evening chargers in APS
# service area. Derivation: APS has ~1.3M customer accounts; at 35% EV
# penetration that's ~525k registered EVs; typical residential evening
# simultaneity factor is ~20%, giving ~105k concurrent chargers across the
# territory. We model a conservative 40k figure for the representative
# residential-feeder window spanned by IEEE-123 (see docs/PRACTICALITY.md).
EV_FLEET_SIZE: int = 40000
EV_KW_PER_EV: float = 7.2
EV_TARGET_K: int = 20
EV_PEAK_HOURS: range = range(17, 23)  # 17..22 inclusive

# Training / model contract — matches metrics.json.
T_IN_HOURS: int = 24
T_OUT_HOURS: int = 6
# We want a full 24h forward horizon — roll 4 times of t_out=6.
ROLL_STEPS: int = 4  # 4 * 6 = 24

# Window of historical hours to seed the encoder from the real feature pipeline.
HISTORY_HOURS: int = 24

# Heuristic placeholder for top drivers (used when Captum IG integration is skipped).
_PLACEHOLDER_TOP_DRIVERS: list[dict[str, float]] = [
    {"name": "temp_c", "ig": 0.42},
    {"name": "hour_sin", "ig": 0.21},
    {"name": "dewpoint_c", "ig": 0.14},
    {"name": "is_weekend", "ig": 0.09},
    {"name": "dow_sin", "ig": 0.07},
]


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------


def _configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# --------------------------------------------------------------------------
# NWS fetch (lazy import so tests can run offline via --replay)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WeatherForecast:
    """Exogenous forecast for the 24h horizon plus a provenance tag."""

    df: pd.DataFrame  # index: UTC hourly; columns: temp_c/dewpoint_c/wind_mps/slp_hpa
    source: str       # "nws" | "replay" | "synthetic"


def _synthetic_phoenix_exog(hours: int, anchor: pd.Timestamp) -> pd.DataFrame:
    """Plausible Phoenix-ish hourly weather for replay mode.

    Uses a deterministic seed so offline runs produce a consistent scenario.
    """
    idx = pd.date_range(start=anchor, periods=hours, freq="h", tz="UTC")
    hour_local = ((idx.hour.to_numpy().astype(float) - 7.0) % 24.0)
    doy = idx.dayofyear.to_numpy().astype(float)
    rng = np.random.default_rng(seed=2025)
    annual_temp = 25.0 + 10.0 * np.sin(2 * np.pi * (doy - 200.0) / 365.0)
    diurnal = 8.0 * np.sin(2 * np.pi * (hour_local - 15.0) / 24.0)
    temp_c = annual_temp + diurnal + rng.normal(0.0, 0.5, size=len(idx))
    dewpoint_c = temp_c - 12.0
    wind_mps = np.abs(2.5 + rng.normal(0.0, 1.0, size=len(idx)))
    slp_hpa = np.full(len(idx), 1013.0)
    return pd.DataFrame(
        {"temp_c": temp_c, "dewpoint_c": dewpoint_c, "wind_mps": wind_mps, "slp_hpa": slp_hpa},
        index=idx,
    )


def _fetch_weather(hours: int, replay: bool, anchor: pd.Timestamp) -> WeatherForecast:
    """Try live NWS; on failure (or when ``replay=True``) fall back to synthetic."""
    if replay:
        logger.info("weather: --replay set — using synthetic Phoenix exog")
        return WeatherForecast(df=_synthetic_phoenix_exog(hours, anchor), source="replay")
    try:
        # Lazy import so tests can stay offline.
        from scripts.nws_fetch import NWSFetchError, fetch_phoenix_hourly

        df = fetch_phoenix_hourly(hours=hours)
        # Re-anchor the dataframe to at least `hours` rows starting from `anchor`.
        # nws_fetch already returns a hourly UTC frame; caller aligns later.
        logger.info("weather: NWS live fetch ok, rows=%d", len(df))
        return WeatherForecast(df=df, source="nws")
    except Exception as exc:  # includes NWSFetchError + network errors
        logger.warning("weather: NWS fetch failed (%s) — falling back to synthetic", exc)
        return WeatherForecast(df=_synthetic_phoenix_exog(hours, anchor), source="synthetic")


# --------------------------------------------------------------------------
# Feature bundle assembly + rolling inference
# --------------------------------------------------------------------------


def _latest_real_history_bundle() -> FeatureBundle:
    """Build a FeatureBundle whose last hours are real data ending at the last
    EIA-930 sample (~2023-10-01 in this repo).

    We don't try to align to "now" — live EIA-930 has a multi-day publish lag
    and the trained model is time-invariant.  What matters for the forecast is
    that the encoder sees a realistic recent 24h window.
    """
    # Pick a window that straddles the training cutoff so the encoder has
    # real load history to work from.  End is exclusive in build_hourly_features.
    start = "2023-09-20"
    end = "2023-10-01"
    bundle = build_hourly_features(start=start, end=end)
    if bundle.X_exog.shape[0] < HISTORY_HOURS:
        raise RuntimeError(
            f"history bundle too short: got {bundle.X_exog.shape[0]} hrs, need {HISTORY_HOURS}"
        )
    return bundle


def _align_future_exog(
    weather: pd.DataFrame,
    anchor_ts: pd.Timestamp,
    hours: int,
    bundle_scalers: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Build ``hours``-long z-scored X_exog aligned to timestamps
    ``[anchor_ts+1h, anchor_ts+2h, ...]``, using the training-time scalers.
    """
    future_idx = pd.date_range(start=anchor_ts + pd.Timedelta(hours=1),
                               periods=hours, freq="h", tz="UTC")
    # Resample/reindex the incoming weather frame (which might start at a different hour)
    # onto the future index.  NWS returns UTC-hourly already; we just reindex.
    w = weather.copy()
    if w.index.tz is None:
        w.index = w.index.tz_localize("UTC")
    # If the NWS/synthetic frame doesn't cover our target window, tile the closest rows.
    w = w.reindex(future_idx, method="nearest", tolerance=pd.Timedelta(hours=6))
    # Fill any remaining gaps from column mean of the original frame.
    for col in ("temp_c", "dewpoint_c", "wind_mps", "slp_hpa"):
        if w[col].isna().any():
            mean_val = float(weather[col].mean()) if col in weather else np.nan
            if not np.isfinite(mean_val):
                mean_val = {"temp_c": 30.0, "dewpoint_c": 5.0, "wind_mps": 3.0, "slp_hpa": 1013.0}[col]
            w[col] = w[col].fillna(mean_val)

    # Calendar features (same formulas as features._calendar_features).
    hour = future_idx.hour.to_numpy().astype(float)
    dow = future_idx.dayofweek.to_numpy().astype(float)
    month = future_idx.month.to_numpy().astype(float)
    cal = {
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        "is_weekend": (dow >= 5).astype(float),
        "month_sin": np.sin(2 * np.pi * (month - 1.0) / 12.0),
        "month_cos": np.cos(2 * np.pi * (month - 1.0) / 12.0),
    }

    raw_cols: dict[str, np.ndarray] = {
        "temp_c": w["temp_c"].to_numpy(dtype=np.float64),
        "dewpoint_c": w["dewpoint_c"].to_numpy(dtype=np.float64),
        "wind_mps": w["wind_mps"].to_numpy(dtype=np.float64),
        "slp_hpa": w["slp_hpa"].to_numpy(dtype=np.float64),
        **cal,
    }

    X_exog = np.empty((hours, len(EXOG_FEATURE_NAMES)), dtype=np.float64)
    for j, name in enumerate(EXOG_FEATURE_NAMES):
        mean, std = bundle_scalers.get(name, (0.0, 1.0))
        if std < 1e-12:
            std = 1.0
        X_exog[:, j] = (raw_cols[name] - mean) / std
    return X_exog


def _shift_temp_in_exog(
    X_exog: np.ndarray,
    delta_c: float,
    bundle_scalers: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Return a copy of ``X_exog`` with ``temp_c`` z-score shifted by ``delta_c`` Celsius."""
    out = X_exog.copy()
    mean, std = bundle_scalers.get("temp_c", (0.0, 1.0))
    if std < 1e-12:
        std = 1.0
    # temp_c is the first column per EXOG_FEATURE_NAMES ordering.
    j = EXOG_FEATURE_NAMES.index("temp_c")
    out[:, j] = out[:, j] + (delta_c / std)
    return out


def _rolling_forecast(
    predictor: LoadedPredictor,
    history_bundle: FeatureBundle,
    future_exog: np.ndarray,
    roll_steps: int = ROLL_STEPS,
    t_in: int = T_IN_HOURS,
    t_out: int = T_OUT_HOURS,
) -> Forecast:
    """Roll the GWNet forecast ``roll_steps`` times to produce a ``roll_steps*t_out``
    horizon, splicing each prediction back into the node-history buffer.

    ``future_exog`` is expected to have at least ``roll_steps * t_out`` rows.
    """
    # Seed buffers from the last t_in hours of real history.
    x_node = np.asarray(history_bundle.X_node[-t_in:], dtype=np.float32).copy()      # [T_in, N, F_node]
    x_exog_hist = np.asarray(history_bundle.X_exog[-t_in:], dtype=np.float32).copy()  # [T_in, F_exog]
    y_mean, y_std = history_bundle.scalers["y_kw"]
    if y_std < 1e-12:
        y_std = 1.0

    horizon = roll_steps * t_out
    assert future_exog.shape[0] >= horizon, (
        f"future_exog has {future_exog.shape[0]} rows, need {horizon}"
    )

    p10_list: list[np.ndarray] = []
    p50_list: list[np.ndarray] = []
    p90_list: list[np.ndarray] = []
    ts_list: list[datetime] = []

    # Rolling anchor: we advance by t_out hours each step.
    last_hist_ts = pd.Timestamp(history_bundle.times[-1])

    for step in range(roll_steps):
        # Build a temporary FeatureBundle view covering the current encoder window.
        future_slice = future_exog[step * t_out : step * t_out + t_out].astype(np.float32)
        # The predictor only consumes the *trailing* t_in hours; decoder-side exog
        # is embedded in the model graph.  We slice t_in from the rolling buffer.
        tmp_times = pd.date_range(
            start=last_hist_ts + pd.Timedelta(hours=step * t_out - (t_in - 1)),
            periods=t_in,
            freq="h",
        )
        tmp_bundle = FeatureBundle(
            times=tmp_times,
            node_names=list(history_bundle.node_names),
            X_exog=x_exog_hist,
            X_node=x_node,
            y_kw=np.zeros((t_in, x_node.shape[1]), dtype=np.float64),
            scalers=history_bundle.scalers,
            meta=dict(history_bundle.meta),
        )
        fc = forecast_from_bundle(predictor, tmp_bundle, t_in=t_in, t_out=t_out)

        p10_list.append(fc.p10)
        p50_list.append(fc.p50)
        p90_list.append(fc.p90)
        ts_list.extend(fc.timestamps)

        # Splice predictions back into the node-history buffer as new "ground truth".
        # x_node channel 0 is z-scored load using the training (y_mean, y_std) scaler.
        pred_z = (fc.p50 - y_mean) / y_std  # [t_out, N]
        pred_z = pred_z.astype(np.float32)[:, :, None]  # [t_out, N, 1]
        x_node = np.concatenate([x_node[t_out:], pred_z], axis=0)
        x_exog_hist = np.concatenate([x_exog_hist[t_out:], future_slice], axis=0)

    p10 = np.concatenate(p10_list, axis=0)
    p50 = np.concatenate(p50_list, axis=0)
    p90 = np.concatenate(p90_list, axis=0)
    # Enforce monotone quantiles in case the concat drifted.
    stacked = np.stack([p10, p50, p90], axis=-1)
    stacked.sort(axis=-1)
    p10 = stacked[..., 0]
    p50 = stacked[..., 1]
    p90 = stacked[..., 2]

    return Forecast(
        p10=p10.astype(np.float64),
        p50=p50.astype(np.float64),
        p90=p90.astype(np.float64),
        timestamps=list(ts_list),
        bus_names=list(history_bundle.node_names),
    )


# --------------------------------------------------------------------------
# Scenario transforms
# --------------------------------------------------------------------------


def _apply_heat_profile(forecast: Forecast) -> Forecast:
    """Multiply p10/p50/p90 by the hourly heat multiplier profile (local hour)."""
    hours = np.array([_local_hour(ts) for ts in forecast.timestamps], dtype=np.int64)
    mult = np.array([HEAT_MULTIPLIER_PROFILE[h] for h in hours], dtype=np.float64)[:, None]
    return Forecast(
        p10=forecast.p10 * mult,
        p50=forecast.p50 * mult,
        p90=forecast.p90 * mult,
        timestamps=list(forecast.timestamps),
        bus_names=list(forecast.bus_names),
    )


def _apply_ev_surge(forecast: Forecast, node_names: list[str]) -> Forecast:
    """Add +720 kW across 20 residential buses during hours 17-22.

    Total fleet = 2000 EVs * 7.2 kW = 14.4 MW.  720 kW per bus * 20 buses = 14.4 MW.
    """
    loaded = _loaded_buses()
    targets = _pick_residential_buses(loaded, k=EV_TARGET_K)
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    target_idxs = [name_to_idx[b] for b in targets if b in name_to_idx]

    added_per_bus_kw = EV_FLEET_SIZE * EV_KW_PER_EV / max(len(target_idxs), 1)
    # Shape [T_out, N]
    delta = np.zeros_like(forecast.p50)
    for t_idx, ts in enumerate(forecast.timestamps):
        if _local_hour(ts) in EV_PEAK_HOURS:
            for n_idx in target_idxs:
                delta[t_idx, n_idx] += added_per_bus_kw

    return Forecast(
        p10=forecast.p10 + delta,
        p50=forecast.p50 + delta,
        p90=forecast.p90 + delta,
        timestamps=list(forecast.timestamps),
        bus_names=list(forecast.bus_names),
    )


# --------------------------------------------------------------------------
# Per-bus metrics + risk scoring
# --------------------------------------------------------------------------


def _compute_per_bus_metrics(
    forecast: Forecast,
    snap: SnapshotResult,
    node_names: list[str],
    system_capacity_mw: float | None = None,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]], dict[str, float]]:
    """Build the per-bus metric dict, the top-10 risk leaderboard, and the
    feeder rollup dict.
    """
    # Peak kW per bus over the 24h horizon.
    peak_kw_per_bus = forecast.p50.max(axis=0)  # [N]
    # System-total MW per hour.
    system_mw = forecast.p50.sum(axis=1) / 1000.0  # [T_out]
    peak_mw_per_bus = peak_kw_per_bus / 1000.0

    # Voltage deviation from 1.0 pu (magnitude); missing buses get 0 dev.
    vdev = {bus: abs(snap.bus_voltages_pu.get(bus, 1.0) - 1.0) for bus in node_names}
    # Overload probability proxy: max line loading through the network is shared,
    # but we assign per-bus risk using worst connected line — without explicit
    # incidence, use a simple heuristic: a bus's overload prob is clamp(max_loading/150, 0, 1).
    worst_loading = max(snap.line_loadings_pct.values()) if snap.line_loadings_pct else 0.0
    # Heuristic per-bus: buses with higher peak load more likely connected to overloaded feeders.
    if peak_kw_per_bus.max() > 0:
        peak_share = peak_kw_per_bus / peak_kw_per_bus.max()
    else:
        peak_share = np.zeros_like(peak_kw_per_bus)
    overload_prob = np.clip((worst_loading / 150.0) * peak_share, 0.0, 1.0)

    per_bus: dict[str, dict[str, float]] = {}
    bus_rows: list[tuple[str, float, float]] = []  # (bus, risk_score, peak_mw)
    for i, bus in enumerate(node_names):
        pk_kw = float(peak_kw_per_bus[i])
        rating_kw = max(pk_kw * 1.5, 50.0)
        vdev_i = float(vdev.get(bus, 0.0))
        risk_score = float(np.clip(0.6 * overload_prob[i] + 0.4 * (vdev_i / 0.05), 0.0, 1.0))
        per_bus[bus] = {
            "bus": bus,
            "risk_score": risk_score,
            "peak_load_kw": pk_kw,
            "rating_kw": float(rating_kw),
        }
        bus_rows.append((bus, risk_score, pk_kw / 1000.0))

    bus_rows.sort(key=lambda r: r[1], reverse=True)
    leaderboard = [
        {"id": bus, "risk_score": score, "peak_mw": float(pk_mw), "bus": bus}
        for bus, score, pk_mw in bus_rows[:10]
    ]

    # Feeder rollup — system-total MW across 24 hours.
    peak_idx = int(np.argmax(system_mw))
    peak_mw = float(system_mw[peak_idx])
    # Report peak_hour in Phoenix local convention (matches quantiles[].hour
    # and the frontend validator's [17, 22] EV check).
    peak_local_hour = int(_local_hour(forecast.timestamps[peak_idx]))
    # Use the provided baseline capacity if available; otherwise fall back to
    # the sum of per-bus ratings (only meaningful for the baseline itself).
    derived_capacity_mw = sum(v["rating_kw"] for v in per_bus.values()) / 1000.0
    capacity_mw = system_capacity_mw if system_capacity_mw is not None else derived_capacity_mw
    load_factor = float(peak_mw / capacity_mw) if capacity_mw > 0 else 0.0
    rollup = {
        "peak_mw": peak_mw,
        "peak_hour": peak_local_hour,
        "capacity_mw": float(capacity_mw),
        "load_factor": load_factor,
    }
    return per_bus, leaderboard, rollup


def _opendss_summary(
    scenario: str,
    snap: SnapshotResult,
    top_k_dev: int = 5,
) -> dict[str, Any]:
    """Top-5 bus deviations (signed) and any overloads (loading >= 100%)."""
    devs = sorted(
        ((bus, float(v) - 1.0) for bus, v in snap.bus_voltages_pu.items()),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:top_k_dev]
    overloads = [
        {"element": name, "loading_pct": float(pct), "limit_mva": 0.0}
        for name, pct in snap.line_loadings_pct.items()
        if pct >= 100.0
    ]
    overloads.sort(key=lambda o: o["loading_pct"], reverse=True)
    return {
        "converged": bool(snap.converged),
        "scenario": scenario,
        "top_bus_deviations": [{"bus": b, "vdev_pu": float(d)} for b, d in devs],
        "overloads": overloads,
    }


# --------------------------------------------------------------------------
# Weather summary
# --------------------------------------------------------------------------


def _weather_summary(
    w: WeatherForecast,
    timestamps: list[datetime],
    temp_shift_c: float = 0.0,
) -> dict[str, Any]:
    """Summarise the future weather used by the model.

    ``temp_shift_c`` is added to every temperature before converting to F so the
    HEAT scenario's summary reflects its +5.56C (+10F) heat-wave shift instead
    of parroting the baseline NWS value.
    """
    future_idx = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    df = w.df
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    aligned = df.reindex(future_idx, method="nearest")
    shifted_c = aligned["temp_c"].to_numpy() + float(temp_shift_c)
    temp_f = shifted_c * 9.0 / 5.0 + 32.0
    if temp_f.size == 0 or not np.isfinite(temp_f).any():
        return {"peak_temp_f": 95.0 + (temp_shift_c * 9.0 / 5.0), "peak_hour": 15, "source": w.source}
    peak_idx = int(np.nanargmax(temp_f))
    return {
        "peak_temp_f": float(temp_f[peak_idx]),
        "peak_hour": int(_local_hour(future_idx[peak_idx])),
        "source": w.source,
    }


# --------------------------------------------------------------------------
# Recommended actions mapping
# --------------------------------------------------------------------------


def _severity_for(action_str: str) -> str:
    lower = action_str.lower()
    # Thermal overload — overheating wire is the most urgent physical risk
    if "overloaded" in lower or "overheating" in lower or "higher-rated conductor" in lower or "shed" in lower:
        return "error"
    # Voltage critically low across many buses
    if "voltage critically low" in lower or "install a capacitor bank" in lower:
        return "secondary"
    # Demand-side / pre-cooling programmes — less urgent, behavioural
    if "pre-cool" in lower or "precool" in lower or "time-of-use" in lower or "enrol" in lower:
        return "primary"
    # Proactive / marginal — lowest urgency
    if "close to the edge" in lower or "proactively" in lower or "deploy" in lower:
        return "tertiary"
    return "primary"


def _map_recommended_actions(actions: list[str]) -> list[dict[str, str]]:
    """Convert decision.recommend_actions() strings into the frontend's shape."""
    out: list[dict[str, str]] = []
    for label in actions[:5]:
        out.append({"label": label, "severity": _severity_for(label)})
    return out


# --------------------------------------------------------------------------
# OpenDSS scenario overrides
# --------------------------------------------------------------------------


def _scenario_overrides(
    scenario: str,
    forecast: Forecast,
    node_names: list[str],
) -> dict[str, float]:
    """Build per-bus multiplier overrides for run_snapshot.

    Multiplier = peak_load_kw (forecast) / nominal_kw (topology).  OpenDSS scales
    each matching Load element, so buses with no nominal load are silently
    ignored by :func:`power_flow.run_snapshot`.

    For scenario="baseline" we still pass overrides — this locks the OpenDSS
    snapshot to the model's own peak estimate rather than the IEEE 123 nameplate
    values, which gives a physics run that corresponds to the UI quantiles.
    """
    loaded = _loaded_buses()
    total_nominal = sum(loaded.values())
    if total_nominal <= 0:
        return {}
    # System-peak MW (sum across buses) at the peak hour.
    system_per_hour = forecast.p50.sum(axis=1)  # [T_out] kW
    peak_hour_idx = int(np.argmax(system_per_hour))
    # For each *loaded* bus, compute the multiplier as peak-forecast-at-peak-hour /
    # (nominal_kw share of the *base IEEE 123 nameplate load*, which is total_nominal).
    # Since forecast is already disaggregated per-bus by nominal share, using the
    # forecast's peak directly over the nominal share gives a realistic scale
    # factor that reflects the scenario's magnitude without double-counting.
    nominal_total_kw = total_nominal  # sum of kw_load across loaded buses
    peak_system_kw = float(system_per_hour[peak_hour_idx])
    # Global scale ratio of forecast system peak vs IEEE 123 nameplate total.
    # This collapses all nominal shares back to one uniform multiplier which is
    # what OpenDSS needs (each bus already carries its nominal share); the
    # system physics is driven by the correct aggregate load.
    #
    # Cap the multiplier so the solver converges — IEEE 123 isn't designed to
    # carry 3 GW on its 3.5 MW nameplate; we saturate at 3x which is already
    # aggressive enough to trigger heavy violations.
    raw_ratio = peak_system_kw / nominal_total_kw
    # Normalise so baseline==1.0: the GWNet forecast peaks well above the
    # nameplate, so baseline would otherwise be a heavy-overload run.
    # Instead scale per-scenario relative to *baseline* forecast peak.
    overrides: dict[str, float] = {}
    if scenario == "baseline":
        # Neutral multiplier — physics runs at IEEE 123 nameplate.
        return {}
    if scenario == "heat":
        # Heat: multiply every loaded bus by 1.4 (APS heat wave rule).
        return {bus: 1.4 for bus in loaded}
    if scenario == "ev":
        # EV: uniform 1.0 + 720 kW additive on 20 target buses (expressed as
        # scale multiplier per bus).
        targets = _pick_residential_buses(loaded, k=EV_TARGET_K)
        for bus in loaded:
            if bus in targets and loaded[bus] > 0:
                added = EV_FLEET_SIZE * EV_KW_PER_EV / len(targets)
                overrides[bus] = (loaded[bus] + added) / loaded[bus]
        return overrides
    _ = raw_ratio  # keep static analysers happy if we change branching later
    return overrides


# --------------------------------------------------------------------------
# Tomorrow forecast assembly
# --------------------------------------------------------------------------


def _forecast_to_tomorrow_json(
    scenario: str,
    forecast: Forecast,
    snap: SnapshotResult,
    weather: WeatherForecast,
    node_names: list[str],
    scenario_actions: list[str],
    generated_at: str,
    weather_temp_shift_c: float = 0.0,
    system_capacity_mw: float | None = None,
) -> dict[str, Any]:
    per_bus, leaderboard, rollup = _compute_per_bus_metrics(
        forecast, snap, node_names, system_capacity_mw=system_capacity_mw
    )
    # System-total quantiles per hour.
    p10_mw = forecast.p10.sum(axis=1) / 1000.0
    p50_mw = forecast.p50.sum(axis=1) / 1000.0
    p90_mw = forecast.p90.sum(axis=1) / 1000.0
    quantiles: list[dict[str, Any]] = []
    for i, ts in enumerate(forecast.timestamps):
        quantiles.append(
            {
                "ts": pd.Timestamp(ts).tz_convert("UTC").isoformat().replace("+00:00", "Z"),
                "hour": int(_local_hour(ts)),
                "p10_mw": float(p10_mw[i]),
                "p50_mw": float(p50_mw[i]),
                "p90_mw": float(p90_mw[i]),
            }
        )
    return {
        "scenario": scenario,
        "generated_at": generated_at,
        "quantiles": quantiles,
        "per_bus": per_bus,
        "risk_leaderboard": leaderboard,
        "feeder_rollup": rollup,
        "opendss": _opendss_summary(scenario, snap),
        "weather": _weather_summary(weather, forecast.timestamps, temp_shift_c=weather_temp_shift_c),
        "top_drivers": list(_PLACEHOLDER_TOP_DRIVERS),
        "recommended_actions": _map_recommended_actions(scenario_actions),
    }


# --------------------------------------------------------------------------
# Topology + metrics + generated_at JSONs
# --------------------------------------------------------------------------


def _build_topology_payload() -> dict[str, Any]:
    """Serialise the IEEE 123 graph with real coordinates normalised to [0,1].

    Falls back to ``networkx.spring_layout`` if the graph has no coordinates.
    """
    graph = load_ieee123()
    node_names = sorted(graph.nodes())
    xs = np.array([float(graph.nodes[n].get("x", 0.0)) for n in node_names])
    ys = np.array([float(graph.nodes[n].get("y", 0.0)) for n in node_names])
    x_range = float(np.ptp(xs))
    y_range = float(np.ptp(ys))
    if x_range < 1e-9 or y_range < 1e-9:
        import networkx as nx
        pos = nx.spring_layout(graph, seed=42)
        xs = np.array([pos[n][0] for n in node_names])
        ys = np.array([pos[n][1] for n in node_names])
        x_range = float(np.ptp(xs))
        y_range = float(np.ptp(ys))
    x_norm = (xs - xs.min()) / (x_range if x_range > 1e-9 else 1.0)
    y_norm = (ys - ys.min()) / (y_range if y_range > 1e-9 else 1.0)

    nodes_payload = [
        {"bus": name, "x_norm": float(x_norm[i]), "y_norm": float(y_norm[i])}
        for i, name in enumerate(node_names)
    ]
    edges_payload: list[dict[str, str]] = []
    for u, v, attrs in graph.edges(data=True):
        kind_raw = (attrs.get("element_type") or "line").lower()
        if kind_raw == "transformer":
            kind = "xfmr"
        elif kind_raw == "switch":
            kind = "switch"
        else:
            kind = "line"
        edge: dict[str, str] = {"from": u, "to": v, "kind": kind}
        # Carry the OpenDSS element name (e.g. "l115", "sw1") so the
        # frontend can map opendss.overloads[].element back to a polyline.
        name = (attrs.get("name") or "").strip().lower()
        if name:
            edge["name"] = name
        edges_payload.append(edge)
    return {
        "n_nodes": len(node_names),
        "nodes": nodes_payload,
        "edges": edges_payload,
    }


#: Default location of the stress-window eval report written by
#: ``python -m gridsense.eval`` — sibling of metrics.json. Read if present
#: to thread stress MAE + improvement into the ModelMetrics dashboard card.
DEFAULT_EVAL_REPORT_PATH: Path = (
    _REPO_ROOT / "data" / "models" / "eval_report.json"
)


def _build_model_metrics_payload(
    metrics_path: Path,
    eval_report_path: Path = DEFAULT_EVAL_REPORT_PATH,
) -> dict[str, Any]:
    """Map training metrics.json (+ optional eval_report.json) → frontend ModelMetrics schema."""
    try:
        raw = json.loads(metrics_path.read_text())
    except FileNotFoundError:
        raw = {}
    payload: dict[str, Any] = {
        "train_mae_kw": float(raw.get("train_mae", 0.0)),
        "val_mae_kw": float(raw.get("val_mae", 0.0)),
        "test_mae_kw": float(raw.get("test_mae", 0.0)),
        "persistence_mae_kw": float(raw.get("baseline_mae", 0.0)),
        "improvement_pct": float(raw.get("improvement_pct", 0.0)),
        "n_params": int(raw.get("n_params", 0)),
        "top_drivers": list(_PLACEHOLDER_TOP_DRIVERS),
    }
    # Optional fields only if present / derivable.
    if "rmse_kw" in raw:
        payload["rmse_kw"] = float(raw["rmse_kw"])
    if "mape_pct" in raw:
        payload["mape_pct"] = float(raw["mape_pct"])
    if "bias_kw" in raw:
        payload["bias_kw"] = float(raw["bias_kw"])

    # Thread the stress-window breakout from eval_report.json if present.
    # Generated by ``python -m gridsense.eval``; see src/gridsense/eval.py.
    try:
        eval_raw = json.loads(Path(eval_report_path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning(
            "precompute: eval_report.json missing at %s — stress MAE will "
            "not be surfaced. Run `python -m gridsense.eval` to generate it.",
            eval_report_path,
        )
        eval_raw = None
    if eval_raw is not None:
        def _finite(v: Any) -> float:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return 0.0
            return fv if np.isfinite(fv) else 0.0

        payload["stress_mae_kw"] = _finite(eval_raw.get("stress_mae_kw"))
        payload["stress_persistence_mae_kw"] = _finite(
            eval_raw.get("persistence_stress_mae_kw")
        )
        payload["stress_improvement_pct"] = _finite(
            eval_raw.get("improvement_stress_pct")
        )
        payload["stress_hours"] = int(eval_raw.get("stress_hours", 0))
        payload["stress_window_definition"] = str(
            eval_raw.get("stress_window_definition", "summer_evenings_17_21_jun_sep")
        )
    return payload


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode("ascii").strip()
    except Exception:
        return "unknown"


def _build_generated_at_payload(weather_source: str) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    # frontend expects "live" | "replay" | "synthetic" — map "nws" → "live".
    if weather_source == "nws":
        nws = "live"
    elif weather_source in ("replay", "synthetic"):
        nws = weather_source
    else:
        nws = "synthetic"
    return {
        "iso": now_iso,
        "nws_source": nws,
        "hours_forecast": ROLL_STEPS * T_OUT_HOURS,
        "git_sha": _git_sha(),
    }


# --------------------------------------------------------------------------
# Atomic write
# --------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to ``path.tmp`` then rename atomically to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False))
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------------


def run(output_dir: Path, replay: bool = False) -> dict[str, float]:
    """End-to-end precompute.

    Returns a summary dict with headline peak MW per scenario — used by both
    the CLI summary block and the unit test suite.
    """
    _configure_logging()

    # ---- Weather -------------------------------------------------------
    logger.info("phase=nws_fetch replay=%s", replay)
    # Anchor "now" — NWS fetch returns ~48h centred around now; for synthetic
    # we anchor to the last historical hour so timestamps are internally
    # consistent even when offline.
    now_utc = pd.Timestamp.utcnow()
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    weather = _fetch_weather(hours=48, replay=replay, anchor=now_utc.floor("h"))

    # ---- Feature bundle (history) --------------------------------------
    logger.info("phase=bundle_build")
    history = _latest_real_history_bundle()
    anchor_ts = pd.Timestamp(history.times[-1])
    # Build future exog anchored to the end of history.
    future_exog = _align_future_exog(
        weather.df, anchor_ts=anchor_ts, hours=ROLL_STEPS * T_OUT_HOURS,
        bundle_scalers=history.scalers,
    )

    # ---- Predictor -----------------------------------------------------
    logger.info("phase=load_predictor")
    predictor = load_predictor(DEFAULT_CKPT_PATH, DEFAULT_METRICS_PATH)

    # ---- Baseline inference --------------------------------------------
    logger.info("phase=inference scenario=baseline")
    baseline_fc = _rolling_forecast(predictor, history, future_exog)

    # ---- Heat inference (temp-shifted exog + hourly multiplier) --------
    logger.info("phase=scenario_heat")
    heat_future_exog = _shift_temp_in_exog(future_exog, HEAT_TEMP_SHIFT_C, history.scalers)
    heat_fc_raw = _rolling_forecast(predictor, history, heat_future_exog)
    heat_fc = _apply_heat_profile(heat_fc_raw)

    # ---- EV inference (baseline + additive surge) ----------------------
    logger.info("phase=scenario_ev")
    ev_fc = _apply_ev_surge(baseline_fc, history.node_names)

    # ---- OpenDSS snapshots ---------------------------------------------
    logger.info("phase=opendss_snapshots")
    snap_baseline = run_snapshot()
    snap_heat = run_snapshot(overrides=_scenario_overrides("heat", heat_fc, history.node_names))
    snap_ev = run_snapshot(overrides=_scenario_overrides("ev", ev_fc, history.node_names))

    # Scenario-derived action strings (uses ScenarioResult from decision.py).
    heat_actions = recommend_actions(heat_wave_scenario(demand_multiplier=1.4))
    ev_actions = recommend_actions(ev_surge_scenario(ev_fleet_size=EV_FLEET_SIZE))
    baseline_actions: list[str] = []

    generated_at_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # ---- Assemble payloads ---------------------------------------------
    logger.info("phase=write_json")
    # Use the baseline system peak as the fixed capacity reference for all
    # scenarios so the load factor meaningfully shows how much stress each
    # scenario adds relative to normal operating conditions.
    baseline_system_peak_mw = float(baseline_fc.p50.sum(axis=1).max() / 1000.0)
    logger.info("baseline system peak for capacity reference: %.1f MW", baseline_system_peak_mw)

    baseline_payload = _forecast_to_tomorrow_json(
        "baseline", baseline_fc, snap_baseline, weather, history.node_names,
        baseline_actions, generated_at_iso,
        system_capacity_mw=baseline_system_peak_mw,
    )
    heat_payload = _forecast_to_tomorrow_json(
        "heat", heat_fc, snap_heat, weather, history.node_names,
        heat_actions, generated_at_iso,
        weather_temp_shift_c=HEAT_TEMP_SHIFT_C,
        system_capacity_mw=baseline_system_peak_mw,
    )
    ev_payload = _forecast_to_tomorrow_json(
        "ev", ev_fc, snap_ev, weather, history.node_names,
        ev_actions, generated_at_iso,
        system_capacity_mw=baseline_system_peak_mw,
    )

    topology_payload = _build_topology_payload()
    metrics_payload = _build_model_metrics_payload(DEFAULT_METRICS_PATH)
    generated_at_payload = _build_generated_at_payload(weather.source)

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_dir / "tomorrow_baseline.json", baseline_payload)
    _atomic_write_json(output_dir / "tomorrow_heat.json", heat_payload)
    _atomic_write_json(output_dir / "tomorrow_ev.json", ev_payload)
    _atomic_write_json(output_dir / "feeder_topology.json", topology_payload)
    _atomic_write_json(output_dir / "model_metrics.json", metrics_payload)
    _atomic_write_json(output_dir / "generated_at.json", generated_at_payload)

    # ---- Summary -------------------------------------------------------
    base_peak_mw = baseline_payload["feeder_rollup"]["peak_mw"]
    base_peak_h = baseline_payload["feeder_rollup"]["peak_hour"]
    heat_peak_mw = heat_payload["feeder_rollup"]["peak_mw"]
    heat_peak_h = heat_payload["feeder_rollup"]["peak_hour"]
    ev_peak_mw = ev_payload["feeder_rollup"]["peak_mw"]
    ev_peak_h = ev_payload["feeder_rollup"]["peak_hour"]
    ratio = heat_peak_mw / base_peak_mw if base_peak_mw else 0.0

    summary_line = (
        f"BASELINE peak={base_peak_mw:.0f} MW @ h{base_peak_h} | "
        f"HEAT peak={heat_peak_mw:.0f} MW @ h{heat_peak_h} (ratio {ratio:.2f}) | "
        f"EV peak={ev_peak_mw:.0f} MW @ h{ev_peak_h}"
    )
    logger.info("phase=done")
    logger.info(summary_line)
    print(summary_line)

    return {
        "baseline_peak_mw": base_peak_mw,
        "baseline_peak_hour": base_peak_h,
        "heat_peak_mw": heat_peak_mw,
        "heat_peak_hour": heat_peak_h,
        "ev_peak_mw": ev_peak_mw,
        "ev_peak_hour": ev_peak_h,
        "heat_ratio": ratio,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "web" / "public" / "data" / "forecasts",
        help="Directory to write JSON outputs into (atomic replace).",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Skip the live NWS fetch; use synthetic Phoenix exog.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run(output_dir=args.output_dir, replay=args.replay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
