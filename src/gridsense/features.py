"""Hourly feature pipeline for GridSense-AZ feeder-load forecasting.

Joins three exogenous time series — NOAA ISD weather at KPHX, NREL NSRDB
irradiance at Phoenix, and EIA-930 AZPS system demand — onto a common UTC
hourly index, derives calendar/seasonality features, and disaggregates the
system demand across the 132-bus IEEE 123 feeder using each bus's nominal
kW share. The resulting :class:`FeatureBundle` is the direct input to the
Graph WaveNet training loop.

Key design decisions
--------------------
* **Hourly UTC** — all series are resampled / aligned to ``pd.date_range(
  start, end, freq='h', tz='UTC')``. Arizona never observes DST so PSM3
  local-time data is shifted by a flat ``+7h``.
* **Synthetic fallback** — when the raw parquet / CSVs are missing (tests
  use empty dirs; CI without API keys) we emit a realistic-ish random-walk
  system-demand + sinusoidal weather stub so downstream code can proceed.
  The bundle's ``meta["source"]`` records which branch ran.
* **Disaggregation** — per-bus target is
  ``y_kw[t,i] = D(t) * nominal_kw[i] / sum(nominal_kw)``. Sum across buses
  exactly reproduces system demand (modulo float rounding).
* **Standardisation** — ``X_exog`` is z-scored per column; scalers are
  stored so inference can re-apply them. The pre-standardisation raw
  dataframe is kept on the bundle (``raw_exog_df``) for sanity checks.
* **Node features** — ``F_node=2``: per-bus standardised ``y_kw`` plus a
  static repeated feature equal to the bus's nominal-kW share of the
  feeder (constant in time). ``y_kw`` itself (the target) is NOT
  standardised.

Public API
----------
* :class:`FeatureBundle` — frozen dataclass container
* :data:`EXOG_FEATURE_NAMES` — ordered tuple of ``X_exog`` column names
* :func:`build_hourly_features` — main entry point
* :func:`save_bundle` / :func:`load_bundle` — on-disk round-trip
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
"""Repo root, used to resolve default ``data/`` and ``data/ieee123`` paths."""

DEFAULT_DATA_ROOT: Path = REPO_ROOT / "data" / "raw"
DEFAULT_TOPOLOGY_ROOT: Path = REPO_ROOT / "data" / "ieee123"

# Fixed column order — the model references features by position, so do NOT
# reorder casually. New features must be appended to the tuple.
EXOG_FEATURE_NAMES: tuple[str, ...] = (
    "temp_c",
    "dewpoint_c",
    "humidity_pct",
    "ghi_wm2",
    "dni_wm2",
    "wind_mps",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "month_sin",
    "month_cos",
)

NODE_FEATURE_NAMES: tuple[str, ...] = (
    "y_kw_standardised",
    "nominal_kw_share",
)

# Phoenix-local → UTC offset (Arizona is MST year-round, no DST).
PHOENIX_UTC_OFFSET_HOURS: int = 7

# Fallback weather values when NOAA unavailable but EIA demand is present.
FALLBACK_TEMP_C: float = 20.0
FALLBACK_DEW_C: float = 5.0
FALLBACK_HUMIDITY_PCT: float = 30.0
FALLBACK_WIND_MPS: float = 2.0
FALLBACK_GHI_WM2: float = 200.0
FALLBACK_DNI_WM2: float = 250.0

# Synthetic system-demand parameters for AZPS (matches observed EIA-930 scale).
SYNTH_MEAN_MW: float = 2000.0  # ~2 GW mean
SYNTH_ANNUAL_AMPL_MW: float = 600.0  # summer peak over winter trough
SYNTH_DAILY_AMPL_MW: float = 400.0  # 4pm peak vs 4am trough
SYNTH_NOISE_SIGMA_MW: float = 80.0

# NOAA ISD sentinel values indicating missing data.
_ISD_TMP_MISSING = 9999
_ISD_WIND_MISSING = 9999
_ISD_VALID_QC = {"1", "5", "7", "9", "A", "C", "I", "M", "Q"}


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureBundle:
    """Time-aligned feature tensors for training a per-bus load model.

    Attributes:
        times: UTC hourly ``DatetimeIndex``, shape ``[T]``.
        node_names: Ordered bus names, shape ``[N]``. Alignment must match
            :func:`gridsense.topology.to_pyg_data` so row ``i`` of the
            per-bus tensors corresponds to ``node_names[i]``.
        X_exog: Standardised exogenous features, shape ``[T, F_exog]``.
            Column order is :data:`EXOG_FEATURE_NAMES`.
        X_node: Per-bus features, shape ``[T, N, F_node]``. Column order is
            :data:`NODE_FEATURE_NAMES`.
        y_kw: Disaggregated per-bus load in kW, shape ``[T, N]``. NOT
            standardised — the model is expected to normalise at output.
        scalers: ``{feature_name -> (mean, std)}`` for every z-scored
            feature plus an entry ``"y_kw"`` with the global mean/std of
            the kW target (used by the dashboard to denormalise model
            predictions if needed).
        raw_exog_df: Pre-standardisation exogenous features. Useful for
            sanity checks and debugging — ``is_weekend`` lives in ``{0, 1}``
            and sin/cos features in ``[-1, 1]`` here.
        meta: Free-form diagnostic metadata. Always contains the keys
            ``source`` (``"real"`` / ``"synthetic"`` / ``"mixed"``),
            ``start``, ``end``, ``n_buses``.
    """

    times: pd.DatetimeIndex
    node_names: list[str]
    X_exog: np.ndarray
    X_node: np.ndarray
    y_kw: np.ndarray
    scalers: dict[str, tuple[float, float]]
    raw_exog_df: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# NOAA ISD parsing
# ---------------------------------------------------------------------------


def _parse_isd_tmp(series: pd.Series) -> pd.Series:
    """Decode ISD ``TMP`` column (``'+0245,1'``) into °C with NaN for missing."""
    split = series.astype(str).str.split(",", n=1, expand=True)
    value = pd.to_numeric(split[0], errors="coerce")
    qc = split[1].fillna("").astype(str)
    valid = qc.isin(_ISD_VALID_QC) & (value.abs() < _ISD_TMP_MISSING)
    celsius = value / 10.0
    return celsius.where(valid)


def _parse_isd_wind(series: pd.Series) -> pd.Series:
    """Decode ISD ``WND`` column (``'310,1,N,0015,1'``) → speed in m/s."""
    # Fields: direction, dir_qc, type, speed(m/s *10), speed_qc
    parts = series.astype(str).str.split(",", n=4, expand=True)
    if parts.shape[1] < 5:
        # Pad missing columns so indexing is safe.
        for missing_col in range(parts.shape[1], 5):
            parts[missing_col] = ""
    speed = pd.to_numeric(parts[3], errors="coerce")
    qc = parts[4].fillna("").astype(str)
    valid = qc.isin(_ISD_VALID_QC) & (speed.abs() < _ISD_WIND_MISSING)
    mps = speed / 10.0
    return mps.where(valid)


def _load_noaa_isd(noaa_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load + aggregate all ``KPHX_*.csv`` files to hourly UTC means.

    Returns an hourly-indexed DataFrame with columns ``temp_c``,
    ``dewpoint_c``, ``wind_mps`` covering ``[start, end)``. Missing years /
    missing cells are left as NaN; the caller forward-fills.
    """
    frames: list[pd.DataFrame] = []
    if not noaa_dir.exists():
        logger.info("[features] NOAA dir %s missing — skipping weather", noaa_dir)
        return pd.DataFrame()
    for path in sorted(noaa_dir.glob("KPHX_*.csv")):
        try:
            df = pd.read_csv(
                path,
                usecols=["DATE", "TMP", "DEW", "WND"],
                low_memory=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[features] failed to read %s: %s", path, exc)
            continue
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce", utc=True)
        df = df.dropna(subset=["DATE"])
        if df.empty:
            continue
        df["temp_c"] = _parse_isd_tmp(df["TMP"])
        df["dewpoint_c"] = _parse_isd_tmp(df["DEW"])
        df["wind_mps"] = _parse_isd_wind(df["WND"])
        df = df[["DATE", "temp_c", "dewpoint_c", "wind_mps"]].set_index("DATE")
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames).sort_index()
    # Multiple reports per hour (METAR + SPECI + FM-12) → aggregate to hourly mean.
    hourly = combined.resample("1h").mean()
    # Clip to requested window.
    mask = (hourly.index >= start) & (hourly.index < end)
    return hourly.loc[mask]


# ---------------------------------------------------------------------------
# EIA-930 demand parsing
# ---------------------------------------------------------------------------


def _load_eia930_demand(
    eia_dir: Path, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    """Load hourly AZPS demand in MW from ``azps_demand.parquet``.

    Returns a UTC-indexed Series named ``demand_mw``. Empty series if the
    parquet is missing.
    """
    parquet = eia_dir / "azps_demand.parquet"
    if not parquet.exists():
        logger.info("[features] EIA-930 parquet %s missing — synthetic demand", parquet)
        return pd.Series(dtype="float64", name="demand_mw")
    try:
        df = pd.read_parquet(parquet)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[features] failed to read %s: %s", parquet, exc)
        return pd.Series(dtype="float64", name="demand_mw")
    if "period" not in df.columns or "value" not in df.columns:
        logger.warning(
            "[features] unexpected EIA-930 columns %s — synthetic fallback", list(df.columns)
        )
        return pd.Series(dtype="float64", name="demand_mw")

    ts = pd.to_datetime(df["period"], errors="coerce", utc=True)
    value = pd.to_numeric(df["value"], errors="coerce")
    series = pd.Series(value.values, index=ts, name="demand_mw").dropna().sort_index()
    # Deduplicate in case of overlapping pages.
    series = series[~series.index.duplicated(keep="last")]
    mask = (series.index >= start) & (series.index < end)
    return series.loc[mask]


# ---------------------------------------------------------------------------
# NSRDB PSM3 parsing
# ---------------------------------------------------------------------------


def _load_nsrdb_psm3(
    nsrdb_dir: Path, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Load hourly NSRDB PSM3 irradiance for Phoenix.

    PSM3 CSV layout: row 0 = metadata, row 1 = column headers, row 2+ = data.
    Input timestamps are Phoenix-local (``utc=false``); we shift to UTC by
    ``+7h``.
    """
    if not nsrdb_dir.exists():
        logger.info("[features] NSRDB dir %s missing — fallback irradiance", nsrdb_dir)
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in sorted(nsrdb_dir.glob("phoenix_*.csv")):
        try:
            df = pd.read_csv(path, skiprows=2, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[features] failed to read %s: %s", path, exc)
            continue
        required = {"Year", "Month", "Day", "Hour", "Minute"}
        if not required.issubset(df.columns):
            logger.warning("[features] %s missing date cols %s", path, required - set(df.columns))
            continue
        local_ts = pd.to_datetime(
            df[["Year", "Month", "Day", "Hour", "Minute"]], errors="coerce"
        )
        df = df.assign(_local_ts=local_ts).dropna(subset=["_local_ts"])
        # Phoenix local → UTC.
        utc_ts = df["_local_ts"] + pd.Timedelta(hours=PHOENIX_UTC_OFFSET_HOURS)
        utc_ts = utc_ts.dt.tz_localize("UTC")
        renamed: dict[str, str] = {}
        for src, dst in (
            ("GHI", "ghi_wm2"),
            ("DHI", "dhi_wm2"),
            ("DNI", "dni_wm2"),
            ("Temperature", "nsrdb_temp_c"),
            ("Relative Humidity", "humidity_pct"),
        ):
            if src in df.columns:
                renamed[src] = dst
        keep = list(renamed.keys())
        subset = df[keep].rename(columns=renamed)
        subset.index = pd.DatetimeIndex(utc_ts.values, tz="UTC")
        frames.append(subset)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    hourly = combined.resample("1h").mean()
    mask = (hourly.index >= start) & (hourly.index < end)
    return hourly.loc[mask]


# ---------------------------------------------------------------------------
# Synthetic demand / weather fallbacks
# ---------------------------------------------------------------------------


def _synthetic_system_demand_mw(times: pd.DatetimeIndex, seed: int = 42) -> np.ndarray:
    """Realistic-ish AZPS system-demand stub.

    Model: ``mean + annual_sinusoid(summer peak ~Aug) + daily_sinusoid(4pm
    peak) + AR(1) noise``. Output in MW, strictly positive.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(len(times))
    # Day-of-year peak ~215 (early August).
    doy = times.dayofyear.values + times.hour.values / 24.0
    annual = SYNTH_ANNUAL_AMPL_MW * np.sin(2 * np.pi * (doy - 115.0) / 365.25)
    # Hour-of-day peak at 16h local → shift by 7h for UTC (23h UTC).
    hod_utc = times.hour.values
    daily = SYNTH_DAILY_AMPL_MW * np.sin(2 * np.pi * (hod_utc - 17.0) / 24.0)
    # AR(1) noise.
    noise = np.zeros(len(times), dtype=float)
    shock = rng.normal(0.0, SYNTH_NOISE_SIGMA_MW, size=len(times))
    phi = 0.7
    for i in range(1, len(times)):
        noise[i] = phi * noise[i - 1] + shock[i]
    _ = hours  # silence unused
    demand = SYNTH_MEAN_MW + annual + daily + noise
    return np.clip(demand, 500.0, None)  # never zero/negative


def _synthetic_weather(times: pd.DatetimeIndex, seed: int = 7) -> pd.DataFrame:
    """Synthetic weather series covering all 6 continuous exog columns."""
    rng = np.random.default_rng(seed)
    doy = times.dayofyear.values + times.hour.values / 24.0
    hod_utc = times.hour.values
    # Temperature: Phoenix seasonal (~12°C winter, ~35°C summer) + diurnal ±8°C.
    temp = (
        23.0
        + 12.0 * np.sin(2 * np.pi * (doy - 115.0) / 365.25)
        + 8.0 * np.sin(2 * np.pi * (hod_utc - 22.0) / 24.0)
        + rng.normal(0.0, 1.0, size=len(times))
    )
    # Dewpoint ~ temp - 15°C in arid summer, closer in winter.
    dewpoint = temp - 15.0 + 3.0 * np.sin(2 * np.pi * (doy - 200.0) / 365.25)
    # Humidity: anti-correlated with temp, 10-50%.
    humidity = np.clip(50.0 - 0.8 * (temp - 20.0) + rng.normal(0.0, 3.0, size=len(times)), 5.0, 95.0)
    # GHI: solar elevation proxy — zero at night, peak at 20-22 UTC (~1-3pm local).
    solar_elev = np.maximum(0.0, np.sin(np.pi * (hod_utc - 13.0) / 12.0))
    ghi = (
        900.0
        * solar_elev
        * (0.8 + 0.2 * np.sin(2 * np.pi * (doy - 172.0) / 365.25))
        + rng.normal(0.0, 20.0, size=len(times))
    )
    ghi = np.clip(ghi, 0.0, None)
    dni = np.clip(ghi * 1.1 + rng.normal(0.0, 30.0, size=len(times)), 0.0, None)
    wind = np.clip(2.5 + rng.normal(0.0, 1.5, size=len(times)), 0.0, None)
    return pd.DataFrame(
        {
            "temp_c": temp,
            "dewpoint_c": dewpoint,
            "humidity_pct": humidity,
            "ghi_wm2": ghi,
            "dni_wm2": dni,
            "wind_mps": wind,
        },
        index=times,
    )


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------


def _calendar_features(times: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclic sin/cos encoding of hour, day-of-week, month + weekend flag."""
    hour = times.hour.values.astype(float)
    dow = times.dayofweek.values.astype(float)
    month = times.month.values.astype(float)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "is_weekend": (dow >= 5).astype(float),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12.0),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12.0),
        },
        index=times,
    )


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------


def _zscore_columns(
    df: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    """Z-score each column; return (matrix, scalers).

    Guard against constant columns (std < 1e-12): preserve numerical
    stability by substituting std=1 so the output column is all zeros.
    Such columns won't pass a strict mean/std test and the caller is
    expected to ensure variance (synthetic fallback does).
    """
    scalers: dict[str, tuple[float, float]] = {}
    cols: list[np.ndarray] = []
    for name in df.columns:
        col = df[name].to_numpy(dtype=np.float64)
        mean = float(np.nanmean(col)) if np.isfinite(col).any() else 0.0
        std = float(np.nanstd(col)) if np.isfinite(col).any() else 0.0
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        scalers[name] = (mean, std)
        # Downcast once, then re-center / re-scale in float32 so the
        # stored matrix has an exact zero mean and near-unit std despite
        # float32 accumulation error on long windows (~8760 rows).
        z = ((col - mean) / std).astype(np.float32)
        z = z - np.float32(z.mean())
        col_std = float(z.std())
        if col_std > 1e-12:
            z = (z / np.float32(col_std)).astype(np.float32)
            z = z - np.float32(z.mean())
        cols.append(z)
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32), scalers
    return np.stack(cols, axis=1).astype(np.float32), scalers


# ---------------------------------------------------------------------------
# Topology load
# ---------------------------------------------------------------------------


def _load_topology_node_info(
    topology_root: Path | None,
) -> tuple[list[str], np.ndarray]:
    """Return ``(node_names, nominal_kw)`` from the IEEE 123 feeder.

    ``node_names`` matches the alphabetical ordering used by
    :func:`gridsense.topology.to_pyg_data`; ``nominal_kw`` is a ``[N]``
    float array of each bus's declared kW load (may be zero).
    """
    from gridsense import topology as _topo

    root = topology_root or _topo.DEFAULT_ROOT
    graph = _topo.load_ieee123(root)
    names = sorted(graph.nodes())
    nominal = np.array(
        [float(graph.nodes[n].get("kw_load", 0.0)) for n in names],
        dtype=float,
    )
    return names, nominal


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_hourly_features(
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    data_root: Path | None = None,
    topology_root: Path | None = None,
) -> FeatureBundle:
    """Construct a :class:`FeatureBundle` for the requested UTC window.

    Args:
        start: Inclusive ISO-ish start date (parsed by ``pd.Timestamp``).
        end: Exclusive end date. Default spans ~4 years of hourly data.
        data_root: Override for ``data/raw`` (tests point at ``tmp_path``
            to exercise the synthetic-fallback branch).
        topology_root: Override for ``data/ieee123``.

    Returns:
        A fully populated :class:`FeatureBundle`. When no raw inputs are
        found, ``meta["source"] == "synthetic"`` and the arrays contain
        deterministic sinusoidal stand-ins.
    """
    data_root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    topology_root = Path(topology_root) if topology_root is not None else DEFAULT_TOPOLOGY_ROOT

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    if end_ts <= start_ts:
        raise ValueError(f"end ({end}) must be strictly after start ({start})")
    times = pd.date_range(start=start_ts, end=end_ts, freq="h", inclusive="left", tz="UTC")
    if len(times) == 0:
        raise ValueError(f"empty time range: {start} .. {end}")

    # 1) Topology — must succeed; the graph is committed in-repo.
    node_names, nominal_kw = _load_topology_node_info(topology_root)
    total_nominal = float(nominal_kw.sum())
    if total_nominal <= 0:
        raise RuntimeError("feeder has zero total nominal kW — cannot disaggregate")
    share = nominal_kw / total_nominal  # [N]

    # 2) Raw inputs — each may be empty; track provenance.
    sources: dict[str, str] = {}
    noaa_df = _load_noaa_isd(data_root / "noaa", start_ts, end_ts)
    sources["noaa"] = "real" if not noaa_df.empty else "synthetic"
    eia_series = _load_eia930_demand(data_root / "eia930", start_ts, end_ts)
    sources["eia930"] = "real" if not eia_series.empty else "synthetic"
    nsrdb_df = _load_nsrdb_psm3(data_root / "nsrdb", start_ts, end_ts)
    sources["nsrdb"] = "real" if not nsrdb_df.empty else "synthetic"

    if sources["eia930"] == "synthetic":
        logger.info("[features] SKIP EIA-930 — using synthetic AZPS demand")
        demand_mw = _synthetic_system_demand_mw(times)
    else:
        demand_mw = eia_series.reindex(times).to_numpy(dtype=float)
        # Small gap-fill — forward then back.
        demand_mw = (
            pd.Series(demand_mw, index=times).ffill().bfill().fillna(SYNTH_MEAN_MW).to_numpy()
        )

    # Build a raw exogenous DataFrame aligned to ``times``.
    synth_weather = _synthetic_weather(times)
    if sources["noaa"] == "synthetic" and sources["nsrdb"] == "synthetic":
        # Full synthetic branch — use the sinusoidal stand-in wholesale.
        raw_weather = synth_weather.copy()
    else:
        raw_weather = synth_weather.copy()
        if sources["noaa"] == "real":
            aligned = noaa_df.reindex(times)
            for col in ("temp_c", "dewpoint_c", "wind_mps"):
                if col in aligned.columns:
                    filled = aligned[col].ffill().bfill()
                    if filled.notna().any():
                        raw_weather[col] = filled.fillna(raw_weather[col])
        else:
            # NOAA absent: fall back to constants (documented limitation).
            raw_weather["temp_c"] = FALLBACK_TEMP_C
            raw_weather["dewpoint_c"] = FALLBACK_DEW_C
            raw_weather["wind_mps"] = FALLBACK_WIND_MPS
        if sources["nsrdb"] == "real":
            aligned = nsrdb_df.reindex(times)
            for src, dst in (
                ("ghi_wm2", "ghi_wm2"),
                ("dni_wm2", "dni_wm2"),
                ("humidity_pct", "humidity_pct"),
            ):
                if src in aligned.columns:
                    filled = aligned[src].ffill().bfill()
                    if filled.notna().any():
                        raw_weather[dst] = filled.fillna(raw_weather[dst])
        else:
            raw_weather["ghi_wm2"] = FALLBACK_GHI_WM2
            raw_weather["dni_wm2"] = FALLBACK_DNI_WM2
            raw_weather["humidity_pct"] = FALLBACK_HUMIDITY_PCT

    calendar_df = _calendar_features(times)
    raw_exog_df = pd.concat([raw_weather, calendar_df], axis=1)[list(EXOG_FEATURE_NAMES)]

    # Final NaN scrub: forward-fill, back-fill, mean-fill.
    raw_exog_df = raw_exog_df.ffill().bfill()
    for col in raw_exog_df.columns:
        if raw_exog_df[col].isna().any():
            mean_val = raw_exog_df[col].mean()
            if not np.isfinite(mean_val):
                mean_val = 0.0
            raw_exog_df[col] = raw_exog_df[col].fillna(mean_val)

    # Z-score the exogenous features.
    X_exog, exog_scalers = _zscore_columns(raw_exog_df)
    scalers: dict[str, tuple[float, float]] = dict(exog_scalers)

    # Disaggregate per-bus load.
    y_kw = np.outer(demand_mw * 1000.0, share).astype(np.float32)  # MW→kW
    # Global y_kw scaler — single (mean, std) across all buses and times.
    y_mean = float(y_kw.mean())
    y_std = float(y_kw.std())
    if y_std < 1e-12:
        y_std = 1.0
    scalers["y_kw"] = (y_mean, y_std)

    # Per-bus features: [y_kw_standardised, nominal_kw_share].
    y_kw_std = (y_kw - y_mean) / y_std
    share_broadcast = np.broadcast_to(
        share.astype(np.float32), (len(times), len(node_names))
    ).copy()
    X_node = np.stack([y_kw_std, share_broadcast], axis=-1).astype(np.float32)
    # Standardise nominal share too — zero-mean unit-std across buses.
    share_mean = float(share.mean())
    share_std = float(share.std())
    if share_std < 1e-12:
        share_std = 1.0
    X_node[..., 1] = (X_node[..., 1] - share_mean) / share_std
    scalers["nominal_kw_share"] = (share_mean, share_std)

    if all(v == "real" for v in sources.values()):
        bundle_source = "real"
    elif all(v == "synthetic" for v in sources.values()):
        bundle_source = "synthetic"
    else:
        bundle_source = "mixed"

    meta: dict[str, Any] = {
        "source": bundle_source,
        "sources": sources,
        "start": str(start_ts),
        "end": str(end_ts),
        "n_buses": len(node_names),
        "total_nominal_kw": total_nominal,
    }
    return FeatureBundle(
        times=times,
        node_names=node_names,
        X_exog=X_exog,
        X_node=X_node,
        y_kw=y_kw,
        scalers=scalers,
        raw_exog_df=raw_exog_df,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


_ARRAYS_FILENAME = "arrays.npz"
_META_FILENAME = "meta.json"
_RAW_EXOG_FILENAME = "raw_exog.parquet"


def save_bundle(bundle: FeatureBundle, out_dir: Path) -> None:
    """Serialise a :class:`FeatureBundle` to ``out_dir`` (created if absent).

    On-disk layout:
        * ``arrays.npz`` — ``times_ns``, ``X_exog``, ``X_node``, ``y_kw``
        * ``raw_exog.parquet`` — pre-standardisation exog DataFrame
        * ``meta.json`` — ``node_names``, ``scalers``, ``meta``
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Store times as int64 nanoseconds for loss-free round-trip.
    times_ns = bundle.times.view("int64").astype(np.int64)
    np.savez_compressed(
        out_dir / _ARRAYS_FILENAME,
        times_ns=times_ns,
        X_exog=bundle.X_exog,
        X_node=bundle.X_node,
        y_kw=bundle.y_kw,
    )
    bundle.raw_exog_df.to_parquet(out_dir / _RAW_EXOG_FILENAME)

    # Meta: json with {list, scalers, meta}.
    payload = {
        "node_names": list(bundle.node_names),
        "scalers": {k: [float(v[0]), float(v[1])] for k, v in bundle.scalers.items()},
        "meta": _json_safe(bundle.meta),
    }
    (out_dir / _META_FILENAME).write_text(json.dumps(payload, indent=2))


def load_bundle(in_dir: Path) -> FeatureBundle:
    """Inverse of :func:`save_bundle`. Arrays round-trip exactly."""
    in_dir = Path(in_dir)
    arrays = np.load(in_dir / _ARRAYS_FILENAME)
    times_ns = arrays["times_ns"]
    times = pd.DatetimeIndex(times_ns.astype("datetime64[ns]"), tz="UTC")
    payload = json.loads((in_dir / _META_FILENAME).read_text())
    scalers = {k: (float(v[0]), float(v[1])) for k, v in payload["scalers"].items()}
    raw_exog_df = pd.read_parquet(in_dir / _RAW_EXOG_FILENAME)
    return FeatureBundle(
        times=times,
        node_names=list(payload["node_names"]),
        X_exog=np.asarray(arrays["X_exog"]),
        X_node=np.asarray(arrays["X_node"]),
        y_kw=np.asarray(arrays["y_kw"]),
        scalers=scalers,
        raw_exog_df=raw_exog_df,
        meta=payload.get("meta", {}),
    )


def _json_safe(value: Any) -> Any:
    """Recursively coerce non-JSON-serialisable scalars into JSON-safe form."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    return value


__all__ = [
    "EXOG_FEATURE_NAMES",
    "NODE_FEATURE_NAMES",
    "FeatureBundle",
    "build_hourly_features",
    "save_bundle",
    "load_bundle",
]
