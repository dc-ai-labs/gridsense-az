"""Feature pipeline — aligns weather, system-load, and calendar series into a
per-hour, per-bus tensor suitable for Graph WaveNet training on the IEEE 123
feeder.

The public entry point is :func:`build_hourly_features`, which returns a
:class:`FeatureBundle` containing:

* ``X_exog``  — z-scored global exogenous features, shape ``[T, F_exog]``.
  The feature order is fixed by :data:`EXOG_FEATURE_NAMES` and is currently::

      ("temp_c", "dewpoint_c", "wind_mps", "slp_hpa",
       "hour_sin", "hour_cos", "dow_sin", "dow_cos",
       "is_weekend", "month_sin", "month_cos")

  (11 features; NSRDB irradiance is intentionally absent — the public PSM3
  endpoint returned 404 during data collection; see ``BLOCKERS.md``.)

* ``X_node``  — z-scored per-bus node features, shape ``[T, N, F_node]``.
  Currently 1 feature: the standardised disaggregated bus load
  (``y_kw_standardised``). Retained as a 3-D tensor so downstream models can
  append additional per-bus channels (e.g. local weather, ResStock shares).

* ``y_kw``    — raw disaggregated bus load in kW, shape ``[T, N]``.
  ``y_kw.sum(axis=1)`` reproduces the (winsorised) AZPS system demand, in kW,
  to within floating-point noise.

* ``scalers`` — per-feature ``(mean, std)`` used for standardisation, plus an
  entry for ``"y_kw"`` covering the raw target.

Disaggregation strategy
-----------------------
Real feeder-level meter data is not public at per-bus granularity. We
disaggregate the EIA-930 AZPS hourly system demand onto the 85 loaded buses
of the IEEE 123 feeder, weighted by the nominal per-bus kW from the OpenDSS
model::

    y_kw[t, i] = D(t) * nominal_kw[i] / total_nominal_kw

This preserves the temporal signal (weather/calendar response) while giving
each bus a plausible load share for graph learning. The feeder is scaled up
to match the AZPS envelope (~2–8 GW vs. 3490 kW nominal); the ratio is a
multiplicative constant and does not distort learning dynamics.

Data sources
------------
* NOAA ISD hourly at KPHX (``data/raw/noaa/KPHX_{year}.csv``).
* EIA-930 AZPS hourly demand (``data/raw/eia930/azps_demand.parquet``) —
  winsorised to ``[500, 10000]`` MW to suppress obvious source outliers
  (max 101 GW, min −22 GW in raw stream; typical AZPS is 2–8 GW).

If the EIA-930 file is missing, demand is synthesised from a diurnal +
annual sinusoid plus white noise, and ``meta["source"] = "synthetic"``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gridsense.topology import DEFAULT_ROOT as TOPOLOGY_DEFAULT_ROOT
from gridsense.topology import load_ieee123

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Default repository root for raw data lookup; resolves to ``<repo>/data/raw``.
DEFAULT_DATA_ROOT: Path = Path(__file__).resolve().parents[2] / "data" / "raw"

#: Fixed ordering of the global exogenous feature vector ``X_exog``.
EXOG_FEATURE_NAMES: tuple[str, ...] = (
    "temp_c",
    "dewpoint_c",
    "wind_mps",
    "slp_hpa",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "month_sin",
    "month_cos",
)

#: Fixed ordering of per-bus node features ``X_node``.
NODE_FEATURE_NAMES: tuple[str, ...] = ("y_kw_standardised",)

#: EIA-930 winsorisation bounds for AZPS hourly demand (MW).
DEMAND_WINSOR_MIN_MW: float = 500.0
DEMAND_WINSOR_MAX_MW: float = 10_000.0

#: NOAA ISD quality-control codes that indicate a trusted reading.
_VALID_QC_CODES: frozenset[str] = frozenset({"1", "5", "7", "9"})

#: Max forward-fill gap (hours) before we fall back to column-mean imputation.
_MAX_FFILL_HOURS: int = 3


# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureBundle:
    """Aligned, z-scored feature tensors for the IEEE 123 feeder.

    Attributes:
        times: Hourly UTC ``DatetimeIndex`` of length ``T``.
        node_names: Ordered bus names of length ``N``; matches the ordering
            produced by :func:`gridsense.topology.to_pyg_data`.
        X_exog: Global exogenous features, ``[T, F_exog]``. Z-scored.
        X_node: Per-bus node features, ``[T, N, F_node]``. Z-scored.
        y_kw: Disaggregated bus load in kW, ``[T, N]``. Raw (not standardised).
        scalers: ``{feature_name: (mean, std)}`` used for z-scoring, plus
            a ``"y_kw"`` entry for the raw target.
        meta: Free-form metadata, minimally including ``source`` (``"real"``
            or ``"synthetic"``), ``n_missing_hours``, and ``n_imputed_rows``.
    """

    times: pd.DatetimeIndex
    node_names: list[str]
    X_exog: np.ndarray
    X_node: np.ndarray
    y_kw: np.ndarray
    scalers: dict[str, tuple[float, float]]
    meta: dict[str, Any]


# ---------------------------------------------------------------------------
# NOAA ISD parsing
# ---------------------------------------------------------------------------


def _parse_tmp(s: object) -> float:
    """Parse an ISD ``TMP`` / ``DEW`` field, e.g. ``"+0245,1"`` → ``24.5`` °C.

    Returns ``NaN`` when the QC code is not in :data:`_VALID_QC_CODES` or the
    value is the ``+9999`` sentinel.
    """
    if not isinstance(s, str) or "," not in s:
        return np.nan
    val, qc = s.split(",", 1)
    qc = qc.strip()
    val = val.strip()
    if qc not in _VALID_QC_CODES or val in {"+9999", "9999", "-9999"}:
        return np.nan
    try:
        return int(val) / 10.0
    except ValueError:
        return np.nan


def _parse_slp(s: object) -> float:
    """Parse an ISD ``SLP`` field, e.g. ``"10148,1"`` → ``1014.8`` hPa."""
    if not isinstance(s, str) or "," not in s:
        return np.nan
    val, qc = s.split(",", 1)
    qc = qc.strip()
    val = val.strip()
    if qc not in _VALID_QC_CODES or val in {"99999", "9999"}:
        return np.nan
    try:
        return int(val) / 10.0
    except ValueError:
        return np.nan


def _parse_wnd_speed(s: object) -> float:
    """Parse an ISD ``WND`` field → wind speed in m/s.

    Format: ``"direction,qc,type,speed_tenths_mps,qc"``. A ``9999`` speed or
    non-trusted QC code yields ``NaN``.
    """
    if not isinstance(s, str):
        return np.nan
    parts = s.split(",")
    if len(parts) < 5:
        return np.nan
    speed_str = parts[3].strip()
    speed_qc = parts[4].strip()
    if speed_qc not in _VALID_QC_CODES or speed_str in {"9999"}:
        return np.nan
    try:
        return int(speed_str) / 10.0
    except ValueError:
        return np.nan


def _synthetic_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Synthesize plausible Phoenix-area hourly weather on ``[start, end)``.

    Used only when no NOAA CSVs are present (e.g. CI / fresh clones). Values
    are tuned to roughly match KPHX climatology: annual temperature swing
    ~15 → 35 °C, daily swing ±8 °C, dewpoint tracking temperature with a
    seasonal offset, light wind, and near-standard sea-level pressure.
    """
    idx = pd.date_range(start=start, end=end, freq="h", inclusive="left", tz="UTC")
    hour_local = ((idx.hour.to_numpy().astype(float) - 7.0) % 24.0)
    doy = idx.dayofyear.to_numpy().astype(float)
    rng = np.random.default_rng(seed=17)
    annual_temp = 25.0 + 10.0 * np.sin(2 * np.pi * (doy - 200.0) / 365.0)
    diurnal_temp = 8.0 * np.sin(2 * np.pi * (hour_local - 15.0) / 24.0)
    temp_c = annual_temp + diurnal_temp + rng.normal(0.0, 1.0, size=len(idx))
    dewpoint_c = temp_c - 12.0 - 4.0 * np.sin(2 * np.pi * (doy - 200.0) / 365.0)
    wind_mps = np.abs(2.5 + rng.normal(0.0, 1.2, size=len(idx)))
    slp_hpa = 1013.0 + rng.normal(0.0, 2.5, size=len(idx))
    return pd.DataFrame(
        {
            "temp_c": temp_c,
            "dewpoint_c": dewpoint_c,
            "wind_mps": wind_mps,
            "slp_hpa": slp_hpa,
        },
        index=idx,
    )


def _load_noaa(data_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Read NOAA ISD CSVs and return an hourly DataFrame on ``[start, end)``.

    Variable-frequency NOAA readings are resampled to hourly means. Returned
    columns: ``temp_c``, ``dewpoint_c``, ``wind_mps``, ``slp_hpa``.
    The returned frame is reindexed to the requested hourly range, introducing
    NaN rows where KPHX has no coverage.
    """
    noaa_dir = data_root / "noaa"
    columns = ["temp_c", "dewpoint_c", "wind_mps", "slp_hpa"]
    if not noaa_dir.exists():
        logger.info("NOAA directory missing: %s — returning empty frame.", noaa_dir)
        idx = pd.date_range(start=start, end=end, freq="h", inclusive="left", tz="UTC")
        return pd.DataFrame(np.nan, index=idx, columns=columns)

    frames: list[pd.DataFrame] = []
    # Load every KPHX_*.csv whose year overlaps [start, end). Cheaper than
    # filtering inside a single global concat.
    years_needed = range(start.year, end.year + 1)
    for year in years_needed:
        path = noaa_dir / f"KPHX_{year}.csv"
        if not path.exists():
            continue
        logger.debug("Loading %s", path)
        raw = pd.read_csv(
            path,
            usecols=["DATE", "TMP", "DEW", "SLP", "WND"],
            low_memory=False,
        )
        if raw.empty:
            continue
        ts = pd.to_datetime(raw["DATE"], utc=True, errors="coerce")
        # Use ``.to_numpy()`` so that the Series' default integer index doesn't
        # clobber alignment with the datetime ``index=ts`` passed below.
        df = pd.DataFrame(
            {
                "temp_c": raw["TMP"].map(_parse_tmp).to_numpy(),
                "dewpoint_c": raw["DEW"].map(_parse_tmp).to_numpy(),
                "slp_hpa": raw["SLP"].map(_parse_slp).to_numpy(),
                "wind_mps": raw["WND"].map(_parse_wnd_speed).to_numpy(),
            },
            index=ts,
        )
        df = df[df.index.notna()]
        frames.append(df)

    if not frames:
        logger.info("No NOAA CSVs found under %s; returning all-NaN frame.", noaa_dir)
        idx = pd.date_range(start=start, end=end, freq="h", inclusive="left", tz="UTC")
        return pd.DataFrame(np.nan, index=idx, columns=columns)

    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    # Resample variable-frequency ISD observations to hourly means.
    hourly = full.resample("1h").mean()
    target_idx = pd.date_range(start=start, end=end, freq="h", inclusive="left", tz="UTC")
    hourly = hourly.reindex(target_idx)
    return hourly[columns]


# ---------------------------------------------------------------------------
# EIA-930 loading + synthetic fallback
# ---------------------------------------------------------------------------


def _load_eia930(data_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    """Load EIA-930 AZPS hourly demand (MW) on ``[start, end)``.

    Returns ``None`` if the parquet is missing — caller falls back to the
    synthetic profile.
    """
    path = data_root / "eia930" / "azps_demand.parquet"
    if not path.exists():
        logger.info("EIA-930 parquet missing at %s — will synthesise demand.", path)
        return None
    df = pd.read_parquet(path)
    # Period is an ISO string like "2019-01-01T00"; EIA v2 docs treat it as UTC.
    idx = pd.to_datetime(df["period"], utc=True, errors="coerce")
    # Value is string-typed in this raw feed — must coerce before winsorise.
    vals = pd.to_numeric(df["value"], errors="coerce")
    ser = pd.Series(vals.to_numpy(), index=idx, name="system_demand_mw")
    ser = ser[ser.index.notna()].sort_index()
    ser = ser[~ser.index.duplicated(keep="first")]
    target_idx = pd.date_range(start=start, end=end, freq="h", inclusive="left", tz="UTC")
    ser = ser.reindex(target_idx)
    return ser


def _synthetic_demand(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Synthesise AZPS-like hourly system demand (MW) on ``[start, end)``.

    * Diurnal sinusoid peaking around 18:00 local Arizona time (= 01:00 UTC,
      since AZ is UTC-7 year-round and does not observe DST).
    * Annual sinusoid peaking around day-of-year 172 (late June).
    * White noise N(0, 100) MW for realism.
    """
    idx = pd.date_range(start=start, end=end, freq="h", inclusive="left", tz="UTC")
    hour_utc = idx.hour.to_numpy().astype(float)
    # Local hour in AZ standard time (UTC-7).
    hour_local = (hour_utc - 7.0) % 24.0
    diurnal = 4000.0 + 1500.0 * np.sin(2 * np.pi * (hour_local - 18.0) / 24.0)
    doy = idx.dayofyear.to_numpy().astype(float)
    annual = 1000.0 * np.sin(2 * np.pi * (doy - 172.0) / 365.0)
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0.0, 100.0, size=len(idx))
    values = diurnal + annual + noise
    return pd.Series(values, index=idx, name="system_demand_mw")


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------


def _calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Build cyclic hour/dow/month encodings + weekend flag."""
    hour = idx.hour.to_numpy().astype(float)
    dow = idx.dayofweek.to_numpy().astype(float)
    month = idx.month.to_numpy().astype(float)
    df = pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "is_weekend": (dow >= 5).astype(float),
            "month_sin": np.sin(2 * np.pi * (month - 1.0) / 12.0),
            "month_cos": np.cos(2 * np.pi * (month - 1.0) / 12.0),
        },
        index=idx,
    )
    return df


# ---------------------------------------------------------------------------
# Imputation + standardisation helpers
# ---------------------------------------------------------------------------


def _impute_numeric(df: pd.DataFrame, max_ffill: int = _MAX_FFILL_HOURS) -> tuple[pd.DataFrame, int]:
    """Forward-fill short NaN gaps, mean-fill the rest.

    Returns (filled_df, total_rows_imputed). A row counts as imputed if any
    of its columns had to be filled.
    """
    before_any_nan = df.isna().any(axis=1)
    filled = df.ffill(limit=max_ffill).bfill(limit=max_ffill)
    for col in filled.columns:
        if filled[col].isna().any():
            mean_val = filled[col].mean()
            if not np.isfinite(mean_val):
                mean_val = 0.0
            filled[col] = filled[col].fillna(mean_val)
    total_imputed = int(before_any_nan.sum())
    return filled, total_imputed


def _zscore(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Z-score a 1-D array; guards against zero variance."""
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std < 1e-12:
        return arr - mean, mean, 1.0
    return (arr - mean) / std, mean, std


# ---------------------------------------------------------------------------
# Disaggregation
# ---------------------------------------------------------------------------


def _load_topology_shares(topology_root: Path | None) -> tuple[list[str], np.ndarray]:
    """Return ``(node_names, nominal_kw)`` ordered like ``to_pyg_data``.

    ``nominal_kw`` is raw per-bus kW; caller divides by its sum for shares.
    """
    if topology_root is None:
        graph = load_ieee123()
    else:
        graph = load_ieee123(root=topology_root)
    node_names = sorted(graph.nodes())
    nominal = np.array(
        [float(graph.nodes[n].get("kw_load", 0.0)) for n in node_names], dtype=np.float64
    )
    return node_names, nominal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_hourly_features(
    start: str = "2020-01-01",
    end: str = "2024-01-01",
    data_root: Path | None = None,
    topology_root: Path | None = None,
) -> FeatureBundle:
    """Build the full aligned feature bundle for training.

    Args:
        start: Inclusive start timestamp (UTC, ISO-parseable).
        end: Exclusive end timestamp (UTC, ISO-parseable).
        data_root: Override for ``data/raw``. Defaults to the repo layout.
        topology_root: Override for the IEEE 123 DSS bundle.

    Returns:
        A fully populated :class:`FeatureBundle`.
    """
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    if end_ts <= start_ts:
        raise ValueError(f"end ({end_ts}) must be strictly after start ({start_ts})")

    root = Path(data_root).resolve() if data_root is not None else DEFAULT_DATA_ROOT
    topo_root = (
        Path(topology_root).resolve() if topology_root is not None else TOPOLOGY_DEFAULT_ROOT
    )

    idx = pd.date_range(start=start_ts, end=end_ts, freq="h", inclusive="left", tz="UTC")

    # --- weather ---------------------------------------------------------
    noaa_df = _load_noaa(root, start_ts, end_ts)
    noaa_is_empty = bool(noaa_df.isna().all(axis=None))
    if noaa_is_empty:
        noaa_df = _synthetic_weather(start_ts, end_ts)

    # --- demand ----------------------------------------------------------
    demand_mw = _load_eia930(root, start_ts, end_ts)
    source = "real"
    if demand_mw is None or demand_mw.isna().all():
        demand_mw = _synthetic_demand(start_ts, end_ts)
        source = "synthetic"
    if noaa_is_empty:
        # If weather had to be synthesised too, mark the whole bundle as synthetic.
        source = "synthetic"
    n_missing_hours = int(demand_mw.isna().sum())
    # Winsorise: real EIA-930 has occasional source-side glitches (101 GW).
    winsorised = demand_mw.clip(lower=DEMAND_WINSOR_MIN_MW, upper=DEMAND_WINSOR_MAX_MW)
    # Winsorise preserves NaNs; median-fill any remaining gaps before use.
    if winsorised.isna().any():
        median_mw = float(np.nanmedian(winsorised.to_numpy()))
        if not np.isfinite(median_mw):
            median_mw = (DEMAND_WINSOR_MIN_MW + DEMAND_WINSOR_MAX_MW) / 2
        winsorised = winsorised.fillna(median_mw)

    # --- calendar --------------------------------------------------------
    cal_df = _calendar_features(idx)

    # --- join + impute ---------------------------------------------------
    exog_raw = pd.concat([noaa_df, cal_df], axis=1)
    # Order columns deterministically per the module contract.
    exog_raw = exog_raw[list(EXOG_FEATURE_NAMES)]
    exog_filled, n_imputed_rows = _impute_numeric(exog_raw)

    # --- topology shares -------------------------------------------------
    node_names, nominal_kw = _load_topology_shares(topo_root)
    total_nominal = float(nominal_kw.sum())
    if total_nominal <= 0:
        raise ValueError("IEEE 123 topology reports total nominal load == 0 — cannot disaggregate.")
    shares = nominal_kw / total_nominal  # [N]

    # --- disaggregate ----------------------------------------------------
    demand_kw = winsorised.to_numpy() * 1000.0  # MW → kW, shape [T]
    y_kw = demand_kw[:, None] * shares[None, :]  # [T, N]
    # Zero-load buses land on zero (shares == 0).

    # --- z-score exogenous features --------------------------------------
    X_exog = np.empty((len(idx), len(EXOG_FEATURE_NAMES)), dtype=np.float64)
    scalers: dict[str, tuple[float, float]] = {}
    for j, name in enumerate(EXOG_FEATURE_NAMES):
        col = exog_filled[name].to_numpy(dtype=np.float64)
        zs, mean, std = _zscore(col)
        X_exog[:, j] = zs
        scalers[name] = (mean, std)

    # --- build node features ---------------------------------------------
    y_flat = y_kw.reshape(-1)
    _, y_mean, y_std = _zscore(y_flat)
    if y_std < 1e-12:
        X_node_std = y_kw - y_mean
    else:
        X_node_std = (y_kw - y_mean) / y_std
    X_node = X_node_std[:, :, None].astype(np.float64)  # [T, N, 1]
    scalers["y_kw"] = (y_mean, y_std)

    meta: dict[str, Any] = {
        "source": source,
        "n_missing_hours": n_missing_hours,
        "n_imputed_rows": n_imputed_rows,
        "start": str(start_ts),
        "end": str(end_ts),
        "n_nodes": len(node_names),
        "n_times": len(idx),
        "exog_feature_names": list(EXOG_FEATURE_NAMES),
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "topology_total_nominal_kw": total_nominal,
        "demand_winsor_mw": [DEMAND_WINSOR_MIN_MW, DEMAND_WINSOR_MAX_MW],
    }

    logger.info(
        "build_hourly_features: T=%d N=%d F_exog=%d F_node=%d source=%s imputed=%d",
        len(idx),
        len(node_names),
        X_exog.shape[1],
        X_node.shape[2],
        source,
        n_imputed_rows,
    )

    return FeatureBundle(
        times=idx,
        node_names=list(node_names),
        X_exog=X_exog,
        X_node=X_node,
        y_kw=y_kw,
        scalers=scalers,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


_BUNDLE_NPZ = "arrays.npz"
_BUNDLE_SIDECAR = "bundle.json"


def save_bundle(bundle: FeatureBundle, out_dir: Path) -> None:
    """Persist a :class:`FeatureBundle` to ``out_dir``.

    Layout:
        ``arrays.npz`` — X_exog, X_node, y_kw, times (as int64 ns).
        ``bundle.json`` — node_names, scalers, meta.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / _BUNDLE_NPZ,
        X_exog=bundle.X_exog,
        X_node=bundle.X_node,
        y_kw=bundle.y_kw,
        times_ns=bundle.times.asi8,
    )
    sidecar = {
        "node_names": list(bundle.node_names),
        "scalers": {k: [float(v[0]), float(v[1])] for k, v in bundle.scalers.items()},
        "meta": _json_safe(bundle.meta),
    }
    (out_dir / _BUNDLE_SIDECAR).write_text(json.dumps(sidecar, indent=2, sort_keys=True))


def load_bundle(in_dir: Path) -> FeatureBundle:
    """Inverse of :func:`save_bundle`. Arrays round-trip exactly."""
    in_dir = Path(in_dir)
    with np.load(in_dir / _BUNDLE_NPZ) as npz:
        X_exog = npz["X_exog"]
        X_node = npz["X_node"]
        y_kw = npz["y_kw"]
        times_ns = npz["times_ns"]
    times = pd.DatetimeIndex(pd.to_datetime(times_ns, utc=True))
    sidecar = json.loads((in_dir / _BUNDLE_SIDECAR).read_text())
    scalers = {k: (float(v[0]), float(v[1])) for k, v in sidecar["scalers"].items()}
    return FeatureBundle(
        times=times,
        node_names=list(sidecar["node_names"]),
        X_exog=X_exog,
        X_node=X_node,
        y_kw=y_kw,
        scalers=scalers,
        meta=dict(sidecar["meta"]),
    )


def _json_safe(obj: Any) -> Any:
    """Recursively coerce numpy/pandas scalars to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


# ---------------------------------------------------------------------------
# CLI entry point (optional)
# ---------------------------------------------------------------------------


def _main() -> None:
    """Convenience CLI: build the default 2020–2024 bundle and print a summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    bundle = build_hourly_features(start="2020-01-01", end="2024-01-01")
    print("times:      ", bundle.times[0], "→", bundle.times[-1], "n=", len(bundle.times))
    print("node_names: ", len(bundle.node_names), "e.g.", bundle.node_names[:5])
    print("X_exog:     ", bundle.X_exog.shape, bundle.X_exog.dtype)
    print("X_node:     ", bundle.X_node.shape, bundle.X_node.dtype)
    print("y_kw:       ", bundle.y_kw.shape, bundle.y_kw.dtype)
    print("scalers:    ", {k: (round(v[0], 3), round(v[1], 3)) for k, v in bundle.scalers.items()})
    print("meta:       ", bundle.meta)


if __name__ == "__main__":
    _main()
