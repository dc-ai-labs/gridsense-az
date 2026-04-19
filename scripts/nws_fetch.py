"""NWS hourly-forecast fetcher for Phoenix.

Thin client around ``api.weather.gov`` that returns a tidy hourly DataFrame with
columns ``temp_c, dewpoint_c, wind_mps, slp_hpa`` on a UTC ``DatetimeIndex``.
Handles the two-hop lookup (points → gridpoints), the retry/backoff dance, and
the ISO-8601 interval expansion that NWS uses to compress flat time series.

This is intentionally a one-function module so ``scripts/precompute_forecasts``
can keep its imports minimal and unit tests can monkey-patch ``requests.get``
without pulling the full forecasting pipeline into the fixture.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

__all__ = ["fetch_phoenix_hourly", "PHOENIX_LAT", "PHOENIX_LON", "NWSFetchError"]

logger = logging.getLogger(__name__)

PHOENIX_LAT: float = 33.4484
PHOENIX_LON: float = -112.0740

_POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"
_DEFAULT_USER_AGENT = "GridSense-AZ/0.1 (dchanda1@asu.edu)"

# ISO-8601 duration like "PT1H", "PT2H", "PT30M". NWS only uses hours+minutes
# in practice, but we parse defensively.
_ISO_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)

# Climatological fallback for Phoenix sea-level pressure when the NWS grid
# returns no pressure samples (which is the common case on this endpoint —
# the PSR office does not publish a pressure layer).
_PHOENIX_FALLBACK_SLP_HPA = 1013.0


class NWSFetchError(RuntimeError):
    """Raised when NWS is unreachable or returns a schema we don't understand."""


def _parse_iso_duration(dur: str) -> pd.Timedelta:
    """Parse a subset of ISO-8601 durations (days + hours + minutes + seconds)."""
    match = _ISO_DURATION_RE.match(dur.strip())
    if not match:
        raise NWSFetchError(f"unrecognised ISO-8601 duration: {dur!r}")
    parts = {k: int(v) if v else 0 for k, v in match.groupdict().items()}
    return pd.Timedelta(
        days=parts["days"],
        hours=parts["hours"],
        minutes=parts["minutes"],
        seconds=parts["seconds"],
    )


def _get_with_retries(
    url: str,
    user_agent: str,
    *,
    max_attempts: int = 3,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """HTTP GET with exponential backoff. Raises :class:`NWSFetchError` on failure.

    Four consecutive failures (= one initial try plus three retries) escalate to
    a raised ``NWSFetchError`` — the caller is expected to fall back to the
    replay path.
    """
    headers = {"User-Agent": user_agent, "Accept": "application/geo+json"}
    last_exc: Exception | None = None
    # Allow one primary attempt + (max_attempts) retries = 4 attempts total.
    total_attempts = max_attempts + 1
    for attempt in range(1, total_attempts + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            # 429/5xx are transient — retry.
            if resp.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(
                    f"NWS transient {resp.status_code} for {url}", response=resp
                )
            # Everything else is terminal (403 / 404 / 400).
            raise NWSFetchError(
                f"NWS {resp.status_code} at {url}: {resp.text[:200]!r}"
            )
        except (requests.RequestException, NWSFetchError) as exc:
            last_exc = exc
            if isinstance(exc, NWSFetchError) and "transient" not in str(exc):
                # Non-transient HTTP error — don't retry.
                raise
            if attempt >= total_attempts:
                break
            sleep_s = 2.0 ** (attempt - 1)
            logger.warning(
                "NWS GET %s failed (attempt %d/%d): %s — sleeping %.1fs",
                url,
                attempt,
                total_attempts,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)
    raise NWSFetchError(
        f"NWS GET {url} failed after {total_attempts} attempts: {last_exc!r}"
    )


def _intervals_to_hourly(
    intervals: list[dict[str, Any]],
    converter,
    target_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Flatten ``[{validTime, value}, ...]`` into an hourly array aligned to ``target_index``.

    ``validTime`` is ISO-8601 ``<start>/<duration>``. Each interval broadcasts its
    value across every whole hour it covers. Missing hours stay NaN.
    """
    ser = pd.Series(np.nan, index=target_index, dtype=np.float64)
    for it in intervals:
        vt = it.get("validTime")
        v = it.get("value")
        if vt is None or v is None:
            continue
        if "/" not in vt:
            continue
        start_str, dur_str = vt.split("/", 1)
        try:
            start = pd.Timestamp(start_str).tz_convert("UTC")
        except (ValueError, TypeError):
            try:
                start = pd.Timestamp(start_str, tz="UTC")
            except Exception:
                continue
        try:
            dur = _parse_iso_duration(dur_str)
        except NWSFetchError:
            continue
        end = start + dur
        # NWS start times are already on-the-hour; clip to the target index.
        hour_start = start.ceil("h") if start != start.floor("h") else start.floor("h")
        hour_end = end.floor("h")
        # Inclusive start, exclusive end — each hour that begins before the
        # interval ends should receive the value.
        for ts in pd.date_range(start=hour_start, end=hour_end, freq="h", tz="UTC"):
            if ts >= end:
                break
            if ts in ser.index:
                try:
                    ser.at[ts] = float(converter(v))
                except (TypeError, ValueError):
                    continue
    return ser.to_numpy()


def fetch_phoenix_hourly(
    hours: int = 48,
    user_agent: str = _DEFAULT_USER_AGENT,
    *,
    lat: float = PHOENIX_LAT,
    lon: float = PHOENIX_LON,
) -> pd.DataFrame:
    """Pull Phoenix NWS gridpoint forecast and return a tidy hourly DataFrame.

    Args:
        hours: Forecast horizon in whole hours, starting from the top of the
            current UTC hour. Default 48 h gives the caller two days of slack
            to pick a tomorrow window in any US-based local timezone.
        user_agent: Contact string NWS requires on every request. Without a
            UA the endpoint returns 403.
        lat: Latitude for the ``/points`` lookup.
        lon: Longitude for the ``/points`` lookup.

    Returns:
        DataFrame with UTC ``DatetimeIndex`` of length ``hours`` and columns
        ``temp_c, dewpoint_c, wind_mps, slp_hpa``. Pressure is imputed to the
        Phoenix climatological mean (1013 hPa) when NWS returns no pressure
        samples, which is the norm on the PSR grid.

    Raises:
        NWSFetchError: Four consecutive transport failures or a schema drift
            (missing ``forecastGridData`` or missing required weather layers).
    """
    if hours <= 0:
        raise ValueError(f"hours must be positive, got {hours}")

    points_url = _POINTS_URL.format(lat=lat, lon=lon)
    points_payload = _get_with_retries(points_url, user_agent)

    grid_url = (
        (points_payload or {}).get("properties", {}).get("forecastGridData")
    )
    if not grid_url:
        raise NWSFetchError(
            f"points payload missing forecastGridData: keys="
            f"{list((points_payload or {}).get('properties', {}).keys())[:10]}"
        )

    grid_payload = _get_with_retries(grid_url, user_agent)
    props = (grid_payload or {}).get("properties", {})
    if not isinstance(props, dict):
        raise NWSFetchError(f"gridpoints payload missing properties (got {type(props).__name__})")

    # Target hourly index — start from the next whole UTC hour so that every
    # intervals bucket is populated on exactly one side.
    now = pd.Timestamp.utcnow().tz_convert("UTC") if pd.Timestamp.utcnow().tzinfo else pd.Timestamp.utcnow().tz_localize("UTC")
    start_idx = now.ceil("h") - pd.Timedelta(hours=1)  # include current hour
    target_index = pd.date_range(start=start_idx, periods=hours, freq="h", tz="UTC")

    def _layer_or_raise(name: str) -> list[dict[str, Any]]:
        v = props.get(name)
        if not isinstance(v, dict):
            raise NWSFetchError(f"gridpoints payload missing '{name}' layer")
        vals = v.get("values")
        if not isinstance(vals, list):
            raise NWSFetchError(f"gridpoints '{name}' has no values array")
        return vals

    # temperature / dewpoint: already °C.
    temp_vals = _layer_or_raise("temperature")
    dew_vals = _layer_or_raise("dewpoint")

    # windSpeed: km/h → m/s.
    wind_vals = _layer_or_raise("windSpeed")

    # pressure: Pa → hPa. Optional — climatological fallback when absent.
    pressure_layer = props.get("pressure", {}) if isinstance(props.get("pressure"), dict) else {}
    pressure_vals = pressure_layer.get("values", []) if isinstance(pressure_layer, dict) else []

    temp_c = _intervals_to_hourly(temp_vals, lambda v: float(v), target_index)
    dewpoint_c = _intervals_to_hourly(dew_vals, lambda v: float(v), target_index)
    wind_mps = _intervals_to_hourly(wind_vals, lambda v: float(v) / 3.6, target_index)
    if pressure_vals:
        slp_hpa = _intervals_to_hourly(
            pressure_vals, lambda v: float(v) / 100.0, target_index
        )
    else:
        # Phoenix is a desert plateau — the real SLP varies by ~5 hPa day to
        # day. Constant-fill is fine for the model's SLP z-scored feature.
        slp_hpa = np.full(len(target_index), _PHOENIX_FALLBACK_SLP_HPA, dtype=np.float64)

    df = pd.DataFrame(
        {
            "temp_c": temp_c,
            "dewpoint_c": dewpoint_c,
            "wind_mps": wind_mps,
            "slp_hpa": slp_hpa,
        },
        index=target_index,
    )
    # Forward-fill short gaps then back-fill any remaining NaN; this matches
    # the tolerance already applied by ``features._impute_numeric``.
    df = df.ffill(limit=6).bfill(limit=6)
    # Last-resort: column-mean, falling back to the climatological defaults.
    fallbacks = {
        "temp_c": 30.0,
        "dewpoint_c": 5.0,
        "wind_mps": 3.0,
        "slp_hpa": _PHOENIX_FALLBACK_SLP_HPA,
    }
    for col, fill in fallbacks.items():
        if df[col].isna().any():
            col_mean = df[col].mean()
            df[col] = df[col].fillna(col_mean if np.isfinite(col_mean) else fill)

    if df.isna().any().any():
        raise NWSFetchError("residual NaNs in NWS hourly frame after imputation")

    return df
