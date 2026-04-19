"""GridSense-AZ inference helper — load a trained GWNet checkpoint and produce
per-bus p10/p50/p90 forecasts.

The trained checkpoint persisted by ``scripts/train.py`` is a raw
:class:`torch.nn.Module.state_dict` (an ``OrderedDict`` of tensors, with no
embedded config). The matching :class:`GWNetConfig` and the target
standardisation scaler live next to it, in ``data/models/metrics.json``
(under ``"config"`` and ``"data.y_scaler"`` respectively). We reconstruct the
model from that sidecar, load the weights, set ``eval()``, and expose two
thin helpers:

* :func:`load_predictor` — load the checkpoint into a CPU ``GWNet`` ready
  for inference. Returns the (model, y_scaler, config-dict) triple bundled
  in a ``LoadedPredictor`` wrapper so callers never have to re-read the
  sidecar on subsequent forecasts.
* :func:`forecast_from_bundle` — slice the last ``t_in`` hours off a
  :class:`FeatureBundle`, run a single forward pass, denormalise back to kW,
  and return a :class:`Forecast` dataclass with ``p10 / p50 / p90`` arrays
  shaped ``[T_out, N_nodes]`` together with timestamp + bus-name
  metadata.

The sidecar lookup is resilient: if ``metrics.json`` is missing we fall
back to :class:`GWNetConfig` defaults (which match the v0 training run) and
to the bundle's own ``y_kw`` scaler; the dashboard can still produce a
forecast in that case.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gridsense.features import FeatureBundle
from gridsense.model import GWNet, GWNetConfig

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CKPT_PATH",
    "DEFAULT_METRICS_PATH",
    "Forecast",
    "LoadedPredictor",
    "load_predictor",
    "forecast_from_bundle",
]


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Default location of the trained Graph-WaveNet checkpoint.
DEFAULT_CKPT_PATH: Path = (
    Path(__file__).resolve().parents[2] / "data" / "models" / "gwnet_v0.pt"
)

#: Default location of the training metrics sidecar carrying ``config`` +
#: ``data.y_scaler`` used at training time. The checkpoint does NOT embed
#: these, so we read them from here at load time.
DEFAULT_METRICS_PATH: Path = (
    Path(__file__).resolve().parents[2] / "data" / "models" / "metrics.json"
)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Forecast:
    """Per-bus quantile forecast over a fixed horizon.

    Attributes:
        p10: Lower-quantile forecast in kW, shape ``[T_out, N_nodes]``.
        p50: Median forecast in kW, shape ``[T_out, N_nodes]``.
        p90: Upper-quantile forecast in kW, shape ``[T_out, N_nodes]``.
        timestamps: Forecast timestamps, one per horizon step (length ``T_out``).
        bus_names: Ordered bus names, one per column of p10/p50/p90.
    """

    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray
    timestamps: list[datetime]
    bus_names: list[str]


@dataclass(frozen=True)
class LoadedPredictor:
    """Bundle of everything :func:`forecast_from_bundle` needs.

    Streamlit callers can hold this behind an ``@st.cache_resource`` so the
    checkpoint is loaded once per process, not once per rerun.
    """

    model: GWNet
    config: GWNetConfig
    #: Training-time target scaler ``(mean, std)`` in kW. Used to denormalise
    #: the model's output back from z-scored to physical units.
    y_scaler: tuple[float, float]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _read_metrics_sidecar(metrics_path: Path) -> dict[str, Any]:
    """Load the training-metrics sidecar (config + y_scaler).

    Returns an empty dict when the sidecar is missing so the caller can
    fall back to defaults.
    """
    try:
        return json.loads(Path(metrics_path).read_text())
    except FileNotFoundError:
        logger.warning("metrics sidecar not found at %s — using defaults", metrics_path)
        return {}
    except json.JSONDecodeError as exc:
        logger.warning("metrics sidecar invalid JSON (%s) — using defaults", exc)
        return {}


def _config_from_metrics(metrics: dict[str, Any]) -> GWNetConfig:
    """Rebuild a :class:`GWNetConfig` from the ``config`` block of metrics.json."""
    raw = metrics.get("config", {}) or {}
    # JSON stores tuples as lists — GWNetConfig's ``quantiles`` must be a tuple.
    kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items()}
    return GWNetConfig(**kwargs) if kwargs else GWNetConfig()


def _y_scaler_from_metrics(metrics: dict[str, Any]) -> tuple[float, float] | None:
    """Return ``(mean, std)`` from ``metrics["data"]["y_scaler"]`` if present."""
    data = metrics.get("data") or {}
    scaler = data.get("y_scaler")
    if not scaler or len(scaler) != 2:
        return None
    return float(scaler[0]), float(scaler[1])


def load_predictor(
    ckpt_path: Path = DEFAULT_CKPT_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    device: str = "cpu",
) -> LoadedPredictor:
    """Load a trained GWNet checkpoint into a ready-for-inference predictor.

    Args:
        ckpt_path: Path to the ``state_dict`` saved by ``scripts/train.py``.
        metrics_path: Path to the training metrics sidecar providing the
            exact :class:`GWNetConfig` and the target ``y_scaler`` used
            during training. Falls back to :class:`GWNetConfig` defaults
            and a ``(0.0, 1.0)`` identity scaler when missing.
        device: PyTorch device string. Forecast payloads are small, so
            ``"cpu"`` is the sensible default even on GPU boxes.

    Returns:
        A :class:`LoadedPredictor` bundling the model (in ``eval()`` mode),
        the reconstructed :class:`GWNetConfig`, and the target scaler.

    Raises:
        FileNotFoundError: If ``ckpt_path`` does not exist.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    metrics = _read_metrics_sidecar(Path(metrics_path))
    config = _config_from_metrics(metrics)
    y_scaler = _y_scaler_from_metrics(metrics) or (0.0, 1.0)

    model = GWNet(config)
    state = torch.load(ckpt_path, map_location=device)
    # ``torch.save(model.state_dict(), ...)`` in ``scripts/train.py`` writes
    # the raw ``OrderedDict`` — no ``state_dict`` key to unwrap.
    if isinstance(state, dict) and "state_dict" in state and "start_conv.weight" not in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    logger.info(
        "load_predictor: ckpt=%s cfg.n_nodes=%d cfg.t_in=%d cfg.t_out=%d y_scaler=%s",
        ckpt_path,
        config.n_nodes,
        config.t_in,
        config.t_out,
        y_scaler,
    )
    return LoadedPredictor(model=model, config=config, y_scaler=y_scaler)


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------


def _standardise_y(y_kw: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    """Apply the training-time z-score to raw kW so it matches ``X_node``."""
    if y_std < 1e-12:
        return y_kw - y_mean
    return (y_kw - y_mean) / y_std


def _denormalise_y(y_std_arr: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    """Inverse of :func:`_standardise_y`."""
    if y_std < 1e-12:
        return y_std_arr + y_mean
    return y_std_arr * y_std + y_mean


def _resolve_y_scaler(
    predictor: LoadedPredictor, bundle: FeatureBundle
) -> tuple[float, float]:
    """Pick the scaler to invert at inference.

    Order of preference:

    1. The scaler baked into the metrics sidecar (exact training scaler).
    2. The scaler inside the feature bundle (``scalers["y_kw"]``) — a sane
       fallback when the sidecar is absent.
    3. An identity ``(0.0, 1.0)`` as last resort.
    """
    if predictor.y_scaler != (0.0, 1.0):
        return predictor.y_scaler
    scaler = bundle.scalers.get("y_kw")
    if scaler is not None:
        return float(scaler[0]), float(scaler[1])
    return 0.0, 1.0


def _future_timestamps(last_ts: pd.Timestamp, horizon: int) -> list[datetime]:
    """Build the ``horizon`` hourly timestamps starting one hour after ``last_ts``."""
    future = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1), periods=horizon, freq="h"
    )
    # ``pd.Timestamp.to_pydatetime()`` returns a ``datetime`` that preserves
    # tz-awareness, which is what the dashboard wants for plotly rendering.
    return [ts.to_pydatetime() for ts in future]


def forecast_from_bundle(
    predictor: LoadedPredictor | GWNet,
    bundle: FeatureBundle,
    t_in: int = 24,
    t_out: int = 6,
    *,
    y_scaler: tuple[float, float] | None = None,
) -> Forecast:
    """Run one forward pass and denormalise to per-bus kW.

    The window taken is the *trailing* ``t_in`` hours of ``bundle`` (i.e. the
    most-recent observations), under the implicit assumption that the bundle
    ends at "now". The forecast's timestamps are then
    ``bundle.times[-1] + [1h, 2h, ..., t_out h]``.

    Args:
        predictor: Either a fully-loaded :class:`LoadedPredictor`, or a bare
            :class:`GWNet` (in which case ``y_scaler`` must be passed
            explicitly or embedded in the bundle).
        bundle: Aligned feature bundle from
            :func:`gridsense.features.build_hourly_features`.
        t_in: Input-history length in hours. Must match the model's
            training config.
        t_out: Output-horizon length in hours. Must match the model's
            training config.
        y_scaler: Optional explicit ``(mean, std)`` to use for denormalising
            predictions back to kW. If ``None``, taken from ``predictor``
            when it is a :class:`LoadedPredictor`, else from
            ``bundle.scalers["y_kw"]``.

    Returns:
        :class:`Forecast` with p10/p50/p90 arrays shaped ``[t_out, N]``,
        ``t_out`` forward timestamps, and the bundle's bus-name ordering.

    Raises:
        ValueError: If the bundle is shorter than ``t_in`` hours, or the
            node count does not match the model's configured ``n_nodes``.
    """
    # ------------------------------------------------------------------
    # Predictor / scaler resolution
    # ------------------------------------------------------------------
    if isinstance(predictor, LoadedPredictor):
        model = predictor.model
        if y_scaler is None:
            y_scaler = _resolve_y_scaler(predictor, bundle)
    else:
        model = predictor
        if y_scaler is None:
            scaler = bundle.scalers.get("y_kw")
            y_scaler = (float(scaler[0]), float(scaler[1])) if scaler else (0.0, 1.0)

    y_mean, y_std = y_scaler

    # ------------------------------------------------------------------
    # Shape guards
    # ------------------------------------------------------------------
    total_steps = bundle.X_exog.shape[0]
    if total_steps < t_in:
        raise ValueError(
            f"bundle only has {total_steps} hours; need at least t_in={t_in}."
        )
    n_nodes_bundle = bundle.X_node.shape[1]
    if n_nodes_bundle != model.config.n_nodes:
        raise ValueError(
            f"bundle N={n_nodes_bundle} does not match model n_nodes="
            f"{model.config.n_nodes}."
        )
    if model.config.t_in != t_in or model.config.t_out != t_out:
        logger.warning(
            "forecast_from_bundle: t_in/t_out (%d/%d) differ from model config "
            "(%d/%d); forward pass will follow the model config.",
            t_in,
            t_out,
            model.config.t_in,
            model.config.t_out,
        )
        t_in = model.config.t_in
        t_out = model.config.t_out

    # ------------------------------------------------------------------
    # Slice last t_in hours and run forward
    # ------------------------------------------------------------------
    x_node_np = np.asarray(bundle.X_node[-t_in:], dtype=np.float32)  # [T_in, N, F_node]
    x_exog_np = np.asarray(bundle.X_exog[-t_in:], dtype=np.float32)  # [T_in, F_exog]

    x_node = torch.from_numpy(x_node_np).unsqueeze(0)  # [1, T_in, N, F_node]
    x_exog = torch.from_numpy(x_exog_np).unsqueeze(0)  # [1, T_in, F_exog]

    device = next(model.parameters()).device
    x_node = x_node.to(device)
    x_exog = x_exog.to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(x_node, x_exog)  # [1, T_out, N, Q]
            # Enforce monotone quantiles on the output axis — mirrors the
            # sort_quantiles path in ``gridsense.model.predict``.
            out, _ = out.sort(dim=-1)
    finally:
        if was_training:
            model.train()

    out_np = out.squeeze(0).detach().cpu().numpy()  # [T_out, N, Q]

    # ------------------------------------------------------------------
    # Denormalise each quantile channel back to kW
    # ------------------------------------------------------------------
    quantiles = tuple(model.config.quantiles)
    # Map the standard p10 / p50 / p90 levels onto the trained quantiles.
    # The training config fixes these at (0.1, 0.5, 0.9), but we look them up
    # defensively so that a re-trained checkpoint with, say, (0.05, 0.5, 0.95)
    # still picks the correct slice.
    def _idx_of(level: float) -> int:
        # Nearest-quantile match; tolerates floating-point drift.
        diffs = [abs(q - level) for q in quantiles]
        return int(np.argmin(diffs))

    i10 = _idx_of(0.1)
    i50 = _idx_of(0.5)
    i90 = _idx_of(0.9)

    p10_std = out_np[..., i10]  # [T_out, N]
    p50_std = out_np[..., i50]
    p90_std = out_np[..., i90]

    p10 = _denormalise_y(p10_std, y_mean, y_std)
    p50 = _denormalise_y(p50_std, y_mean, y_std)
    p90 = _denormalise_y(p90_std, y_mean, y_std)

    # Re-sort after denorm (monotone transform preserves order but only when
    # y_std > 0; be safe when the scaler is degenerate).
    stacked = np.stack([p10, p50, p90], axis=-1)
    stacked.sort(axis=-1)
    p10, p50, p90 = stacked[..., 0], stacked[..., 1], stacked[..., 2]

    # ------------------------------------------------------------------
    # Timestamps + bus names
    # ------------------------------------------------------------------
    last_ts = pd.Timestamp(bundle.times[-1])
    timestamps = _future_timestamps(last_ts, t_out)

    return Forecast(
        p10=p10.astype(np.float64),
        p50=p50.astype(np.float64),
        p90=p90.astype(np.float64),
        timestamps=timestamps,
        bus_names=list(bundle.node_names),
    )
