"""Tests for :mod:`gridsense.predictor`.

Covers:

* Loading the v0 checkpoint at ``data/models/gwnet_v0.pt`` — the model is
  reconstructed with the *exact* :class:`GWNetConfig` used at training time
  (pulled from ``data/models/metrics.json``), not the module defaults.
* The forecast shape contract: ``[T_out, N_nodes]`` per quantile, with
  ``T_out`` future timestamps and ``N_nodes`` bus names.
* The monotone-quantile guarantee (``p10 <= p50 <= p90``).
* The training contract: the model outputs raw kW, so predictions are NOT
  denormalised at inference — magnitudes should already be plausible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from gridsense.features import FeatureBundle, build_hourly_features
from gridsense.model import GWNet, GWNetConfig
from gridsense.predictor import (
    DEFAULT_CKPT_PATH,
    DEFAULT_METRICS_PATH,
    Forecast,
    LoadedPredictor,
    forecast_from_bundle,
    load_predictor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_bundle(
    t: int = 48,
    n: int = 132,
    f_node: int = 1,
    f_exog: int = 11,
    seed: int = 0,
) -> FeatureBundle:
    """Build a :class:`FeatureBundle` of random but shape-correct arrays."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2023-01-01", periods=t, freq="h", tz="UTC")
    node_names = [f"bus{i}" for i in range(n)]
    x_node = rng.normal(size=(t, n, f_node)).astype(np.float32)
    x_exog = rng.normal(size=(t, f_exog)).astype(np.float32)
    # y_kw scale chosen to match the v0 training distribution (MW-class).
    y_kw = (rng.normal(size=(t, n)) * 1000.0 + 30_000.0).astype(np.float32)
    scalers = {"y_kw": (30_000.0, 5_000.0)}
    return FeatureBundle(
        times=times,
        node_names=node_names,
        X_exog=x_exog,
        X_node=x_node,
        y_kw=y_kw,
        scalers=scalers,
        meta={"source": "synthetic"},
    )


# ---------------------------------------------------------------------------
# Checkpoint reconstruction
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not DEFAULT_CKPT_PATH.exists(),
    reason="trained checkpoint not present",
)
def test_load_predictor_reconstructs_config() -> None:
    """``load_predictor`` rebuilds the exact training-time config + scaler."""
    predictor = load_predictor()
    assert isinstance(predictor, LoadedPredictor)
    assert isinstance(predictor.model, GWNet)
    # Frozen-by-training contract:
    assert predictor.config.n_nodes == 132
    assert predictor.config.t_in == 24
    assert predictor.config.t_out == 6
    assert predictor.config.f_exog == 11
    assert tuple(predictor.config.quantiles) == (0.1, 0.5, 0.9)
    # Model is in eval() mode so BN stats / dropout stay frozen.
    assert predictor.model.training is False
    # The scaler surfaced on the predictor is the input-channel scaler
    # (used by features.py on X_node[..., 0]), retained for diagnostics
    # only. We still sanity-check it came from metrics.json (non-identity)
    # when the sidecar is present.
    y_mean, y_std = predictor.y_scaler
    assert y_std > 0.0
    assert y_mean > 0.0  # AZPS demand is strictly positive in the real bundle


@pytest.mark.skipif(
    not DEFAULT_CKPT_PATH.exists(),
    reason="trained checkpoint not present",
)
def test_load_predictor_returns_matching_state_dict_keys() -> None:
    """The reconstructed model accepts the checkpoint state_dict without surprises."""
    predictor = load_predictor()
    # A second (strict=True) reload of the same state file must succeed — this
    # catches silent config drift that leaves stale parameters behind.
    state = torch.load(DEFAULT_CKPT_PATH, map_location="cpu")
    result = predictor.model.load_state_dict(state, strict=True)
    # ``load_state_dict`` returns a ``_IncompatibleKeys`` NamedTuple.
    assert not result.missing_keys
    assert not result.unexpected_keys


# ---------------------------------------------------------------------------
# Forecast contract
# ---------------------------------------------------------------------------


def test_forecast_shape() -> None:
    """Forecast arrays match ``[T_out, N_nodes]`` and quantiles are monotone.

    Runs against a freshly-initialised (not checkpoint-loaded) model so the
    test exercises the inference pipeline even when the checkpoint file is
    absent — which is the common case on CI / a fresh clone.
    """
    torch.manual_seed(0)
    cfg = GWNetConfig()
    model = GWNet(cfg)
    model.eval()
    predictor = LoadedPredictor(model=model, config=cfg, y_scaler=(30_000.0, 5_000.0))

    bundle = _synthetic_bundle(t=cfg.t_in + 8, n=cfg.n_nodes)
    forecast = forecast_from_bundle(predictor, bundle, t_in=cfg.t_in, t_out=cfg.t_out)

    assert isinstance(forecast, Forecast)
    assert forecast.p10.shape == (cfg.t_out, cfg.n_nodes)
    assert forecast.p50.shape == (cfg.t_out, cfg.n_nodes)
    assert forecast.p90.shape == (cfg.t_out, cfg.n_nodes)
    assert len(forecast.timestamps) == cfg.t_out
    assert len(forecast.bus_names) == cfg.n_nodes
    assert forecast.bus_names == bundle.node_names

    # Quantiles must be monotone elementwise.
    assert np.all(forecast.p10 <= forecast.p50 + 1e-6)
    assert np.all(forecast.p50 <= forecast.p90 + 1e-6)

    # First forecast timestamp is exactly 1 h after the bundle's last time.
    expected_first = pd.Timestamp(bundle.times[-1]) + pd.Timedelta(hours=1)
    assert pd.Timestamp(forecast.timestamps[0]) == expected_first


def test_forecast_from_bundle_accepts_bare_model() -> None:
    """``forecast_from_bundle`` also works when passed a bare ``GWNet``."""
    torch.manual_seed(0)
    cfg = GWNetConfig()
    model = GWNet(cfg)
    model.eval()

    bundle = _synthetic_bundle(t=cfg.t_in + 4, n=cfg.n_nodes)
    forecast = forecast_from_bundle(model, bundle, t_in=cfg.t_in, t_out=cfg.t_out)
    assert forecast.p50.shape == (cfg.t_out, cfg.n_nodes)


def test_forecast_raises_on_short_bundle() -> None:
    """A bundle shorter than ``t_in`` must be rejected with a clear error."""
    cfg = GWNetConfig()
    model = GWNet(cfg)
    predictor = LoadedPredictor(model=model, config=cfg, y_scaler=(0.0, 1.0))

    bundle = _synthetic_bundle(t=cfg.t_in - 1, n=cfg.n_nodes)
    with pytest.raises(ValueError, match="at least t_in"):
        forecast_from_bundle(predictor, bundle, t_in=cfg.t_in, t_out=cfg.t_out)


def test_forecast_output_is_raw_kw_no_denormalisation() -> None:
    """The y_scaler on LoadedPredictor must NOT affect the forecast.

    Training contract: the model's output is already in raw kW. Any value
    sitting in ``LoadedPredictor.y_scaler`` is a diagnostic tag for the
    input-channel scaler — :func:`forecast_from_bundle` must ignore it.
    Two predictors that share the same model weights but carry different
    ``y_scaler`` tags must produce identical forecasts.
    """
    torch.manual_seed(0)
    cfg = GWNetConfig()
    model = GWNet(cfg)
    model.eval()
    bundle = _synthetic_bundle(t=cfg.t_in + 4, n=cfg.n_nodes)

    predictor_identity = LoadedPredictor(model=model, config=cfg, y_scaler=(0.0, 1.0))
    predictor_scaled = LoadedPredictor(
        model=model, config=cfg, y_scaler=(30_000.0, 5_000.0)
    )

    fc_identity = forecast_from_bundle(
        predictor_identity, bundle, t_in=cfg.t_in, t_out=cfg.t_out
    )
    fc_scaled = forecast_from_bundle(
        predictor_scaled, bundle, t_in=cfg.t_in, t_out=cfg.t_out
    )

    # Bitwise-identical forecasts regardless of the tag: same model, same
    # inputs, no denormalisation step that could pick up the tag.
    assert np.array_equal(fc_identity.p10, fc_scaled.p10)
    assert np.array_equal(fc_identity.p50, fc_scaled.p50)
    assert np.array_equal(fc_identity.p90, fc_scaled.p90)


def test_load_predictor_missing_file() -> None:
    """A clearly-missing checkpoint raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        load_predictor(ckpt_path=Path("/tmp/does/not/exist.pt"))


# ---------------------------------------------------------------------------
# Magnitude regression — catches the 43 000× blow-up bug
# ---------------------------------------------------------------------------


_REAL_BUNDLE_READY = (
    DEFAULT_CKPT_PATH.exists()
    and (Path(__file__).resolve().parent.parent / "data" / "raw").exists()
)


@pytest.mark.skipif(
    not _REAL_BUNDLE_READY,
    reason="trained checkpoint or data/raw not present",
)
def test_forecast_magnitudes_realistic() -> None:
    """End-to-end sanity check: system-total MW lands in a plausible AZPS range.

    The previous predictor (pre-fix) multiplied the model output by the
    input-channel std (~43 000) and added the mean, which inflated
    system-total forecasts to ~200 GW — ~30× above the AZPS all-time peak.
    This regression asserts that a summer-window forecast sums to between
    2 000 and 12 000 MW (AZPS summer typical: 4–8 GW).
    """
    data_root = Path(__file__).resolve().parent.parent / "data" / "raw"
    predictor = load_predictor()
    bundle = build_hourly_features(
        "2023-08-01", "2023-08-08", data_root=data_root
    )
    forecast = forecast_from_bundle(
        predictor,
        bundle,
        t_in=predictor.config.t_in,
        t_out=predictor.config.t_out,
    )

    # System total in MW per forecast hour.
    system_mw = forecast.p50.sum(axis=1) / 1000.0
    assert system_mw.shape == (predictor.config.t_out,)
    assert np.all(system_mw > 2_000.0), (
        f"system total {system_mw.tolist()} MW below AZPS floor — "
        "possible mis-scaled forecast."
    )
    assert np.all(system_mw < 12_000.0), (
        f"system total {system_mw.tolist()} MW above AZPS ceiling — "
        "the output denormalisation bug has regressed."
    )

    # Per-bus p50 should sit in the kW range — not GW, not W.
    assert forecast.p50.min() > -5_000.0  # tolerate tiny negatives
    assert forecast.p50.max() < 500_000.0
    # Mean per-bus load should be comfortably positive.
    assert forecast.p50.mean() > 1_000.0
