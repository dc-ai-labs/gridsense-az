"""Tests for the hourly feature pipeline (gridsense.features).

These must pass in a hermetic environment — no network, no API keys — so
every test points at an empty ``tmp_path`` and exercises the synthetic-
fallback branch of :func:`gridsense.features.build_hourly_features`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Test window must span a full year so every exogenous column — including
# month_sin/cos and dow_sin/cos — carries variance, otherwise the z-score
# assertion trivially fails on constant calendar columns. 8760 hours rebuilds
# in ~0.4s on CI.
_TEST_START = "2022-01-01"
_TEST_END = "2023-01-01"


@pytest.fixture(scope="module")
def synthetic_bundle(tmp_path_factory):
    """Build a FeatureBundle pointing at an empty data dir → synthetic fallback."""
    from gridsense.features import build_hourly_features

    empty_root = tmp_path_factory.mktemp("raw_empty")
    return build_hourly_features(
        start=_TEST_START,
        end=_TEST_END,
        data_root=empty_root,
    )


# ---------------------------------------------------------------------------
# 1. Importability
# ---------------------------------------------------------------------------


def test_importable() -> None:
    """Module loads and exposes the public API."""
    import gridsense.features as feat

    assert hasattr(feat, "FeatureBundle")
    assert hasattr(feat, "build_hourly_features")
    assert hasattr(feat, "save_bundle")
    assert hasattr(feat, "load_bundle")
    assert hasattr(feat, "EXOG_FEATURE_NAMES")


# ---------------------------------------------------------------------------
# 2. Synthetic fallback
# ---------------------------------------------------------------------------


def test_build_synthetic_fallback(tmp_path):
    """With an empty raw dir, a valid bundle is produced and flagged synthetic."""
    from gridsense.features import FeatureBundle, build_hourly_features

    b = build_hourly_features(start=_TEST_START, end=_TEST_END, data_root=tmp_path)
    assert isinstance(b, FeatureBundle)
    assert b.meta["source"] == "synthetic"

    # Shapes consistent with a 1-year hourly window on the 132-bus feeder.
    T = 365 * 24
    assert len(b.times) == T
    assert b.X_exog.shape[0] == T
    assert b.X_node.shape == (T, 132, b.X_node.shape[-1])
    assert b.y_kw.shape == (T, 132)


# ---------------------------------------------------------------------------
# 3. Shape alignment
# ---------------------------------------------------------------------------


def test_shapes_align(synthetic_bundle):
    """Per-bus tensors line up with the 132-node feeder and the time axis."""
    b = synthetic_bundle
    T = len(b.times)

    assert b.X_exog.shape[0] == T
    assert b.X_node.shape == (T, 132, b.X_node.shape[-1])
    assert b.y_kw.shape == (T, 132)
    assert len(b.node_names) == 132

    # Monotonic hourly UTC index.
    assert b.times.is_monotonic_increasing
    assert str(b.times.tz) == "UTC"
    deltas = b.times[1:] - b.times[:-1]
    assert (deltas == pd.Timedelta("1h")).all()


# ---------------------------------------------------------------------------
# 4. Disaggregation conservation
# ---------------------------------------------------------------------------


def test_disaggregation_sums_to_system(synthetic_bundle):
    """Sum of per-bus kW at each hour equals system demand (float32 tolerance)."""
    b = synthetic_bundle
    # Back out the synthetic system demand used to build y_kw — it's the only
    # source when meta.source == 'synthetic'.
    from gridsense.features import _synthetic_system_demand_mw

    assert b.meta["source"] == "synthetic"
    demand_kw = _synthetic_system_demand_mw(b.times) * 1000.0
    totals = b.y_kw.sum(axis=1).astype(np.float64)
    max_abs_err = float(np.max(np.abs(totals - demand_kw)))
    # Each hour aggregates float32 rounding over 132 buses → ~1 kW tolerance.
    assert max_abs_err < 1.0, f"disaggregation residual too large: {max_abs_err} kW"


# ---------------------------------------------------------------------------
# 5. Z-score correctness
# ---------------------------------------------------------------------------


def test_zscore_mean_std(synthetic_bundle):
    """Each X_exog column is standardised to zero-mean unit-std."""
    b = synthetic_bundle
    means = b.X_exog.mean(axis=0)
    stds = b.X_exog.std(axis=0)
    assert np.all(np.abs(means) < 1e-6), f"non-zero means: {means}"
    assert np.all(np.abs(stds - 1.0) < 1e-3), f"non-unit stds: {stds}"


# ---------------------------------------------------------------------------
# 6. No NaNs in feature matrices
# ---------------------------------------------------------------------------


def test_no_nan_in_features(synthetic_bundle):
    """None of the three float tensors contain NaN/Inf."""
    b = synthetic_bundle
    for name, arr in [("X_exog", b.X_exog), ("X_node", b.X_node), ("y_kw", b.y_kw)]:
        assert not np.isnan(arr).any(), f"{name} contains NaN"
        assert not np.isinf(arr).any(), f"{name} contains Inf"


# ---------------------------------------------------------------------------
# 7. Save/load round-trip
# ---------------------------------------------------------------------------


def test_roundtrip(tmp_path):
    """save_bundle → load_bundle reproduces the arrays byte-for-byte."""
    from gridsense.features import build_hourly_features, load_bundle, save_bundle

    raw = tmp_path / "raw"
    raw.mkdir()
    b = build_hourly_features(start=_TEST_START, end=_TEST_END, data_root=raw)

    out = tmp_path / "bundle"
    save_bundle(b, out)
    b2 = load_bundle(out)

    np.testing.assert_array_equal(b.X_exog, b2.X_exog)
    np.testing.assert_array_equal(b.X_node, b2.X_node)
    np.testing.assert_array_equal(b.y_kw, b2.y_kw)
    assert list(b.times) == list(b2.times)
    assert b.node_names == b2.node_names
    assert b.scalers == b2.scalers
    assert b.meta["source"] == b2.meta["source"]


# ---------------------------------------------------------------------------
# 8. Calendar-feature ranges (inspected on raw_exog_df)
# ---------------------------------------------------------------------------


def test_calendar_features_ranges(synthetic_bundle):
    """Raw (pre-standardisation) calendar features live in their natural ranges."""
    b = synthetic_bundle
    df = b.raw_exog_df

    for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"):
        values = df[col].to_numpy()
        assert values.min() >= -1.0 - 1e-9, f"{col} below -1: {values.min()}"
        assert values.max() <= 1.0 + 1e-9, f"{col} above 1: {values.max()}"

    weekend = df["is_weekend"].to_numpy()
    unique = set(np.unique(weekend).tolist())
    assert unique.issubset({0.0, 1.0}), f"is_weekend outside {{0,1}}: {unique}"
    # A full year obviously spans both weekdays and weekend days.
    assert unique == {0.0, 1.0}, f"expected both 0 and 1 in window, got {unique}"
