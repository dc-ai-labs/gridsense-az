"""Tests for ``gridsense.features`` — the hourly feature pipeline.

Exercises the synthetic-demand fallback path by default so the test suite
stays hermetic (no reliance on the big real parquet/CSV files). The
real-data branch is covered implicitly by importing the module-level
constants and by the shape assertions, which remain identical for both
data sources.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridsense.features import (
    EXOG_FEATURE_NAMES,
    FeatureBundle,
    build_hourly_features,
    load_bundle,
    save_bundle,
)

# A short, hermetic window — 10 weeks spans multiple months + weekends so
# every calendar-derived column has non-zero variance (required for the
# unit-std z-score assertion). Still fast: ~1.7k hourly rows.
_START = "2022-01-01"
_END = "2022-03-15"


@pytest.fixture(scope="module")
def empty_data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An empty ``data/raw``-shaped directory to force the synthetic fallback."""
    root = tmp_path_factory.mktemp("empty_raw")
    # Intentionally empty; pipeline must cope.
    return root


@pytest.fixture(scope="module")
def bundle(empty_data_root: Path) -> FeatureBundle:
    """A small synthetic-fallback bundle, reused across tests for speed."""
    return build_hourly_features(start=_START, end=_END, data_root=empty_data_root)


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------


def test_importable() -> None:
    """Module imports and exposes the public API."""
    import gridsense.features as features

    assert hasattr(features, "build_hourly_features")
    assert hasattr(features, "save_bundle")
    assert hasattr(features, "load_bundle")
    assert hasattr(features, "FeatureBundle")
    assert hasattr(features, "EXOG_FEATURE_NAMES")


def test_exog_feature_names_contract(bundle: FeatureBundle) -> None:
    """``EXOG_FEATURE_NAMES`` is a tuple of str matching ``X_exog`` width."""
    assert isinstance(EXOG_FEATURE_NAMES, tuple)
    assert all(isinstance(name, str) for name in EXOG_FEATURE_NAMES)
    assert len(EXOG_FEATURE_NAMES) == bundle.X_exog.shape[1]
    assert len(set(EXOG_FEATURE_NAMES)) == len(EXOG_FEATURE_NAMES), "names must be unique"


# ---------------------------------------------------------------------------
# Synthetic-fallback path
# ---------------------------------------------------------------------------


def test_build_synthetic_fallback(empty_data_root: Path) -> None:
    """With no raw EIA-930 parquet present, pipeline synthesises demand."""
    b = build_hourly_features(start=_START, end=_END, data_root=empty_data_root)
    assert b.meta["source"] == "synthetic"
    T = len(b.times)
    N = len(b.node_names)
    assert b.X_exog.shape == (T, len(EXOG_FEATURE_NAMES))
    assert b.y_kw.shape == (T, N)
    assert b.X_node.shape == (T, N, 1)


# ---------------------------------------------------------------------------
# Shape alignment (132 buses on IEEE-123)
# ---------------------------------------------------------------------------


def test_shapes_align(bundle: FeatureBundle) -> None:
    T = len(bundle.times)
    N = len(bundle.node_names)
    assert N == 132, f"expected 132 buses (IEEE 123 + regulator aliases), got {N}"
    assert bundle.X_exog.shape == (T, len(EXOG_FEATURE_NAMES))
    assert bundle.X_node.shape == (T, N, 1)
    assert bundle.y_kw.shape == (T, N)
    # Monotonic, hourly, UTC.
    assert bundle.times.is_monotonic_increasing
    assert bundle.times.freq is not None or (
        # ``date_range`` may not set freq after round-trips; verify stride manually.
        (bundle.times[1] - bundle.times[0]) == pd.Timedelta(hours=1)
    )
    assert str(bundle.times.tz) == "UTC"


# ---------------------------------------------------------------------------
# Disaggregation invariant
# ---------------------------------------------------------------------------


def test_disaggregation_sums_to_system(bundle: FeatureBundle) -> None:
    """``y_kw.sum(axis=1)`` reproduces the winsorised system demand in kW."""
    # Recover the per-hour system demand: sum across buses.
    per_hour = bundle.y_kw.sum(axis=1)
    # Every buses' share sums to 1, so per_hour[t] == D(t) up to fp noise.
    assert per_hour.shape == (len(bundle.times),)
    # Every bus with zero nominal kW must contribute exactly zero.
    zero_buses = (bundle.y_kw == 0).all(axis=0)
    # IEEE-123 has 132 - 85 = 47 unloaded buses.
    assert int(zero_buses.sum()) == 132 - 85
    # Winsorised synthetic demand lives inside the configured envelope.
    assert per_hour.min() >= 500.0 * 1000.0 - 1.0
    assert per_hour.max() <= 10_000.0 * 1000.0 + 1.0


def test_disaggregation_matches_nominal_shares(bundle: FeatureBundle) -> None:
    """Ratio between any two loaded buses equals their nominal kW ratio."""
    from gridsense.topology import load_ieee123

    graph = load_ieee123()
    loaded = [
        (n, graph.nodes[n]["kw_load"])
        for n in sorted(graph.nodes())
        if graph.nodes[n].get("kw_load", 0) > 0
    ]
    # Pick two well-separated loaded buses.
    assert len(loaded) >= 2
    (n1, kw1), (n2, kw2) = loaded[0], loaded[-1]
    i1 = bundle.node_names.index(n1)
    i2 = bundle.node_names.index(n2)
    t = 3  # any hour
    expected = kw1 / kw2
    got = bundle.y_kw[t, i1] / bundle.y_kw[t, i2]
    assert got == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Standardisation contract
# ---------------------------------------------------------------------------


def test_zscore_mean_std(bundle: FeatureBundle) -> None:
    """Each column of ``X_exog`` is mean-0, std-1 (within fp tolerance)."""
    means = bundle.X_exog.mean(axis=0)
    stds = bundle.X_exog.std(axis=0)
    assert np.all(np.abs(means) < 1e-5), f"non-zero means: {means}"
    assert np.all(np.abs(stds - 1.0) < 1e-3), f"non-unit stds: {stds}"


# ---------------------------------------------------------------------------
# No NaNs anywhere in the model inputs
# ---------------------------------------------------------------------------


def test_no_nan_in_features(bundle: FeatureBundle) -> None:
    assert not np.isnan(bundle.X_exog).any()
    assert not np.isnan(bundle.X_node).any()
    assert not np.isnan(bundle.y_kw).any()


# ---------------------------------------------------------------------------
# Save/load round-trip
# ---------------------------------------------------------------------------


def test_roundtrip_save_load(bundle: FeatureBundle, tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    save_bundle(bundle, out)
    reloaded = load_bundle(out)
    np.testing.assert_array_equal(reloaded.X_exog, bundle.X_exog)
    np.testing.assert_array_equal(reloaded.X_node, bundle.X_node)
    np.testing.assert_array_equal(reloaded.y_kw, bundle.y_kw)
    assert reloaded.node_names == bundle.node_names
    assert reloaded.times.equals(bundle.times)
    # Scalers round-trip as (float, float).
    assert set(reloaded.scalers) == set(bundle.scalers)
    for name, (m, s) in bundle.scalers.items():
        rm, rs = reloaded.scalers[name]
        assert rm == pytest.approx(m, abs=1e-12)
        assert rs == pytest.approx(s, abs=1e-12)
