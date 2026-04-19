"""Tests for :mod:`gridsense.eval`.

Covers:

* The stress mask definition (Phoenix summer evening, UTC→local with no DST).
* Shape + agreement contracts of :func:`compute_stress_window_mae` on a
  synthetic :class:`FeatureBundle` (fast — runs on every CI).
* CLI plumbing: ``--help`` works, ``--history /nonexistent`` exits with a
  clean error + non-zero code.
* On a machine that has the v0 checkpoint AND the real raw data, an
  end-to-end run agrees with ``metrics.json`` on overall test MAE.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from gridsense.eval import (
    PHOENIX_UTC_OFFSET_HOURS,
    STRESS_HOURS_LOCAL,
    STRESS_MONTHS,
    STRESS_WINDOW_TAG,
    compute_stress_window_mae,
    stress_mask_for_timestamps,
)
from gridsense.features import FeatureBundle
from gridsense.model import GWNet, GWNetConfig
from gridsense.predictor import DEFAULT_CKPT_PATH, DEFAULT_METRICS_PATH, LoadedPredictor


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------


def test_stress_mask_phoenix_summer_evening_utc_conversion() -> None:
    """UTC hours map to Phoenix local (UTC-7, no DST). A July 22:00 UTC
    timestamp is 15:00 MST — *not* stress. A July 03:00 UTC timestamp is
    20:00 MST the *previous* day — stress.
    """
    # Crafted UTC timestamps with known local hour.
    ts = pd.DatetimeIndex(
        [
            "2023-07-15 22:00:00+00:00",  # 15 MST (Jul) — not evening → False
            "2023-07-16 00:00:00+00:00",  # 17 MST (Jul 15) — stress → True
            "2023-07-16 03:00:00+00:00",  # 20 MST (Jul 15) — stress → True
            "2023-07-16 05:00:00+00:00",  # 22 MST (Jul 15) — not evening → False
            "2023-05-16 02:00:00+00:00",  # 19 MST but May — False
            "2023-09-30 04:00:00+00:00",  # 21 MST Sep → True
            "2023-10-01 04:00:00+00:00",  # 21 MST Oct — out of season → False
        ]
    )
    mask = stress_mask_for_timestamps(ts)
    np.testing.assert_array_equal(mask, [False, True, True, False, False, True, False])


def test_stress_mask_bounds_exact() -> None:
    """Every (month, local_hour) pair in the product maps as expected."""
    # Sweep a full year at 1h cadence.
    idx = pd.date_range("2023-01-01", "2024-01-01", freq="h", tz="UTC", inclusive="left")
    mask = stress_mask_for_timestamps(idx)
    # Independently recompute the mask.
    local_hour = (idx.hour.to_numpy() + PHOENIX_UTC_OFFSET_HOURS) % 24
    expected = np.isin(idx.month, list(STRESS_MONTHS)) & np.isin(local_hour, list(STRESS_HOURS_LOCAL))
    np.testing.assert_array_equal(mask, expected)
    # Sanity: the stress window is 4 months * 5 hours / (12*24) ≈ 6.9% of the year.
    frac = mask.mean()
    assert 0.06 < frac < 0.08, f"stress fraction {frac:.3f} out of expected band"


def test_stress_window_tag_matches_frontend_contract() -> None:
    """The tag we write to JSON must match the frontend type literal."""
    assert STRESS_WINDOW_TAG == "summer_evenings_17_21_jun_sep"


# ---------------------------------------------------------------------------
# compute_stress_window_mae on a synthetic bundle
# ---------------------------------------------------------------------------


def _synthetic_bundle(
    t: int,
    n: int = 8,
    f_node: int = 1,
    f_exog: int = 11,
    seed: int = 0,
    start: str = "2023-06-01",
) -> FeatureBundle:
    """Construct a small but time-correct feature bundle for eval tests."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=t, freq="h", tz="UTC")
    node_names = [f"bus{i}" for i in range(n)]
    x_node = rng.normal(size=(t, n, f_node)).astype(np.float32)
    x_exog = rng.normal(size=(t, f_exog)).astype(np.float32)
    # Realistic kW-scale targets so MAE numbers come out in a sensible range.
    y_kw = (rng.normal(size=(t, n)) * 500.0 + 20_000.0).astype(np.float32)
    scalers = {"y_kw": (20_000.0, 500.0)}
    return FeatureBundle(
        times=times,
        node_names=node_names,
        X_exog=x_exog,
        X_node=x_node,
        y_kw=y_kw,
        scalers=scalers,
        meta={"source": "synthetic"},
    )


def _tiny_predictor(n_nodes: int, t_in: int = 24, t_out: int = 6) -> LoadedPredictor:
    cfg = GWNetConfig(n_nodes=n_nodes, t_in=t_in, t_out=t_out)
    torch.manual_seed(0)
    model = GWNet(cfg)
    model.eval()
    return LoadedPredictor(model=model, config=cfg, y_scaler=(0.0, 1.0))


def test_compute_stress_window_mae_schema_and_sanity() -> None:
    """Report contains all required keys; overall MAE >= 0 and finite."""
    # ~1600 hours starting June 1 2023 spans plenty of summer evenings.
    bundle = _synthetic_bundle(t=24 * 70, n=8, start="2023-06-01")
    predictor = _tiny_predictor(n_nodes=8)
    report = compute_stress_window_mae(predictor, bundle, batch_size=16)

    required_keys = {
        "overall_mae_kw",
        "stress_mae_kw",
        "persistence_overall_mae_kw",
        "persistence_stress_mae_kw",
        "improvement_overall_pct",
        "improvement_stress_pct",
        "stress_hours",
        "total_hours",
        "stress_window_definition",
    }
    assert required_keys.issubset(report.keys())
    assert report["stress_window_definition"] == STRESS_WINDOW_TAG

    # Non-negative, finite MAEs.
    for key in ("overall_mae_kw", "persistence_overall_mae_kw"):
        val = report[key]
        assert np.isfinite(val), f"{key}={val} not finite"
        assert val >= 0.0

    # The synthetic bundle is dense with summer evenings → stress coverage > 0.
    assert report["stress_hours"] > 0
    assert report["total_hours"] >= report["stress_hours"]


def test_compute_stress_window_mae_handles_empty_stress_bucket() -> None:
    """If the test split covers no stress hours, the report still returns
    the overall MAE and NaNs for the stress-only entries."""
    # Bundle entirely in February → zero stress hours for any test window.
    bundle = _synthetic_bundle(t=24 * 70, n=8, start="2023-02-01")
    predictor = _tiny_predictor(n_nodes=8)
    report = compute_stress_window_mae(predictor, bundle, batch_size=16)

    assert report["stress_hours"] == 0
    assert np.isnan(report["stress_mae_kw"])
    assert np.isnan(report["persistence_stress_mae_kw"])
    assert np.isnan(report["improvement_stress_pct"])
    # The overall metrics must still be finite.
    assert np.isfinite(report["overall_mae_kw"])
    assert np.isfinite(report["persistence_overall_mae_kw"])


def test_persistence_improvement_signs_are_consistent() -> None:
    """Persistence MAE >= 0; improvement_pct has the expected (baseline-model)/baseline form."""
    bundle = _synthetic_bundle(t=24 * 70, n=8, start="2023-06-01")
    predictor = _tiny_predictor(n_nodes=8)
    report = compute_stress_window_mae(predictor, bundle, batch_size=16)

    overall = report["overall_mae_kw"]
    base = report["persistence_overall_mae_kw"]
    imp = report["improvement_overall_pct"]
    expected = 100.0 * (base - overall) / base
    assert abs(imp - expected) < 1e-6


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_help_runs_cleanly() -> None:
    """``python -m gridsense.eval --help`` exits 0 with usage info on stdout."""
    result = subprocess.run(
        [sys.executable, "-m", "gridsense.eval", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert result.returncode == 0, result.stderr
    assert "stress-window" in result.stdout.lower() or "stress" in result.stdout.lower()
    assert "--ckpt" in result.stdout
    assert "--history" in result.stdout
    assert "--out" in result.stdout


def test_cli_missing_history_emits_clean_error(tmp_path: Path) -> None:
    """A non-existent --history triggers exit code 2 with a helpful message."""
    bogus = tmp_path / "does_not_exist.parquet"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gridsense.eval",
            "--ckpt",
            str(DEFAULT_CKPT_PATH),
            "--history",
            str(bogus),
            "--out",
            str(tmp_path / "report.json"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    # If the checkpoint is missing on this machine we get exit code 2 from the
    # ckpt check (also acceptable — that's a different "clean error"). Either
    # way the CLI must not crash with an unhandled traceback.
    assert result.returncode in (2, 3)
    combined = (result.stdout + result.stderr).lower()
    assert "error" in combined


# ---------------------------------------------------------------------------
# End-to-end (only when real data + checkpoint are present)
# ---------------------------------------------------------------------------


_RAW_DATA_READY = (
    DEFAULT_CKPT_PATH.exists()
    and (REPO_ROOT / "data" / "raw" / "eia930" / "azps_demand.parquet").exists()
    and (REPO_ROOT / "data" / "raw" / "noaa").exists()
)


@pytest.mark.slow
@pytest.mark.skipif(
    not _RAW_DATA_READY,
    reason="trained checkpoint or raw data not present",
)
def test_end_to_end_agrees_with_training_metrics() -> None:
    """The overall MAE from ``compute_stress_window_mae`` agrees with
    ``metrics.json.test_mae`` recorded by ``scripts/train.py`` to within 1%.
    """
    from gridsense.eval import run_eval

    report = run_eval()
    metrics_raw = json.loads(DEFAULT_METRICS_PATH.read_text())
    train_test_mae = float(metrics_raw["test_mae"])
    eval_test_mae = float(report["overall_mae_kw"])
    # Rounding / batching can introduce <1% drift; anything more indicates a
    # schema mismatch between the training loader and the eval loader.
    rel_err = abs(eval_test_mae - train_test_mae) / max(train_test_mae, 1e-9)
    assert rel_err < 0.01, (
        f"overall MAE drift too large: eval={eval_test_mae:.2f} kW vs "
        f"training-recorded {train_test_mae:.2f} kW (rel_err={rel_err:.4f})"
    )
    # Sanity: stress MAE should be reported (not NaN) on the real bundle.
    assert np.isfinite(report["stress_mae_kw"])
    assert report["stress_hours"] > 0
