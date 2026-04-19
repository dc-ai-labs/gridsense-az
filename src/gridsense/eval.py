"""Stress-period evaluation for GridSense-AZ.

This module computes forecast accuracy (MAE, in raw kW) for the trained
Graph-WaveNet checkpoint, with an explicit *stress-window* breakout that
the APS rubric rewards: Phoenix summer evenings, when A/C load peaks and
solar generation drops off a cliff.

Stress window definition
------------------------
A forecast *target* timestep is considered a stress row when::

    month in {6, 7, 8, 9}   AND   local_hour in {17, 18, 19, 20, 21}

Phoenix does not observe DST, so the local hour is always
``(utc_hour - 7) mod 24``.

The stress mask is applied at the level of the model's output grid
(shape ``[N_windows, T_out, N_nodes]``): a single test window contributes
some of its ``T_out`` horizon steps to the stress bucket and some to the
"normal" bucket, based on the UTC timestamp of each horizon step.

Baseline
--------
We report MAE for the trained model *and* for a flat-persistence baseline
(``y_hat[t+h] = y[t-1]`` for every horizon step). The persistence baseline
is the same one ``scripts/train.py`` uses — the rubric wants a comparable
stress-window improvement, not just an absolute number.

CLI
---
``python -m gridsense.eval`` rebuilds the test split from the real raw
data (EIA-930 + NOAA ISD), runs the checkpoint once over the test
windows, and writes a JSON report with both overall and stress
breakouts::

    {
      "overall_mae_kw": 4574.0,
      "stress_mae_kw": 3921.3,
      "persistence_overall_mae_kw": 5603.7,
      "persistence_stress_mae_kw": 6844.8,
      "improvement_overall_pct": 18.4,
      "improvement_stress_pct": 42.7,
      "stress_hours": 412,
      "total_hours": 10350,
      "stress_window_definition": "summer_evenings_17_21_jun_sep"
    }

If the raw data needed to rebuild the feature bundle is missing, the CLI
exits non-zero with a pointer at ``scripts/pull_all.sh`` /
``scripts/train.py`` — we never silently skip or fake numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gridsense.features import FeatureBundle, build_hourly_features
from gridsense.model import GWNet, make_dataloader, predict
from gridsense.predictor import (
    DEFAULT_CKPT_PATH,
    DEFAULT_METRICS_PATH,
    LoadedPredictor,
    load_predictor,
)

__all__ = [
    "STRESS_MONTHS",
    "STRESS_HOURS_LOCAL",
    "STRESS_WINDOW_TAG",
    "PHOENIX_UTC_OFFSET_HOURS",
    "stress_mask_for_timestamps",
    "compute_stress_window_mae",
    "run_eval",
    "main",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stress window constants
# ---------------------------------------------------------------------------

#: Months included in the "summer" portion of the stress window.
STRESS_MONTHS: frozenset[int] = frozenset({6, 7, 8, 9})

#: Local hours-of-day included in the "evening" portion of the stress window.
STRESS_HOURS_LOCAL: frozenset[int] = frozenset({17, 18, 19, 20, 21})

#: Arizona does not observe DST — always UTC-7.
PHOENIX_UTC_OFFSET_HOURS: int = -7

#: Tag we persist alongside the stress MAE so downstream consumers (dashboard,
#: report) can render the human definition without hard-coding it.
STRESS_WINDOW_TAG: str = "summer_evenings_17_21_jun_sep"


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------


def stress_mask_for_timestamps(timestamps: pd.DatetimeIndex | np.ndarray) -> np.ndarray:
    """Boolean mask marking which timestamps fall in the stress window.

    Args:
        timestamps: Hourly UTC timestamps (tz-aware or tz-naive, interpreted
            as UTC). Accepts a :class:`pd.DatetimeIndex` or an array of
            numpy datetime64 values.

    Returns:
        Boolean array of the same length as ``timestamps``; ``True`` where
        the UTC timestamp, converted to Phoenix local time (UTC-7, no DST),
        falls on a summer (Jun–Sep) evening hour (17:00–21:00 local).
    """
    idx = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    utc_hour = idx.hour.to_numpy().astype(np.int64)
    local_hour = (utc_hour + PHOENIX_UTC_OFFSET_HOURS) % 24
    month = idx.month.to_numpy().astype(np.int64)
    in_month = np.isin(month, list(STRESS_MONTHS))
    in_hour = np.isin(local_hour, list(STRESS_HOURS_LOCAL))
    return np.logical_and(in_month, in_hour)


# ---------------------------------------------------------------------------
# Low-level MAE helpers
# ---------------------------------------------------------------------------


def _masked_mae(pred: np.ndarray, target: np.ndarray, mask_2d: np.ndarray) -> float:
    """Mean absolute error over an ``[N_windows, T_out, N_nodes]`` prediction,
    restricted to the windows/horizons selected by ``mask_2d`` (shape
    ``[N_windows, T_out]``).

    Every node in each selected (window, horizon) cell is counted; the mask
    lives on the time axis only. Returns ``nan`` if the mask selects no
    elements.
    """
    if pred.shape[:2] != mask_2d.shape:
        raise ValueError(
            f"shape mismatch: pred leading {pred.shape[:2]} vs mask {mask_2d.shape}"
        )
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    if not mask_2d.any():
        return float("nan")
    # Broadcast mask along the node axis.
    mask = np.broadcast_to(mask_2d[..., None], pred.shape)
    abs_err = np.abs(pred - target)
    return float(abs_err[mask].mean())


def _persistence_predictions(
    y_full: np.ndarray,
    starts: np.ndarray,
    t_in: int,
    t_out: int,
) -> np.ndarray:
    """Build flat-persistence predictions for every test window.

    For each window at start ``i`` we forecast every one of the ``t_out``
    horizon steps as ``y_full[i + t_in - 1]`` — the last observed input
    timestep. Output shape: ``[N_windows, t_out, N_nodes]``.
    """
    n_windows = starts.size
    n_nodes = y_full.shape[1]
    pred = np.empty((n_windows, t_out, n_nodes), dtype=np.float64)
    for w, i in enumerate(starts.astype(np.int64)):
        last_input = y_full[int(i) + t_in - 1]  # [N]
        pred[w, :, :] = last_input[None, :]
    return pred


def _window_timestamps(
    times: pd.DatetimeIndex,
    starts: np.ndarray,
    t_in: int,
    t_out: int,
) -> pd.DatetimeIndex:
    """Return the flat array of UTC timestamps for every (window, horizon)
    cell in row-major order.

    Output length: ``N_windows * t_out``. The ``h``-th horizon step of the
    ``w``-th window corresponds to ``times[starts[w] + t_in + h]``.
    """
    offsets = np.asarray(starts, dtype=np.int64)[:, None] + (t_in + np.arange(t_out, dtype=np.int64))[None, :]
    flat_idx = offsets.reshape(-1)
    return pd.DatetimeIndex(times[flat_idx])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_stress_window_mae(
    predictor: LoadedPredictor | GWNet,
    bundle: FeatureBundle,
    *,
    t_in: int | None = None,
    t_out: int | None = None,
    batch_size: int = 32,
    device: str = "cpu",
    split: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, float | int | str]:
    """Compute overall and stress-window MAE on the test split of ``bundle``.

    The function reproduces the test split used by ``scripts/train.py``:
    it calls :func:`gridsense.model.make_dataloader` with ``which="test"``
    using the same ``(t_in, t_out, split)`` defaults as training. This is
    the same test set whose MAE is recorded in ``metrics.json`` under
    ``test_mae`` — the *overall* MAE returned here should agree with it to
    rounding, and the *stress* MAE is the new breakout.

    The persistence baseline is computed on the exact same window grid so
    the improvement percentages are directly comparable.

    Args:
        predictor: A loaded :class:`LoadedPredictor` or a bare :class:`GWNet`.
            Either form is accepted; the model's config supplies ``t_in`` /
            ``t_out`` defaults when not given explicitly.
        bundle: Aligned feature bundle covering the same window as training
            (``metrics.data.start`` → ``metrics.data.end``).
        t_in, t_out: Override the input/output horizons. Default to the
            model config.
        batch_size: DataLoader batch size (memory-bound; 32 is plenty).
        device: Torch device string. ``"cpu"`` is typically fine for
            one-shot eval runs.
        split: ``(train, val, test)`` fractions matching the training run.

    Returns:
        A dict with the following keys, all JSON-serialisable::

            overall_mae_kw            # model MAE on all test horizon steps
            stress_mae_kw             # model MAE on stress horizon steps
            persistence_overall_mae_kw  # flat-persistence baseline, overall
            persistence_stress_mae_kw   # flat-persistence baseline, stress-only
            improvement_overall_pct   # (baseline - model) / baseline * 100
            improvement_stress_pct
            stress_hours              # count of (window, horizon) cells in stress
            total_hours               # count of all (window, horizon) cells
            stress_window_definition  # always STRESS_WINDOW_TAG

        Values are ``nan`` when the underlying mask selects zero cells
        (e.g. if the test window somehow excludes all summer evenings —
        which never happens for the real training budget but matters for
        synthetic-bundle tests).
    """
    if isinstance(predictor, LoadedPredictor):
        model = predictor.model
        cfg = predictor.config
    else:
        model = predictor
        cfg = predictor.config
    if t_in is None:
        t_in = int(cfg.t_in)
    if t_out is None:
        t_out = int(cfg.t_out)

    # Rebuild the *exact* test loader train.py uses. The split + stride
    # defaults are matched to ``make_dataloader``'s defaults so the window
    # indices are reproduced bit-for-bit.
    test_loader: DataLoader = make_dataloader(
        bundle,
        t_in=t_in,
        t_out=t_out,
        batch_size=batch_size,
        which="test",
        split=split,
        shuffle=False,
    )
    dataset = test_loader.dataset  # _BundleWindowDataset
    starts = np.asarray(dataset.starts, dtype=np.int64)
    n_windows = int(starts.size)
    if n_windows == 0:
        raise RuntimeError("test split is empty — cannot evaluate stress MAE")

    logger.info(
        "compute_stress_window_mae: test windows=%d, t_in=%d, t_out=%d",
        n_windows,
        t_in,
        t_out,
    )

    # --- Model predictions (p50 slice) ---------------------------------------
    pred_tensor, target_tensor = predict(model, test_loader, device=device, sort_quantiles=True)
    # pred: [W, T_out, N, Q], target: [W, T_out, N]
    q = pred_tensor.shape[-1]
    pred_p50 = pred_tensor[..., q // 2].numpy().astype(np.float64)
    target_np = target_tensor.numpy().astype(np.float64)

    # --- Persistence predictions on the same window grid ---------------------
    y_full = np.asarray(bundle.y_kw, dtype=np.float64)
    persist_pred = _persistence_predictions(y_full, starts, t_in=t_in, t_out=t_out)

    # --- Stress mask keyed on UTC-hour per (window, horizon) cell ------------
    flat_ts = _window_timestamps(bundle.times, starts, t_in=t_in, t_out=t_out)
    flat_mask = stress_mask_for_timestamps(flat_ts)
    mask_2d = flat_mask.reshape(n_windows, t_out)
    all_mask = np.ones_like(mask_2d, dtype=bool)

    overall_mae = _masked_mae(pred_p50, target_np, all_mask)
    stress_mae = _masked_mae(pred_p50, target_np, mask_2d)
    persist_overall_mae = _masked_mae(persist_pred, target_np, all_mask)
    persist_stress_mae = _masked_mae(persist_pred, target_np, mask_2d)

    def _improve(model_mae: float, baseline_mae: float) -> float:
        if not np.isfinite(model_mae) or not np.isfinite(baseline_mae) or baseline_mae <= 0:
            return float("nan")
        return float(100.0 * (baseline_mae - model_mae) / baseline_mae)

    improvement_overall = _improve(overall_mae, persist_overall_mae)
    improvement_stress = _improve(stress_mae, persist_stress_mae)

    total_hours = int(mask_2d.size)
    stress_hours = int(mask_2d.sum())

    result: dict[str, float | int | str] = {
        "overall_mae_kw": float(overall_mae),
        "stress_mae_kw": float(stress_mae),
        "persistence_overall_mae_kw": float(persist_overall_mae),
        "persistence_stress_mae_kw": float(persist_stress_mae),
        "improvement_overall_pct": float(improvement_overall),
        "improvement_stress_pct": float(improvement_stress),
        "stress_hours": stress_hours,
        "total_hours": total_hours,
        "stress_window_definition": STRESS_WINDOW_TAG,
    }
    logger.info(
        "eval: overall=%.1f kW (baseline %.1f, %.2f%%) | stress=%.1f kW "
        "(baseline %.1f, %.2f%%) | stress_hours=%d / %d",
        overall_mae,
        persist_overall_mae,
        improvement_overall,
        stress_mae,
        persist_stress_mae,
        improvement_stress,
        stress_hours,
        total_hours,
    )
    return result


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def _missing_raw_data_error(raw_root: Path, missing: list[str]) -> str:
    """Produce a human-readable hint when raw data needed for eval is absent."""
    lines = [
        "Cannot reproduce the training feature bundle: required raw inputs are missing.",
        f"  raw data root: {raw_root}",
        "  missing:",
    ]
    lines.extend(f"    - {m}" for m in missing)
    lines += [
        "",
        "Fix one of:",
        "  1. Pull raw data: bash scripts/pull_all.sh",
        "  2. Re-run training which writes the same bundle: "
        ".venv/bin/python scripts/train.py",
        "  3. Pass an explicit --history parquet (not yet implemented for custom "
        "bundles — current CLI rebuilds from raw).",
    ]
    return "\n".join(lines)


def _check_raw_data_present(raw_root: Path) -> list[str]:
    """Return a list of missing required raw files (empty list ⇒ all present)."""
    missing: list[str] = []
    eia = raw_root / "eia930" / "azps_demand.parquet"
    if not eia.exists():
        missing.append(str(eia))
    noaa_dir = raw_root / "noaa"
    if not noaa_dir.exists() or not any(noaa_dir.glob("KPHX_*.csv")):
        missing.append(f"{noaa_dir}/KPHX_*.csv")
    return missing


def run_eval(
    ckpt_path: Path = DEFAULT_CKPT_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    history_path: Path | None = None,
    out_path: Path | None = None,
    device: str = "cpu",
    batch_size: int = 32,
) -> dict[str, float | int | str]:
    """Build features from the real raw data, run the checkpoint, write a JSON report.

    Args:
        ckpt_path: Path to ``gwnet_v0.pt`` (or equivalent).
        metrics_path: Path to the training metrics sidecar, used to pull the
            exact training window (``data.start``, ``data.end``).
        history_path: Currently only accepted for forward-compat; when set to a
            ``.parquet`` we verify the file exists but we still rebuild the
            bundle from raw (the training pipeline does not serialise a
            reusable history parquet yet). Passing a non-existent path
            triggers a clean error rather than a silent skip.
        out_path: If set, write the JSON report here; otherwise the caller
            gets only the in-memory dict.
        device: Torch device for the forward pass.
        batch_size: DataLoader batch size.

    Returns:
        The eval report dict (same schema as
        :func:`compute_stress_window_mae`).

    Raises:
        FileNotFoundError: If the checkpoint is missing, or ``history_path``
            was provided but does not exist.
        RuntimeError: If required raw data to rebuild the bundle is missing —
            the error message points at ``scripts/pull_all.sh`` and
            ``scripts/train.py`` rather than silently skipping.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt_path}. Train it with .venv/bin/python "
            "scripts/train.py first."
        )

    if history_path is not None:
        hp = Path(history_path)
        if not hp.exists():
            raise FileNotFoundError(
                f"--history pointed at {hp}, but that file does not exist. "
                "Either omit --history (to rebuild the bundle from data/raw/), "
                "run `bash scripts/pull_all.sh` to populate raw data, or run "
                "`.venv/bin/python scripts/train.py` to regenerate the training "
                "artefacts."
            )
        logger.info("history parquet located at %s (informational; using raw rebuild)", hp)

    predictor = load_predictor(ckpt_path=ckpt_path, metrics_path=metrics_path, device=device)

    # Pull training window from the sidecar so we reproduce the same split.
    metrics_raw: dict[str, Any] = {}
    try:
        metrics_raw = json.loads(Path(metrics_path).read_text())
    except FileNotFoundError:
        logger.warning(
            "metrics sidecar %s not found — using feature-pipeline defaults", metrics_path
        )
    data_block = metrics_raw.get("data") or {}
    start = str(data_block.get("start") or "2022-06-01")
    end = str(data_block.get("end") or "2023-10-01")

    raw_root = Path(__file__).resolve().parents[2] / "data" / "raw"
    missing = _check_raw_data_present(raw_root)
    if missing:
        raise RuntimeError(_missing_raw_data_error(raw_root, missing))

    logger.info("rebuilding feature bundle: %s → %s", start, end)
    bundle = build_hourly_features(start=start, end=end)

    report = compute_stress_window_mae(
        predictor,
        bundle,
        t_in=predictor.config.t_in,
        t_out=predictor.config.t_out,
        batch_size=batch_size,
        device=device,
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2, sort_keys=True))
        tmp.replace(out_path)
        logger.info("wrote eval report to %s", out_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m gridsense.eval",
        description=(
            "Evaluate the trained GridSense-AZ checkpoint on the test split, "
            "with an explicit stress-window (Phoenix summer evenings, "
            "Jun-Sep 17:00-21:00 MST) MAE breakout alongside the overall MAE."
        ),
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=DEFAULT_CKPT_PATH,
        help="Path to the trained checkpoint (default: data/models/gwnet_v0.pt).",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to training metrics sidecar (default: data/models/metrics.json).",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help=(
            "Optional path to a history parquet. If provided and missing, the "
            "CLI exits with a clean error pointing at scripts/pull_all.sh / "
            "scripts/train.py. The current training pipeline does not write a "
            "reusable history parquet — we rebuild from data/raw/ either way."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "data"
            / "models"
            / "eval_report.json"
        ),
        help="Where to write the JSON report (default: data/models/eval_report.json).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for the forward pass (default: cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="DataLoader batch size (default: 32).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        report = run_eval(
            ckpt_path=args.ckpt,
            metrics_path=args.metrics,
            history_path=args.history,
            out_path=args.out,
            device=args.device,
            batch_size=args.batch_size,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    # Human-friendly summary (the JSON report is the programmatic output).
    print("=" * 72)
    print("GridSense-AZ stress-window evaluation")
    print("=" * 72)
    print(f"  stress window: {report['stress_window_definition']}")
    print(
        f"  overall MAE    : {report['overall_mae_kw']:>10.1f} kW   "
        f"(persistence {report['persistence_overall_mae_kw']:.1f} kW, "
        f"{report['improvement_overall_pct']:+.2f}%)"
    )
    print(
        f"  stress MAE     : {report['stress_mae_kw']:>10.1f} kW   "
        f"(persistence {report['persistence_stress_mae_kw']:.1f} kW, "
        f"{report['improvement_stress_pct']:+.2f}%)"
    )
    print(
        f"  stress coverage: {report['stress_hours']:,} / "
        f"{report['total_hours']:,} hours"
    )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
