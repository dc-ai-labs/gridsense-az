#!/usr/bin/env python3
"""GridSense-AZ — train GWNet on real AZ summer data.

Pipeline:
    1. Build hourly features via :func:`gridsense.features.build_hourly_features`.
    2. Build train/val/test loaders via :func:`gridsense.model.make_dataloader`.
    3. Fit a :class:`gridsense.model.GWNet` with pinball loss.
    4. Evaluate on the test loader at the p50 quantile; compare against a
       flat-persistence baseline (``y_hat[t+h] = y[t-1]`` for every horizon step).
    5. Persist ``gwnet_v0.pt``, ``metrics.json`` (includes full config dict),
       and ``history.json`` under ``--out-dir``.

The CLI defaults cover two Arizona summer peaks (2022-06-01 → 2023-10-01,
~11.7k hourly steps × 132 buses). Training runs on CPU by default; expect
10-20 min for 15 epochs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Make ``src/`` importable when invoked as ``python scripts/train.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
for _p in (_SRC, _REPO_ROOT):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from gridsense.features import build_hourly_features  # noqa: E402
from gridsense.model import (  # noqa: E402
    GWNet,
    GWNetConfig,
    fit,
    make_dataloader,
    predict,
)

__all__ = ["run", "main"]

logger = logging.getLogger("gridsense.train")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_START = "2022-06-01"
DEFAULT_END = "2023-10-01"
DEFAULT_EPOCHS = 15
DEFAULT_OUT_DIR = "data/models"
DEFAULT_DEVICE = "cpu"
DEFAULT_T_IN = 24
DEFAULT_T_OUT = 6
DEFAULT_BATCH = 32
DEFAULT_LR = 1e-3
DEFAULT_SCHEDULER = "cosine"
DEFAULT_WARMUP_EPOCHS = 10

CKPT_NAME = "gwnet_v0.pt"
METRICS_NAME = "metrics.json"
HISTORY_NAME = "history.json"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _p50_mae_from_predict(pred: torch.Tensor, target: torch.Tensor) -> float:
    """MAE of the p50 quantile (middle slot) vs ground truth."""
    # pred: [W, T_out, N, Q], target: [W, T_out, N]
    q = pred.shape[-1]
    p50 = pred[..., q // 2]
    return float(torch.mean(torch.abs(p50 - target)).item())


def _persistence_mae(loader: torch.utils.data.DataLoader) -> float:
    """Flat persistence baseline: ``y_hat[t+h] = y[t-1]`` for every horizon step.

    For each window in ``loader``, the forecast for every horizon step equals
    the last observed target value at the last input time step. Because the
    bundle stores the target in ``X_node[..., 0]`` (z-scored) we instead
    read the raw load from the window's own target path: we pull the last
    input step's target from the loader's underlying dataset via ``y[t_in - 1]``,
    computed from the bundle's raw ``y_kw`` at the appropriate offset.

    We use the simpler route: the dataset's ``x_node`` at the last timestep is
    the standardised target, but we want **raw** y_kw. The bundle stores
    ``y_kw`` directly on the dataset; we index it using each window's start.
    """
    ds = loader.dataset  # _BundleWindowDataset
    y_raw = ds.y  # torch [T, N] — raw y_kw
    t_in = int(ds.t_in)
    t_out = int(ds.t_out)
    starts = np.asarray(ds.starts)
    total_abs = 0.0
    total_count = 0
    for i in starts:
        i = int(i)
        last_input = y_raw[i + t_in - 1]  # [N]
        target_window = y_raw[i + t_in : i + t_in + t_out]  # [t_out, N]
        pred_window = last_input.unsqueeze(0).expand_as(target_window)
        total_abs += float(torch.abs(pred_window - target_window).sum().item())
        total_count += int(target_window.numel())
    if total_count == 0:
        return float("nan")
    return total_abs / total_count


def _p50_mae_raw(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_mean: float,
    y_std: float,
) -> float:
    """Decode model p50 predictions from z-space back to raw kW, then MAE.

    The model is trained on ``y_kw`` (raw kW) directly — features.py only
    z-scores the per-bus **input channel** (``X_node[..., 0]``), not the
    target. So ``pred`` is already in kW and we can compare to ``target``
    directly. ``y_mean``/``y_std`` kept in the signature for forward
    compatibility if the feature pipeline starts standardising the target.
    """
    del y_mean, y_std  # currently unused — target is raw kW by contract
    return _p50_mae_from_predict(pred, target)


# ---------------------------------------------------------------------------
# Core entry point
# ---------------------------------------------------------------------------


def run(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    epochs: int = DEFAULT_EPOCHS,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    device: str = DEFAULT_DEVICE,
    t_in: int = DEFAULT_T_IN,
    t_out: int = DEFAULT_T_OUT,
    batch_size: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    resume: bool = False,
    seed: int = 1337,
    scheduler: str = "none",
    warmup_epochs: int = 0,
) -> dict[str, Any]:
    """Train GWNet on the given window and save metrics/checkpoint.

    Returns the metrics dict (also written to ``{out_dir}/metrics.json``).
    """
    out_path = Path(out_dir)
    if not out_path.is_absolute():
        out_path = (_REPO_ROOT / out_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    logger.info("building features: %s → %s", start, end)
    bundle = build_hourly_features(start=start, end=end)
    logger.info(
        "bundle: T=%d N=%d F_exog=%d F_node=%d source=%s (%.1fs)",
        len(bundle.times),
        len(bundle.node_names),
        bundle.X_exog.shape[1],
        bundle.X_node.shape[2],
        bundle.meta.get("source"),
        time.time() - t0,
    )

    train_loader = make_dataloader(
        bundle, t_in=t_in, t_out=t_out, batch_size=batch_size, which="train"
    )
    val_loader = make_dataloader(
        bundle, t_in=t_in, t_out=t_out, batch_size=batch_size, which="val"
    )
    test_loader = make_dataloader(
        bundle, t_in=t_in, t_out=t_out, batch_size=batch_size, which="test"
    )
    logger.info(
        "loaders: train=%d val=%d test=%d windows",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    cfg = GWNetConfig(
        n_nodes=len(bundle.node_names),
        f_node=bundle.X_node.shape[2],
        f_exog=bundle.X_exog.shape[1],
        t_in=t_in,
        t_out=t_out,
    )
    model = GWNet(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model: %d params, config=%s", n_params, asdict(cfg))

    ckpt_path = out_path / CKPT_NAME
    if resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        logger.info("resumed from %s", ckpt_path)

    t1 = time.time()
    history = fit(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        scheduler=scheduler,
        warmup_epochs=warmup_epochs,
    )
    train_secs = time.time() - t1
    logger.info("fit complete in %.1fs (%.2fs/epoch)", train_secs, train_secs / max(epochs, 1))

    # ---- evaluation --------------------------------------------------------
    y_mean, y_std = bundle.scalers.get("y_kw", (0.0, 1.0))

    def _eval_loader(dl: torch.utils.data.DataLoader) -> float:
        pred, target = predict(model, dl, device=device, sort_quantiles=True)
        if pred.numel() == 0:
            return float("nan")
        return _p50_mae_raw(pred, target, y_mean, y_std)

    train_mae = _eval_loader(train_loader)
    val_mae = _eval_loader(val_loader)
    test_mae = _eval_loader(test_loader)
    baseline_mae = _persistence_mae(test_loader)

    if baseline_mae > 0 and np.isfinite(baseline_mae):
        improvement_pct = 100.0 * (baseline_mae - test_mae) / baseline_mae
    else:
        improvement_pct = float("nan")

    # ---- persist -----------------------------------------------------------
    torch.save(model.state_dict(), ckpt_path)
    logger.info("saved checkpoint %s", ckpt_path)

    metrics: dict[str, Any] = {
        "train_mae": float(train_mae),
        "val_mae": float(val_mae),
        "test_mae": float(test_mae),
        "baseline_mae": float(baseline_mae),
        "improvement_pct": float(improvement_pct),
        "epochs": int(epochs),
        "train_seconds": float(train_secs),
        "n_params": int(n_params),
        "config": asdict(cfg),
        "data": {
            "start": start,
            "end": end,
            "source": bundle.meta.get("source"),
            "n_times": len(bundle.times),
            "n_nodes": len(bundle.node_names),
            "y_scaler": [float(y_mean), float(y_std)],
            "train_windows": len(train_loader.dataset),
            "val_windows": len(val_loader.dataset),
            "test_windows": len(test_loader.dataset),
        },
        "hyper": {
            "lr": float(lr),
            "batch_size": int(batch_size),
            "t_in": int(t_in),
            "t_out": int(t_out),
            "device": device,
            "seed": int(seed),
            "scheduler": str(scheduler),
            "warmup_epochs": int(warmup_epochs),
        },
    }
    (out_path / METRICS_NAME).write_text(json.dumps(metrics, indent=2, sort_keys=True))
    (out_path / HISTORY_NAME).write_text(json.dumps(history, indent=2, sort_keys=True))
    logger.info("wrote %s and %s", out_path / METRICS_NAME, out_path / HISTORY_NAME)

    # ---- summary -----------------------------------------------------------
    _print_summary(metrics)
    return metrics


def _print_summary(metrics: dict[str, Any]) -> None:
    """Terse summary table for the console."""
    imp = metrics["improvement_pct"]
    imp_str = f"{imp:+.2f}%" if np.isfinite(imp) else "n/a"
    rows = [
        ("train MAE (kW)", f"{metrics['train_mae']:.2f}"),
        ("val   MAE (kW)", f"{metrics['val_mae']:.2f}"),
        ("test  MAE (kW)", f"{metrics['test_mae']:.2f}"),
        ("persistence MAE (kW)", f"{metrics['baseline_mae']:.2f}"),
        ("improvement vs persistence", imp_str),
        ("epochs", str(metrics["epochs"])),
        ("params", f"{metrics['n_params']:,}"),
        ("train secs", f"{metrics['train_seconds']:.1f}"),
    ]
    width = max(len(k) for k, _ in rows)
    print("=" * (width + 20))
    for k, v in rows:
        print(f"{k:<{width}}  {v}")
    print("=" * (width + 20))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train GWNet on GridSense-AZ features.")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--t-in", type=int, default=DEFAULT_T_IN)
    ap.add_argument("--t-out", type=int, default=DEFAULT_T_OUT)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument(
        "--scheduler",
        choices=("none", "cosine"),
        default=DEFAULT_SCHEDULER,
        help="LR schedule (default: cosine). 'none' preserves constant LR.",
    )
    ap.add_argument(
        "--warmup-epochs",
        type=int,
        default=DEFAULT_WARMUP_EPOCHS,
        help="Linear warmup epochs when --scheduler=cosine (default: 10).",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(
        start=args.start,
        end=args.end,
        epochs=args.epochs,
        out_dir=args.out_dir,
        device=args.device,
        t_in=args.t_in,
        t_out=args.t_out,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
        seed=args.seed,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
