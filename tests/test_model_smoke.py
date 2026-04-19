"""Smoke tests for :mod:`gridsense.model`.

Covers the contract that ``scripts/train.py`` (next SDE) will rely on:
- importability of the public API,
- ``GWNetConfig`` defaults,
- forward-pass shapes + parameter budget,
- pinball-loss sign / zero-at-match / quantile symmetry,
- dataloader window shapes + time-based split + no-temporal-leakage guard,
- a 2-epoch ``fit(...)`` actually reduces the loss on a tiny bundle,
- the monotonic-quantile guard in ``predict(...)``.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from gridsense.model import (
    GWNet,
    GWNetConfig,
    fit,
    make_dataloader,
    pinball_loss,
    predict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_bundle(
    total_steps: int = 300,
    n_nodes: int = 132,
    f_node: int = 1,
    f_exog: int = 11,
    seed: int = 0,
) -> types.SimpleNamespace:
    """Build a duck-typed FeatureBundle for dataloader / training tests.

    Signals are smooth (diurnal + random walk) so a tiny model can fit them
    in very few epochs — keeps ``test_fit_decreases_loss_cpu`` fast.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(total_steps, dtype=np.float32)
    base = np.sin(2 * np.pi * t / 24.0) + 2.0
    # Per-bus modulation + noise.
    node_scale = rng.uniform(0.5, 1.5, size=n_nodes).astype(np.float32)
    y = np.outer(base, node_scale).astype(np.float32)
    y += rng.normal(scale=0.05, size=y.shape).astype(np.float32)

    x_node = np.stack([y], axis=-1) if f_node == 1 else rng.normal(
        size=(total_steps, n_nodes, f_node)
    ).astype(np.float32)

    # Exog: diurnal + weather-ish noise columns.
    x_exog = np.zeros((total_steps, f_exog), dtype=np.float32)
    x_exog[:, 0] = np.sin(2 * np.pi * t / 24.0)
    x_exog[:, 1] = np.cos(2 * np.pi * t / 24.0)
    x_exog[:, 2:] = rng.normal(size=(total_steps, f_exog - 2)).astype(np.float32)

    node_names = [f"bus{i}" for i in range(n_nodes)]
    times = np.array(t, dtype=np.int64)
    return types.SimpleNamespace(
        X_exog=x_exog, X_node=x_node, y_kw=y, node_names=node_names, times=times
    )


# ---------------------------------------------------------------------------
# Static-surface tests
# ---------------------------------------------------------------------------


def test_importable() -> None:
    """Every documented public symbol is importable."""
    import gridsense.model as m  # noqa: F401

    for name in ("GWNet", "GWNetConfig", "pinball_loss", "fit", "predict", "make_dataloader"):
        assert hasattr(m, name), f"missing public symbol: {name}"


def test_config_defaults() -> None:
    """GWNetConfig defaults match the manager's contract."""
    cfg = GWNetConfig()
    assert cfg.n_nodes == 132
    assert cfg.f_node == 1
    assert cfg.f_exog == 11
    assert cfg.t_in == 24
    assert cfg.t_out == 6
    assert cfg.d_hidden == 32
    assert cfg.n_blocks == 4
    assert cfg.n_layers_per_block == 2
    assert cfg.quantiles == (0.1, 0.5, 0.9)
    assert cfg.learn_adj is True


# ---------------------------------------------------------------------------
# Forward / loss tests
# ---------------------------------------------------------------------------


def test_forward_shapes() -> None:
    """Forward on a [4, 24, 132, 1] + [4, 24, 11] batch returns [4, 6, 132, 3]."""
    torch.manual_seed(0)
    model = GWNet(GWNetConfig())
    x_node = torch.randn(4, 24, 132, 1)
    x_exog = torch.randn(4, 24, 11)
    y = model(x_node, x_exog)
    assert y.shape == (4, 6, 132, 3), f"unexpected shape {tuple(y.shape)}"
    assert torch.isfinite(y).all()


def test_param_count_reasonable() -> None:
    """The default model must fit a sub-500k-param budget for CPU training."""
    model = GWNet(GWNetConfig())
    n = sum(p.numel() for p in model.parameters())
    assert n < 500_000, f"param count {n} too large"
    # And not degenerately tiny either.
    assert n > 5_000, f"param count {n} suspiciously small"


def test_pinball_loss_perfect() -> None:
    """pred == target on every quantile -> loss exactly zero."""
    q = (0.1, 0.5, 0.9)
    target = torch.randn(2, 6, 10)
    pred = target.unsqueeze(-1).expand(-1, -1, -1, len(q)).contiguous()
    loss = pinball_loss(pred, target, q)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_pinball_loss_positive_at_wrong() -> None:
    """pred != target yields a strictly positive loss that grows with error."""
    q = (0.1, 0.5, 0.9)
    target = torch.zeros(3, 4, 7)
    small = torch.full((3, 4, 7, 3), 0.1)
    big = torch.full((3, 4, 7, 3), 1.0)
    l_small = pinball_loss(small, target, q).item()
    l_big = pinball_loss(big, target, q).item()
    assert l_small > 0.0
    assert l_big > l_small


# ---------------------------------------------------------------------------
# Dataloader tests
# ---------------------------------------------------------------------------


def test_make_dataloader_shapes() -> None:
    """Batch shapes on a synthetic (T=300, N=132, Fx=11, Fn=1) bundle."""
    bundle = _synthetic_bundle(total_steps=300)
    train = make_dataloader(bundle, t_in=24, t_out=6, batch_size=8, which="train")
    val = make_dataloader(bundle, t_in=24, t_out=6, batch_size=8, which="val")
    test = make_dataloader(bundle, t_in=24, t_out=6, batch_size=8, which="test")
    assert len(train.dataset) > 0
    assert len(val.dataset) > 0
    assert len(test.dataset) > 0

    x_node, x_exog, y = next(iter(train))
    assert x_node.shape[1:] == (24, 132, 1)
    assert x_exog.shape[1:] == (24, 11)
    assert y.shape[1:] == (6, 132)
    assert x_node.shape[0] == x_exog.shape[0] == y.shape[0]


def test_no_temporal_leakage() -> None:
    """last_train + t_in + t_out <= first_val (and same for val -> test)."""
    bundle = _synthetic_bundle(total_steps=300)
    t_in, t_out = 24, 6
    train_loader = make_dataloader(bundle, t_in=t_in, t_out=t_out, which="train")
    val_loader = make_dataloader(bundle, t_in=t_in, t_out=t_out, which="val")
    test_loader = make_dataloader(bundle, t_in=t_in, t_out=t_out, which="test")
    train_starts = np.asarray(train_loader.dataset.starts)
    val_starts = np.asarray(val_loader.dataset.starts)
    test_starts = np.asarray(test_loader.dataset.starts)
    assert train_starts.size > 0 and val_starts.size > 0 and test_starts.size > 0
    # Train's last window consumes starts through last_train + t_in + t_out - 1.
    # First val window starts at first_val. No overlap means:
    assert int(train_starts.max()) + t_in + t_out <= int(val_starts.min()), (
        f"train→val leakage: last_train={train_starts.max()}, "
        f"first_val={val_starts.min()}"
    )
    assert int(val_starts.max()) + t_in + t_out <= int(test_starts.min()), (
        f"val→test leakage: last_val={val_starts.max()}, "
        f"first_test={test_starts.min()}"
    )


# ---------------------------------------------------------------------------
# Fit / predict tests
# ---------------------------------------------------------------------------


def test_fit_decreases_loss_cpu() -> None:
    """2 epochs of fit() on a tiny bundle should reduce the training loss."""
    torch.manual_seed(0)
    bundle = _synthetic_bundle(total_steps=200)
    train_loader = make_dataloader(
        bundle, t_in=24, t_out=6, batch_size=8, which="train", shuffle=False
    )

    cfg = GWNetConfig(d_hidden=16, n_blocks=2, n_layers_per_block=1, dropout=0.0)
    model = GWNet(cfg)

    # Baseline loss before any fit() step.
    quantiles = tuple(cfg.quantiles)
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        x_node, x_exog, y = batch
        initial = float(pinball_loss(model(x_node, x_exog), y, quantiles))

    history = fit(
        model,
        train_loader=train_loader,
        val_loader=None,
        epochs=2,
        lr=1e-2,
        device="cpu",
    )
    assert "train_loss" in history and len(history["train_loss"]) == 2
    final = history["train_loss"][-1]
    assert final < initial, f"loss did not decrease: initial={initial:.4f} final={final:.4f}"


def test_fit_cosine_schedule_with_warmup() -> None:
    """``scheduler='cosine'`` ramps LR up over warmup_epochs then anneals.

    Default ``fit()`` behaviour (no scheduler) must also leave ``history``
    without an ``'lr'`` key — the legacy contract is unchanged.
    """
    torch.manual_seed(0)
    bundle = _synthetic_bundle(total_steps=200)
    train_loader = make_dataloader(
        bundle, t_in=24, t_out=6, batch_size=8, which="train", shuffle=False
    )
    cfg = GWNetConfig(d_hidden=16, n_blocks=2, n_layers_per_block=1, dropout=0.0)

    # Legacy path: no scheduler kwargs -> no 'lr' in history.
    model_legacy = GWNet(cfg)
    hist_legacy = fit(
        model_legacy, train_loader=train_loader, val_loader=None, epochs=2, lr=1e-3
    )
    assert "lr" not in hist_legacy, "legacy fit() must not record LR history"

    # Cosine + warmup: LR starts low, peaks at target around end of warmup,
    # then anneals toward 0.
    torch.manual_seed(0)
    model_cos = GWNet(cfg)
    target_lr = 2e-3
    warmup = 2
    epochs = 6
    hist_cos = fit(
        model_cos,
        train_loader=train_loader,
        val_loader=None,
        epochs=epochs,
        lr=target_lr,
        scheduler="cosine",
        warmup_epochs=warmup,
    )
    lrs = hist_cos["lr"]
    assert len(lrs) == epochs
    # Warmup: first-epoch LR is below target.
    assert lrs[0] < target_lr
    # Peak sits at the warmup boundary (epoch index == warmup).
    assert lrs[warmup] == pytest.approx(target_lr, rel=1e-6)
    # Anneal: final epoch LR is well below the peak.
    assert lrs[-1] < lrs[warmup]


def test_predict_monotonic_quantiles() -> None:
    """After predict(..., sort_quantiles=True): pred[...,0] <= pred[...,1] <= pred[...,2]."""
    torch.manual_seed(0)
    bundle = _synthetic_bundle(total_steps=200)
    loader = make_dataloader(
        bundle, t_in=24, t_out=6, batch_size=4, which="val", shuffle=False
    )
    cfg = GWNetConfig(d_hidden=16, n_blocks=2, n_layers_per_block=1, dropout=0.0)
    model = GWNet(cfg)
    pred, target = predict(model, loader, device="cpu", sort_quantiles=True)
    assert pred.dim() == 4 and pred.shape[-1] == len(cfg.quantiles)
    assert target.shape == pred.shape[:-1]
    # Monotonic along quantile axis.
    diff_01 = pred[..., 1] - pred[..., 0]
    diff_12 = pred[..., 2] - pred[..., 1]
    assert (diff_01 >= -1e-6).all()
    assert (diff_12 >= -1e-6).all()
