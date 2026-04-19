"""Smoke tests for ``gridsense.models.gwnet``.

Guards:
- shape correctness of a tiny forward pass,
- pinball-loss sanity (non-negative + monotone in the residual),
- one SGD step actually reduces the loss vs. random init.
"""

from __future__ import annotations

import torch

from gridsense.models.gwnet import GraphWaveNetQuantile, pinball_loss


def _tiny_model(horizon: int = 6, nodes: int = 5, hidden: int = 8) -> GraphWaveNetQuantile:
    torch.manual_seed(42)
    return GraphWaveNetQuantile(
        num_nodes=nodes,
        adj_init=None,
        input_dim=5,
        hidden_dim=hidden,
        num_blocks=3,
        kernel_size=2,
        dilation_growth=2,
        horizon=horizon,
        quantile_levels=(0.1, 0.5, 0.9),
        dropout=0.0,
        adaptive_adj_dim=4,
        skip_dim=16,
        end_dim=16,
        gcn_order=2,
    )


def test_model_forward_shape() -> None:
    """GraphWaveNetQuantile(tiny) on zeros emits [B, H, N, Q]."""
    model = _tiny_model(horizon=6, nodes=5, hidden=8)
    model.eval()
    x = torch.zeros(2, 24, 5, 5)  # [B, T_in, N, F_in]
    y = model(x)
    assert y.shape == (2, 6, 5, 3), f"unexpected shape {tuple(y.shape)}"
    assert torch.isfinite(y).all()


def test_pinball_loss_monotone() -> None:
    """Pinball loss is non-negative, zero at match, and grows with residual."""
    q = (0.1, 0.5, 0.9)
    target = torch.zeros(4, 6, 5)  # [B, H, N]

    # Exact match -> zero loss.
    pred_exact = torch.zeros(4, 6, 5, 3)
    l0 = pinball_loss(pred_exact, target, q)
    assert l0.item() == 0.0

    # Small miss vs. big miss.
    pred_small = torch.full((4, 6, 5, 3), 0.1)
    pred_big = torch.full((4, 6, 5, 3), 1.0)
    l_small = pinball_loss(pred_small, target, q).item()
    l_big = pinball_loss(pred_big, target, q).item()
    assert l_small >= 0.0 and l_big >= 0.0
    assert l_big > l_small, "bigger residual must yield larger pinball loss"


def test_model_trainable() -> None:
    """One SGD step reduces pinball loss vs. random init."""
    torch.manual_seed(123)
    model = _tiny_model(horizon=6, nodes=5, hidden=8)
    model.train()

    b, t_in, n, f_in = 4, 24, 5, 5
    x = torch.randn(b, t_in, n, f_in)
    target = torch.randn(b, 6, n)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    q = (0.1, 0.5, 0.9)

    # Baseline loss.
    with torch.no_grad():
        loss_0 = pinball_loss(model(x), target, q).item()

    # A few training steps to smooth noise from dropout/bn.
    for _ in range(5):
        opt.zero_grad()
        loss = pinball_loss(model(x), target, q)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        loss_1 = pinball_loss(model(x), target, q).item()

    assert loss_1 < loss_0, f"loss did not decrease: {loss_0:.6f} -> {loss_1:.6f}"


def test_importable() -> None:
    """Keep the original import-smoke guard."""
    import gridsense.models  # noqa: F401
    import gridsense.models.dcrnn  # noqa: F401
    import gridsense.models.gwnet  # noqa: F401
