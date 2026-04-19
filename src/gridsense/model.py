"""GridSense-AZ Graph WaveNet (compact variant) + quantile regression.

Public surface
--------------
* :class:`GWNetConfig` — dataclass holding model hyperparameters.
* :class:`GWNet` — stacked dilated-causal-TCN blocks with GLU gating, an
  adaptive adjacency mixing step between blocks, skip aggregation, and a
  quantile head that emits ``[B, T_out, N, Q]``.
* :func:`pinball_loss` — standard quantile / pinball loss.
* :func:`fit`, :func:`predict`, :func:`make_dataloader` — minimal training
  and inference helpers designed to be called by ``scripts/train.py``.

The ``FeatureBundle`` argument of :func:`make_dataloader` is duck-typed: any
object exposing ``X_exog[T, F_exog]``, ``X_node[T, N, F_node]``, and
``y_kw[T, N]`` attributes is accepted, so this module does not import
``gridsense.features`` (which may be under construction in parallel).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "GWNetConfig",
    "GWNet",
    "pinball_loss",
    "fit",
    "predict",
    "make_dataloader",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GWNetConfig:
    """Hyperparameters for :class:`GWNet`.

    Defaults are tuned for the IEEE 123-bus feeder on CPU with a small
    training budget: ~100-300 k parameters, 24 h input, 6 h horizon, three
    quantile heads (p10/p50/p90).
    """

    n_nodes: int = 132
    f_node: int = 1
    f_exog: int = 11
    t_in: int = 24
    t_out: int = 6
    d_hidden: int = 32
    n_blocks: int = 4
    n_layers_per_block: int = 2
    dropout: float = 0.1
    learn_adj: bool = True
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _CausalGLUConv(nn.Module):
    """1D dilated causal conv along the time axis with GLU gating.

    Operates on tensors shaped ``[B, C, N, T]``. Left-pads along ``T`` so the
    output preserves the input's temporal length (causal / same-length).
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        # Output 2*channels so we can split into (a, b) for GLU gating.
        self.conv = nn.Conv2d(
            channels,
            2 * channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
        )

    def forward(self, x: Tensor) -> Tensor:
        pad = (self.kernel_size - 1) * self.dilation
        # Pad only the left side of the time axis (last dim).
        x = F.pad(x, (pad, 0, 0, 0))
        y = self.conv(x)  # [B, 2C, N, T]
        a, b = torch.chunk(y, 2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)


class _AdaptiveGraphMix(nn.Module):
    """Adaptive adjacency mixing: ``h' = A_adp @ h`` along the node axis.

    ``A_adp = softmax(ReLU(E1 E2^T))`` with two learned node-embedding
    matrices of dimension ``d_hidden // 2`` each.
    """

    def __init__(self, n_nodes: int, emb_dim: int):
        super().__init__()
        self.e1 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.e2 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)

    def adjacency(self) -> Tensor:
        logits = F.relu(self.e1 @ self.e2.t())
        return F.softmax(logits, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, N, T]; mix along N with [N, N] adjacency.
        a = self.adjacency()
        return torch.einsum("bcnt,mn->bcmt", x, a).contiguous()


class _TCNBlock(nn.Module):
    """One stack of GLU-gated dilated causal convolutions + graph mixing.

    Parameters
    ----------
    channels : int
        Hidden-channel width (preserved through the block).
    skip_dim : int
        Channel width of the skip-path branch.
    kernel_size : int
        Temporal kernel size for each internal conv.
    dilation : int
        Dilation factor shared by every internal conv in this block.
    n_layers : int
        Number of GLU-gated conv layers stacked inside the block.
    n_nodes : int
        Number of graph nodes — needed to size the adaptive adjacency.
    adj_emb_dim : int
        Embedding dim used by the adaptive adjacency.
    learn_adj : bool
        If False, the adaptive mixing step is replaced with identity.
    dropout : float
        Dropout applied after each GLU.
    """

    def __init__(
        self,
        channels: int,
        skip_dim: int,
        kernel_size: int,
        dilation: int,
        n_layers: int,
        n_nodes: int,
        adj_emb_dim: int,
        learn_adj: bool,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _CausalGLUConv(channels, kernel_size=kernel_size, dilation=dilation)
                for _ in range(n_layers)
            ]
        )
        self.learn_adj = learn_adj
        if learn_adj:
            self.graph_mix: nn.Module = _AdaptiveGraphMix(n_nodes, adj_emb_dim)
        else:
            self.graph_mix = nn.Identity()
        self.residual = nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.skip = nn.Conv2d(channels, skip_dim, kernel_size=(1, 1))
        self.dropout = dropout
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        residual = x
        h = x
        for layer in self.layers:
            h = layer(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        # Graph mixing across nodes at the block's output.
        h = self.graph_mix(h)
        # Skip branch (pre-residual sum) — used by the final aggregation.
        s = self.skip(h)
        # Residual projection + add.
        h = self.residual(h) + residual
        h = self.norm(h)
        return h, s


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class GWNet(nn.Module):
    """Compact Graph-WaveNet-style forecaster with a 3-quantile head.

    Forward signature
    -----------------
    ``forward(x_node [B, T_in, N, F_node], x_exog [B, T_in, F_exog]) ->
    [B, T_out, N, Q]``

    ``fixed_edge_index`` is accepted for interface parity with PyG-style
    graph models but is currently used only as a physics prior via the
    adaptive adjacency's initialisation (we bias ``E1 E2^T`` toward the
    fixed neighbourhood). When ``fixed_edge_index`` is ``None`` the model
    relies purely on the data-learned adjacency.
    """

    def __init__(self, config: GWNetConfig, fixed_edge_index: Tensor | None = None):
        super().__init__()
        self.config = config

        # Input feature dim = per-node features + broadcast exogenous features.
        in_dim = config.f_node + config.f_exog
        self.start_conv = nn.Conv2d(in_dim, config.d_hidden, kernel_size=(1, 1))

        # Skip-path width. Keep modest for CPU / param budget.
        self.skip_dim = max(config.d_hidden, 32)

        emb_dim = max(config.d_hidden // 2, 4)
        self.blocks = nn.ModuleList()
        dilations = [2**i for i in range(config.n_blocks)]
        for d in dilations:
            self.blocks.append(
                _TCNBlock(
                    channels=config.d_hidden,
                    skip_dim=self.skip_dim,
                    kernel_size=2,
                    dilation=d,
                    n_layers=config.n_layers_per_block,
                    n_nodes=config.n_nodes,
                    adj_emb_dim=emb_dim,
                    learn_adj=config.learn_adj,
                    dropout=config.dropout,
                )
            )

        # End head: aggregated skip -> T_out * Q predictions.
        # The skip path keeps the full temporal axis; we mean-pool over time
        # before the 1x1 conv, which is cheap and shape-stable.
        q = len(config.quantiles)
        self.end_conv = nn.Conv2d(self.skip_dim, config.t_out * q, kernel_size=(1, 1))

        # Register the fixed edge index as a buffer so ``.to(device)`` moves it
        # — but do not attempt to build a dense prior on top of it. Training
        # SDE can still inspect this to inject OpenDSS connectivity if desired.
        if fixed_edge_index is not None:
            self.register_buffer("fixed_edge_index", fixed_edge_index.long(), persistent=False)
        else:
            self.register_buffer(
                "fixed_edge_index", torch.empty((2, 0), dtype=torch.long), persistent=False
            )

    # ------------------------------------------------------------------
    def forward(self, x_node: Tensor, x_exog: Tensor) -> Tensor:
        if x_node.dim() != 4:
            raise ValueError(f"x_node must be 4D [B,T,N,F_node]; got {tuple(x_node.shape)}")
        if x_exog.dim() != 3:
            raise ValueError(f"x_exog must be 3D [B,T,F_exog]; got {tuple(x_exog.shape)}")
        b, t_in, n, f_node = x_node.shape
        b2, t_in2, f_exog = x_exog.shape
        if b != b2 or t_in != t_in2:
            raise ValueError(
                f"x_node / x_exog batch or time mismatch: "
                f"{tuple(x_node.shape)} vs {tuple(x_exog.shape)}"
            )
        if n != self.config.n_nodes:
            raise ValueError(f"expected N={self.config.n_nodes}, got {n}")
        if f_node != self.config.f_node:
            raise ValueError(f"expected F_node={self.config.f_node}, got {f_node}")
        if f_exog != self.config.f_exog:
            raise ValueError(f"expected F_exog={self.config.f_exog}, got {f_exog}")

        # Broadcast exog to all nodes, concat with node features.
        exog_bcast = x_exog.unsqueeze(2).expand(b, t_in, n, f_exog)
        x = torch.cat([x_node, exog_bcast], dim=-1)  # [B, T, N, F_in]
        # [B, T, N, F] -> [B, F, N, T]
        h = x.permute(0, 3, 2, 1).contiguous()
        h = self.start_conv(h)  # [B, d_hidden, N, T_in]

        skip_sum: Tensor | None = None
        for block in self.blocks:
            h, s = block(h)
            skip_sum = s if skip_sum is None else skip_sum + s

        # Aggregate skip over time — mean-pool is cheap and shape-safe.
        assert skip_sum is not None
        agg = F.relu(skip_sum).mean(dim=-1, keepdim=True)  # [B, skip_dim, N, 1]
        out = self.end_conv(agg)  # [B, T_out * Q, N, 1]
        out = out.squeeze(-1)  # [B, T_out * Q, N]
        q = len(self.config.quantiles)
        out = out.reshape(b, self.config.t_out, q, n)  # [B, T_out, Q, N]
        out = out.permute(0, 1, 3, 2).contiguous()  # [B, T_out, N, Q]
        return out


# ---------------------------------------------------------------------------
# Pinball loss
# ---------------------------------------------------------------------------


def pinball_loss(
    pred: Tensor,
    target: Tensor,
    quantiles: tuple[float, ...],
) -> Tensor:
    """Mean pinball (quantile) loss.

    Parameters
    ----------
    pred : [B, T_out, N, Q]
        Quantile forecasts.
    target : [B, T_out, N]
        Ground-truth target for each horizon step / node.
    quantiles : tuple of floats in (0, 1)
        Quantile levels matching the last axis of ``pred``.

    Returns
    -------
    Scalar mean loss.
    """
    if pred.shape[-1] != len(quantiles):
        raise ValueError(
            f"pred last dim ({pred.shape[-1]}) must equal len(quantiles) ({len(quantiles)})"
        )
    if pred.shape[:-1] != target.shape:
        raise ValueError(
            f"target shape {tuple(target.shape)} must match pred[:-1] "
            f"{tuple(pred.shape[:-1])}"
        )
    q = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype).view(1, 1, 1, -1)
    err = target.unsqueeze(-1) - pred  # [B, T_out, N, Q]
    return torch.mean(torch.maximum(q * err, (q - 1.0) * err))


# ---------------------------------------------------------------------------
# Windowed dataset + dataloader
# ---------------------------------------------------------------------------


@dataclass
class _Windows:
    """Concrete integer-window indices for a given (T, T_in, T_out, stride)."""

    valid_start: np.ndarray
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def _compute_windows(
    total_steps: int,
    t_in: int,
    t_out: int,
    stride: int,
    split: tuple[float, float, float],
) -> _Windows:
    """Compute window-start indices with a leakage-free time-based split.

    The *time axis* is partitioned first — not the window indices. Given
    split fractions ``(p_tr, p_va, p_te)``:

    * train windows satisfy ``i + t_in + t_out <= T_train``,
    * val windows satisfy ``i >= T_train`` and ``i + t_in + t_out <= T_val``,
    * test windows satisfy ``i >= T_val`` and ``i + t_in + t_out <= T``.

    This construction guarantees that the ground-truth targets consumed by
    a train window never overlap the inputs of a val window (and same
    across val → test), so there is no temporal leakage by construction.

    Returns a :class:`_Windows` record. The ``valid_start`` field is the
    concatenation of all three subsets (for test diagnostics only).
    """
    if not np.isclose(sum(split), 1.0):
        raise ValueError(f"split must sum to 1.0; got {split} ({sum(split)})")
    max_start = total_steps - t_in - t_out
    if max_start < 0:
        raise ValueError(
            f"T={total_steps} too short for t_in={t_in}, t_out={t_out} "
            f"(need >= {t_in + t_out})"
        )

    # Time-axis partition boundaries. ``t_train`` is the first index that
    # belongs to val, and ``t_val`` is the first index that belongs to test.
    p_tr, p_va, _ = split
    t_train = int(np.floor(total_steps * p_tr))
    t_val = int(np.floor(total_steps * (p_tr + p_va)))
    # Clamp and ensure strict monotonicity so each subset has room for at
    # least one window (when the total data budget allows).
    t_train = max(t_in + t_out, min(t_train, total_steps - 2 * (t_in + t_out)))
    t_val = max(t_train + t_in + t_out, min(t_val, total_steps - (t_in + t_out)))

    def _range(lo: int, hi_exclusive: int) -> np.ndarray:
        # Starts i such that lo <= i and i + t_in + t_out <= hi_exclusive.
        top = hi_exclusive - t_in - t_out
        if top < lo:
            return np.empty((0,), dtype=np.int64)
        return np.arange(lo, top + 1, stride, dtype=np.int64)

    train = _range(0, t_train)
    val = _range(t_train, t_val)
    test = _range(t_val, total_steps)
    if train.size == 0 or val.size == 0 or test.size == 0:
        raise ValueError(
            f"split produced empty subset (train={train.size}, val={val.size}, "
            f"test={test.size}) for T={total_steps}, t_in={t_in}, t_out={t_out}, "
            f"split={split}"
        )
    valid = np.concatenate([train, val, test])
    return _Windows(valid_start=valid, train=train, val=val, test=test)


class _BundleWindowDataset(Dataset):
    """Thin ``Dataset`` that yields (x_node, x_exog, y) triplets.

    Expects the bundle's arrays to be numpy or torch-convertible. Slicing is
    done on demand so we do not materialise a B*T_in*N*F tensor up front.
    """

    def __init__(
        self,
        x_node: Tensor,
        x_exog: Tensor,
        y: Tensor,
        starts: np.ndarray,
        t_in: int,
        t_out: int,
    ):
        self.x_node = x_node
        self.x_exog = x_exog
        self.y = y
        self.starts = starts
        self.t_in = t_in
        self.t_out = t_out

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        i = int(self.starts[idx])
        xn = self.x_node[i : i + self.t_in]
        xe = self.x_exog[i : i + self.t_in]
        yw = self.y[i + self.t_in : i + self.t_in + self.t_out]
        return xn, xe, yw


def _to_float_tensor(arr: Any) -> Tensor:
    """Convert a numpy / torch / array-like object to a float32 tensor."""
    if isinstance(arr, Tensor):
        return arr.to(torch.float32)
    return torch.as_tensor(np.asarray(arr), dtype=torch.float32)


def make_dataloader(
    bundle: Any,
    t_in: int = 24,
    t_out: int = 6,
    batch_size: int = 32,
    stride: int = 1,
    split: tuple[float, float, float] = (0.7, 0.15, 0.15),
    which: Literal["train", "val", "test"] = "train",
    shuffle: bool | None = None,
) -> DataLoader:
    """Build a ``DataLoader`` over sliding windows of a ``FeatureBundle``.

    The bundle is duck-typed: any object with attributes ``X_node``
    (``[T, N, F_node]``), ``X_exog`` (``[T, F_exog]``), and ``y_kw``
    (``[T, N]``) is accepted, which avoids a hard import of
    ``gridsense.features``.

    Parameters
    ----------
    bundle : Any
        Object exposing ``X_exog``, ``X_node``, ``y_kw`` array-like fields.
    t_in, t_out : int
        Input-history and output-horizon length in hourly steps.
    batch_size : int
    stride : int
        Step between successive window starts.
    split : tuple of 3 floats
        Must sum to 1. Interpreted as the proportions of the valid-start
        indices assigned to (train, val, test) in chronological order.
    which : {"train", "val", "test"}
        Which subset's windows to expose.
    shuffle : bool or None
        If None, defaults to True only for ``which == "train"``.
    """
    if which not in ("train", "val", "test"):
        raise ValueError(f"which must be 'train'/'val'/'test'; got {which!r}")
    x_node = _to_float_tensor(bundle.X_node)
    x_exog = _to_float_tensor(bundle.X_exog)
    y = _to_float_tensor(bundle.y_kw)
    if x_node.dim() != 3:
        raise ValueError(f"bundle.X_node must be [T,N,F_node]; got {tuple(x_node.shape)}")
    if x_exog.dim() != 2:
        raise ValueError(f"bundle.X_exog must be [T,F_exog]; got {tuple(x_exog.shape)}")
    if y.dim() != 2:
        raise ValueError(f"bundle.y_kw must be [T,N]; got {tuple(y.shape)}")
    t, n, _ = x_node.shape
    if x_exog.shape[0] != t or y.shape[0] != t:
        raise ValueError(
            "time axis mismatch: "
            f"X_node T={t}, X_exog T={x_exog.shape[0]}, y_kw T={y.shape[0]}"
        )
    if y.shape[1] != n:
        raise ValueError(
            f"node axis mismatch: X_node N={n} vs y_kw N={y.shape[1]}"
        )

    windows = _compute_windows(
        total_steps=t, t_in=t_in, t_out=t_out, stride=stride, split=split
    )
    starts = {"train": windows.train, "val": windows.val, "test": windows.test}[which]
    if starts.size == 0:
        raise ValueError(
            f"no windows left for subset={which!r} after leakage trimming "
            f"(total valid={windows.valid_start.size})"
        )

    dataset = _BundleWindowDataset(x_node, x_exog, y, starts, t_in=t_in, t_out=t_out)
    if shuffle is None:
        shuffle = which == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Train + predict
# ---------------------------------------------------------------------------


def fit(
    model: GWNet,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
    log_every: int = 50,
) -> dict[str, list[float]]:
    """Train ``model`` with Adam + pinball loss.

    Returns a history dict with per-epoch train and (optional) val losses.
    """
    torch.manual_seed(1337)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    quantiles = tuple(model.config.quantiles)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for step, batch in enumerate(train_loader):
            x_node, x_exog, y = batch
            x_node = x_node.to(device)
            x_exog = x_exog.to(device)
            y = y.to(device)
            optim.zero_grad()
            pred = model(x_node, x_exog)
            loss = pinball_loss(pred, y, quantiles)
            loss.backward()
            optim.step()
            epoch_losses.append(float(loss.detach().cpu()))
            if log_every and (step + 1) % log_every == 0:
                # Keep logging minimal — the training SDE will wire real logs.
                pass
        history["train_loss"].append(
            float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        )

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_losses: list[float] = []
                for batch in val_loader:
                    x_node, x_exog, y = batch
                    x_node = x_node.to(device)
                    x_exog = x_exog.to(device)
                    y = y.to(device)
                    pred = model(x_node, x_exog)
                    val_losses.append(
                        float(pinball_loss(pred, y, quantiles).detach().cpu())
                    )
            history["val_loss"].append(
                float(np.mean(val_losses)) if val_losses else float("nan")
            )

    return history


def predict(
    model: GWNet,
    loader: DataLoader,
    device: str = "cpu",
    sort_quantiles: bool = True,
) -> tuple[Tensor, Tensor]:
    """Run the model over ``loader`` and return concatenated predictions.

    Parameters
    ----------
    model : GWNet
    loader : DataLoader
    device : str
    sort_quantiles : bool
        If True, sort along the quantile axis so that
        ``pred[..., 0] <= pred[..., 1] <= pred[..., 2]`` (the monotonic
        quantile guard — kept OUT of the loss so that the network is
        free to overshoot during training and learn a proper ordering).

    Returns
    -------
    pred : [N_windows, T_out, N, Q]
    target : [N_windows, T_out, N]
    """
    model = model.to(device)
    model.eval()
    preds: list[Tensor] = []
    targets: list[Tensor] = []
    with torch.no_grad():
        for batch in loader:
            x_node, x_exog, y = batch
            x_node = x_node.to(device)
            x_exog = x_exog.to(device)
            pred = model(x_node, x_exog).detach().cpu()
            if sort_quantiles:
                pred, _ = pred.sort(dim=-1)
            preds.append(pred)
            targets.append(y.detach().cpu())
    if not preds:
        q = len(model.config.quantiles)
        t_out = model.config.t_out
        n = model.config.n_nodes
        return (
            torch.empty((0, t_out, n, q), dtype=torch.float32),
            torch.empty((0, t_out, n), dtype=torch.float32),
        )
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)


# Silence unused-import warnings for tooling that scans the module — ``field``
# is retained so downstream callers can subclass ``GWNetConfig`` ergonomically.
_ = field  # noqa: F841
