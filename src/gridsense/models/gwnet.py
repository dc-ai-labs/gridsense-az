"""Graph WaveNet with 3-quantile head (p10 / p50 / p90).

Hero model for GridSense-AZ feeder-load forecasting.

Architecture (adapted from Zonghan Wu et al. 2019 — ``nnzhan/Graph-WaveNet``, MIT):

    x : [B, T_in, N, F_in]
        │  permute → [B, F_in, N, T_in]
        ▼
    1x1 start_conv → hidden_dim channels
        │
        ├── stacked dilated-gated TCN blocks, each followed by a
        │   graph-convolution block that combines a *fixed* normalized
        │   adjacency with a *learned* adaptive adjacency built from two
        │   node-embedding matrices (softmax-normalized product).
        │   Residual + skip connections accumulate into ``skip``.
        ▼
    1x1 end_conv on skip → horizon * Q channels, reshaped to
        [B, horizon, N, Q]   where Q = len(quantile_levels) (default 3).

The quantile head lets us optimise the pinball loss directly and produce
calibrated p10/p50/p90 forecasts with a single forward pass — no ensemble.

Exports
-------
- :class:`GraphWaveNetQuantile`
- :func:`pinball_loss`
- :func:`build_adaptive_adj`
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GraphWaveNetQuantile", "pinball_loss", "build_adaptive_adj"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def build_adaptive_adj(node_emb_source: torch.Tensor, node_emb_target: torch.Tensor) -> torch.Tensor:
    """Build a learned adaptive adjacency matrix from two node-embedding tables.

    Parameters
    ----------
    node_emb_source : [N, d]
    node_emb_target : [N, d]

    Returns
    -------
    adj : [N, N] — softmax over the row axis of ``ReLU(E_s @ E_t.T)``.
    """
    logits = F.relu(torch.mm(node_emb_source, node_emb_target.t()))
    return F.softmax(logits, dim=1)


def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric-ish row normalization with self-loops — safe on non-sym graphs."""
    adj = adj + torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
    d = adj.sum(dim=1).clamp(min=1e-6)
    return adj / d.unsqueeze(1)


# ----------------------------------------------------------------------------
# Graph convolution over an [N, N] dense adjacency
# ----------------------------------------------------------------------------
class GraphConv(nn.Module):
    """K-hop diffusion conv over a *list* of dense adjacencies.

    Input  : x [B, C_in, N, T]
    Output :    [B, C_out, N, T]

    For each adjacency ``A_k``, computes ``x @ A_k`` repeatedly up to ``order``
    hops and concatenates them along the channel axis before a 1x1 mixer.
    """

    def __init__(self, c_in: int, c_out: int, num_adj: int, order: int = 2, dropout: float = 0.3):
        super().__init__()
        self.order = order
        self.num_adj = num_adj
        # +1 for the identity pass-through term (x itself).
        total_in = c_in * (1 + num_adj * order)
        self.mlp = nn.Conv2d(total_in, c_out, kernel_size=(1, 1))
        self.dropout = dropout

    @staticmethod
    def _nconv(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N, T], a: [N, N] -> [B, C, N, T] via einsum over the N axis.
        return torch.einsum("bcnt,nm->bcmt", x, a).contiguous()

    def forward(self, x: torch.Tensor, adjs: Sequence[torch.Tensor]) -> torch.Tensor:
        out = [x]
        for a in adjs:
            h = x
            for _ in range(self.order):
                h = self._nconv(h, a)
                out.append(h)
        h_cat = torch.cat(out, dim=1)
        h_cat = self.mlp(h_cat)
        return F.dropout(h_cat, p=self.dropout, training=self.training)


# ----------------------------------------------------------------------------
# Dilated gated TCN block + graph conv + residual + skip
# ----------------------------------------------------------------------------
class GWNetBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        skip_dim: int,
        kernel_size: int,
        dilation: int,
        num_adj: int,
        gcn_order: int,
        dropout: float,
    ):
        super().__init__()
        self.filter_conv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=(1, kernel_size), dilation=(1, dilation)
        )
        self.gate_conv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=(1, kernel_size), dilation=(1, dilation)
        )
        self.gconv = GraphConv(hidden_dim, hidden_dim, num_adj=num_adj, order=gcn_order, dropout=dropout)
        self.residual_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))
        self.skip_conv = nn.Conv2d(hidden_dim, skip_dim, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        adjs: Sequence[torch.Tensor],
        skip: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, N, T]
        residual = x
        f = torch.tanh(self.filter_conv(x))
        g = torch.sigmoid(self.gate_conv(x))
        h = f * g  # [B, C, N, T']; T' = T - (kernel_size - 1) * dilation

        # Skip path — aligned to the block's output temporal length.
        s = self.skip_conv(h)
        if skip is None:
            skip = s
        else:
            # The newer block has a shorter T'. Slice the running skip tail to match.
            skip = skip[..., -s.size(-1):] + s

        # Graph conv + residual.
        h = self.gconv(h, adjs)
        # Trim the residual to the new temporal length.
        residual = residual[..., -h.size(-1):]
        h = h + residual
        h = self.bn(h)
        return h, skip


# ----------------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------------
class GraphWaveNetQuantile(nn.Module):
    """Graph WaveNet with a 3-quantile output head.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes (feeder buses).
    adj_init : Optional[torch.Tensor]
        Fixed [N, N] adjacency (can be unnormalized; will be normalized inside).
        If None, only the adaptive adjacency is used.
    input_dim : int
        Number of input features per node per timestep.
    hidden_dim : int
        Channel width of the backbone.
    num_blocks : int
        Number of TCN+GCN blocks. Dilation grows geometrically.
    kernel_size : int
        Temporal kernel size in each block.
    dilation_growth : int
        Multiplicative dilation growth per block (1, g, g^2, ...).
    horizon : int
        Forecast horizon (number of output steps).
    quantile_levels : tuple[float, ...]
        Quantile levels to predict (e.g. (0.1, 0.5, 0.9)).
    dropout : float
        Dropout inside the graph conv.
    adaptive_adj_dim : int
        Embedding dimension for the learned adjacency.
    skip_dim : int
        Channel width of the skip-path 1x1 conv outputs.
    end_dim : int
        Channel width of the first end-conv layer.
    """

    def __init__(
        self,
        num_nodes: int,
        adj_init: torch.Tensor | None = None,
        input_dim: int = 5,
        hidden_dim: int = 32,
        num_blocks: int = 4,
        kernel_size: int = 2,
        dilation_growth: int = 2,
        horizon: int = 24,
        quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
        dropout: float = 0.3,
        adaptive_adj_dim: int = 10,
        skip_dim: int = 64,
        end_dim: int = 128,
        gcn_order: int = 2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.quantile_levels = tuple(quantile_levels)
        self.num_quantiles = len(self.quantile_levels)
        self.dropout = dropout

        # Fixed (topology) adjacency — buffer so it moves with .to(device).
        if adj_init is not None:
            adj_init = adj_init.float()
            fixed = _normalize_adj(adj_init)
            self.register_buffer("fixed_adj", fixed)
            self._has_fixed = True
        else:
            self.register_buffer("fixed_adj", torch.eye(num_nodes))
            self._has_fixed = False

        # Adaptive (learned) adjacency embeddings.
        self.node_emb_src = nn.Parameter(torch.randn(num_nodes, adaptive_adj_dim) * 0.1)
        self.node_emb_tgt = nn.Parameter(torch.randn(num_nodes, adaptive_adj_dim) * 0.1)

        num_adj = 2 if self._has_fixed else 1

        # 1x1 start conv lifts input_dim -> hidden_dim.
        self.start_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1))

        # Stack of dilated-gated TCN + GCN blocks.
        self.blocks = nn.ModuleList()
        dilation = 1
        for _ in range(num_blocks):
            self.blocks.append(
                GWNetBlock(
                    hidden_dim=hidden_dim,
                    skip_dim=skip_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    num_adj=num_adj,
                    gcn_order=gcn_order,
                    dropout=dropout,
                )
            )
            dilation *= dilation_growth

        # End head: skip -> ReLU -> 1x1 -> ReLU -> 1x1 -> (horizon * num_quantiles).
        self.end_conv_1 = nn.Conv2d(skip_dim, end_dim, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(end_dim, horizon * self.num_quantiles, kernel_size=(1, 1))

        # Minimum input length required so every block still has >=1 output step.
        # Each block with dilation d and kernel k consumes (k-1)*d steps.
        d = 1
        consumed = 0
        for _ in range(num_blocks):
            consumed += (kernel_size - 1) * d
            d *= dilation_growth
        self.receptive_field = consumed + 1

    # ------------------------------------------------------------------
    def _current_adjs(self) -> list[torch.Tensor]:
        adp = build_adaptive_adj(self.node_emb_src, self.node_emb_tgt)
        if self._has_fixed:
            return [self.fixed_adj, adp]
        return [adp]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : [B, T_in, N, F_in]

        Returns
        -------
        y : [B, horizon, N, Q]
        """
        if x.dim() != 4:
            raise ValueError(f"expected 4D [B,T,N,F], got {tuple(x.shape)}")
        b, t_in, n, f_in = x.shape
        if n != self.num_nodes:
            raise ValueError(f"expected N={self.num_nodes} nodes, got {n}")
        if f_in != self.input_dim:
            raise ValueError(f"expected F={self.input_dim} features, got {f_in}")

        # Pad if input is shorter than the model's receptive field.
        if t_in < self.receptive_field:
            pad = self.receptive_field - t_in
            x = F.pad(x, (0, 0, 0, 0, pad, 0))  # pad along time axis on the left

        # [B, T, N, F] -> [B, F, N, T]
        h = x.permute(0, 3, 2, 1).contiguous()
        h = self.start_conv(h)

        adjs = self._current_adjs()
        skip: torch.Tensor | None = None
        for block in self.blocks:
            h, skip = block(h, adjs, skip)

        # End head.
        out = F.relu(skip)
        out = F.relu(self.end_conv_1(out))
        out = self.end_conv_2(out)  # [B, H*Q, N, T']

        # Collapse the temporal axis (take the last step — it is the one that
        # has seen the full receptive field) and reshape into [B, H, N, Q].
        out = out[..., -1]  # [B, H*Q, N]
        out = out.reshape(b, self.horizon, self.num_quantiles, n)
        out = out.permute(0, 1, 3, 2).contiguous()  # [B, H, N, Q]
        return out


# ----------------------------------------------------------------------------
# Pinball / quantile loss
# ----------------------------------------------------------------------------
def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: Sequence[float],
) -> torch.Tensor:
    """Mean pinball loss.

    Parameters
    ----------
    pred : [..., Q]
        Predicted quantiles along the last axis.
    target : [...]
        Ground-truth values; same leading dims as ``pred``.
    quantiles : sequence of floats in (0, 1).

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
            f"target shape {tuple(target.shape)} must match pred[:-1] {tuple(pred.shape[:-1])}"
        )
    q = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype)
    # broadcast target over the Q axis
    diff = target.unsqueeze(-1) - pred  # [..., Q]
    loss = torch.maximum(q * diff, (q - 1.0) * diff)
    return loss.mean()
