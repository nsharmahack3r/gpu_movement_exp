"""Dilated causal Temporal Convolutional Network (Bai et al., 2018).

The reference CNN baseline for sequence forecasting. Causal convolutions
(left-pad by ``(kernel-1)·dilation``, trim the right overhang) guarantee timestep
*t* never sees *t+1*; residual blocks stack with exponentially growing dilation
so the receptive field covers the full input.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from movement.models.base import ForecastModel

logger = logging.getLogger(__name__)


class CausalConv1d(nn.Module):
    """1-D convolution with causal padding: output[t] depends only on input[<=t]."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.padding, 0))  # pad only the left (causal)
        return self.conv(x)


class _TemporalBlock(nn.Module):
    """One residual block: two causal convs + skip path with 1×1 conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        activation: Literal["gelu", "relu"],
        norm: Literal["none", "layernorm"],
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.relu2 = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(out_channels) if norm == "layernorm" else nn.Identity()
        self.norm2 = nn.LayerNorm(out_channels) if norm == "layernorm" else nn.Identity()
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )
        # WeightNorm on every causal conv (the skip 1×1 stays plain — it never
        # sees temporally-padded input, so weight norm buys nothing there).
        self.conv1.conv = nn.utils.parametrizations.weight_norm(self.conv1.conv)
        self.conv2.conv = nn.utils.parametrizations.weight_norm(self.conv2.conv)

    def _norm(self, x: Tensor, norm: nn.Module) -> Tensor:
        """LayerNorm over the channel dim: (B, C, T) -> permute -> norm -> back."""
        if isinstance(norm, nn.LayerNorm):
            x = x.transpose(1, 2)  # (B, T, C)
            x = norm(x)
            return x.transpose(1, 2)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """``x: (B, C, T) -> (B, C_out, T)`` (length preserved, causal)."""
        residual = self.skip(x)
        out = self.relu1(self._norm(self.conv1(x), self.norm1))
        out = self.dropout1(out)
        out = self.relu2(self._norm(self.conv2(out), self.norm2))
        out = self.dropout2(out)
        return out + residual


class TCN(ForecastModel):
    """Dilated causal TCN for trajectory displacement forecasting.

    Parameters mirror ``configs/model/tcn.yaml``. The receptive field is computed
    programmatically from ``channels/kernel_size`` and asserted to cover
    ``input_len``; the effective field is logged at construction.
    """

    def __init__(
        self,
        input_len: int,
        horizon: int,
        n_features: int,
        channels: list[int] = (64, 64, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.2,
        activation: Literal["gelu", "relu"] = "gelu",
        norm: Literal["none", "layernorm"] = "layernorm",
        head: Literal["direct", "autoregressive"] = "direct",
    ):
        super().__init__()
        if kernel_size < 2:
            raise ValueError("kernel_size must be >= 2 for causal convolution.")
        self.input_len = input_len
        self.horizon = horizon
        self.kernel_size = kernel_size
        self.head = head

        # Input projection to the first block's channel width.
        self.input_proj = nn.Conv1d(n_features, channels[0], 1)

        blocks: list[nn.Module] = []
        dilations: list[int] = []
        for i, c in enumerate(channels):
            d = 2**i
            dilations.append(d)
            in_c = channels[i - 1] if i > 0 else channels[0]
            blocks.append(
                _TemporalBlock(
                    in_c,
                    c,
                    kernel_size,
                    d,
                    dropout,
                    activation,
                    norm,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.dilations = dilations

        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)
        if self.receptive_field < input_len:
            raise ValueError(
                f"TCN receptive field {self.receptive_field} < input_len {input_len}. "
                f"Increase channels/kernel_size or add more blocks."
            )
        logger.info(
            "TCN receptive field: %d (input_len=%d, dilations=%s)",
            self.receptive_field,
            input_len,
            dilations,
        )

        # Direct multi-step head: final timestep repr -> (horizon, 2).
        self.final_act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.head_linear = nn.Linear(channels[-1], horizon * 2)
        if head == "autoregressive":
            # Autoregressive head predicts one step at a time.
            self.ar_head = nn.Linear(channels[-1], 2)

    def forward(self, x: Tensor, *, context: dict | None = None) -> Tensor:
        """``x: (B, input_len, F) -> (B, horizon, 2)``."""
        if x.dim() != 3 or x.size(1) != self.input_len:
            raise ValueError(f"Expected x of shape (B, {self.input_len}, F), got {tuple(x.shape)}")
        x = x.transpose(1, 2)  # (B, F, T)
        h = self.input_proj(x)
        h = self.blocks(h)
        if self.head == "direct":
            last = h[:, :, -1]  # (B, C)
            out = self.head_linear(self.final_act(last))
            return out.view(-1, self.horizon, 2)
        # Autoregressive head (ablation only): predict one step, append, re-run.
        return self._autoregressive(x)

    def _autoregressive(self, x: Tensor) -> Tensor:
        """Step-by-step prediction re-feeding the last output as the next input.

        The first ``input_len`` features are treated as the observation window
        (displacements); each predicted step appends ``(Δx, Δy)`` and slides.
        Slower and drift-prone — kept as an ablation, not the default.
        """
        B, F, T = x.shape
        # The first two channels are (Δx, Δy); the rest are context features.
        context_cols = x[:, 2:, :] if F > 2 else None
        seq = x[:, :2, :].clone()  # (B, 2, T)
        preds: list[Tensor] = []
        for _ in range(self.horizon):
            h = self.input_proj(seq)
            h = self.blocks(h)
            last = h[:, :, -1]
            step = self.ar_head(self.final_act(last))  # (B, 2)
            preds.append(step)
            # Slide: append the new displacement, drop the oldest.
            seq = torch.cat([seq[:, :, 1:], step.unsqueeze(2)], dim=2)
            if context_cols is not None:
                context_cols = context_cols[:, :, 1:]
                seq = torch.cat([seq, context_cols], dim=1)
        return torch.stack(preds, dim=1)  # (B, horizon, 2)

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    @receptive_field.setter
    def receptive_field(self, value: int) -> None:
        self._receptive_field = value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
