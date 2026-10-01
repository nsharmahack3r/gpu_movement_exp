"""Transformer encoder arm for trajectory displacement forecasting.

The self-attention counterpart to the TCN and LSTM. Two modes, selected by
config:

- **Encoder-only (default)**: input projection → positional encoding → transformer
  encoder stack → pooling → MLP head to ``(horizon, 2)``. The direct analogue of
  the TCN head and the LSTM direct head.
- **Encoder–decoder (ablation)**: a causal-masked decoder over ``horizon`` learned
  query positions, cross-attending to the encoder output. Non-autoregressive (all
  steps in one pass) so no ground truth is ever fed at any stage.

Positional encodings: ``sinusoidal`` (default), ``learned``, or ``time_aware``
(encodes the actual elapsed Δt of each fix — the most defensible option for
irregularly sampled GPS data).
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from movement.models.base import ForecastModel

logger = logging.getLogger(__name__)


def _sinusoidal_positions(length: int, d_model: int) -> Tensor:
    """Fixed sinusoidal positional encodings, shape ``(1, length, d_model)``."""
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)  # (L, 1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1, L, d)


class TimeAwarePositionalEncoding(nn.Module):
    """Encode the actual elapsed Δt of each fix rather than its ordinal index.

    ``forward(dt_seconds)`` where ``dt_seconds`` is ``(B, T)`` of cumulative
    seconds since the first fix. Each timestep gets a sinusoidal encoding of its
    elapsed time, so positional differences reflect real temporal distance.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, dt_seconds: Tensor) -> Tensor:
        div = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=dt_seconds.dtype, device=dt_seconds.device)
            * (-math.log(10000.0) / self.d_model)
        )
        angles = dt_seconds.unsqueeze(-1) * div  # (B, T, d_model/2)
        pe = torch.zeros(*dt_seconds.shape, self.d_model, dtype=dt_seconds.dtype, device=dt_seconds.device)
        pe[..., 0::2] = torch.sin(angles)
        pe[..., 1::2] = torch.cos(angles)
        return pe


class PositionalEncoding(nn.Module):
    """Positional encoding module for the encoder input."""

    def __init__(
        self,
        d_model: int,
        input_len: int,
        kind: Literal["sinusoidal", "learned", "time_aware"],
    ):
        super().__init__()
        self.kind = kind
        if kind == "sinusoidal":
            self.register_buffer("pe", _sinusoidal_positions(input_len, d_model))
        elif kind == "learned":
            self.embed = nn.Embedding(input_len, d_model)
        elif kind == "time_aware":
            self.pe = TimeAwarePositionalEncoding(d_model)

    def forward(self, x: Tensor, dt_seconds: Tensor | None = None) -> Tensor:
        """``x: (B, T, d_model) -> (B, T, d_model)`` with positions added."""
        if self.kind == "sinusoidal":
            return x + self.pe[:, : x.size(1)]
        if self.kind == "learned":
            positions = torch.arange(x.size(1), device=x.device)
            return x + self.embed(positions).unsqueeze(0)
        if dt_seconds is None:
            raise ValueError("time_aware positional encoding requires dt_seconds in context.")
        return x + self.pe(dt_seconds)


class TransformerForecaster(ForecastModel):
    """Transformer encoder(-decoder) for trajectory displacement forecasting.

    Parameters mirror ``configs/model/transformer.yaml``. ``context`` may carry
    ``stage`` ("train"/"val"/"test") and ``dt_seconds`` (for ``time_aware``
    positional encoding), computed by the datamodule from the raw timestamps.
    """

    def __init__(
        self,
        input_len: int,
        horizon: int,
        n_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        mode: Literal["encoder_only", "encoder_decoder"] = "encoder_only",
        pos_encoding: Literal["sinusoidal", "learned", "time_aware"] = "sinusoidal",
        pooling: Literal["last", "mean", "cls"] = "last",
        causal_mask: bool = False,
        norm_first: bool = True,
    ):
        super().__init__()
        self.input_len = input_len
        self.horizon = horizon
        self.mode = mode
        self.pos_encoding = pos_encoding
        self.pooling = pooling
        self.causal_mask = causal_mask
        self.d_model = d_model

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, input_len, pos_encoding)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            norm_first=norm_first,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        if mode == "encoder_only":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, horizon * 2),
            )
        else:
            # Non-autoregressive decoder: horizon learned query positions,
            # causal-masked self-attention, cross-attention to encoder output.
            self.decoder_query = nn.Parameter(torch.randn(horizon, d_model) * 0.02)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                norm_first=norm_first,
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
            self.head = nn.Linear(d_model, 2)

    @property
    def receptive_field(self) -> int:
        """Self-attention sees the whole window."""
        return self.input_len

    def forward(self, x: Tensor, *, context: dict | None = None) -> Tensor:
        """``x: (B, input_len, F) -> (B, horizon, 2)``."""
        if x.dim() != 3 or x.size(1) != self.input_len:
            raise ValueError(f"Expected x of shape (B, {self.input_len}, F), got {tuple(x.shape)}")
        context = context or {}
        dt_seconds = context.get("dt_seconds")

        h = self.input_proj(x)  # (B, T, d)
        h = self.pos_enc(h, dt_seconds)

        if self.mode == "encoder_only":
            mem = self.encoder(h, mask=self._causal_mask(h))
            pooled = self._pool(mem)
            out = self.head(pooled)
            return out.view(-1, self.horizon, 2)

        # Encoder-decoder mode: no causal mask on the encoder (observed history).
        mem = self.encoder(h)
        queries = self.decoder_query.unsqueeze(0).expand(x.size(0), -1, -1)  # (B, H, d)
        tgt_mask = self._causal_mask(queries)  # causal over the horizon
        dec_out = self.decoder(queries, mem, tgt_mask=tgt_mask)
        return self.head(dec_out)  # (B, horizon, 2)

    def _causal_mask(self, t: Tensor) -> Tensor | None:
        """Lower-triangular causal mask for the given sequence length."""
        if not self.causal_mask:
            return None
        return torch.triu(torch.full((t.size(1), t.size(1)), float("-inf")), diagonal=1).to(t.device)

    def _pool(self, mem: Tensor) -> Tensor:
        """Pool the encoder output to a single representation."""
        if self.pooling == "last":
            return mem[:, -1]
        if self.pooling == "mean":
            return mem.mean(dim=1)
        # cls: a learnable token attends over the whole window.
        cls = self.cls_token.expand(mem.size(0), -1, -1)
        return torch.cat([cls, mem], dim=1)[:, 0]

    def attention_entropy(self, x: Tensor, dt_seconds: Tensor | None = None) -> float:
        """Mean attention entropy over all heads/layers for one batch (diagnostic).

        Cheap check for attention collapse: entropy near 0 means heads attend to
        a single position, which usually indicates broken training. Recomputes
        each encoder layer's self-attention with ``need_weights=True`` to get
        the weight matrices (the layers themselves run with weights off).
        """
        self.eval()
        with torch.no_grad():
            h = self.input_proj(x)
            h = self.pos_enc(h, dt_seconds)
            entropies: list[Tensor] = []
            for layer in self.encoder.layers:
                attn = layer.self_attn
                norm = layer.norm1 if layer.norm_first else None
                src = norm(h) if norm is not None else h
                _, weights = attn(src, src, src, need_weights=True, average_attn_weights=False)
                if weights is None:
                    continue
                probs = torch.softmax(weights, dim=-1).clamp_min(1e-9)
                entropies.append(-(probs * probs.log()).sum(dim=-1))
            if not entropies:
                return float("nan")
            return float(torch.cat(entropies).mean())
