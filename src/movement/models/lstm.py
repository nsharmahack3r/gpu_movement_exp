"""Encoder–decoder LSTM arm for trajectory displacement forecasting.

The recurrent counterpart to the TCN. Two forecast heads, selected by config:

- **Direct multi-step (default)**: the final encoder hidden state is projected
  through an MLP to ``horizon × 2`` and reshaped to ``(horizon, 2)``. One
  forward pass, no exposure bias — the direct analogue of the TCN head.
- **Autoregressive decoder (ablation)**: a second ``nn.LSTM`` initialised from
  the encoder's final ``(h, c)`` emits one displacement per step and feeds its
  own prediction back as the next input. Teacher forcing with scheduled sampling
  is applied during training only; ground truth is *never* fed at validation or
  test time.
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


def scheduled_sampling_ratio(
    epoch: int,
    *,
    max_ratio: float,
    mode: Literal["constant", "linear", "inverse_sigmoid"],
    total_epochs: int,
) -> float:
    """Teacher-forcing ratio at a given epoch for each decay mode.

    - ``"constant"``: always ``max_ratio``.
    - ``"linear"``: ``max_ratio`` → 0 linearly over ``total_epochs``.
    - ``"inverse_sigmoid"``: sigmoid-decay from ``max_ratio`` → 0 (inverse
      sigmoid in the sense that the ratio is high early and decays).

    The returned ratio is clamped to ``[0, 1]``.
    """
    if mode == "constant":
        return float(max_ratio)
    k = total_epochs
    if mode == "linear":
        ratio = max_ratio * (1.0 - min(1.0, epoch / k))
    else:  # inverse_sigmoid
        # Inverse-sigmoid decay: ratio(k) = max_ratio / (1 + exp(k - epoch)).
        # At epoch 0 it is high, at k it is ~max_ratio/2, and beyond k it
        # approaches 0.
        ratio = max_ratio / (1.0 + math.exp(epoch - k))
    return float(min(1.0, max(0.0, ratio)))


class LSTMForecaster(ForecastModel):
    """Encoder–decoder LSTM for trajectory displacement forecasting.

    Parameters mirror ``configs/model/lstm.yaml``. The encoder consumes
    ``(B, input_len, F)``; the decoder head produces ``(B, horizon, 2)``
    displacement predictions.
    """

    def __init__(
        self,
        input_len: int,
        horizon: int,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        decoder: Literal["direct", "autoregressive"] = "direct",
        teacher_forcing_ratio: float = 1.0,
        scheduled_sampling: Literal["constant", "linear", "inverse_sigmoid"] = "constant",
        scheduled_sampling_epochs: int = 50,
        attention: Literal["none", "additive", "dot"] = "none",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if dropout > 0.0 and num_layers == 1:
            logger.warning(
                "LSTM dropout=%.2f is ignored by PyTorch when num_layers=1; "
                "set num_layers>=2 or dropout=0 to silence this.",
                dropout,
            )
        self.input_len = input_len
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.decoder = decoder
        self.attention = attention
        self._dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self._dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        enc_dim = hidden_size * (2 if bidirectional else 1)

        if decoder == "direct":
            # Direct head: project the final encoder state to (horizon, 2).
            self.head = nn.Sequential(
                nn.Linear(enc_dim, enc_dim),
                nn.GELU(),
                nn.Linear(enc_dim, horizon * 2),
            )
        else:
            # Autoregressive decoder LSTM emits one displacement per step.
            dec_input = 2 + (hidden_size if attention != "none" else 0)
            self.decoder_rnn = nn.LSTM(
                input_size=dec_input,
                hidden_size=enc_dim,
                num_layers=1,
                batch_first=True,
            )
            self.decoder_out = nn.Linear(enc_dim, 2)
            if attention == "additive":
                # Additive (Bahdanau) attention: score = v^T tanh(W_q·q + W_k·k)
                self.attn_Wq = nn.Linear(enc_dim, enc_dim)
                self.attn_Wk = nn.Linear(enc_dim, enc_dim)
                self.attn_v = nn.Linear(enc_dim, 1, bias=False)
            elif attention == "dot":
                self.attn_Wq = None  # dot-product uses encoder outputs directly
            # Teacher forcing engages only when context["stage"] == "train".
            self.teacher_forcing_ratio = teacher_forcing_ratio
            self.scheduled_sampling = scheduled_sampling
            self.scheduled_sampling_epochs = scheduled_sampling_epochs

    @property
    def receptive_field(self) -> None:
        """An LSTM has unbounded theoretical context — no fixed receptive field."""
        return None

    def forward(self, x: Tensor, *, context: dict | None = None) -> Tensor:
        """``x: (B, input_len, F) -> (B, horizon, 2)``.

        ``context`` may carry ``stage`` ("train"/"val"/"test"), ``epoch``, and
        ``targets`` (the ``(B, horizon, 2)`` ground-truth deltas). The
        autoregressive decoder uses teacher forcing **only** when
        ``context["stage"] == "train"``; at val/test time it always feeds its
        own predictions, so no ground truth leaks into evaluation.
        """
        if x.dim() != 3 or x.size(1) != self.input_len:
            raise ValueError(f"Expected x of shape (B, {self.input_len}, F), got {tuple(x.shape)}")
        context = context or {}
        stage = context.get("stage", "train")

        enc_out, (h, c) = self.encoder(x)
        if self.decoder == "direct":
            return self._direct_head(h, c)

        # Autoregressive decoder.
        if stage == "train":
            targets = context.get("targets")
            if targets is None:
                raise ValueError(
                    "Autoregressive LSTM training requires context['targets'] for "
                    "teacher forcing. The trainer must pass targets during training."
                )
            epoch = context.get("epoch", 0)
            ratio = scheduled_sampling_ratio(
                epoch,
                max_ratio=self.teacher_forcing_ratio,
                mode=self.scheduled_sampling,
                total_epochs=self.scheduled_sampling_epochs,
            )
            return self._autoregressive(x, enc_out, (h, c), targets=targets, ratio=ratio)
        return self._autoregressive(x, enc_out, (h, c), targets=None, ratio=0.0)

    def _direct_head(self, h: Tensor, c: Tensor) -> Tensor:
        """Project the final encoder state to (horizon, 2)."""
        # h: (num_layers*num_directions, B, hidden); take the top layer.
        last = h[-1]  # (B, enc_dim)
        out = self.head(last)
        return out.view(-1, self.horizon, 2)

    def _autoregressive(
        self,
        x: Tensor,
        enc_out: Tensor,
        enc_state: tuple[Tensor, Tensor],
        *,
        targets: Tensor | None,
        ratio: float,
    ) -> Tensor:
        """Step-by-step decoding with optional teacher forcing.

        ``targets`` are the ground-truth per-step displacement deltas
        ``(B, horizon, 2)``; at ratio ``r`` each decoder input is the ground
        truth with probability ``r`` (train only) and the model's own previous
        prediction otherwise. ``targets=None`` forces pure free-running.
        """
        B = x.size(0)
        h, c = enc_state
        # Initialise the decoder state from the encoder's final state.
        dec_h = h[-1:]  # (1, B, enc_dim)
        dec_c = c[-1:]
        prev = x[:, -1, :2].unsqueeze(1)  # (B, 1, 2) — last observed displacement
        preds: list[Tensor] = []

        for t in range(self.horizon):
            attn_vec = self._attention(enc_out, dec_h) if self.attention != "none" else None
            dec_in = prev if attn_vec is None else torch.cat([prev, attn_vec], dim=-1)
            out, (dec_h, dec_c) = self.decoder_rnn(dec_in, (dec_h, dec_c))
            step = self.decoder_out(out)  # (B, 1, 2)
            preds.append(step)

            # Teacher forcing only when targets are provided (train stage).
            if targets is not None:
                use_teacher = torch.rand(B, 1, device=x.device) < ratio
                forced = targets[:, t : t + 1, :]
                prev = torch.where(use_teacher.unsqueeze(-1), forced, step)
            else:
                prev = step

        return torch.cat(preds, dim=1)  # (B, horizon, 2)

    def _attention(self, enc_out: Tensor, dec_h: Tensor) -> Tensor:
        """Attention context vector over encoder outputs (ablation only)."""
        # enc_out: (B, T, enc_dim); dec_h: (1, B, enc_dim)
        query = dec_h[0]  # (B, enc_dim)
        if self.attention == "dot":
            scores = torch.bmm(enc_out, query.unsqueeze(2)).squeeze(2)  # (B, T)
            weights = torch.softmax(scores, dim=1).unsqueeze(1)  # (B, 1, T)
            return torch.bmm(weights, enc_out)  # (B, 1, enc_dim)
        # additive (Bahdanau)
        q = self.attn_Wq(query).unsqueeze(1)  # (B, 1, enc_dim)
        k = self.attn_Wk(enc_out)  # (B, T, enc_dim)
        scores = self.attn_v(torch.tanh(q + k)).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=1).unsqueeze(1)  # (B, 1, T)
        return torch.bmm(weights, enc_out)  # (B, 1, enc_dim)
