"""ForecastModel interface every model arm implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class ForecastModel(nn.Module, ABC):
    """Interface all arms (CNN, LSTM, Transformer) satisfy.

    The trainer, metrics, and eval entrypoint know nothing beyond this contract,
    so adding a new model means one model file plus one config file.
    """

    @abstractmethod
    def forward(self, x: Tensor, *, context: dict | None = None) -> Tensor:
        """Map a batch of windows to displacement predictions.

        Parameters
        ----------
        x:
            ``(B, input_len, F)`` — per-fix features (deltas + extras), already
            standardised where configured.
        context:
            Optional categorical/static conditioning (reserved; unused by CNN).

        Returns
        -------
        Tensor:
            ``(B, horizon, 2)`` displacement predictions in the model's target
            space (metres, relative to the last observed fix when delta-encoded).
        """

    @property
    @abstractmethod
    def receptive_field(self) -> int | None:
        """Number of input timesteps a single output depends on, if defined."""

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
