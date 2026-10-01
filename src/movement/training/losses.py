"""Loss functions for displacement forecasting."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def displacement_mse(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error over (B, horizon, 2) displacement predictions."""
    return F.mse_loss(pred, target)


def displacement_huber(pred: Tensor, target: Tensor, delta: float = 1.0) -> Tensor:
    """Huber loss over displacement predictions (robust to outlier steps)."""
    return F.smooth_l1_loss(pred, target, beta=delta)
