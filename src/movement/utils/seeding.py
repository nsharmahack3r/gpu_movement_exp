"""Deterministic seeding for Python, NumPy, and PyTorch."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed all RNGs and optionally enable cuDNN determinism.

    ``deterministic=True`` trades speed for reproducible convolutions; only
    meaningful on CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("cuDNN deterministic mode enabled (slower).")
    else:
        torch.backends.cudnn.benchmark = True
