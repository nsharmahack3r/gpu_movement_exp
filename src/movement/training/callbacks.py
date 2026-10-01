"""Training callbacks: early stopping, checkpointing, LR scheduling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

logger = logging.getLogger(__name__)


@dataclass
class EarlyStopping:
    """Stop training when a monitored metric stops improving for ``patience`` epochs."""

    patience: int = 15
    min_delta: float = 1e-4
    mode: str = "min"
    best: float | None = field(default=None, init=False)
    counter: int = field(default=0, init=False)
    stopped_epoch: int = field(default=-1, init=False)

    def __call__(self, value: float, epoch: int) -> bool:
        """Update state; return True when training should stop."""
        if self.best is None or (self.mode == "min" and value < self.best - self.min_delta) or (
            self.mode == "max" and value > self.best + self.min_delta
        ):
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                logger.info("Early stopping triggered at epoch %d (best=%s).", epoch, self.best)
                return True
        return False


class CheckpointManager:
    """Save best-by-val-metric and last checkpoints with full state."""

    def __init__(self, run_dir: Path, checkpoint_dir: Path, metric: str = "val_ade", mode: str = "min"):
        self.dir = checkpoint_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.mode = mode
        self.best_value: float | None = None

    def _path(self, tag: str) -> Path:
        return self.dir / f"{tag}.pt"

    def maybe_save(
        self,
        state: dict,
        value: float,
        epoch: int,
    ) -> Path | None:
        """Save the best checkpoint when ``value`` improves; always save ``last``."""
        self._save("last", state, epoch)
        improved = self.best_value is None or (
            self.mode == "min" and value < self.best_value
        ) or (self.mode == "max" and value > self.best_value)
        if improved:
            self.best_value = value
            self._save("best", state, epoch)
            return self._path("best")
        return None

    def _save(self, tag: str, state: dict, epoch: int) -> None:
        state = {**state, "epoch": epoch, "best_value": self.best_value, "checkpoint_tag": tag}
        torch.save(state, self._path(tag))
        logger.debug("Saved %s checkpoint (epoch %d) to %s", tag, epoch, self._path(tag))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine schedule with linear warmup over ``total_steps``."""
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
