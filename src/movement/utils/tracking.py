"""Run tracking: per-run directories, manifest, TensorBoard/W&B wrappers.

Every run writes into ``runs/<timestamp>-<model>-<seed>/``:

- ``config.yaml``     — full merged config snapshot
- ``manifest.json``   — git commit, seed, hardware, timing, VRAM, params
- ``split.json``      — exact train/val/test individual assignment
- ``scalers.json``    — per-feature standardisation statistics (fit on train only)
- ``checkpoints/``    — best/last model checkpoints
- ``tensorboard/``    — TensorBoard event logs
- ``metrics.json``    — test metrics (written by the eval entrypoint)
- ``per_horizon.csv`` — error breakdown by horizon step
- ``figures/``        — predicted-vs-actual plots
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from movement.config import Config

logger = logging.getLogger(__name__)

try:  # wandb is an optional dependency
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


def git_commit_hash() -> str | None:
    """Short hash of the current git HEAD, or ``None`` outside a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes; None outside git."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def gpu_info() -> dict[str, Any]:
    """Name, total VRAM, and free VRAM of the first visible CUDA device."""
    if not torch.cuda.is_available():
        return {}
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info(0)
    return {
        "name": name,
        "total_memory_mb": round(total / 1024**2, 1),
        "free_memory_mb": round(free / 1024**2, 1),
        "capability": f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}",
    }


def peak_vram_mb() -> float:
    """Peak CUDA VRAM allocated so far (MB), 0 when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated(0) / 1024**2, 1)


def make_run_dir(config: Config) -> Path:
    """Create and return the per-run directory for this configuration."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = config.trainer.run_dir / f"{stamp}-{config.model.name}-s{config.trainer.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for sub in (
        config.trainer.checkpoint_dir,
        config.trainer.tensorboard_dir,
        "figures",
    ):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_config_snapshot(run_dir: Path, config: Config) -> Path:
    """Persist the full merged config as ``config.yaml`` in the run dir."""
    import yaml

    path = run_dir / "config.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    return path


def write_manifest(
    run_dir: Path,
    config: Config,
    *,
    start_time: float,
    end_time: float,
    params: int,
    train_metrics: dict[str, Any] | None = None,
    test_metrics: dict[str, Any] | None = None,
) -> Path:
    """Write ``manifest.json`` with hardware, timing, and run metadata."""
    manifest = {
        "git_commit": git_commit_hash(),
        "git_dirty": git_dirty(),
        "seed": config.trainer.seed,
        "deterministic": config.trainer.deterministic,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu_info(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "wall_clock_seconds": round(end_time - start_time, 1),
        "parameter_count": params,
        "peak_vram_mb": peak_vram_mb(),
        "train_metrics": train_metrics or {},
        "test_metrics": test_metrics or {},
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


class Tracker:
    """Wraps TensorBoard (default) and optionally W&B behind one interface.

    Works fully offline; W&B only activates when ``config.tracking.backend ==
    "wandb"`` *and* wandb is installed.
    """

    def __init__(self, config: Config, run_dir: Path):
        self.config = config
        self.run_dir = run_dir
        self.backend = config.tracking.backend
        self._wandb = None
        if self.backend == "wandb":
            if not _WANDB_AVAILABLE:
                raise RuntimeError(
                    "tracking.backend=wandb but wandb is not installed. "
                    "Install with `uv sync --extra wandb` or set backend=tensorboard."
                )
            mode = "offline" if config.tracking.offline else "online"
            self._wandb = wandb.init(  # type: ignore[union-attr]
                project=config.tracking.project,
                entity=config.tracking.entity,
                mode=mode,
                dir=str(run_dir),
                config=config.model_dump(mode="json"),
            )
            logger.info("W&B run started (%s mode): %s", mode, self._wandb.name)
        else:
            from torch.utils.tensorboard import SummaryWriter

            self._tensorboard = SummaryWriter(log_dir=str(run_dir / config.trainer.tensorboard_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._wandb is not None:
            self._wandb.log({tag: value}, step=step)
        else:
            self._tensorboard.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        if self._wandb is not None:
            self._wandb.log({tag: wandb.Histogram(values.detach().cpu().numpy())}, step=step)  # type: ignore[union-attr]
        else:
            self._tensorboard.add_histogram(tag, values, step)

    def log_figure(self, tag: str, fig: Any, step: int) -> None:
        if self._wandb is not None:
            self._wandb.log({tag: wandb.Image(fig)}, step=step)  # type: ignore[union-attr]
        else:
            self._tensorboard.add_figure(tag, fig, step)

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()  # type: ignore[union-attr]
        else:
            self._tensorboard.close()
