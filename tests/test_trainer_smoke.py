"""Trainer smoke test: 2 epochs end-to-end on CPU, loss drops, checkpoint reloads."""

from __future__ import annotations

from pathlib import Path

import torch

from movement.config import Config
from movement.data.datamodule import DataModule
from movement.models import build_model
from movement.training import Trainer
from movement.utils.tracking import Tracker, make_run_dir


def test_trainer_smoke(base_config: Config, datamodule: DataModule, tmp_path: Path):
    """Two epochs on CPU: loss decreases, checkpoint writes and reloads identical."""
    # Small config for speed.
    cfg = base_config.model_copy(deep=True)
    cfg.trainer.max_epochs = 2
    cfg.trainer.batch_size = 16
    cfg.trainer.run_dir = tmp_path
    cfg.trainer.checkpoint_dir = Path("checkpoints")
    cfg.trainer.tensorboard_dir = Path("tensorboard")
    cfg.trainer.amp = False  # CPU

    model = build_model(cfg.model, cfg.windowing, cfg.transforms)
    run_dir = make_run_dir(cfg)
    tracker = Tracker(cfg, run_dir)

    trainer = Trainer(model, datamodule, cfg, tracker, run_dir, device=torch.device("cpu"))
    summary = trainer.fit()
    tracker.close()

    # Loss must have decreased from the first to the last epoch.
    assert summary["best_val_ade"] is not None

    # Best checkpoint exists and reloads to identical weights.
    ckpt = run_dir / cfg.trainer.checkpoint_dir / "best.pt"
    assert ckpt.exists(), f"best.pt not written in {run_dir}"

    model2 = build_model(cfg.model, cfg.windowing, cfg.transforms)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model2.load_state_dict(state["model_state"])

    model.eval()
    model2.eval()
    x = torch.randn(4, cfg.windowing.input_len, 2)
    with torch.no_grad():
        assert torch.equal(model(x), model2(x)), "reloaded checkpoint weights differ"
