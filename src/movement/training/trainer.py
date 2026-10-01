"""Model-agnostic training loop.

The trainer knows nothing about convolutions, LSTMs, or attention: it accepts
any :class:`ForecastModel` plus config/datamodule/tracker and runs the standard
train/val loop with AMP, gradient clipping, gradient accumulation, checkpointing,
early stopping, and LR scheduling. Adding a new model arm never touches this file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from movement.config import Config
from movement.data.datamodule import DataModule, unpack_batch
from movement.evaluation.scoring import energy_score, spatial_median
from movement.models.base import ForecastModel
from movement.training.callbacks import CheckpointManager, EarlyStopping, build_scheduler
from movement.training.losses import displacement_mse
from movement.utils.seeding import seed_everything
from movement.utils.tracking import Tracker

logger = logging.getLogger(__name__)


class Trainer:
    """Standard supervised training loop for any ForecastModel."""

    def __init__(
        self,
        model: ForecastModel,
        datamodule: DataModule,
        config: Config,
        tracker: Tracker,
        run_dir: Path,
        device: torch.device | None = None,
    ):
        self.model = model
        self.dm = datamodule
        self.config = config
        self.tracker = tracker
        self.run_dir = run_dir
        tc = config.trainer

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.trainer.amp and self.device.type != "cuda":
            logger.warning("AMP requested but device is %s; running without AMP.", self.device.type)
        self.amp_enabled = config.trainer.amp and self.device.type == "cuda"
        self.model.to(self.device)

        # Parameters flagged ``no_weight_decay`` (e.g. FaunaFormer's fusion gate) get
        # their own group without decay; every other model is unaffected.
        decay = [p for p in model.parameters() if not getattr(p, "no_weight_decay", False)]
        no_decay = [p for p in model.parameters() if getattr(p, "no_weight_decay", False)]
        groups = [{"params": decay, "weight_decay": tc.weight_decay}]
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        self.optimizer = AdamW(groups, lr=tc.lr, weight_decay=tc.weight_decay)
        train_len = len(datamodule.train_windows)
        steps_per_epoch = max(1, train_len // tc.batch_size // tc.gradient_accumulation_steps)
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=tc.max_epochs * steps_per_epoch,
            warmup_fraction=tc.warmup_fraction,
        )
        self.scaler = torch.amp.GradScaler(enabled=self.amp_enabled)
        self.early_stopping = EarlyStopping(patience=tc.early_stopping_patience)
        self.checkpoints = CheckpointManager(run_dir, run_dir / tc.checkpoint_dir, metric="val_ade", mode="min")

        # Probabilistic models are selected (and early-stopped) on the validation
        # energy score, the quantity they are trained on; point models on ADE.
        self.probabilistic = bool(getattr(model, "is_probabilistic", False))
        self.selection_metric = "val_es" if self.probabilistic else "val_ade"
        # Destination-hexagon head: joint loss = energy score + weight * cross-entropy.
        self.hex_grid = None
        self.hex_loss_weight = float(getattr(config.model, "hex_loss_weight", 0.0))
        if getattr(model, "hex_rings", 0) > 0:
            if not self.probabilistic:
                raise ValueError("model.hex_rings > 0 needs model.probabilistic=true.")
            from movement.evaluation.hexgrid import HexGrid

            self.hex_grid = HexGrid(model.hex_rings, model.hex_edge_m)
        self.best_val_metric: float | None = None
        self.best_val_ade: float | None = None
        self.best_epoch: int | None = None
        self.epochs_run = 0
        self.start_epoch = 0

    def _to_device(self, x: Tensor) -> Tensor:
        return x.to(self.device)

    def fit(self, epochs: int | None = None) -> dict:
        """Run the train/val loop; returns final training summary."""
        seed_everything(self.config.trainer.seed, deterministic=self.config.trainer.deterministic)
        epochs = epochs or self.config.trainer.max_epochs
        train_loader = self.dm.dataloader("train", shuffle=True)
        val_loader = self.dm.dataloader("val", shuffle=False)

        for epoch in range(self.start_epoch, epochs):
            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics = self._validate(val_loader, epoch)
            self.scheduler.step()

            self.epochs_run = epoch + 1
            summary = {**train_metrics, **val_metrics, "epoch": epoch}
            self._log_epoch(summary)
            self._checkpoint(epoch, val_metrics, train_loader)
            if self.early_stopping(val_metrics[self.selection_metric], epoch):
                logger.info("Early stopping at epoch %d.", epoch)
                break

        return self._summary()

    def _train_epoch(self, loader: DataLoader, epoch: int) -> dict:
        import tqdm

        self.model.train()
        total_loss = 0.0
        n_batches = 0
        accum = self.config.trainer.gradient_accumulation_steps
        total_grad_norm = 0.0
        bar = tqdm.tqdm(loader, desc=f"Epoch {epoch} [train]", unit="batch", leave=False)
        for step, batch in enumerate(bar):
            x, y, dt_seconds, extras = unpack_batch(batch, self.device)
            with torch.autocast(device_type="cuda", enabled=self.amp_enabled):
                pred = self.model(
                    x,
                    context={"stage": "train", "epoch": epoch, "targets": y, "dt_seconds": dt_seconds, **extras},
                )
            if self.probabilistic:
                loss = self._energy_loss(pred, y)
                if self.hex_grid is not None:
                    loss = loss + self.hex_loss_weight * self._hex_loss(y)
            else:
                with torch.autocast(device_type="cuda", enabled=self.amp_enabled):
                    loss = displacement_mse(pred, y)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss ({loss.item()}) at epoch {epoch}, step {step}. "
                    "Aborting — see README for AMP/RNN caveats."
                )
            self.scaler.scale(loss).backward()
            total_loss += loss.item()
            n_batches += 1
            if (step + 1) % accum == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.trainer.clip_grad_norm
                )
                total_grad_norm += float(grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            bar.set_postfix(loss=f"{loss.item():.4f}")
        # Flush any partial accumulation.
        if n_batches % accum != 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.trainer.clip_grad_norm
            )
            total_grad_norm += float(grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        return {
            "train_loss": total_loss / max(1, n_batches),
            "grad_norm": total_grad_norm / max(1, n_batches // accum),
        }

    def _energy_loss(self, samples: Tensor, y: Tensor) -> Tensor:
        """Mean energy score of sampled paths, in units of the train RMS step.

        Computed in float32 outside autocast (square roots of near-zero
        pairwise distances are not fp16-safe).
        """
        scale = self.model.target_scale.float()
        pos = torch.cumsum(samples.float(), dim=2) / scale  # (B, M, H, 2)
        tgt = torch.cumsum(y.float(), dim=1) / scale  # (B, H, 2)
        return energy_score(pos, tgt).mean()

    def _hex_loss(self, y: Tensor) -> Tensor:
        """Cross-entropy of the hex head for the cell of the final true position."""
        dest = torch.sum(y.detach().float(), dim=1).cpu().numpy()  # (B, 2) metres
        cells = torch.from_numpy(self.hex_grid.assign(dest)).to(self.device)
        return torch.nn.functional.cross_entropy(self.model.last_hex_logits.float(), cells)

    def _validate(self, loader: DataLoader, epoch: int) -> dict:
        import tqdm

        self.model.eval()
        from movement.evaluation.metrics import ade, fde

        total_ade = 0.0
        total_fde = 0.0
        total_es = 0.0
        n = 0
        with torch.no_grad():
            bar = tqdm.tqdm(loader, desc=f"Epoch {epoch} [val]", unit="batch", leave=False)
            for batch in bar:
                x, y, dt_seconds, extras = unpack_batch(batch, self.device)
                pred = self.model(
                    x,
                    context={"stage": "val", "epoch": epoch, "dt_seconds": dt_seconds, **extras},
                )
                if self.probabilistic:
                    # Point forecast = per-step spatial median of the sampled positions.
                    pos = torch.cumsum(pred.float(), dim=2)
                    ypos = torch.cumsum(y.float(), dim=1)
                    total_es += float(energy_score(pos, ypos).mean()) * len(x)
                    pred = torch.diff(spatial_median(pos), dim=1,
                                      prepend=torch.zeros_like(pos[:, 0, :1]))
                # Targets are per-step deltas; cumsum for conventional ADE/FDE on
                # cumulative positions (consistent with the eval entrypoint).
                err = torch.cumsum(y.float() - pred.float(), dim=1).detach().cpu().numpy()
                total_ade += ade(err) * len(x)
                total_fde += fde(err) * len(x)
                n += len(x)
        out = {"val_ade": total_ade / max(1, n), "val_fde": total_fde / max(1, n)}
        if self.probabilistic:
            out["val_es"] = total_es / max(1, n)
        return out

    def _log_epoch(self, summary: dict) -> None:
        self.tracker.log_scalar("train/loss", summary["train_loss"], summary["epoch"])
        self.tracker.log_scalar("train/grad_norm", summary.get("grad_norm", 0.0), summary["epoch"])
        self.tracker.log_scalar("val/ade", summary["val_ade"], summary["epoch"])
        self.tracker.log_scalar("val/fde", summary["val_fde"], summary["epoch"])
        self.tracker.log_scalar("lr", self.optimizer.param_groups[0]["lr"], summary["epoch"])
        if "val_es" in summary:
            self.tracker.log_scalar("val/es", summary["val_es"], summary["epoch"])
        logger.info(
            "Epoch %d | train_loss=%.4f | grad_norm=%.3f | val_ade=%.3f m | val_fde=%.3f m%s",
            summary["epoch"], summary["train_loss"], summary.get("grad_norm", 0.0),
            summary["val_ade"], summary["val_fde"],
            f" | val_es={summary['val_es']:.3f} m" if "val_es" in summary else "",
        )

    def _checkpoint(self, epoch: int, val_metrics: dict, _train_loader: DataLoader) -> None:
        value = val_metrics[self.selection_metric]
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "epoch": epoch,
            "config": self.config.model_dump(mode="json"),
        }
        path = self.checkpoints.maybe_save(state, value, epoch)
        if path is not None:
            self.best_val_metric = value
            self.best_val_ade = val_metrics["val_ade"]
            self.best_epoch = epoch
            logger.info("New best %s: %.3f m (saved %s)", self.selection_metric, value, path)

    def _summary(self) -> dict:
        return {
            "best_val_ade": self.best_val_ade,
            "selection_metric": self.selection_metric,
            "best_val_metric": self.best_val_metric,
            "stopped_epoch": self.early_stopping.stopped_epoch,
            "best_epoch": self.best_epoch,
            "epochs_run": self.epochs_run,
            "max_epochs": self.config.trainer.max_epochs,
            "last_checkpoint": str(self.checkpoints._path("last")),
            "best_checkpoint": str(self.checkpoints._path("best")),
        }

    @classmethod
    def load_checkpoint(cls, checkpoint_path: Path, model: ForecastModel, device: torch.device) -> dict:
        """Load model weights from a checkpoint; returns the saved state dict."""
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        return state
