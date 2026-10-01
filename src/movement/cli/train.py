"""Training entrypoint: ``uv run train --config configs/model/tcn.yaml [overrides...]``."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from movement.config import Config, load_config
from movement.data.datamodule import DataModule
from movement.data.transforms import REPRESENTATION_FILE, WINDOW_REPRESENTATION
from movement.models import build_model
from movement.training import Trainer
from movement.utils.env import load_env
from movement.utils.logging import setup_logging
from movement.utils.seeding import seed_everything
from movement.utils.tracking import (
    Tracker,
    make_run_dir,
    write_config_snapshot,
    write_manifest,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a movement-forecasting model.")
    p.add_argument("--config", default=None, help="Model-arm YAML (e.g. configs/model/tcn.yaml).")
    p.add_argument("--model", default=None, help="Model name (e.g. tcn). Overrides config.")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override as --override key=value (repeatable).",
    )
    p.add_argument("--split-file", default=None, help="Reuse a persisted split.json (exact same IDs).")
    p.add_argument("--no-amp", action="store_true", help="Disable AMP mixed precision.")
    p.add_argument("--deterministic", action="store_true", help="Enable cuDNN determinism (slower).")
    p.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds for a multi-seed sweep, e.g. 42,7,123. "
        "Each seed trains a separate run dir; the split file is reused as-is.",
    )
    return p


def _run_one_seed(config: Config, split_file: Path | None, device: torch.device) -> None:
    """Train a single seed into its own run directory (shared plumbing)."""
    seed_everything(config.trainer.seed, deterministic=config.trainer.deterministic)

    if split_file:
        # Covariate statistics are refit on this split's training animals: the
        # split's source run may have used a different covariate set.
        dm = DataModule.from_split_file(config, split_file, refit_covariate_scaler=True)
        logger.info("Reusing split file: %s", split_file)
    else:
        dm = DataModule.build(config)
    logger.info(
        "Splits: train=%d windows (%d individuals), val=%d, test=%d",
        len(dm.train_windows), len({w.individual_id for w in dm.train_windows}),
        len(dm.val_windows), len(dm.test_windows),
    )

    model = build_model(config.model, config.windowing, config.transforms, data_spec=dm.model_data_spec())
    n_params = model.count_parameters()
    logger.info("Model '%s': %d parameters", config.model.name, n_params)

    run_dir = make_run_dir(config)
    write_config_snapshot(run_dir, config)
    (run_dir / REPRESENTATION_FILE).write_text(WINDOW_REPRESENTATION + "\n", encoding="utf-8")
    dm.persist_split(run_dir)
    if dm.scaler is not None:
        dm.persist_scaler(run_dir)
    dm.persist_covariates(run_dir)

    tracker = Tracker(config, run_dir)
    start = time.time()
    trainer = Trainer(model, dm, config, tracker, run_dir, device=device)
    summary = trainer.fit()
    end = time.time()

    write_manifest(
        run_dir,
        config,
        start_time=start,
        end_time=end,
        params=n_params,
        train_metrics={
            "best_val_ade": summary["best_val_ade"],
            "best_epoch": summary.get("best_epoch"),
            "selection_metric": summary.get("selection_metric"),
            "best_val_metric": summary.get("best_val_metric"),
            "epochs_run": summary.get("epochs_run"),
            "max_epochs": summary.get("max_epochs"),
            "stopped_early": summary.get("stopped_epoch", -1) >= 0,
            "n_train_windows": len(dm.train_windows),
            "n_val_windows": len(dm.val_windows),
            "window_representation": WINDOW_REPRESENTATION,
        },
    )
    tracker.close()
    logger.info(
        "Training finished. Best val ADE: %s m. Artifacts in %s",
        summary["best_val_ade"],
        run_dir,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    load_env()

    overrides = list(args.override)
    if args.model:
        overrides.append(f"model.name={args.model}")
    if args.no_amp:
        overrides.append("trainer.amp=false")
    if args.deterministic:
        overrides.append("trainer.deterministic=true")

    config = load_config(args.config, overrides)
    setup_logging(config.trainer.log_level)
    logger.info("Loaded config: %s", args.config or "base only")

    # Fail loudly when CUDA was expected but missing.
    if torch.cuda.is_available():
        logger.info(
            "CUDA device: %s (capability %s, %.0f MB)",
            torch.cuda.get_device_name(0),
            ".".join(map(str, torch.cuda.get_device_capability(0))),
            torch.cuda.get_device_properties(0).total_memory / 1024**2,
        )
    else:
        logger.warning("CUDA is NOT available — running on CPU (slow, but valid for smoke runs).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_file = Path(args.split_file) if args.split_file else None

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
        for seed in seeds:
            logger.info("=== Seed %d / %s ===", seed, seeds)
            seed_config = config.model_copy(deep=True)
            seed_config.trainer.seed = seed
            _run_one_seed(seed_config, split_file, device)
        return

    _run_one_seed(config, split_file, device)


if __name__ == "__main__":
    main()
