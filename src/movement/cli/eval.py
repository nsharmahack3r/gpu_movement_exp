"""Evaluation entrypoint: ``uv run eval --run runs/<id>``.

Reloads a checkpoint plus its persisted split and scalers, runs the test set,
and writes ``metrics.json`` plus per-horizon CSV into the run directory. Also
saves a few predicted-vs-actual trajectory figures.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from movement.config import Config
from movement.data.datamodule import DataModule
from movement.data.transforms import REPRESENTATION_FILE, WINDOW_REPRESENTATION
from movement.evaluation import evaluate_model, save_plots, write_eval_outputs
from movement.models import build_model
from movement.utils.env import load_env
from movement.utils.logging import setup_logging
from movement.utils.tracking import Tracker

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained model checkpoint on the test set.")
    p.add_argument("--run", required=True, help="Run directory under runs/ (e.g. runs/20260818-123456-tcn-s42).")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--checkpoint", default="best", choices=["best", "last"], help="Which checkpoint to load.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    load_env()
    setup_logging("INFO")

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml snapshot in {run_dir} — cannot rebuild the model.")
    config = Config.model_validate(_read_yaml(config_path))

    split_path = run_dir / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"No split.json in {run_dir} — cannot rebuild the split.")
    scalers_path = run_dir / "scalers.json"
    if not scalers_path.exists() and config.transforms.scale:
        raise FileNotFoundError(f"No scalers.json in {run_dir} but transforms.scale=true.")

    check_representation(run_dir)

    dm = DataModule.from_split_file(config, split_path)
    logger.info("Loaded datamodule: test=%d windows", len(dm.test_windows))

    model = build_model(config.model, config.windowing, config.transforms, data_spec=dm.model_data_spec())
    ckpt_dir = run_dir / config.trainer.checkpoint_dir
    ckpt = ckpt_dir / f"{args.checkpoint}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    logger.info("Loaded checkpoint '%s' (epoch %s) from %s", args.checkpoint, state.get("epoch"), ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Seeded so probabilistic samples (and hence ES / coverage) are reproducible.
    from movement.utils.seeding import seed_everything

    seed_everything(config.trainer.seed)
    report, inv, samples = evaluate_model(
        model, dm, device, split=args.split, per_horizon=config.evaluation.per_horizon, return_details=True
    )

    gate = model.fusion_gate_value() if hasattr(model, "fusion_gate_value") else None
    if gate is not None:
        report["fusion_gate"] = gate
    write_eval_outputs(run_dir, report, model_name=config.model.name, split=args.split)

    if args.split == "test":
        save_plots(inv, run_dir, n_plots=config.evaluation.n_plots, seed=config.evaluation.plot_seed,
                   samples=samples)

    if hasattr(model, "covariate_selection_summary") and dm.covariate_columns:
        summary = model.covariate_selection_summary(dm, device, split=args.split)
        if summary is not None:
            path = run_dir / "covariate_selection.csv"
            summary.to_csv(path, index=False)
            logger.info("Wrote mean variable-selection weights per covariate to %s", path)

    tracker = Tracker(config, run_dir)
    for k, v in report.items():
        if k == "per_horizon":
            continue
        if isinstance(v, float):
            tracker.log_scalar(f"eval/{k}", v, 0)
    tracker.close()

    logger.info(
        "Evaluation complete. metrics.json: ADE=%.3f m, FDE=%.3f m, ES=%.3f m (stay put %.3f m, "
        "hour-of-day climatology %.3f m)",
        report["ade"], report["fde"], report["es"], report["es_cp"], report.get("es_clim_hour", float("nan")),
    )


def check_representation(run_dir: Path) -> None:
    """Refuse to evaluate a checkpoint trained under a different window representation.

    Runs made before ``window_representation.txt`` existed used the legacy
    centroid frame (``centroid_v1``), which leaks the target; evaluating them
    with the current transform would feed the model inputs it never saw.
    """
    marker = run_dir / REPRESENTATION_FILE
    found = marker.read_text(encoding="utf-8").strip() if marker.exists() else "centroid_v1"
    if found != WINDOW_REPRESENTATION:
        raise SystemExit(
            f"{run_dir} was trained with window representation '{found}', but this code uses "
            f"'{WINDOW_REPRESENTATION}'. Re-train the run; its existing metrics.json (if any) "
            "belongs to the old representation."
        )


def _read_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
