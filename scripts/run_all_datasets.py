"""Run all three model arms on every dataset in RAW_DATASET_PATH.

For each CSV in the raw directory:

1. Detect its nominal sampling interval (median interval across individuals).
2. Scale the 24h/12h windowing protocol to that interval → input_len/horizon.
3. Build a per-dataset split (seed 42), persisted under ``runs/<dataset>/split.json``.
4. Train all three arms (tcn, lstm, transformer) against that same split.
5. Evaluate each arm and write a per-dataset summary into ``reports/by_dataset/``.

Existing runs under ``runs/`` (including the original deer experiment) are
never touched — each dataset's artifacts land in its own ``runs/<dataset>/``
subtree, and ``--datasets deer.csv`` lets you target a single file.

Usage:
    uv run python scripts/run_all_datasets.py                          # all CSVs
    uv run python scripts/run_all_datasets.py --datasets deer.csv      # one dataset
    uv run python scripts/run_all_datasets.py --datasets a.csv,b.csv --seeds 42,7
    uv run python scripts/run_all_datasets.py --dry-run                # plan only, no training
    uv run python scripts/run_all_datasets.py --epochs 1 --batch-size 64   # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

from movement.data.sampling import SamplingProfile, detect_sampling, scale_windowing
from movement.utils.env import load_env, raw_dataset_path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
# The three arms of the study.
ARM_CONFIGS = {
    "tcn": "configs/model/tcn.yaml",
    "lstm": "configs/model/lstm.yaml",
    "transformer": "configs/model/transformer.yaml",
}
DEFAULT_SEEDS = [42]
DATASET_SPLIT_SEED = 42


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run all three arms on every raw dataset.")
    p.add_argument("--datasets", default=None, help="Comma-separated CSV names to run (default: all in raw/).")
    p.add_argument("--seeds", default=None, help="Comma-separated seeds per arm (default: 42).")
    p.add_argument("--epochs", type=int, default=None, help="Override trainer.max_epochs (smoke tests).")
    p.add_argument("--batch-size", type=int, default=None, help="Override trainer.batch_size.")
    p.add_argument("--raw-path", default=None, help="Override the raw dataset dir (default: RAW_DATASET_PATH).")
    p.add_argument(
        "--max-speed",
        type=float,
        default=None,
        help="Max plausible step speed in m/s (default: auto — 25 m/s when the "
        "dataset's nominal interval is finer than the 15-min floor, else off). "
        "Steps exceeding this split the trajectory like a gap.",
    )
    p.add_argument("--dry-run", action="store_true", help="Detect sampling + print the plan, run nothing.")
    p.add_argument("--skip-eval", action="store_true", help="Train only; skip evaluation pass.")
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Do not train/eval; only regenerate reports/by_dataset/<name>.md from existing metrics.json.",
    )
    return p


def _dataset_name(csv: Path) -> str:
    """Stem of the CSV, lowercased and sanitised for use as a directory name."""
    return csv.stem.lower().replace(" ", "_")


def _make_dataset_config(
    dataset_name: str,
    csv: Path,
    profile: SamplingProfile,
    raw_path: Path,
    max_speed: float | None,
) -> dict:
    """Per-dataset config overrides: windowing scaled to the detected sampling.

    The train stride is scaled by how much finer the raw sampling is than the
    effective (floored) interval: for burst GPS (10-second fixes) the raw dt is
    ~90× finer than the 15-minute floor, so we keep every ~90th window instead
    of generating ~90× more near-identical ones (which OOMs on long segments).

    ``max_speed`` defaults to 25 m/s for fine-sampled data (burst GPS has
    relocation glitches that produce kilometre-scale fake steps) and is left
    off otherwise.
    """
    from movement.data.sampling import MIN_DT_HOURS

    input_len, horizon, max_gap = scale_windowing(profile.nominal_dt_hours)
    # Stride = how many *raw* fixes make one effective (floored) fix. For
    # 10-second fixes vs the 15-min floor that is 0.25/0.00278 ≈ 90 — take
    # every 90th window instead of generating 90× more near-identical ones
    # (which OOMs on long segments). Capped at input_len to keep overlap.
    stride = max(1, min(input_len, int(round(MIN_DT_HOURS / max(profile.nominal_dt_hours, 1e-9)))))
    if stride > 1:
        logger.info(
            "  Sampling %.4g h is finer than the %.2f h floor — using train stride=%d "
            "to keep the window count bounded.",
            profile.nominal_dt_hours, MIN_DT_HOURS, stride,
        )
    if max_speed is None and stride > 1:
        max_speed = 25.0  # fine-sampled data: split at relocation-speed jumps
        logger.info("  Enabling speed filter (max_speed_mps=%.0f) for fine-sampled data.", max_speed)
    return {
        "data": {
            "raw_path": str(raw_path),
            "raw_csv": csv.name,
            "nominal_dt_hours": profile.nominal_dt_hours,
            "max_gap_multiplier": max_gap,
            "max_speed_mps": max_speed,
        },
        "windowing": {"input_len": input_len, "horizon": horizon, "stride": stride, "eval_stride": horizon},
        "trainer": {"run_dir": f"runs/{dataset_name}"},
    }


def _write_dataset_manifest(
    dataset_dir: Path,
    csv: Path,
    profile: SamplingProfile,
    input_len: int,
    horizon: int,
    max_gap: float,
    max_speed: float | None,
) -> None:
    """Persist the detected sampling + windowing plan for a dataset."""
    payload = {
        "dataset": csv.name,
        "nominal_dt_hours": profile.nominal_dt_hours,
        "median_dt_hours": profile.median_dt_hours,
        "mode_dt_hours": profile.mode_dt_hours,
        "fraction_regular": profile.fraction_regular,
        "scaled_input_len": input_len,
        "scaled_horizon": horizon,
        "max_gap_multiplier": max_gap,
        "max_speed_mps": max_speed,
    }
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "sampling.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_split(dataset_cfg: dict, dataset_dir: Path) -> Path:
    """Build (or reuse) the per-dataset split file via a 0-epoch training run.

    Reuses the datamodule's split logic by running ``train`` with the given
    dataset config and a fresh run dir; the split (and its train-fitted
    scalers) are persisted by the trainer. Returns the path to split.json.
    """
    run_dir = Path(dataset_cfg["trainer"]["run_dir"]) / f"splitbuild-{int(time.time())}"
    cmd = [
        "uv", "run", "train",
        "--config", "configs/model/tcn.yaml",  # config content irrelevant; split only
        "--override", f"data.raw_path={dataset_cfg['data']['raw_path']}",
        "--override", f"data.raw_csv={dataset_cfg['data']['raw_csv']}",
        "--override", f"data.nominal_dt_hours={dataset_cfg['data']['nominal_dt_hours']}",
        "--override", f"data.max_gap_multiplier={dataset_cfg['data']['max_gap_multiplier']}",
        "--override", f"windowing.input_len={dataset_cfg['windowing']['input_len']}",
        "--override", f"windowing.horizon={dataset_cfg['windowing']['horizon']}",
        "--override", f"windowing.stride={dataset_cfg['windowing']['stride']}",
        "--override", f"windowing.eval_stride={dataset_cfg['windowing']['eval_stride']}",
        "--override", f"trainer.run_dir={run_dir}",
        "--override", "trainer.max_epochs=0",
    ]
    speed = dataset_cfg["data"].get("max_speed_mps")
    if speed is not None:
        cmd += ["--override", f"data.max_speed_mps={speed}"]
    subprocess.run(cmd, check=True)
    # The trainer nests a timestamped run dir under run_dir; locate split.json.
    split_candidates = list(run_dir.rglob("split.json"))
    if not split_candidates:
        raise RuntimeError(f"Split not produced under {run_dir}")
    split = split_candidates[0]
    nested = split.parent
    # Promote the split AND its train-fitted scalers up to the dataset dir (eval
    # and later arms load both from there), then discard the throwaway run dir.
    target = dataset_dir / "split.json"
    split.replace(target)
    scalers = nested / "scalers.json"
    if scalers.exists():
        scalers.replace(dataset_dir / "scalers.json")
    import shutil

    shutil.rmtree(run_dir, ignore_errors=True)
    return target


def _train_arm(arm: str, dataset_cfg: dict, dataset_dir: Path, seed: int, epochs: int | None, batch_size: int | None) -> None:
    """Train one arm on one dataset seed with the shared split."""
    run_dir = Path(dataset_cfg["trainer"]["run_dir"])
    split = dataset_dir / "split.json"
    cmd = [
        "uv", "run", "train",
        "--config", ARM_CONFIGS[arm],
        "--split-file", str(split),
        "--seeds", str(seed),
        "--override", f"data.raw_path={dataset_cfg['data']['raw_path']}",
        "--override", f"data.raw_csv={dataset_cfg['data']['raw_csv']}",
        "--override", f"data.nominal_dt_hours={dataset_cfg['data']['nominal_dt_hours']}",
        "--override", f"data.max_gap_multiplier={dataset_cfg['data']['max_gap_multiplier']}",
        "--override", f"windowing.input_len={dataset_cfg['windowing']['input_len']}",
        "--override", f"windowing.horizon={dataset_cfg['windowing']['horizon']}",
        "--override", f"windowing.stride={dataset_cfg['windowing']['stride']}",
        "--override", f"windowing.eval_stride={dataset_cfg['windowing']['eval_stride']}",
        "--override", f"trainer.run_dir={run_dir}",
    ]
    speed = dataset_cfg["data"].get("max_speed_mps")
    if speed is not None:
        cmd += ["--override", f"data.max_speed_mps={speed}"]
    if epochs is not None:
        cmd += ["--override", f"trainer.max_epochs={epochs}"]
    if batch_size is not None:
        cmd += ["--override", f"trainer.batch_size={batch_size}"]
    logger.info("Training %s (seed %d) on %s", arm, seed, dataset_cfg["data"]["raw_csv"])
    subprocess.run(cmd, check=True)


def _eval_arm(arm: str, dataset_dir: Path) -> None:
    """Evaluate the best checkpoint of every seed run of an arm under a dataset dir."""
    run_dirs = sorted(
        p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("2") and f"-{arm}-s" in p.name
    )
    if not run_dirs:
        logger.warning("No %s run dirs under %s; skipping eval.", arm, dataset_dir)
        return
    for run_dir in run_dirs:
        if (run_dir / "checkpoints" / "best.pt").exists():
            logger.info("Evaluating %s", run_dir)
            subprocess.run(["uv", "run", "eval", "--run", str(run_dir)], check=True)


def _collect_metrics(dataset_dir: Path) -> dict:
    """Aggregate per-arm metrics.json under a dataset dir into a summary dict."""
    summary: dict[str, dict] = {}
    for arm in ARM_CONFIGS:
        run_dirs = sorted(
            p for p in dataset_dir.iterdir() if p.is_dir() and f"-{arm}-s" in p.name
        )
        for run_dir in run_dirs:
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            summary[arm] = {
                "ade": data.get("ade"),
                "fde": data.get("fde"),
                "ade_cp": data.get("ade_cp"),
                "ade_cv": data.get("ade_cv"),
                "run": run_dir.name,
            }
    return summary


def _write_dataset_report(dataset_name: str, dataset_dir: Path, summary: dict) -> None:
    """Write reports/by_dataset/<dataset>.md with the three-arm table."""
    out_dir = REPO_ROOT / "reports" / "by_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {dataset_name} — three-arm comparison",
        "",
        "| arm | ADE (m) | FDE (m) | ADE_cp | ADE_cv | run |",
        "|-----|---------|---------|--------|--------|-----|",
    ]
    for arm in ARM_CONFIGS:
        row = summary.get(arm)
        if row:
            lines.append(
                f"| {arm} | {row['ade']:.2f} | {row['fde']:.2f} | {row['ade_cp']:.2f} | {row['ade_cv']:.2f} | {row['run']} |"
            )
        else:
            lines.append(f"| {arm} | — | — | — | — | — |")
    (out_dir / f"{dataset_name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    load_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    raw = Path(args.raw_path) if args.raw_path else raw_dataset_path()
    csvs = sorted(raw.glob("*.csv"))
    if args.datasets:
        wanted = {d.strip().lower() for d in args.datasets.split(",") if d.strip()}
        csvs = [c for c in csvs if c.stem.lower() in wanted or c.name.lower() in wanted]
    if not csvs:
        raise SystemExit(f"No CSVs found in {raw}")

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(DEFAULT_SEEDS)

    for csv in csvs:
        name = _dataset_name(csv)
        logger.info("=== Dataset: %s (%s) ===", name, csv.name)
        profile = detect_sampling(csv)
        input_len, horizon, max_gap = scale_windowing(profile.nominal_dt_hours)
        logger.info(
            "  nominal_dt=%.4g h, regular=%.1f%%, windowing -> input_len=%d horizon=%d (gap>%d x)",
            profile.nominal_dt_hours, profile.fraction_regular * 100, input_len, horizon, max_gap,
        )
        if profile.fraction_regular < 0.5:
            logger.warning(
                "  Dataset %s is irregular (only %.0f%% of intervals within +/-10%% of the median). "
                "Results may be dominated by the gap-splitting policy.",
                csv.name, profile.fraction_regular * 100,
            )

        dataset_dir = REPO_ROOT / "runs" / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _write_dataset_manifest(dataset_dir, csv, profile, input_len, horizon, max_gap, args.max_speed)

        dataset_cfg = _make_dataset_config(name, csv, profile, raw, args.max_speed)
        if args.dry_run:
            continue
        if args.report_only:
            summary = _collect_metrics(dataset_dir)
            _write_dataset_report(name, dataset_dir, summary)
            logger.info("  Report (from existing metrics): reports/by_dataset/%s.md", name)
            continue

        split = _build_split(dataset_cfg, dataset_dir)
        logger.info("  Split: %s", split)
        for arm in ARM_CONFIGS:
            for seed in seeds:
                _train_arm(arm, dataset_cfg, dataset_dir, seed, args.epochs, args.batch_size)
            if not args.skip_eval:
                _eval_arm(arm, dataset_dir)

        summary = _collect_metrics(dataset_dir)
        _write_dataset_report(name, dataset_dir, summary)
        logger.info("  Report: reports/by_dataset/%s.md", name)


if __name__ == "__main__":
    main()
