"""Hyperparameter-validation sweep: plan | run | summarise (+capacity-match, finalise-test).

Builds and executes the four-arm study described in ``prompts/new_exp.md``:

- **sensitivity** (Arm A): one-dimensional learning-rate sweep at default config.
- **search** (Arm B): equal-budget random search over a shared space + per-arm
  capacity axis (20 trials/arm/dataset).
- **capacity** (Arm C): capacity-matched LSTM/Transformer vs the TCN at defaults.
- **seeds** (Arm D): multi-seed confirmation (42, 7, 123) of the configs that
  survive A–C: the default config, the search-selected config per arm/dataset,
  and the capacity-matched config.

The test split is off limits during search. Every selection decision is made on
val ADE; test is evaluated exactly once, on the finally-selected config per arm,
via the explicit ``finalise-test`` step. The search path physically cannot read
test: it never invokes eval with ``--split test`` (that is the ``finalise-test``
step alone), and the trial runner only ever evaluates the val split.

Usage:
    uv run python scripts/sweep.py plan --study search
    uv run python scripts/sweep.py run --study search
    uv run python scripts/sweep.py capacity-match --dataset wolf_reshaped
    uv run python scripts/sweep.py finalise-test
    uv run python scripts/sweep.py summarise [--fake]

``plan`` writes nothing and runs nothing. ``run`` is resumable: each trial is
keyed by a hash of its resolved config; trials recorded complete in trials.csv
are skipped. Per-trial failure (non-finite loss, OOM) is recorded with a reason
and the sweep continues — a failed trial is never silently dropped or recorded
as success. Pre-flight VRAM estimates against the 8 GB ceiling skip and record
configs that would OOM rather than crashing mid-sweep.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from movement.utils.env import load_env, repo_root

logger = logging.getLogger("sweep")

REPO_ROOT = repo_root()
DEFAULT_SPACE = REPO_ROOT / "configs" / "sweep" / "space.yaml"
RESULTS_ROOT = REPO_ROOT / "results" / "sweeps"
ARMS = ["tcn", "lstm", "transformer"]
ARM_CONFIGS = {
    "tcn": "configs/model/tcn.yaml",
    "lstm": "configs/model/lstm.yaml",
    "transformer": "configs/model/transformer.yaml",
}
STUDIES = ("sensitivity", "search", "capacity", "seeds")

# Time/cost calibration from the 27 completed runs
# (runs/<dataset>/<run>/manifest.json): total wall_clock_seconds per arm per
# dataset at the default config, used by `plan` scaled by relative parameter
# count (the spec's estimate rule).
REFERENCE_WALL_SECONDS: dict[str, dict[str, float]] = {
    "wolf_reshaped": {"tcn": 109.7, "lstm": 25.7, "transformer": 103.4},
    "boar_reshaped": {"tcn": 270.7, "lstm": 231.4, "transformer": 260.4},
    "cougars_reshaped": {"tcn": 334.9, "lstm": 279.9, "transformer": 326.3},
    "wild_pig_reshaped": {"tcn": 605.3, "lstm": 556.5, "transformer": 547.7},
    "african_elephant_reshaped": {"tcn": 388.2, "lstm": 357.3, "transformer": 508.3},
}

# Reference parameter counts at the default config (recorded in manifests and
# verified by instantiating on CPU).
REFERENCE_PARAMS: dict[str, dict[str, int]] = {
    "wolf_reshaped": {"tcn": 102104, "lstm": 219288, "transformer": 417432},
    "boar_reshaped": {"tcn": 102104, "lstm": 219288, "transformer": 417432},
    "cougars_reshaped": {"tcn": 101324, "lstm": 217740, "transformer": 415884},
    "wild_pig_reshaped": {"tcn": 128752, "lstm": 222384, "transformer": 420528},
    "african_elephant_reshaped": {"tcn": 156960, "lstm": 228576, "transformer": 426720},
}

# Reference peak VRAM (MB) at the default config (recorded manifests).
REFERENCE_VRAM: dict[str, dict[str, float]] = {
    "wolf_reshaped": {"tcn": 333.8, "lstm": 159.5, "transformer": 127.9},
    "boar_reshaped": {"tcn": 333.8, "lstm": 159.6, "transformer": 127.9},
    "cougars_reshaped": {"tcn": 104.9, "lstm": 92.8, "transformer": 92.6},
    "wild_pig_reshaped": {"tcn": 1225.3, "lstm": 291.3, "transformer": 204.3},
    "african_elephant_reshaped": {"tcn": 4735.4, "lstm": 557.2, "transformer": 379.6},
}

# Input length at the reference config for each dataset (for TCN VRAM scaling).
REFERENCE_LEN: dict[str, int] = {
    "wolf_reshaped": 24,
    "boar_reshaped": 24,
    "cougars_reshaped": 12,
}

# Trials CSV columns (spec §4).
TRIAL_COLUMNS = [
    "study", "trial_id", "dataset", "arm", "seed", "config_hash",
    "lr", "weight_decay", "dropout", "clip_grad_norm", "warmup_fraction",
    "capacity", "parameter_count", "val_ade", "val_fde", "test_ade", "test_fde",
    "epochs_run", "best_epoch", "wall_clock_s", "peak_vram_mb",
    "status", "failure_reason", "git_commit", "timestamp",
]


# ---------------------------------------------------------------------------
# Space loading
# ---------------------------------------------------------------------------
def load_space(path: Path = DEFAULT_SPACE) -> dict[str, Any]:
    """Load and validate the sweep space YAML."""
    if not path.exists():
        raise FileNotFoundError(f"Sweep space file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for key in ("study_name", "datasets", "shared", "capacity", "seeds"):
        if key not in data:
            raise ValueError(f"Sweep space {path} is missing required key: {key}")
    return data


def studies_path(space: dict[str, Any], study: str) -> Path:
    return RESULTS_ROOT / space["study_name"] / study


# ---------------------------------------------------------------------------
# Trial identity: config_hash
# ---------------------------------------------------------------------------
def canonical_json(obj: Any) -> str:
    """Deterministic JSON for hashing (sorted keys, compact separators)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: dict[str, Any]) -> str:
    """SHA-256 (first 16 hex) of the canonical JSON of a resolved config."""
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Trial building
# ---------------------------------------------------------------------------
@dataclass
class Trial:
    """One sweep trial: everything needed to resolve a config and run it."""

    study: str
    dataset: str
    arm: str
    seed: int
    overrides: dict[str, Any]  # dot-key -> value, applied on top of the arm config
    capacity: str = ""  # human label of the capacity option (if any)
    param_count: int | None = None  # filled by counting on CPU

    @property
    def trial_id(self) -> str:
        h = hashlib.sha256(canonical_json(self.overrides).encode("utf-8")).hexdigest()[:8]
        return f"{self.study}-{self.dataset}-{self.arm}-s{self.seed}-{h}"

    @property
    def run_root(self) -> Path:
        """Directory the CLI writes the timestamped run dir into."""
        return RESULTS_ROOT / self.study / self.dataset / self.trial_id


def apply_dot_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply dot-key overrides to a config dict (deep copy, list values kept)."""
    out = json.loads(json.dumps(cfg))
    for key, value in overrides.items():
        parts = key.split(".")
        node = out
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return out


def dataset_overrides(space: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Data/windowing overrides for a dataset (mirrors the persisted run configs)."""
    ds = space["datasets"][dataset]
    return {
        "data.raw_csv": ds["raw_csv"],
        "data.nominal_dt_hours": ds["nominal_dt_hours"],
        "data.max_gap_multiplier": ds["max_gap_multiplier"],
        "data.max_speed_mps": ds.get("max_speed_mps"),
        "windowing.input_len": ds["input_len"],
        "windowing.horizon": ds["horizon"],
        "windowing.stride": ds["stride"],
        "windowing.eval_stride": ds["eval_stride"],
    }


def resolve_config(space: dict[str, Any], trial: Trial) -> dict[str, Any]:
    """Resolve a trial's full merged config dict (base + arm + dataset + trial)."""
    from movement.config import load_config

    cfg = load_config(ARM_CONFIGS[trial.arm]).model_dump(mode="json")
    cfg = apply_dot_overrides(cfg, dataset_overrides(space, trial.dataset))
    cfg = apply_dot_overrides(cfg, trial.overrides)
    cfg["trainer"]["seed"] = trial.seed
    cfg["trainer"]["run_dir"] = str(trial.run_root)
    return cfg


def build_cli_overrides(space: dict[str, Any], trial: Trial) -> list[str]:
    """Dot-overrides for the train CLI: dataset windowing + trial hyperparams.

    ``None`` values (e.g. max_speed_mps) are skipped — the CLI would coerce the
    string "None" and break pydantic.
    """
    merged = {**dataset_overrides(space, trial.dataset), **trial.overrides}
    out: list[str] = []
    for key, value in merged.items():
        if value is None:
            continue
        if isinstance(value, list):
            out.append(f"{key}={json.dumps(value)}")
        else:
            out.append(f"{key}={value}")
    return out


def count_params_for_cfg(cfg: dict[str, Any]) -> int:
    """Instantiate the model from a resolved config dict on CPU and count params."""
    from movement.config import Config
    from movement.models import build_model

    validated = Config.model_validate(cfg)
    model = build_model(validated.model, validated.windowing, validated.transforms)
    return model.count_parameters()


# ---------------------------------------------------------------------------
# Sampling from the shared space
# ---------------------------------------------------------------------------
def _sample_log_uniform(rng: random.Random, axis: dict[str, Any]) -> float:
    lo, hi = axis["min"], axis["max"]
    return float(10 ** rng.uniform(math.log10(lo), math.log10(hi)))


def sample_search_overrides(space: dict[str, Any], arm: str, rng: random.Random) -> tuple[dict[str, Any], str]:
    """One random draw from the shared space + the arm's capacity axis.

    Returns ``(overrides, capacity_label)``.
    """
    shared = space["shared"]
    overrides = {
        shared["lr"]["key"]: _sample_log_uniform(rng, shared["lr"]),
        shared["weight_decay"]["key"]: _sample_log_uniform(rng, shared["weight_decay"]),
        shared["dropout"]["key"]: rng.choice(shared["dropout"]["options"]),
        shared["clip_grad_norm"]["key"]: rng.choice(shared["clip_grad_norm"]["options"]),
        shared["warmup_fraction"]["key"]: rng.choice(shared["warmup_fraction"]["options"]),
    }
    cap = rng.choice(space["capacity"][arm]["options"])
    for key, value in zip(space["capacity"][arm]["keys"], cap["value"]):
        overrides[key] = value
    return overrides, cap["label"]


# ---------------------------------------------------------------------------
# Capacity matching (Arm C)
# ---------------------------------------------------------------------------
def capacity_match_overrides(space: dict[str, Any], dataset: str, arm: str) -> dict[str, Any] | None:
    """Solve for the arm config whose parameter count matches the TCN within ±tolerance.

    Counts real models on CPU (never an analytic estimate) and prefers the
    config closest to the TCN count. Returns dot-overrides, or None if no config
    lands inside the window.
    """
    tol = space["capacity_match"]["tolerance"]
    tcn_trial = Trial(study="capacity", dataset=dataset, arm="tcn", seed=42, overrides={})
    tcn_params = count_params_for_cfg(resolve_config(space, tcn_trial))
    lo, hi = tcn_params * (1 - tol), tcn_params * (1 + tol)

    candidates: list[dict[str, Any]] = []
    if arm == "lstm":
        for hidden in range(32, 257, 4):
            for layers in (1, 2, 3):
                candidates.append({"model.hidden_size": hidden, "model.num_layers": layers})
    else:  # transformer
        for d_model in range(32, 161, 4):
            for layers in (1, 2, 3):
                for ff_mult in (1, 2):
                    # nhead must divide d_model (embed_dim % num_heads == 0).
                    for nhead in (1, 2, 4, 8):
                        if d_model % nhead == 0:
                            candidates.append({
                                "model.d_model": d_model,
                                "model.num_layers": layers,
                                "model.dim_feedforward": d_model * ff_mult,
                                "model.nhead": nhead,
                            })

    best: tuple[dict[str, Any], int] | None = None
    mid = (lo + hi) / 2
    for overrides in candidates:
        trial = Trial(study="capacity", dataset=dataset, arm=arm, seed=42, overrides=overrides)
        p = count_params_for_cfg(resolve_config(space, trial))
        if lo <= p <= hi:
            if best is None or abs(p - mid) < abs(best[1] - mid):
                best = (overrides, p)
    return best[0] if best else None


# ---------------------------------------------------------------------------
# Trial list construction
# ---------------------------------------------------------------------------
def build_trials(space: dict[str, Any], study: str) -> list[Trial]:
    """Build the ordered trial list for a study (no execution, no writes)."""
    datasets = sorted(space["datasets"])
    trials: list[Trial] = []

    if study == "sensitivity":
        for ds in datasets:
            for arm in ARMS:
                for lr in space["sensitivity"]["lr_values"]:
                    trials.append(Trial(
                        study=study, dataset=ds, arm=arm, seed=42,
                        overrides={"trainer.lr": lr}, capacity=f"lr={lr:g}",
                    ))
        return trials

    if study == "search":
        rng = random.Random(space.get("search_seed", 12345))
        n = space["search_trials_per_arm_dataset"]
        for ds in datasets:
            for arm in ARMS:
                for _ in range(n):
                    overrides, cap_label = sample_search_overrides(space, arm, rng)
                    trials.append(Trial(
                        study=study, dataset=ds, arm=arm, seed=42,
                        overrides=overrides, capacity=cap_label,
                    ))
        return trials

    if study == "capacity":
        for ds in datasets:
            for arm in ("lstm", "transformer"):
                matched = capacity_match_overrides(space, ds, arm)
                if matched is None:
                    raise RuntimeError(
                        f"No capacity-matched config found for {arm} on {ds} "
                        f"within ±{space['capacity_match']['tolerance']:.0%} of the TCN. "
                        "Adjust configs/sweep/space.yaml capacity_match."
                    )
                trials.append(Trial(
                    study=study, dataset=ds, arm=arm, seed=42,
                    overrides=matched, capacity="matched",
                ))
        return trials

    if study == "seeds":
        # Arm D: three config families per arm/dataset — default, search-selected,
        # capacity-matched — × the three seeds.
        for ds in datasets:
            for arm in ARMS:
                families: list[tuple[dict[str, Any], str]] = [({}, "default")]
                sel = _search_selected_overrides(space, ds, arm)
                if sel is not None:
                    families.append((sel, "search-selected"))
                else:
                    logger.info(
                        "seeds plan: no search trials for %s/%s yet — default-only family.",
                        ds, arm,
                    )
                if arm != "tcn":
                    matched = capacity_match_overrides(space, ds, arm)
                    if matched is not None:
                        families.append((matched, "capacity-matched"))
                for overrides, label in families:
                    for seed in space["seeds"]:
                        trials.append(Trial(
                            study=study, dataset=ds, arm=arm, seed=seed,
                            overrides=overrides, capacity=label,
                        ))
        return trials

    raise ValueError(f"Unknown study: {study}")


def _search_selected_overrides(space: dict[str, Any], dataset: str, arm: str) -> dict[str, Any] | None:
    """Best (lowest val ADE) search trial's overrides for an arm/dataset, if any."""
    path = studies_path(space, "search") / "trials.csv"
    if not path.exists():
        return None
    best: dict[str, str] | None = None
    best_ade = float("inf")
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (
                row.get("status") == "complete"
                and row.get("dataset") == dataset
                and row.get("arm") == arm
                and row.get("val_ade")
            ):
                ade = float(row["val_ade"])
                if ade < best_ade:
                    best_ade = ade
                    best = row
    if best is None:
        return None
    shared = space["shared"]
    return {
        shared["lr"]["key"]: float(best["lr"]),
        shared["weight_decay"]["key"]: float(best["weight_decay"]),
        shared["dropout"]["key"]: float(best["dropout"]),
        shared["clip_grad_norm"]["key"]: float(best["clip_grad_norm"]),
        shared["warmup_fraction"]["key"]: float(best["warmup_fraction"]),
    }


# ---------------------------------------------------------------------------
# Wall-clock / VRAM estimation
# ---------------------------------------------------------------------------
def estimate_wall_clock(space: dict[str, Any], trial: Trial) -> float:
    """Estimate trial wall-clock: reference total wall × relative parameter count.

    Reference totals come from the recorded manifests (spec §3: "Estimate from
    the recorded wall_clock_seconds in the existing manifests, scaled by relative
    parameter count"). Early stopping usually cuts the true time below this.
    """
    ref_wall = REFERENCE_WALL_SECONDS.get(trial.dataset, {}).get(trial.arm, 200.0)
    ref_params = REFERENCE_PARAMS.get(trial.dataset, {}).get(trial.arm, 1)
    params = trial.param_count or ref_params
    return ref_wall * (params / ref_params)


def estimate_vram_mb(space: dict[str, Any], trial: Trial) -> float:
    """Pre-flight VRAM estimate, scaled by relative parameter count.

    The TCN's memory grows with both channel width and window length (spec §3:
    it peaked at 6,288 MB on a 128-step window). The window factor applies only
    to the TCN, whose activations scale with input length, relative to that
    dataset's own reference length.
    """
    ref_vram = REFERENCE_VRAM.get(trial.dataset, {}).get(trial.arm, 0.0)
    ref_params = REFERENCE_PARAMS.get(trial.dataset, {}).get(trial.arm, 1)
    params = trial.param_count or ref_params
    scale = params / ref_params
    if trial.arm == "tcn":
        ds = space["datasets"][trial.dataset]
        return ref_vram * scale * (ds["input_len"] / REFERENCE_LEN.get(trial.dataset, 24))
    return ref_vram * scale


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
def plan(space: dict[str, Any], study: str) -> None:
    """Print the trial table and estimated wall-clock; write nothing, run nothing."""
    trials = build_trials(space, study)
    if not trials:
        raise SystemExit(f"Study {study!r} has no trials.")

    total_seconds = 0.0
    per_arm: dict[str, int] = {}
    per_dataset: dict[str, int] = {}
    rows: list[tuple[Trial, int, float]] = []
    for t in trials:
        t.param_count = count_params_for_cfg(resolve_config(space, t))
        est = estimate_wall_clock(space, t)
        total_seconds += est
        per_arm[t.arm] = per_arm.get(t.arm, 0) + 1
        per_dataset[t.dataset] = per_dataset.get(t.dataset, 0) + 1
        rows.append((t, t.param_count, est))

    print(f"# Sweep plan: {study}  ({space['study_name']})")
    print(f"Total trials: {len(trials)}")
    print(f"\n{'trial':<58} {'dataset':<20} {'arm':<12} {'params':>10} {'est_min':>10}")
    print("-" * 115)
    for t, params, est in rows:
        print(f"{t.trial_id:<58} {t.dataset:<20} {t.arm:<12} {params:>10,} {est / 60:>10.1f}")

    print(f"\nTrials per arm:     {per_arm}")
    print(f"Trials per dataset: {per_dataset}")
    print(f"Estimated wall-clock total: {total_seconds / 3600:.2f} GPU-hours")
    print("(plan writes nothing and runs nothing)")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def _load_completed_hashes(trials_path: Path) -> set[str]:
    if not trials_path.exists():
        return set()
    completed: set[str] = set()
    with trials_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "complete" and row.get("config_hash"):
                completed.add(row["config_hash"])
    return completed


def _append_trial_row(trials_path: Path, row: dict[str, Any]) -> None:
    new = not trials_path.exists()
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    with trials_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in TRIAL_COLUMNS})


def _run_train_cli(space: dict[str, Any], trial: Trial, split_file: Path) -> None:
    """Train one trial via the existing train CLI (single seed).

    On Ctrl-C the child is killed so no orphaned training process survives; the
    ``KeyboardInterrupt`` propagates to ``run``, which records the trial as
    interrupted (it is retried on restart, never silently lost).
    """
    cmd = [
        sys.executable, "-m", "movement.cli.train",
        "--config", ARM_CONFIGS[trial.arm],
        "--split-file", str(split_file),
    ]
    for override in build_cli_overrides(space, trial):
        cmd += ["--override", override]
    cmd += ["--override", f"trainer.seed={trial.seed}"]
    cmd += ["--override", f"trainer.run_dir={trial.run_root}"]
    logger.info("Trial %s: %s", trial.trial_id, " ".join(cmd))
    try:
        proc = subprocess.Popen(cmd)
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"train exited {proc.returncode}")


def _find_run_dir(run_root: Path) -> Path:
    """The CLI nests a timestamped run dir under run_root; locate the real one."""
    if (run_root / "manifest.json").exists():
        return run_root
    candidates = sorted(run_root.glob("*/manifest.json"))
    if not candidates:
        raise RuntimeError(f"No manifest.json under {run_root}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple run dirs under {run_root} — ambiguous resume state. "
            f"Clean up partial runs or inspect: {[c.parent.name for c in candidates]}"
        )
    return candidates[0].parent


def _evaluate_split(run_dir: Path, split: str) -> dict[str, Any]:
    """Run the eval CLI on a split; returns its metrics.json contents.

    Same Ctrl-C-safe child handling as ``_run_train_cli``: a KeyboardInterrupt
    kills the child and propagates to ``run`` (which records the trial as
    interrupted) instead of leaving an orphaned eval process.
    """
    cmd = [sys.executable, "-m", "movement.cli.eval", "--run", str(run_dir), "--split", split]
    logger.info("Eval %s: %s", split, " ".join(cmd))
    try:
        proc = subprocess.Popen(cmd)
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        raise
    metrics_path = run_dir / "metrics.json"
    if proc.returncode != 0 or not metrics_path.exists():
        raise RuntimeError(f"eval --split {split} failed for {run_dir}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))

def _cfg_value(cfg: dict[str, Any], dot_key: str) -> Any:
    """Read a dot-key from a config dict (e.g. 'model.dropout' -> 0.2)."""
    node: Any = cfg
    for part in dot_key.split("."):
        node = node[part]
    return node


def _row_for_trial(
    space: dict[str, Any],
    trial: Trial,
    cfg: dict[str, Any],
    *,
    status: str,
    failure_reason: str = "",
    metrics: dict[str, Any] | None = None,
    wall: float = 0.0,
    peak_vram: float = 0.0,
    epochs_run: str = "",
    best_epoch: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "study": trial.study,
        "trial_id": trial.trial_id,
        "dataset": trial.dataset,
        "arm": trial.arm,
        "seed": trial.seed,
        "config_hash": config_hash(cfg),
        "lr": _cfg_value(cfg, space["shared"]["lr"]["key"]),
        "weight_decay": _cfg_value(cfg, space["shared"]["weight_decay"]["key"]),
        "dropout": _cfg_value(cfg, space["shared"]["dropout"]["key"]),
        "clip_grad_norm": _cfg_value(cfg, space["shared"]["clip_grad_norm"]["key"]),
        "warmup_fraction": _cfg_value(cfg, space["shared"]["warmup_fraction"]["key"]),
        "capacity": trial.capacity,
        "parameter_count": trial.param_count or "",
        "val_ade": "",
        "val_fde": "",
        "test_ade": "",
        "test_fde": "",
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "wall_clock_s": round(wall, 1) if wall else "",
        "peak_vram_mb": peak_vram if peak_vram else "",
        "status": status,
        "failure_reason": failure_reason,
        "git_commit": _git_commit(),
        "timestamp": _now(),
    }
    if metrics:
        row["val_ade"] = metrics.get("ade", "")
        row["val_fde"] = metrics.get("fde", "")
    return row


def run(space: dict[str, Any], study: str, *, limit: int | None = None) -> None:
    """Execute the study schedule, resumably. Ctrl-C safe: completed rows persist."""
    trials_path = studies_path(space, study) / "trials.csv"
    completed = _load_completed_hashes(trials_path)
    trials = build_trials(space, study)
    logger.info("Study %s: %d trials, %d already complete.", study, len(trials), len(completed))

    executed = 0
    for trial in trials:
        if limit is not None and executed >= limit:
            break
        cfg = resolve_config(space, trial)
        h = config_hash(cfg)
        if h in completed:
            logger.info("Skip %s (complete, hash %s)", trial.trial_id, h)
            continue

        # Recover: if this trial was recorded failed but its run dir already
        # finished (training + val eval), promote it instead of retraining.
        if _recover_failed_trial(space, trial, trials_path):
            continue

        # Pre-flight VRAM estimate against the 8 GB ceiling.
        trial.param_count = count_params_for_cfg(cfg)
        vram_est = estimate_vram_mb(space, trial)
        if vram_est > space["vram_ceiling_mb"]:
            logger.warning(
                "Skip %s: estimated peak VRAM %.0f MB > ceiling %.0f MB (params=%d)",
                trial.trial_id, vram_est, space["vram_ceiling_mb"], trial.param_count,
            )
            _append_trial_row(trials_path, _row_for_trial(
                space, trial, cfg, status="failed", failure_reason="estimated_vram_oom",
            ))
            continue

        split_file = REPO_ROOT / "runs" / trial.dataset / "split.json"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file} — the 27 existing runs are the "
                f"evidence; their split/scalers are reused as-is."
            )

        start = time.time()
        try:
            _run_train_cli(space, trial, split_file)
            run_dir = _find_run_dir(trial.run_root)
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            # Selection metric: evaluate the val split (test is off limits here).
            val_metrics = _evaluate_split(run_dir, "val")
            wall = time.time() - start
            row = _row_for_trial(
                space, trial, cfg,
                status="complete",
                metrics=val_metrics,
                wall=wall,
                peak_vram=manifest.get("peak_vram_mb", 0.0),
                epochs_run=manifest.get("epochs_run", ""),
                best_epoch=manifest.get("best_epoch", ""),
            )
        except KeyboardInterrupt:
            # Ctrl-C: record the in-flight trial as interrupted (retried on
            # restart) and stop the sweep cleanly — no traceback, nothing lost.
            logger.warning("Interrupted during trial %s — recording and stopping.", trial.trial_id)
            row = _row_for_trial(
                space, trial, cfg,
                status="failed",
                failure_reason="interrupted",
                wall=time.time() - start,
            )
            _append_trial_row(trials_path, row)
            raise SystemExit(
                f"Interrupted during trial {trial.trial_id}. It was recorded as "
                f"failed/interrupted and will be retried on the next `run`."
            )
        except Exception as e:  # noqa: BLE001 — sweep-level fallback is the one deliberate exception
            logger.error("Trial %s FAILED: %s", trial.trial_id, e)
            row = _row_for_trial(
                space, trial, cfg,
                status="failed",
                failure_reason=str(e)[:200],
                wall=time.time() - start,
            )
        _append_trial_row(trials_path, row)
        executed += 1


# ---------------------------------------------------------------------------
# Finalise test (the ONLY place test is evaluated)
# ---------------------------------------------------------------------------
def finalise_test(space: dict[str, Any]) -> None:
    """Evaluate test exactly once, on the finally-selected config per arm/dataset.

    Selection: lowest mean val ADE across the Arm D (seeds) runs; fall back to
    the best search trial if seeds has not run. Fills test_ade/test_fde on the
    selected rows in trials.csv.
    """
    selected = _select_final_configs(space)
    if not selected:
        raise SystemExit("No completed search/seeds trials to select from.")
    print(f"Evaluating test for {len(selected)} finally-selected config(s).")

    for (dataset, arm), trial_id in selected.items():
        trials_path = studies_path(space, "seeds") / "trials.csv"
        run_dir = _locate_run_dir_for_trial(space, dataset, arm, trial_id)
        if run_dir is None:
            logger.warning("No run dir for selected %s/%s/%s — skipping.", dataset, arm, trial_id)
            continue
        metrics = _evaluate_split(run_dir, "test")
        print(f"  {dataset}/{arm} ({trial_id}): test ADE={metrics.get('ade')}, FDE={metrics.get('fde')}")
        _record_test_metrics(trials_path, trial_id, metrics)


def _select_final_configs(space: dict[str, Any]) -> dict[tuple[str, str], str]:
    """trial_id of the finally-selected config per (dataset, arm)."""
    selected: dict[tuple[str, str], str] = {}
    for ds in sorted(space["datasets"]):
        for arm in ARMS:
            best_id: str | None = None
            best_ade = float("inf")
            seeds_path = studies_path(space, "seeds") / "trials.csv"
            if seeds_path.exists():
                with seeds_path.open("r", encoding="utf-8", newline="") as f:
                    rows = [r for r in csv.DictReader(f)
                            if r.get("status") == "complete" and r.get("dataset") == ds
                            and r.get("arm") == arm and r.get("val_ade")]
                if rows:
                    # Mean val ADE per config family (capacity label identifies it).
                    sums: dict[str, list[float]] = defaultdict(list)
                    for r in rows:
                        sums[r["capacity"]].append(float(r["val_ade"]))
                    for key, vals in sums.items():
                        mean = sum(vals) / len(vals)
                        if mean < best_ade:
                            best_ade = mean
                            best_id = next(r["trial_id"] for r in rows if r["capacity"] == key)
            if best_id is None:
                best_id = _best_search_trial_id(space, ds, arm)
            if best_id:
                selected[(ds, arm)] = best_id
    return selected


def _best_search_trial_id(space: dict[str, Any], dataset: str, arm: str) -> str | None:
    path = studies_path(space, "search") / "trials.csv"
    if not path.exists():
        return None
    best_id, best_ade = None, float("inf")
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "complete" and row.get("dataset") == dataset \
                    and row.get("arm") == arm and row.get("val_ade"):
                ade = float(row["val_ade"])
                if ade < best_ade:
                    best_ade, best_id = ade, row["trial_id"]
    return best_id


def _locate_run_dir_for_trial(space: dict[str, Any], dataset: str, arm: str, trial_id: str) -> Path | None:
    """Find the run dir for a trial id across the seeds/search/capacity trees.

    Run dirs live under ``RESULTS_ROOT/<study>/<dataset>/<trial_id>/`` (the
    same path ``Trial.run_root`` uses — note: no ``study_name`` prefix).
    """
    for study in ("seeds", "search", "capacity"):
        root = RESULTS_ROOT / study / dataset / trial_id
        if root.is_dir():
            try:
                return _find_run_dir(root)
            except RuntimeError:
                return None
    return None


def _record_test_metrics(trials_path: Path, trial_id: str, metrics: dict[str, Any]) -> None:
    """Update the selected row's test_ade/test_fde in trials.csv (in place)."""
    if not trials_path.exists():
        return
    rows = list(csv.DictReader(trials_path.open("r", encoding="utf-8")))
    with trials_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_COLUMNS)
        writer.writeheader()
        for row in rows:
            if row["trial_id"] == trial_id:
                row["test_ade"] = metrics.get("ade", "")
                row["test_fde"] = metrics.get("fde", "")
            writer.writerow(row)


def _recover_failed_trial(space: dict[str, Any], trial: Trial, trials_path: Path) -> bool:
    """Promote a failed-row trial to complete if its run dir already finished.

    The harness can fail *after* training and the val-eval both completed (e.g.
    a transient Windows subprocess crash in the eval step). Re-running such a
    trial would waste GPU time. If the run dir has a manifest + best.pt + val
    metrics.json, the row is updated in place and the trial is NOT retrained.
    """
    if not trials_path.exists():
        return False
    run_dir = _find_run_dir_for_trial(space, trial)
    if run_dir is None:
        return False
    manifest_path = run_dir / "manifest.json"
    best_path = run_dir / trial_arm_checkpoint_dir(trial, run_dir)
    metrics_path = run_dir / "metrics.json"
    if not (manifest_path.exists() and best_path.exists() and metrics_path.exists()):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if metrics.get("split") != "val":
        return False

    rows = list(csv.DictReader(trials_path.open("r", encoding="utf-8")))
    updated = False
    with trials_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_COLUMNS)
        writer.writeheader()
        for row in rows:
            if row["trial_id"] == trial.trial_id:
                row["status"] = "complete"
                row["failure_reason"] = ""
                row["val_ade"] = metrics.get("ade", "")
                row["val_fde"] = metrics.get("fde", "")
                row["parameter_count"] = manifest.get("parameter_count", row.get("parameter_count", ""))
                row["wall_clock_s"] = manifest.get("wall_clock_seconds", row.get("wall_clock_s", ""))
                row["peak_vram_mb"] = manifest.get("peak_vram_mb", row.get("peak_vram_mb", ""))
                updated = True
            writer.writerow(row)
    if updated:
        logger.info("Recovered %s: run dir already complete (training + val eval done).", trial.trial_id)
    return updated


def trial_arm_checkpoint_dir(trial: Trial, run_dir: Path) -> Path:
    """Resolve the checkpoint dir from the run's own config snapshot."""
    cfg = _load_run_config(run_dir)
    return Path(cfg.get("trainer", {}).get("checkpoint_dir", "checkpoints")) / "best.pt"


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        try:
            return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            return {}
    return {}


def _find_run_dir_for_trial(space: dict[str, Any], trial: Trial) -> Path | None:
    """Locate the completed run dir for a trial (run_root or its nested child)."""
    root = trial.run_root
    if (root / "manifest.json").exists():
        return root
    candidates = sorted(root.glob("*/manifest.json"))
    if not candidates:
        return None
    if len(candidates) > 1:
        return None
    return candidates[0].parent


# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------
def summarise(space: dict[str, Any], study: str | None = None) -> None:
    """Read trials.csv and emit the markdown report + figures. Standalone on partial sweeps."""
    report_dir = REPO_ROOT / "results" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / "hyperparameter_validation.md"

    all_rows: list[dict[str, str]] = []
    for s in STUDIES:
        p = studies_path(space, s) / "trials.csv"
        if p.exists():
            with p.open("r", encoding="utf-8", newline="") as f:
                all_rows += list(csv.DictReader(f))
    # Dedup by trial_id (last row wins) so interrupted/recovered appends never
    # double-count a trial in the report.
    deduped: dict[str, dict[str, str]] = {}
    for r in all_rows:
        deduped[r["trial_id"]] = r
    all_rows = list(deduped.values())
    if study:
        all_rows = [r for r in all_rows if r.get("study") == study]
    if not all_rows:
        raise SystemExit("No trials found under results/sweeps/<study>/ — run the sweep first (or use --fake).")

    lines: list[str] = [
        "# Hyperparameter Validation Study",
        "",
        "## Protocol statement",
        "",
        "Each arm receives an identical search budget over an identical shared search "
        "space, selected on validation ADE. The capacity-matched arm additionally "
        "controls for parameter count.",
        "",
        "## LR sensitivity",
        "",
        "_See figures/lr_sensitivity_<dataset>.png — the default lr's position on "
        "its curve is the headline evidence._",
        "",
    ]

    # Search results.
    search_rows = [r for r in all_rows if r.get("study") == "search" and r.get("status") == "complete"]
    if search_rows:
        lines += ["## Search results", "",
                  "| dataset | arm | best val ADE | capacity | default rank |",
                  "|---|---|---|---|---|"]
        for ds in sorted({r["dataset"] for r in search_rows}):
            for arm in ARMS:
                arm_rows = [r for r in search_rows if r["dataset"] == ds and r["arm"] == arm]
                if not arm_rows:
                    continue
                arm_rows.sort(key=lambda r: float(r["val_ade"]))
                best = arm_rows[0]
                rank = _default_rank(arm_rows, arm)
                lines.append(
                    f"| {ds} | {arm} | {float(best['val_ade']):.2f} | {best['capacity']} | "
                    f"{rank} / {len(arm_rows)} |"
                )
        lines.append("")

    # Ranking change: default vs tuned vs capacity-matched.
    lines += ["## Does the ranking change?", "",
              "| dataset | default ranking | tuned ranking | capacity-matched ranking |",
              "|---|---|---|---|"]
    for ds in sorted(space["datasets"]):
        lines.append(
            f"| {ds} | {_rank_arm(all_rows, ds, 'sensitivity')} | "
            f"{_rank_arm(all_rows, ds, 'search')} | {_rank_arm(all_rows, ds, 'capacity')} |"
        )
    lines.append("")

    # Multi-seed table.
    seed_rows = [r for r in all_rows if r.get("study") == "seeds" and r.get("status") == "complete"]
    lines += ["## Multi-seed confirmation (mean ± std)", ""]
    if seed_rows:
        lines += ["| dataset | arm | family | ADE mean ± std | test ADE |", "|---|---|---|---|---|"]
        for ds in sorted({r["dataset"] for r in seed_rows}):
            for arm in ARMS:
                fam_rows = [r for r in seed_rows if r["dataset"] == ds and r["arm"] == arm and r["val_ade"]]
                if not fam_rows:
                    continue
                by_fam: dict[str, list[float]] = {}
                test_by_fam: dict[str, str] = {}
                for r in fam_rows:
                    fam = r["capacity"] or "default"
                    by_fam.setdefault(fam, []).append(float(r["val_ade"]))
                    if r.get("test_ade"):
                        test_by_fam[fam] = f"{float(r['test_ade']):.2f}"
                for fam, vals in by_fam.items():
                    mean = sum(vals) / len(vals)
                    std = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
                    overlap = " (no clear difference)" if _intervals_overlap(vals) else ""
                    lines.append(
                        f"| {ds} | {arm} | {fam} | {mean:.2f} ± {std:.2f}{overlap} | "
                        f"{test_by_fam.get(fam, '—')} |"
                    )
        lines.append("")
    else:
        lines += ["_Pending — run `sweep.py run --study seeds`._", ""]

    # Full appendix.
    lines += ["## Appendix: all trials", "",
              "| study | trial_id | dataset | arm | status | val ADE | test ADE | params |",
              "|---|---|---|---|---|---|---|---|"]
    for r in all_rows:
        lines.append(
            f"| {r.get('study')} | {r.get('trial_id')} | {r.get('dataset')} | {r.get('arm')} | "
            f"{r.get('status')} | {r.get('val_ade') or '—'} | {r.get('test_ade') or '—'} | "
            f"{r.get('parameter_count') or '—'} |"
        )
    lines.append("")

    # Honest statement of what was not searched.
    lines += [
        "## What was NOT searched",
        "",
        "- Architecture depth for the TCN is receptive-field-driven, not free; "
        "block count stays determined by the window.",
        "- Window sizes are fixed by the sampling-rate rule (24 h in / 12 h out, "
        "scaled to the detected nominal interval).",
        "- AMP stays off for the LSTM because of the cuDNN caveat.",
        "",
    ]

    _plot_lr_sensitivity(all_rows, figures_dir)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {out_path}")
    print(f"Figures written to {figures_dir}")


def _intervals_overlap(vals: list[float]) -> bool:
    """Overlap heuristic: mean±std of the family is wider than ±10% of the mean."""
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
    return std > mean * 0.1


def _default_rank(arm_rows: list[dict[str, str]], arm: str) -> int:
    """Rank (1-based) of the trial closest to the arm's default hyperparameters."""
    defaults = {
        "tcn": {"lr": 1e-3, "weight_decay": 1e-4},
        "lstm": {"lr": 1e-3, "weight_decay": 1e-4},
        "transformer": {"lr": 3e-4, "weight_decay": 0.01},
    }
    d = defaults[arm]
    best_idx, best_dist = 0, float("inf")
    for i, r in enumerate(arm_rows):
        dist = abs(float(r["lr"]) - d["lr"]) + abs(float(r["weight_decay"]) - d["weight_decay"])
        if dist < best_dist:
            best_dist, best_idx = dist, i
    return best_idx + 1


def _rank_arm(rows: list[dict[str, str]], ds: str, study: str) -> str:
    """Ranking of arms for a dataset under a study, e.g. 'tcn > lstm > transformer'."""
    vals: dict[str, float] = {}
    for r in rows:
        if r.get("dataset") == ds and r.get("study") == study and r.get("status") == "complete" and r.get("val_ade"):
            arm = r["arm"]
            v = float(r["val_ade"])
            if arm not in vals or v < vals[arm]:
                vals[arm] = v
    if not vals:
        return "—"
    return " > ".join(sorted(vals, key=lambda a: vals[a]))


def _plot_lr_sensitivity(rows: list[dict[str, str]], figures_dir: Path) -> None:
    """Val ADE vs lr per arm per dataset (log-x), one panel per dataset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sens = [r for r in rows if r.get("study") == "sensitivity" and r.get("status") == "complete"]
    if not sens:
        return
    for ds in sorted({r["dataset"] for r in sens}):
        fig, ax = plt.subplots(figsize=(8, 5))
        for arm in ARMS:
            pts = [(float(r["lr"]), float(r["val_ade"])) for r in sens
                   if r["dataset"] == ds and r["arm"] == arm and r["val_ade"]]
            if pts:
                pts.sort()
                xs, ys = zip(*pts)
                ax.semilogx(xs, ys, marker="o", label=arm)
        ax.set_xlabel("learning rate")
        ax.set_ylabel("val ADE (m)")
        ax.set_title(f"LR sensitivity — {ds}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / f"lr_sensitivity_{ds}.png", dpi=110)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Synthetic trials (test helper for summarise)
# ---------------------------------------------------------------------------
def _generate_fake_trials(space: dict[str, Any]) -> Path:
    """Write a synthetic trials.csv for testing summarise (spec acceptance)."""
    rng = random.Random(0)
    written: list[Path] = []
    for s in STUDIES:
        path = studies_path(space, s) / "trials.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRIAL_COLUMNS)
            writer.writeheader()
            for ds in sorted(space["datasets"]):
                for arm in ARMS:
                    if s == "sensitivity":
                        for lr in space["sensitivity"]["lr_values"]:
                            writer.writerow(_fake_row(s, ds, arm, 42, {"lr": lr}, f"lr={lr:g}",
                                                      rng.uniform(100, 500), rng.uniform(200, 600)))
                    elif s == "search":
                        for i in range(space["search_trials_per_arm_dataset"]):
                            writer.writerow(_fake_row(s, ds, arm, 42, {"i": i}, "c64",
                                                      rng.uniform(100, 500), rng.uniform(200, 600)))
                    elif s == "capacity":
                        if arm != "tcn":
                            writer.writerow(_fake_row(s, ds, arm, 42, {}, "matched",
                                                      rng.uniform(100, 500), rng.uniform(200, 600)))
                    else:  # seeds
                        for seed in space["seeds"]:
                            writer.writerow(_fake_row(s, ds, arm, seed, {"seed": seed}, "default",
                                                      rng.uniform(100, 500), rng.uniform(200, 600)))
        written.append(path)
    return written[0]


def _fake_row(study: str, ds: str, arm: str, seed: int, overrides: dict[str, Any],
              capacity: str, val_ade: float, val_fde: float) -> dict[str, Any]:
    # Vary lr so the sensitivity figure has real x-spread across the log axis.
    lr = overrides.get("lr", 10 ** random.Random(hash((study, ds, arm)) % 1000).uniform(-4, -2.5))
    return {
        "study": study, "trial_id": f"fake-{study}-{ds}-{arm}-{seed}-{hash(str(overrides)) & 0xffff:04x}",
        "dataset": ds, "arm": arm, "seed": seed,
        "config_hash": config_hash({"s": study, "d": ds, "a": arm, "o": overrides}),
        "lr": lr, "weight_decay": 1e-4, "dropout": 0.2, "clip_grad_norm": 10.0,
        "warmup_fraction": 0.05, "capacity": capacity, "parameter_count": 100000,
        "val_ade": f"{val_ade:.2f}", "val_fde": f"{val_fde:.2f}",
        "test_ade": "", "test_fde": "",
        "epochs_run": 20, "best_epoch": 12, "wall_clock_s": 300, "peak_vram_mb": 200,
        "status": "complete", "failure_reason": "", "git_commit": "fake",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _git_commit() -> str:
    from movement.utils.tracking import git_commit_hash

    return git_commit_hash() or ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hyperparameter-validation sweep (plan | run | summarise).")
    p.add_argument("--space", default=str(DEFAULT_SPACE), help="Path to space.yaml.")
    sub = p.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="Print the trial schedule + wall-clock estimate; writes nothing.")
    p_plan.add_argument("--study", required=True, choices=STUDIES)

    p_run = sub.add_parser("run", help="Execute the schedule (resumable).")
    p_run.add_argument("--study", required=True, choices=STUDIES)
    p_run.add_argument("--limit", type=int, default=None, help="Stop after N trials (testing).")

    p_cap = sub.add_parser("capacity-match", help="Find capacity-matched configs (Arm C).")
    p_cap.add_argument("--dataset", required=True)
    p_cap.add_argument("--arm", default=None, choices=["lstm", "transformer"])

    sub.add_parser("finalise-test", help="Evaluate test ONCE on finally-selected configs.")

    p_sum = sub.add_parser("summarise", help="Emit the report + figures from trials.csv.")
    p_sum.add_argument("--study", default=None, choices=STUDIES)
    p_sum.add_argument("--fake", action="store_true", help="Generate a synthetic trials.csv for testing.")
    return p


def _capacity_match_all(space: dict[str, Any], dataset: str) -> None:
    """Print matched configs for both arms."""
    tcn_trial = Trial(study="capacity", dataset=dataset, arm="tcn", seed=42, overrides={})
    tcn_params = count_params_for_cfg(resolve_config(space, tcn_trial))
    tol = space["capacity_match"]["tolerance"]
    print(f"TCN reference params on {dataset}: {tcn_params:,}  "
          f"(target window [{tcn_params * (1 - tol):,.0f}, {tcn_params * (1 + tol):,.0f}])")
    for arm in ("lstm", "transformer"):
        matched = capacity_match_overrides(space, dataset, arm)
        if matched is None:
            print(f"  {arm}: NO config within ±{tol:.0%} — adjust capacity_match in space.yaml")
            continue
        trial = Trial(study="capacity", dataset=dataset, arm=arm, seed=42, overrides=matched)
        p = count_params_for_cfg(resolve_config(space, trial))
        print(f"  {arm}: params={p:,} (ratio {p / tcn_params:.3f})  overrides={matched}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    load_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    # Model construction logs at INFO on every CPU instantiation (planning counts
    # hundreds of models); keep the sweep's own output readable.
    for noisy in ("movement", "sweep"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    space = load_space(Path(args.space))

    if args.command == "plan":
        plan(space, args.study)
    elif args.command == "run":
        run(space, args.study, limit=args.limit)
    elif args.command == "capacity-match":
        if args.arm:
            matched = capacity_match_overrides(space, args.dataset, args.arm)
            print(f"{args.arm} on {args.dataset}: {matched}")
        else:
            _capacity_match_all(space, args.dataset)
    elif args.command == "finalise-test":
        finalise_test(space)
    elif args.command == "summarise":
        if args.fake:
            path = _generate_fake_trials(space)
            print(f"Generated synthetic trials.csv at {path}")
        summarise(space, study=args.study)


if __name__ == "__main__":
    main()
