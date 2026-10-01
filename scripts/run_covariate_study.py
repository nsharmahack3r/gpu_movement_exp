"""Covariate study: does per-fix Sentinel-2 help the Transformer forecast?

For one dataset with a GEE-induced covariate CSV in ``GEE_DATASET_PATH``, train
and evaluate on animal-disjoint splits:

1. ``cov``         — ``cov_transformer`` with covariates. Whichever arm is trained
                     first in a fold writes the split file the other arms reuse.
2. ``nocov``       — the identical architecture minus the covariate branch.
3. ``transformer`` — the original Transformer arm, and optionally ``tcn`` / ``lstm``.
4. ``pcov`` / ``pnocov`` — ``cov`` / ``nocov`` with the probabilistic head
   (sampled paths, trained on the energy score) and random rotation/mirror
   augmentation of training windows. ``nocov_aug`` (optional) is ``nocov`` with
   the augmentation only, to separate the two changes.
5. ``pcov_idx`` (and optional point arm ``cov_idx``) — covariates restricted to the
   six spectral indices (NDVI, EVI, SAVI, NDWI, NDMI, NBR) at the fix.
6. ``faunaformer`` — the six indices fed as within-window *changes* and fused late
   through a near-closed learnable gate, with the probabilistic head and rotation
   augmentation. Optional ablations: ``ff_levels`` (late fusion of raw levels) and
   ``ff_early`` (changes, fused early); ``ff_no_nbr`` (FaunaFormer with NDVI, EVI,
   SAVI, NDWI and NDMI only — NBR removed).

7. ``pmem`` — ``pnocov`` plus memory features (the animal's own position at the
   same clock time on the previous 7 days, and a 14-day home-range summary;
   ``movement.data.memory``). ``pmem_hex`` adds the destination-hexagon head
   (``movement.evaluation.hexgrid``); ``phex`` (optional) is the head without memory.
   Every probabilistic arm is also scored on the hexagon grid (cell NLL, top-1/top-5).

Every run's ``metrics.json`` also carries the energy score (``es``; equal to ADE
for point forecasts) and two no-skill probabilistic baselines built from
randomly rotated training futures: ``es_clim_all`` and ``es_clim_hour`` (same
clock hour). The report's energy-score section compares all arms against them.

Two split modes:

- ``--mode kfold`` (default): animals are dealt into ``--folds`` folds balanced by
  *fix count*; fold k is test, fold k+1 validation, the rest train. Running every
  fold tests each animal exactly once, and every validation set is a full fold.
  This replaces random draws, which on boar (18 animals, 56–6,588 fixes each)
  produced validation sets from 58 to ~1,000 windows — and a 58-window set
  early-stopped three arms at the "no movement" solution.
- ``--val tail`` (k-fold only): no validation fold. Every non-test animal trains,
  and the last 15% (by time) of each training animal is the validation set for
  early stopping. ~80% of animals train instead of ~60%; the test fold (unseen
  animals, each tested once across folds) is unchanged. Runs go to a separate
  ``..._tailval_...`` directory and report.
- ``--mode random``: the earlier protocol (random 70/15/15 animal draws per seed),
  kept so the first study's runs can still be re-reported.

The report (``reports/covariate_study/<dataset>_<tag>.md``) flags two failure
modes instead of averaging them in silently:

- **collapsed** — test ADE within 3% of constant-position: the run converged on
  "predict no movement". Excluded from means and paired comparisons.
- **still improving** — best validation epoch in the last 10% of ``max_epochs``
  without early stopping: undertrained; its number is an upper bound.

Notifications (optional): set ``NTFY_NOTIFICATION_TOPIC`` in ``.env`` to a topic
name and the study reports to ``https://ntfy.sh/<topic>`` — on start, after each
experiment (with its ADE/FDE, flagged when the run collapsed), on failure, and
when the report is rebuilt. Self-host with ``NTFY_NOTIFICATION_URL`` and protect
the topic with ``NTFY_NOTIFICATION_TOKEN``. Unset topic = no notifications.

Usage (on the GPU machine):
    uv run python scripts/run_covariate_study.py --dataset boar_reshaped
    uv run python scripts/run_covariate_study.py --dataset boar_reshaped --arms cov,nocov,transformer,tcn,lstm
    uv run python scripts/run_covariate_study.py --dataset boar_reshaped --folds 6 --fold-seed 7
    uv run python scripts/run_covariate_study.py --dataset boar_reshaped --dry-run
    uv run python scripts/run_covariate_study.py --dataset boar_reshaped --report-only
    uv run python scripts/run_covariate_study.py --dataset boar_reshaped --mode random --report-only   # first study
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from movement.data.sampling import MIN_DT_HOURS, detect_sampling, scale_windowing
from movement.data.transforms import WINDOW_REPRESENTATION
from movement.utils.env import load_env, raw_dataset_path
from movement.utils.notify import notify

logger = logging.getLogger("covariate_study")
REPO_ROOT = Path(__file__).resolve().parents[1]

INDICES_ONLY = 'covariates.include=["^s2_(ndvi|evi|savi|ndwi|ndmi|nbr)$"]'
INDICES_NO_NBR = 'covariates.include=["^s2_(ndvi|evi|savi|ndwi|ndmi)$"]'
# Green-wave covariates (pre-registered, reports/preregistration_green_wave.md):
# NDVI, green-up rate (IRG proxy) and snow fraction at the fix, as levels + changes.
GREEN_WAVE = 'covariates.include=["^s2_(ndvi|ndvi_rate|snow_fraction)$"]'
PROBABILISTIC = ["model.probabilistic=true", "trainer.augment_rotation=true"]
NO_COV = ["covariates.enabled=false", "model.use_covariates=false"]
HEX_RINGS = 10

# arm -> (config file, extra overrides)
ARMS: dict[str, tuple[str, list[str]]] = {
    "cov": ("configs/model/cov_transformer.yaml", []),
    "nocov": ("configs/model/cov_transformer.yaml",
              ["covariates.enabled=false", "model.use_covariates=false"]),
    "transformer": ("configs/model/transformer.yaml", []),
    "tcn": ("configs/model/tcn.yaml", []),
    "lstm": ("configs/model/lstm.yaml", []),
    "pcov": ("configs/model/cov_transformer.yaml",
             ["model.probabilistic=true", "trainer.augment_rotation=true"]),
    "pnocov": ("configs/model/cov_transformer.yaml",
               ["covariates.enabled=false", "model.use_covariates=false",
                "model.probabilistic=true", "trainer.augment_rotation=true"]),
    # The six spectral indices only (no raw bands, 30 m buffers, snow, green-up
    # rate, scene count or composite age): the ecologically interpretable subset.
    "cov_idx": ("configs/model/cov_transformer.yaml", [INDICES_ONLY]),
    "pcov_idx": ("configs/model/cov_transformer.yaml",
                 [INDICES_ONLY, "model.probabilistic=true", "trainer.augment_rotation=true"]),
    # FaunaFormer: six indices as within-window changes, late gated fusion,
    # probabilistic head, rotation augmentation (configs/model/faunaformer.yaml).
    "faunaformer": ("configs/model/faunaformer.yaml", []),
    # Ablations of its two covariate changes (optional arms).
    "ff_levels": ("configs/model/faunaformer.yaml", ["model.covariate_features=levels"]),
    "ff_early": ("configs/model/faunaformer.yaml", ["model.covariate_fusion=early"]),
    # FaunaFormer without NBR: NDVI, EVI, SAVI, NDWI, NDMI only (optional arm).
    "ff_no_nbr": ("configs/model/faunaformer.yaml", [INDICES_NO_NBR]),
    # Green-wave experiment (multi-day, mule deer). Imagery embargo of 9 days in the
    # primary arm: no composite containing imagery after the forecast origin.
    "gw_nocov": ("configs/model/cov_transformer.yaml",
                 ["covariates.enabled=false", "model.use_covariates=false", *PROBABILISTIC]),
    "gw_cov": ("configs/model/cov_transformer.yaml",
               [GREEN_WAVE, "model.covariate_features=both", "covariates.embargo_days=9", *PROBABILISTIC]),
    "gw_cov_centred": ("configs/model/cov_transformer.yaml",
                       [GREEN_WAVE, "model.covariate_features=both", *PROBABILISTIC]),
    "nocov_aug": ("configs/model/cov_transformer.yaml",
                  ["covariates.enabled=false", "model.use_covariates=false", "trainer.augment_rotation=true"]),
    # Memory features and the destination-hexagon head (GPS only, no covariates).
    "pmem": ("configs/model/cov_transformer.yaml", [*NO_COV, *PROBABILISTIC, "transforms.memory=true"]),
    "pmem_hex": ("configs/model/cov_transformer.yaml",
                 [*NO_COV, *PROBABILISTIC, "transforms.memory=true", f"model.hex_rings={HEX_RINGS}"]),
    "phex": ("configs/model/cov_transformer.yaml", [*NO_COV, *PROBABILISTIC, f"model.hex_rings={HEX_RINGS}"]),
}
DEFAULT_ARMS = ["cov", "nocov", "transformer", "pcov", "pnocov", "pcov_idx", "faunaformer"]
PROBABILISTIC_ARMS = {"pcov", "pnocov", "pcov_idx", "faunaformer", "ff_levels", "ff_early", "ff_no_nbr",
                      "gw_nocov", "gw_cov", "gw_cov_centred", "pmem", "pmem_hex", "phex"}
DEFAULT_SPLIT_SEEDS = [42, 7, 123]
# A run whose ADE is this close to constant-position learned nothing.
COLLAPSE_RATIO = 0.97
# Best epoch in the last this-fraction of max_epochs (and no early stop) = undertrained.
LATE_BEST_FRACTION = 0.9


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", required=True, help="Raw CSV stem in RAW_DATASET_PATH, e.g. boar_reshaped.")
    p.add_argument("--mode", choices=["kfold", "random"], default="kfold")
    p.add_argument("--folds", type=int, default=5, help="kfold: number of fix-balanced animal folds.")
    p.add_argument("--val", choices=["fold", "tail"], default="fold",
                   help="kfold: validation = the next fold of animals (fold) or the last 15%% of every "
                        "training animal's track (tail; more training animals).")
    p.add_argument("--fold-seed", type=int, default=42, help="kfold: seed of the fold assignment (and training).")
    p.add_argument("--split-seeds", default=",".join(map(str, DEFAULT_SPLIT_SEEDS)),
                   help="random mode: one random animal split per seed.")
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS), help=f"Subset of {sorted(ARMS)}. The first arm trained in a fold writes the split; "
                        "every other arm (and later runs) reuse it.")
    p.add_argument("--epochs", type=int, default=200, help="trainer.max_epochs (first study used 100).")
    p.add_argument("--patience", type=int, default=30, help="trainer.early_stopping_patience (first study used 15).")
    p.add_argument("--override", action="append", default=[], help="Extra override for every arm (repeatable).")
    p.add_argument("--input-len", type=int, default=None,
                   help="Observed fixes per window (default: 24 h at the detected sampling interval).")
    p.add_argument("--horizon", type=int, default=None,
                   help="Forecast fixes per window (default: 12 h at the detected sampling interval).")
    p.add_argument("--stride", type=int, default=None, help="Training-window stride in fixes.")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and the commands; run nothing.")
    p.add_argument("--report-only", action="store_true", help="Only rebuild the report from existing runs.")
    p.add_argument("--reeval", default="",
                   help="Re-run evaluation (no training) on the latest run of these arms in every unit, "
                        "e.g. 'pnocov,faunaformer' or 'prob' for every probabilistic arm; then rebuild "
                        "the report. Use after evaluation code changes (e.g. calibration metrics).")
    return p


def dataset_overrides(csv: Path, *, input_len: int | None = None, horizon: int | None = None,
                      stride: int | None = None) -> list[str]:
    """Sampling-scaled windowing, identical to scripts/run_all_datasets.py.

    ``input_len`` / ``horizon`` / ``stride`` (in fixes) replace the scaled 24 h / 12 h
    defaults, e.g. for multi-day experiments; validation/test stride stays = horizon.
    """
    profile = detect_sampling(csv)
    auto_in, auto_h, max_gap = scale_windowing(profile.nominal_dt_hours)
    auto_stride = max(1, min(auto_in, int(round(MIN_DT_HOURS / max(profile.nominal_dt_hours, 1e-9)))))
    input_len, horizon = input_len or auto_in, horizon or auto_h
    stride = stride or auto_stride
    out = [
        f"data.raw_path={csv.parent}",
        f"data.raw_csv={csv.name}",
        f"data.nominal_dt_hours={profile.nominal_dt_hours}",
        f"data.max_gap_multiplier={max_gap}",
        f"windowing.input_len={input_len}",
        f"windowing.horizon={horizon}",
        f"windowing.stride={stride}",
        f"windowing.eval_stride={horizon}",
    ]
    if auto_stride > 1:  # high-frequency data subsampled by default: drop GPS glitches
        out.append("data.max_speed_mps=25.0")
    logger.info("%s: nominal dt %.4g h -> input_len=%d horizon=%d stride=%d",
                csv.name, profile.nominal_dt_hours, input_len, horizon, stride)
    return out


def study_layout(args: argparse.Namespace) -> tuple[Path, str, list[tuple[str, list[str]]]]:
    """(study dir, report tag, [(unit label, unit-specific overrides)])."""
    root = REPO_ROOT / "runs" / "covariate_study" / args.dataset
    # Runs are grouped by window representation so a re-run under a new
    # representation never resumes from (or reports) runs made under an old one.
    # Legacy centroid_v1 runs stay where they were (no suffix).
    rep = "" if WINDOW_REPRESENTATION == "centroid_v1" else f"_{WINDOW_REPRESENTATION}"
    if args.mode == "kfold":
        tail = getattr(args, "val", "fold") == "tail"
        tag = f"kfold{args.folds}{'_tailval' if tail else ''}_seed{args.fold_seed}{rep}"
        unit = "individual_kfold_tailval" if tail else "individual_kfold"
        units = [
            (f"fold{k}", [f"data.split_unit={unit}", f"data.n_folds={args.folds}",
                          f"data.fold={k}", f"trainer.seed={args.fold_seed}"])
            for k in range(args.folds)
        ]
        return root / tag, tag, units
    # Legacy random-draw layout: runs/covariate_study/<dataset>/split<seed>/<arm>/.
    units = [(f"split{s}", ["data.split_unit=individual", f"trainer.seed={s}"])
             for s in (int(x) for x in args.split_seeds.split(","))]
    if rep:
        return root / f"random{rep}", f"random{rep}", units
    return root, "random", units


def _latest_run(arm_dir: Path) -> Path | None:
    runs = sorted(p for p in arm_dir.glob("2*") if (p / "checkpoints" / "best.pt").exists())
    return runs[-1] if runs else None


def _run(cmd: list[str], dry: bool) -> None:
    logger.info("$ %s", " ".join(cmd))
    if not dry:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _metrics_line(run: Path) -> tuple[str, bool]:
    """One-line summary of a finished run, and whether it collapsed.

    Mirrors the report's collapse rule: a point arm whose ADE is at least
    ``COLLAPSE_RATIO`` of constant-position learned nothing. Never raises —
    a notification is best-effort and must not fail an otherwise good run.
    """
    try:
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "metrics unavailable", False
    ade, ade_cp = m.get("ade"), m.get("ade_cp")
    bits = [f"ADE {ade:.1f} m" if ade is not None else None,
            f"FDE {m['fde']:.1f} m" if m.get("fde") is not None else None]
    ratio = ade / ade_cp if ade is not None and ade_cp else None
    if ratio is not None:
        bits.append(f"{ratio:.3f}× const-pos")
    if m.get("probabilistic") and m.get("es") is not None:
        bits.append(f"ES {m['es']:.1f} m")
    collapsed = ratio is not None and ratio >= COLLAPSE_RATIO and not m.get("probabilistic", False)
    return " | ".join(b for b in bits if b) or "metrics unavailable", collapsed


def run_unit(label: str, unit_overrides: list[str], arms: list[str], common: list[str],
             study_dir: Path, dataset: str, dry: bool) -> None:
    unit_dir = study_dir / label
    split_file = existing_split(unit_dir)
    for arm in arms:
        arm_dir = unit_dir / arm
        done = _latest_run(arm_dir)
        if done is not None and (done / "metrics.json").exists():
            logger.info("[%s] %s already done: %s", label, arm, done.name)
        else:
            config, extra = ARMS[arm]
            cmd = [sys.executable, "-m", "movement.cli.train", "--config", config]
            if split_file is not None:
                cmd += ["--split-file", str(split_file)]
            for o in [*common, *unit_overrides, *extra, f"trainer.run_dir={arm_dir}"]:
                cmd += ["--override", o]
            _run(cmd, dry)
            done = _latest_run(arm_dir)
            if not dry:
                if done is None:
                    raise RuntimeError(f"No finished run under {arm_dir}")
                _run([sys.executable, "-m", "movement.cli.eval", "--run", str(done)], dry)
                summary, collapsed = _metrics_line(done)
                notify(f"{dataset}: {label}/{arm} done",
                       f"{summary}\n{study_dir.name}/{label}/{arm}",
                       tags=["warning"] if collapsed else ["white_check_mark"],
                       priority="high" if collapsed else "default")
        # The first arm trained in a unit writes the split every later arm reuses.
        if split_file is None and done is not None and (done / "split.json").exists():
            split_file = done / "split.json"
        if split_file is None and not dry:
            raise RuntimeError(f"[{label}] {arm} finished without a split.json; cannot share the split.")


def existing_split(unit_dir: Path) -> Path | None:
    """Split file already written in this unit (preferring the ``cov`` arm), so arms added later
    reuse the exact animals of earlier arms."""
    for arm in ["cov", *sorted(p.name for p in unit_dir.glob("*") if p.is_dir())]:
        run = _latest_run(unit_dir / arm)
        if run is not None and (run / "split.json").exists():
            return run / "split.json"
    return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def collect(study_dir: Path, unit_prefix: str) -> pd.DataFrame:
    rows = []
    for metrics in sorted(study_dir.glob(f"{unit_prefix}*/*/2*/metrics.json")):
        run = metrics.parent
        m = json.loads(metrics.read_text(encoding="utf-8"))
        manifest = run / "manifest.json"
        man = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        tm = man.get("train_metrics", {})
        cfg_path = run / "config.yaml"
        max_epochs = tm.get("max_epochs")
        if max_epochs is None and cfg_path.exists():
            max_epochs = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["trainer"]["max_epochs"]
        split = json.loads((run / "split.json").read_text(encoding="utf-8"))
        rows.append({
            "unit": run.parent.parent.name,
            "arm": run.parent.name,
            "ade": m["ade"], "fde": m["fde"], "ade_cp": m["ade_cp"], "ade_cv": m["ade_cv"],
            # Point forecasts: energy score == ADE (runs evaluated before ES existed lack the key).
            "es": m.get("es", m["ade"]),
            "es_clim_hour": m.get("es_clim_hour"),
            "es_clim_all": m.get("es_clim_all"),
            "probabilistic": bool(m.get("probabilistic", False)),
            "fusion_gate": m.get("fusion_gate"),
            **_calibration_fields(m),
            **_hex_fields(m),
            "n_windows": m.get("n_windows"),
            "params": man.get("parameter_count"),
            "best_epoch": tm.get("best_epoch"),
            "epochs_run": tm.get("epochs_run"),
            "max_epochs": max_epochs,
            "stopped_early": tm.get("stopped_early"),
            "n_val_windows": tm.get("n_val_windows"),
            "n_test_animals": len(split["test"]),
            "run": str(run.relative_to(study_dir)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ratio_cp"] = df["ade"] / df["ade_cp"]
    # The collapse rule is about point forecasts; probabilistic arms are judged on ES.
    df["collapsed"] = (df["ratio_cp"] >= COLLAPSE_RATIO) & ~df["probabilistic"]
    # Climatology baselines depend only on the unit's split: share them across its arms.
    for col in ("es_clim_hour", "es_clim_all"):
        df[col] = df.groupby("unit")[col].transform(lambda v: v.dropna().iloc[0] if v.notna().any() else np.nan)
    late = (df["best_epoch"].notna() & df["max_epochs"].notna()
            & (df["best_epoch"] >= LATE_BEST_FRACTION * df["max_epochs"] - 1)
            & (df["stopped_early"] != True))  # noqa: E712 - NaN-safe comparison
    df["still_improving"] = late
    return df


def _calibration_fields(m: dict) -> dict:
    """Flatten coverage / radius from metrics.json (absent for point runs and older evals)."""
    out: dict = {}
    for prefix, block in (("", m.get("calibration")), ("clim_", m.get("calibration_clim_hour"))):
        if not block:
            continue
        for a, v in block.get("coverage", {}).items():
            out[f"{prefix}cov{a}"] = v
        for a, v in block.get("radius", {}).items():
            out[f"{prefix}rad{a}"] = v
    for prefix, curve in (("", m.get("calibration_curve")), ("clim_", m.get("calibration_curve_clim_hour"))):
        if curve:
            out[f"{prefix}curve"] = json.dumps(curve)
    return out


def _hex_fields(m: dict) -> dict:
    """Flatten destination-hexagon scores (probabilistic runs evaluated with hex scoring)."""
    out: dict = {}
    for source, sc in (m.get("hex") or {}).items():
        if isinstance(sc, dict):
            for k in ("nll", "top1", "top5", "outside"):
                if k in sc:
                    out[f"hex_{source}_{k}"] = sc[k]
    return out


def _hex_section(df: pd.DataFrame) -> list[str]:
    """Destination-hexagon scores: which ~0.08 km² cell holds the animal at the last horizon step."""
    if "hex_samples_nll" not in df or df["hex_samples_nll"].isna().all():
        return []
    prob = df[df["hex_samples_nll"].notna() & df["n_windows"].notna()]

    def pooled(g: pd.DataFrame, col: str) -> float:
        g = g[g[col].notna()] if col in g else g.iloc[0:0]
        return float((g[col] * g["n_windows"]).sum() / g["n_windows"].sum()) if len(g) else float("nan")

    lines = ["", "## Destination hexagon (final horizon step)", "",
             "The forecast frame is tiled with hexagons of H3 resolution-9 size (edge 174 m, ~0.08 km²), "
             "10 rings around the last observed fix (331 cells, ~3 km) plus an *outside* class. Score: "
             "probability given to the cell where the animal actually is at the last step. "
             "**NLL** = mean −log p (lower is better; uniform over 332 classes = 5.81); **top-1 / top-5** = "
             "share of windows whose true cell is the most / one of the five most probable. "
             "*samples* = kernel density of the arm's 64 sampled destinations (every probabilistic arm); "
             "*head* = the arm's own hexagon classifier. `clim (hour)` uses its 64 rotated samples the same way. "
             "Pooled over all test windows.", "",
             "| arm | source | NLL | top-1 | top-5 |", "|---|---|---|---|---|"]
    for arm, g in prob.groupby("arm"):
        for source in ("samples", "head"):
            if f"hex_{source}_nll" in g and g[f"hex_{source}_nll"].notna().any():
                lines.append(f"| {arm} | {source} | {pooled(g, f'hex_{source}_nll'):.3f} | "
                             f"{pooled(g, f'hex_{source}_top1'):.1%} | {pooled(g, f'hex_{source}_top5'):.1%} |")
    one = prob.drop_duplicates("unit")
    if "hex_clim_hour_nll" in one and one["hex_clim_hour_nll"].notna().any():
        lines.append(f"| clim (hour) | samples | {pooled(one, 'hex_clim_hour_nll'):.3f} | "
                     f"{pooled(one, 'hex_clim_hour_top1'):.1%} | {pooled(one, 'hex_clim_hour_top5'):.1%} |")
    if "hex_stay_put_top1" in one and one["hex_stay_put_top1"].notna().any():
        lines.append(f"| stay put | centre cell | — | {pooled(one, 'hex_stay_put_top1'):.1%} | — |")
    if "hex_samples_outside" in one:
        lines += ["", f"True destination outside the grid: {pooled(one, 'hex_samples_outside'):.1%} of windows."]
    # Per-unit NLL, best source per arm.
    best = prob.assign(hex_best_nll=prob[[c for c in ("hex_samples_nll", "hex_head_nll") if c in prob]].min(axis=1))
    wide = best.pivot_table(index="unit", columns="arm", values="hex_best_nll")
    lines += ["", "Per unit, NLL of each arm's better source:", "",
              "| unit | " + " | ".join(wide.columns) + " |", "|---|" + "---|" * len(wide.columns)]
    for unit in sorted(wide.index, key=_unit_key):
        lines.append(f"| {unit} | " + " | ".join(f"{v:.3f}" if pd.notna(v) else "—" for v in wide.loc[unit]) + " |")
    return lines


def _unit_key(label: str) -> tuple[str, int]:
    digits = "".join(ch for ch in label if ch.isdigit())
    return (label.rstrip("0123456789"), int(digits) if digits else 0)


def _fmt_ms(values: pd.Series, *, signed: bool = False) -> str:
    if len(values) == 0:
        return "—"
    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    mean = f"{values.mean():+.1f}" if signed else f"{values.mean():.1f}"
    return f"{mean} ± {sd:.1f}"


def _paired(df: pd.DataFrame, a: str, b: str) -> list[str]:
    wide = df.pivot_table(index="unit", columns="arm", values="ade")
    bad = df[df["collapsed"]].groupby("unit")["arm"].apply(set).to_dict()
    if not {a, b} <= set(wide.columns):
        return []
    lines = ["", f"## Paired: `{a}` − `{b}` (same animals, same split)", "",
             "| unit | Δ ADE (m) | Δ % | note |", "|---|---|---|---|"]
    kept = []
    for unit in sorted(wide.index, key=_unit_key):
        if pd.isna(wide.loc[unit, a]) or pd.isna(wide.loc[unit, b]):
            continue
        d = wide.loc[unit, a] - wide.loc[unit, b]
        collapsed = bad.get(unit, set()) & {a, b}
        note = f"excluded: {', '.join(sorted(collapsed))} collapsed" if collapsed else ""
        if not collapsed:
            kept.append(d)
        lines.append(f"| {unit} | {d:+.1f} | {100 * d / wide.loc[unit, b]:+.1f}% | {note} |")
    if kept:
        s = pd.Series(kept)
        wins = int((s < 0).sum())
        lines.append(f"| **mean ± std ({len(kept)} units)** | **{_fmt_ms(s, signed=True)}** | | "
                     f"`{a}` better in {wins}/{len(kept)} |")
    lines += ["", "Negative = the first arm has lower error. A mean whose ± std spans 0 is "
                  "*no clear difference*."]
    return lines


def _energy_section(df: pd.DataFrame) -> list[str]:
    """Energy-score comparison: every arm vs stay-put and the climatology baselines."""
    if df["es_clim_hour"].isna().all():
        return []
    lines = ["", "## Energy score (probabilistic comparison)", "",
             "Step-averaged energy score in metres, lower is better. For a single-path forecast it "
             "equals ADE, so point arms and stay-put are on the same scale. *Climatology* baselines "
             "resample training futures with a random rotation: `clim (all)` knows typical movement, "
             "`clim (hour)` also knows the clock hour. Beating `clim (hour)` means knowing more than "
             "the time of day. No runs are excluded here.", "",
             "| unit | " + " | ".join(sorted(df["arm"].unique())) + " | stay put | clim (all) | clim (hour) |",
             "|---|" + "---|" * (df["arm"].nunique() + 3)]
    wide = df.pivot_table(index="unit", columns="arm", values="es")
    base = df.groupby("unit")[["ade_cp", "es_clim_all", "es_clim_hour"]].first()
    for unit in sorted(wide.index, key=_unit_key):
        cells = [f"{wide.loc[unit, a]:.1f}" if pd.notna(wide.loc[unit, a]) else "—" for a in wide.columns]
        b = base.loc[unit]
        cells += [f"{b['ade_cp']:.1f}"] + [f"{b[c]:.1f}" if pd.notna(b[c]) else "—" for c in ("es_clim_all", "es_clim_hour")]
        lines.append(f"| {unit} | " + " | ".join(cells) + " |")
    # Pooled (window-weighted) over units.
    if df["n_windows"].notna().all():
        def pooled(g, col):
            g = g[g[col].notna()]
            return (g[col] * g["n_windows"]).sum() / g["n_windows"].sum() if len(g) else float("nan")
        cells = [f"**{pooled(df[df['arm'] == a], 'es'):.1f}**" for a in wide.columns]
        one = df.drop_duplicates("unit")
        cells += [f"**{pooled(one, c):.1f}**" for c in ("ade_cp", "es_clim_all", "es_clim_hour")]
        lines.append("| **pooled** | " + " | ".join(cells) + " |")

    def paired(a: str, b: str | None, label: str) -> list[str]:
        if a not in wide.columns or (b is not None and b not in wide.columns):
            return []
        ref = base["es_clim_hour"] if b is None else wide[b]
        d = (wide[a] - ref).dropna()
        if d.empty:
            return []
        rel = (d / ref.loc[d.index]) * 100
        return [f"| `{a}` − {label} | {d.mean():+.1f} ± {d.std(ddof=1) if len(d) > 1 else 0.0:.1f} | "
                f"{rel.mean():+.1f}% | {int((d < 0).sum())}/{len(d)} |"]
    rows = []
    rows += paired("pcov", "pnocov", "`pnocov`")
    rows += paired("pcov_idx", "pnocov", "`pnocov`")
    rows += paired("pcov_idx", "pcov", "`pcov`")
    rows += paired("faunaformer", "pnocov", "`pnocov`")
    rows += paired("faunaformer", "pcov_idx", "`pcov_idx`")
    rows += paired("ff_levels", "faunaformer", "`faunaformer`")
    rows += paired("ff_early", "faunaformer", "`faunaformer`")
    rows += paired("ff_no_nbr", "faunaformer", "`faunaformer`")
    rows += paired("ff_no_nbr", "pnocov", "`pnocov`")
    rows += paired("pmem", "pnocov", "`pnocov`")
    rows += paired("pmem_hex", "pmem", "`pmem`")
    rows += paired("pmem_hex", "pnocov", "`pnocov`")
    rows += paired("phex", "pnocov", "`pnocov`")
    for a in ("faunaformer", "pnocov", "pcov", "pcov_idx", "nocov", "transformer", "pmem", "pmem_hex"):
        rows += paired(a, None, "clim (hour)")
    if rows:
        lines += ["", "| comparison | Δ ES mean ± std (m) | Δ % | first better in (units) |", "|---|---|---|---|", *rows]
    return lines


def _calibration_section(df: pd.DataFrame, fig_path: Path) -> list[str]:
    """Pooled coverage of centre-outward sample regions per probabilistic arm, plus a curve figure."""
    if "cov0.5" not in df or df["cov0.5"].isna().all():
        return []
    levels = ["0.5", "0.8", "0.9", "0.95"]
    prob = df[df["cov0.5"].notna() & df["n_windows"].notna()]

    def pooled(g: pd.DataFrame, col: str) -> float:
        g = g[g[col].notna()]
        return float((g[col] * g["n_windows"]).sum() / g["n_windows"].sum()) if len(g) else float("nan")

    lines = ["", "## Calibration (probabilistic arms)", "",
             "Share of true positions inside the model's central α-region: for each window and step, "
             "the disc around the samples' spatial median whose radius is the α-quantile of the samples' "
             "distances to it. Calibrated = coverage ≈ α; below α = spread too narrow (overconfident), "
             "above = too wide. Radius (m) is the mean region size (sharpness). Pooled over all test "
             "windows; averaged over the 12 steps.", "",
             "| arm | " + " | ".join(f"{float(a):.0%} region" for a in levels) + " | 50% radius (m) | 90% radius (m) |",
             "|---|" + "---|" * (len(levels) + 2)]
    for arm, g in prob.groupby("arm"):
        cells = [f"{pooled(g, 'cov' + a):.1%}" for a in levels]
        lines.append(f"| {arm} | " + " | ".join(cells) + f" | {pooled(g, 'rad0.5'):.0f} | {pooled(g, 'rad0.9'):.0f} |")
    clim = prob.drop_duplicates("unit")
    if "clim_cov0.5" in clim and clim["clim_cov0.5"].notna().any():
        cells = [f"{pooled(clim, 'clim_cov' + a):.1%}" for a in levels]
        lines.append("| clim (hour) | " + " | ".join(cells) +
                     f" | {pooled(clim, 'clim_rad0.5'):.0f} | {pooled(clim, 'clim_rad0.9'):.0f} |")

    curves = prob[prob["curve"].notna()] if "curve" in prob else prob.iloc[0:0]
    if len(curves):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.8, 4.6))
        ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="perfect calibration")

        def pooled_curve(g: pd.DataFrame, col: str):
            parsed = [(json.loads(c), n) for c, n in zip(g[col], g["n_windows"]) if isinstance(c, str)]
            keys = sorted(parsed[0][0], key=float)
            tot = sum(n for _, n in parsed)
            return [float(k) for k in keys], [sum(c[k] * n for c, n in parsed) / tot for k in keys]

        for arm, g in curves.groupby("arm"):
            x, y = pooled_curve(g, "curve")
            ax.plot(x, y, marker="o", ms=3, lw=1.4, label=arm)
        if "clim_curve" in clim and clim["clim_curve"].notna().any():
            x, y = pooled_curve(clim[clim["clim_curve"].notna()], "clim_curve")
            ax.plot(x, y, color="0.3", lw=1.2, ls=":", label="clim (hour)")
        ax.set_xlabel("Nominal coverage α")
        ax.set_ylabel("Observed coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        lines += ["", f"![Calibration curves](figures/{fig_path.name})"]
    return lines


def write_report(dataset: str, study_dir: Path, tag: str, unit_prefix: str) -> Path:
    df = collect(study_dir, unit_prefix)
    name = dataset if tag == "random" else f"{dataset}_{tag}"
    out = REPO_ROOT / "reports" / "covariate_study" / f"{name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    if tag.startswith("random"):
        mode_line = "Random animal-disjoint splits (`data.split_unit=individual`), one per seed."
    elif "_tailval" in tag:
        mode_line = ("Fix-balanced animal folds with tail validation (`data.split_unit=individual_kfold_tailval`): "
                     "fold k is test (unseen animals); every other animal trains, and the last 15% of each "
                     "training animal's track (by time) is the validation set. Across folds every animal is "
                     "tested exactly once.")
    else:
        mode_line = ("Fix-balanced animal folds (`data.split_unit=individual_kfold`): fold k is test, fold k+1 "
                     "validation; across folds every animal is tested exactly once.")
    lines = [f"# Covariate study — {dataset} ({tag})", "", mode_line,
             "Within a unit, every arm shares one split file. Test ADE/FDE in metres (lower is better).", ""]
    if df.empty:
        lines.append("_No evaluated runs yet._")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out

    lines += ["## Per unit", "",
              "| unit | arm | params | ADE | FDE | const-pos ADE | ADE/const-pos | best epoch | val windows | test animals | flag |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    df = df.assign(_k=df["unit"].map(_unit_key)).sort_values(["_k", "arm"]).drop(columns="_k")
    for r in df.itertuples():
        params = f"{int(r.params):,}" if pd.notna(r.params) else "—"
        best = (f"{int(r.best_epoch)}/{int(r.epochs_run)}" if pd.notna(r.best_epoch) and pd.notna(r.epochs_run)
                else "—")
        val_w = f"{int(r.n_val_windows):,}" if pd.notna(r.n_val_windows) else "—"
        flags = [f for f, on in (("**collapsed**", r.collapsed), ("still improving", r.still_improving)) if on]
        lines.append(f"| {r.unit} | {r.arm} | {params} | {r.ade:.1f} | {r.fde:.1f} | {r.ade_cp:.1f} | "
                     f"{r.ratio_cp:.3f} | {best} | {val_w} | {r.n_test_animals} | {', '.join(flags)} |")

    lines += ["", "## Summary per arm", "",
              "Means exclude collapsed runs. *Pooled* weights every test window equally across all units "
              "(including collapsed ones); in k-fold mode that is every animal exactly once.", "",
              "| arm | units | collapsed | still improving | ADE mean ± std | FDE mean ± std | ADE/const-pos | pooled ADE |",
              "|---|---|---|---|---|---|---|---|"]
    for arm, g in df.groupby("arm"):
        ok = g[~g["collapsed"]]
        pooled = "—"
        if g["n_windows"].notna().all():
            pooled = f"{(g['ade'] * g['n_windows']).sum() / g['n_windows'].sum():.1f}"
        lines.append(f"| {arm} | {len(g)} | {int(g['collapsed'].sum())} | {int(g['still_improving'].sum())} | "
                     f"{_fmt_ms(ok['ade'])} | {_fmt_ms(ok['fde'])} | "
                     f"{ok['ratio_cp'].mean() if len(ok) else float('nan'):.3f} | {pooled} |")

    lines += _paired(df, "cov", "nocov")
    lines += _paired(df, "cov", "transformer")
    lines += _paired(df, "nocov", "transformer")
    lines += _energy_section(df)
    lines += _calibration_section(df, out.parent / "figures" / f"{name}_calibration.png")
    lines += _hex_section(df)
    gates = df[df["fusion_gate"].notna()] if "fusion_gate" in df else df.iloc[0:0]
    if len(gates):
        lines += ["", "## Late-fusion gate (FaunaFormer family)", "",
                  "Mean opening of the covariate gate after training: 0 = covariates ignored, "
                  "1 = fully used. It starts at sigmoid(-4) ≈ 0.018.", "",
                  "| unit | arm | gate |", "|---|---|---|"]
        for r in gates.sort_values(["unit", "arm"]).itertuples():
            lines.append(f"| {r.unit} | {r.arm} | {r.fusion_gate:.3f} |")

    for sel_arm in ("cov", "faunaformer", "ff_no_nbr"):
        sel = sorted(study_dir.glob(f"{unit_prefix}*/{sel_arm}/2*/covariate_selection.csv"))
        if not sel:
            continue
        weights = pd.concat([pd.read_csv(p) for p in sel]).groupby("covariate")["mean_weight"].mean()
        lines += ["", f"## Variable-selection weights ({sel_arm} arm, mean over units)", "",
                  "How the model routes covariate information — not a causal importance.", "",
                  "| covariate | mean weight |", "|---|---|"]
        for cov_name, w in weights.sort_values(ascending=False).head(15).items():
            lines.append(f"| `{cov_name}` | {w:.3f} |")

    lines += ["", "## Flags", "",
              f"- **collapsed**: ADE ≥ {COLLAPSE_RATIO:.0%} of constant-position — the run learned "
              "\"no movement\". Excluded from means and paired rows.",
              f"- **still improving**: best validation epoch in the last {1 - LATE_BEST_FRACTION:.0%} of "
              "`max_epochs` with no early stop — undertrained; re-run with a larger `--epochs`."]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def reeval(study_dir: Path, unit_prefix: str, spec: str, dry: bool) -> int:
    """Re-evaluate existing runs (latest per unit and arm) with the current eval code.

    Returns the number of runs re-evaluated.
    """
    wanted = sorted(PROBABILISTIC_ARMS) if spec.strip() == "prob" else [a.strip() for a in spec.split(",") if a.strip()]
    unknown = set(wanted) - set(ARMS)
    if unknown:
        raise SystemExit(f"Unknown arm(s) {sorted(unknown)}; choose from {sorted(ARMS)} or 'prob'")
    n = 0
    for unit_dir in sorted(study_dir.glob(f"{unit_prefix}*"), key=lambda p: _unit_key(p.name)):
        for arm in wanted:
            run = _latest_run(unit_dir / arm)
            if run is None:
                continue
            _run([sys.executable, "-m", "movement.cli.eval", "--run", str(run)], dry)
            n += 1
    return n


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    load_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    study_dir, tag, units = study_layout(args)
    unit_prefix = "fold" if args.mode == "kfold" else "split"
    if args.reeval:
        n = reeval(study_dir, unit_prefix, args.reeval, args.dry_run)
        if not args.dry_run:
            notify(f"{args.dataset}: re-evaluation done",
                   f"Re-evaluated {n} run(s) in {study_dir.name}", tags=["white_check_mark"])
    elif not args.report_only:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = set(arms) - set(ARMS)
        if unknown:
            raise SystemExit(f"Unknown arm(s) {sorted(unknown)}; choose from {sorted(ARMS)}")
        csv = raw_dataset_path() / f"{args.dataset}.csv"
        if not csv.exists():
            raise SystemExit(f"Raw CSV not found: {csv}")
        common = dataset_overrides(csv, input_len=args.input_len, horizon=args.horizon,
                                   stride=args.stride) + [
            f"trainer.max_epochs={args.epochs}",
            f"trainer.early_stopping_patience={args.patience}",
            *args.override,
        ]
        if not args.dry_run:
            notify(f"{args.dataset}: study started",
                   f"{tag} | {len(units)} units × {len(arms)} arms\n{study_dir}",
                   tags=["rocket"], priority="low")
        label = None
        try:
            for label, unit_overrides in units:
                run_unit(label, unit_overrides, arms, common, study_dir, args.dataset, args.dry_run)
        except Exception as exc:  # noqa: BLE001 - report the unit that failed, then stop
            if not args.dry_run:
                notify(f"{args.dataset}: study failed",
                       f"{label or 'setup'}: {type(exc).__name__}: {exc}",
                       tags=["rotating_light"], priority="urgent")
            raise
    if not args.dry_run:
        report = write_report(args.dataset, study_dir, tag, unit_prefix)
        logger.info("Report: %s", report)
        notify(f"{args.dataset}: report ready",
               f"{report.relative_to(REPO_ROOT)}", tags=["checkered_flag"])


if __name__ == "__main__":
    main()
