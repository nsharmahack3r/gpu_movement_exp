"""Datamodule: splits + Dataset + DataLoader factory.

The datamodule owns the comparison protocol: split-by-individual, fit scalers
on train only, deterministic window construction. Persisting the split and
scalers to the run directory is the caller's job (``cli/train.py``) so the eval
entrypoint can reload them.

Batches are ``(x, y, dt_seconds)`` for the original arms. When per-fix time
context (``transforms.time_context``) or covariates (``covariates.enabled``) are
configured, a fourth element — a dict of extra tensors — is appended; use
:func:`unpack_batch` to consume either form. With both off, batches are
byte-identical to before.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from movement.config import Config
from movement.data.covariates import (
    COVARIATE_SCALER_FILE,
    N_TIME_FEATURES,
    CovariateScaler,
    window_time_context,
)
from movement.data.loading import (
    DISJOINT_SPLIT_UNITS,
    Trajectory,
    apply_tail_cuts,
    assert_disjoint_ids,
    assert_tailval_split,
    discard_short,
    load_dataset_with_covariates,
    save_split,
    split_individuals_kfold_tailval,
    split_trajectories,
)
from movement.data.memory import (
    GLOBAL_FEATURES,
    GLOBAL_VECTOR_PAIRS,
    STEP_FEATURES,
    STEP_VECTOR_PAIRS,
    AnimalHistory,
    build_histories,
    memory_features,
    rotate_pairs,
)

from movement.data.transforms import FeatureScaler, WindowTransform
from movement.data.windowing import Window, build_windows

logger = logging.getLogger(__name__)

# Below this many validation windows, early stopping reacts to noise: on boar a
# 58-window validation set stopped three arms at the "no movement" solution.
MIN_VAL_WINDOWS = 200


def unpack_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """``(x, y, dt[, extras])`` → tensors on ``device`` + extras dict (possibly empty)."""
    x, y, dt = batch[0], batch[1], batch[2]
    extras = batch[3] if len(batch) > 3 else {}
    return (
        x.to(device),
        y.to(device),
        dt.to(device),
        {k: v.to(device) for k, v in extras.items()},
    )


@dataclass
class MovementDataset(Dataset):
    """Map-style dataset over pre-built windows.

    Applies the window transform and the (train-fitted) scaler on access so
    memory stays bounded and scaler fitting stays train-only.
    """

    windows: list[Window]
    transform: WindowTransform
    scaler: FeatureScaler | None = None
    # Optional extras (see module docstring).
    time_context: bool = False
    nominal_dt_hours: float = 1.0
    covariate_scaler: CovariateScaler | None = None
    # Random rotation + mirror of inputs and targets (train split only).
    augment_rotation: bool = False
    # Hide covariates of fixes closer than this to the forecast origin (seconds).
    covariate_embargo_s: float = 0.0
    # Memory features (movement.data.memory): per-animal fix histories + settings.
    histories: dict[str, AnimalHistory] | None = None
    memory_days: int = 7
    memory_lookback_days: float = 14.0
    memory_near_m: float = 200.0

    def __post_init__(self) -> None:
        if self.augment_rotation and (self.transform.turning_angle or not self.transform.delta_encoding):
            raise ValueError(
                "trainer.augment_rotation needs delta_encoding=true and turning_angle=false "
                "(a mirrored track flips the sign of its turning angles)."
            )
        self._rng: np.random.Generator | None = None
        self._memory_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _rotation(self) -> np.ndarray:
        from movement.evaluation.scoring import random_rotations

        if self._rng is None:  # seeded from the global numpy state set by seed_everything
            self._rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        return random_rotations(1, self._rng)[0].astype(np.float32)

    def _rotate(self, x: np.ndarray, y: np.ndarray, rot: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        rot = self._rotation() if rot is None else rot
        x = x.copy()
        x[:, :2] = x[:, :2] @ rot.T
        return x, (y @ rot.T).astype(np.float32)

    def _memory(self, idx: int, w: Window) -> tuple[np.ndarray, np.ndarray]:
        cached = self._memory_cache.get(idx)
        if cached is None:
            hist = self.histories.get(w.individual_id)
            if hist is None or w.timestamps is None:
                raise RuntimeError(f"No fix history / timestamps for {w.individual_id}; memory features need both.")
            cached = memory_features(
                hist, int(w.timestamps[-1]), (float(w.features[-1, 0]), float(w.features[-1, 1])),
                horizon=self.transform.horizon, dt_s=self.nominal_dt_hours * 3600.0,
                days=self.memory_days, lookback_days=self.memory_lookback_days, near_m=self.memory_near_m,
            )
            self._memory_cache[idx] = cached
        return cached

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        out = self.transform.apply(w.features, w.target, w.timestamp)
        x = out["x"]
        y = out["y"]
        rot = self._rotation() if self.augment_rotation else None
        if rot is not None:
            x, y = self._rotate(x, y, rot)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        dt = self._dt_seconds(w)
        base = (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(dt))

        extras: dict[str, torch.Tensor] = {}
        if self.time_context:
            obs, fut = window_time_context(
                w.timestamps, w.features[:, 1], horizon=self.transform.horizon,
                nominal_dt_hours=self.nominal_dt_hours,
            )
            extras["time_feats"] = torch.from_numpy(obs)
            extras["future_time_feats"] = torch.from_numpy(fut)
        if self.covariate_scaler is not None:
            if w.covariates is None:
                raise RuntimeError("Covariates are enabled but a window carries none.")
            cov, missing = self.covariate_scaler.transform(w.covariates)
            if self.covariate_embargo_s > 0:
                ts = self._dt_seconds(w)  # seconds since the window's first fix
                recent = (ts[-1] - ts) < self.covariate_embargo_s
                cov[recent] = 0.0
                missing[recent] = 1.0
            extras["covariates"] = torch.from_numpy(cov)
            extras["covariate_missing"] = torch.from_numpy(missing)
        if self.histories is not None:
            step, glob = self._memory(idx, w)
            if rot is not None:
                step = rotate_pairs(step, rot, STEP_VECTOR_PAIRS)
                glob = rotate_pairs(glob, rot, GLOBAL_VECTOR_PAIRS)
            extras["memory_step"] = torch.from_numpy(np.ascontiguousarray(step, dtype=np.float32))
            extras["memory_global"] = torch.from_numpy(np.ascontiguousarray(glob, dtype=np.float32))
        return (*base, extras) if extras else base

    @staticmethod
    def _dt_seconds(w: Window) -> np.ndarray:
        """Cumulative seconds since the window's first fix (for time_aware PE)."""
        n = len(w.features)
        if w.timestamps is not None and len(w.timestamps) == n:
            ts = w.timestamps.astype(np.float64)
        else:
            # Nominal hourly sampling fallback.
            ts = np.arange(n, dtype=np.float64) * 3600.0
        return ts - ts[0]


def fit_scaler(windows: list[Window], transform: WindowTransform) -> FeatureScaler:
    """Fit a standardisation scaler on the *training* windows only."""
    scaler = FeatureScaler()
    cols: list[np.ndarray] = []
    for w in windows:
        out = transform.apply(w.features, w.target, w.timestamp)
        cols.append(out["x"])
    x = np.concatenate(cols, axis=0)
    return scaler.fit(x)


def fit_covariate_scaler(train: list[Trajectory], columns: list[str], *, clip: float) -> CovariateScaler:
    """Fit covariate statistics on the training animals' *fixes* (not windows).

    Fitting on fixes rather than stride-1 windows counts every fix once, so
    long segments are not over-weighted by window overlap.
    """
    rows = np.concatenate([t.df[columns].to_numpy(dtype=np.float64) for t in train], axis=0)
    return CovariateScaler.fit(rows, columns, clip=clip)


def _window_transform(config: Config) -> WindowTransform:
    wcfg = config.windowing
    return WindowTransform(
        input_len=wcfg.input_len,
        horizon=wcfg.horizon,
        delta_encoding=config.transforms.delta_encoding,
        step_length=config.transforms.step_length,
        turning_angle=config.transforms.turning_angle,
        speed=config.transforms.speed,
        delta_t=config.transforms.delta_t,
        cyclical_time=config.transforms.cyclical_time,
    )


def _windows(config: Config, train, val, test, covariate_columns):
    wcfg = config.windowing
    kw = {"input_len": wcfg.input_len, "horizon": wcfg.horizon, "covariate_columns": covariate_columns}
    return (
        build_windows(train, stride=wcfg.stride, **kw),
        build_windows(val, stride=wcfg.eval_stride, **kw),
        build_windows(test, stride=wcfg.eval_stride, **kw),
    )


def _warn_small_val(val_windows: list[Window]) -> None:
    if len(val_windows) < MIN_VAL_WINDOWS:
        logger.warning(
            "Only %d validation windows (from %d animal(s)). Early stopping and checkpoint "
            "selection will be noisy; consider data.split_unit=individual_kfold, which "
            "balances folds by fix count.",
            len(val_windows), len({w.individual_id for w in val_windows}),
        )


def _load(config: Config, raw_csv_pattern: str | None) -> tuple[list[Trajectory], list[str]]:
    trajectories, columns = load_dataset_with_covariates(
        config.data, raw_csv_pattern or config.data.raw_csv, config.covariates
    )
    min_len = config.windowing.input_len + config.windowing.horizon
    trajectories = discard_short(trajectories, min_len)
    if not trajectories:
        raise ValueError(
            f"No trajectories survive the minimum length of {min_len} fixes "
            f"(input_len + horizon). Check the data and windowing config."
        )
    return trajectories, columns


@dataclass
class DataModule:
    """Owns trajectories, splits, datasets, and dataloaders for one config."""

    config: Config
    trajectories: list[Trajectory]
    train_windows: list[Window]
    val_windows: list[Window]
    test_windows: list[Window]
    scaler: FeatureScaler
    covariate_columns: list[str] = field(default_factory=list)
    covariate_scaler: CovariateScaler | None = None
    # Tail-validation splits: per training animal, the time its validation tail starts.
    val_cuts: dict | None = None

    def __post_init__(self) -> None:
        tc = self.config.transforms
        self.histories = build_histories(self.trajectories) if tc.memory else None

    @property
    def n_covariates(self) -> int:
        return len(self.covariate_columns)

    def model_data_spec(self) -> dict:
        """Data-dependent quantities a model builder may need.

        ``target_scale`` is the RMS of the per-step target displacement (metres)
        over training windows — train split only, like every other statistic.
        """
        return {
            "n_covariates": self.n_covariates,
            "n_time_features": N_TIME_FEATURES if self.config.transforms.time_context else 0,
            "covariate_columns": list(self.covariate_columns),
            "target_scale": self.target_rms(),
            "covariate_change_scale": self.covariate_change_std(),
            "n_memory_step": len(STEP_FEATURES) if self.histories is not None else 0,
            "n_memory_global": len(GLOBAL_FEATURES) if self.histories is not None else 0,
        }

    def covariate_change_std(self, max_windows: int = 5000) -> list[float] | None:
        """Train-split std of each within-window covariate change input.

        Order matches :func:`movement.models.cov_transformer.covariate_changes`:
        the anomaly of every covariate, then the step change of every covariate.
        Computed on standardised covariates over observed entries only. ``None``
        without covariates.
        """
        if self.covariate_scaler is None or not self.train_windows:
            return None
        from movement.models.cov_transformer import covariate_changes

        windows = self.train_windows
        idx = np.linspace(0, len(windows) - 1, num=min(max_windows, len(windows))).astype(int)
        pairs = [self.covariate_scaler.transform(windows[i].covariates) for i in idx]
        values = torch.from_numpy(np.stack([p[0] for p in pairs]).astype(np.float64))
        missing = torch.from_numpy(np.stack([p[1] for p in pairs]).astype(np.float64))
        chg, chg_missing = covariate_changes(values, missing)
        obs = 1.0 - chg_missing
        n = obs.sum(dim=(0, 1)).clamp_min(1.0)
        mean = (chg * obs).sum(dim=(0, 1)) / n
        var = (((chg - mean) ** 2) * obs).sum(dim=(0, 1)) / n
        std = var.sqrt()
        return [float(v) if float(v) > 1e-8 else 1.0 for v in std]

    def target_rms(self, max_windows: int = 5000) -> float:
        """RMS of per-step (Δx, Δy) targets over (an even subsample of) train windows."""
        windows = self.train_windows
        if not windows:
            return 1.0
        idx = np.linspace(0, len(windows) - 1, num=min(max_windows, len(windows))).astype(int)
        transform = _window_transform(self.config)
        ys = np.concatenate([transform.apply(windows[i].features, windows[i].target, windows[i].timestamp)["y"]
                             for i in idx])
        rms = float(np.sqrt(np.mean(np.square(ys))))
        return rms if np.isfinite(rms) and rms > 1e-6 else 1.0

    @classmethod
    def build(cls, config: Config, *, raw_csv_pattern: str | None = None) -> "DataModule":
        """Construct from config: load → split → window → scale.

        Uses ``data.raw_path`` unless ``raw_csv_pattern`` is overridden (tests
        pass a fixture path). Discards segments shorter than
        ``input_len + horizon`` before splitting.
        """
        all_traj, cov_cols = _load(config, raw_csv_pattern)
        val_cuts = None
        if config.data.split_unit == "individual_kfold_tailval":
            d = config.data
            train, val, test, val_cuts = split_individuals_kfold_tailval(
                all_traj, n_folds=d.n_folds, fold=d.fold, seed=config.trainer.seed,
                val_tail_fraction=d.val_tail_fraction,
            )
        else:
            train, val, test = split_trajectories(all_traj, config.data, seed=config.trainer.seed)
        if config.data.split_unit == "segment":
            logger.warning(
                "data.split_unit=segment (legacy): animals can appear in more than one "
                "split (progress_report.md §7.1). Use data.split_unit=individual for new results."
            )
        train_windows, val_windows, test_windows = _windows(config, train, val, test, cov_cols)
        _warn_small_val(val_windows)
        transform = _window_transform(config)
        scaler = fit_scaler(train_windows, transform) if config.transforms.scale else None
        cov_scaler = (
            fit_covariate_scaler(train, cov_cols, clip=config.covariates.clip) if cov_cols else None
        )
        return cls(
            config=config,
            trajectories=all_traj,
            train_windows=train_windows,
            val_windows=val_windows,
            test_windows=test_windows,
            scaler=scaler,
            covariate_columns=cov_cols,
            covariate_scaler=cov_scaler,
            val_cuts=val_cuts,
        )

    def dataset(self, split: str, *, scaled: bool = True) -> MovementDataset:
        windows = {"train": self.train_windows, "val": self.val_windows, "test": self.test_windows}[split]
        return MovementDataset(
            windows=windows,
            transform=_window_transform(self.config),
            scaler=self.scaler if (scaled and self.config.transforms.scale) else None,
            time_context=self.config.transforms.time_context,
            nominal_dt_hours=self.config.data.nominal_dt_hours,
            covariate_scaler=self.covariate_scaler,
            augment_rotation=self.config.trainer.augment_rotation and split == "train",
            covariate_embargo_s=self.config.covariates.embargo_days * 86400.0,
            histories=self.histories,
            memory_days=self.config.transforms.memory_days,
            memory_lookback_days=self.config.transforms.memory_lookback_days,
            memory_near_m=self.config.transforms.memory_near_m,
        )

    def dataloader(self, split: str, *, shuffle: bool | None = None, batch_size: int | None = None) -> DataLoader:
        ds = self.dataset(split)
        if batch_size is None:
            batch_size = (
                self.config.trainer.batch_size if split == "train" else self.config.trainer.eval_batch_size
            )
        if shuffle is None:
            shuffle = split == "train"
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=split == "train")

    def persist_split(self, run_dir: Path) -> Path:
        """Write ``split.json`` with the exact individual assignment."""
        return save_split(run_dir, self.train_windows, self.val_windows, self.test_windows,
                          val_cuts=self.val_cuts)

    def persist_scaler(self, run_dir: Path) -> Path:
        """Write ``scalers.json``; raises if no scaler was fitted."""
        if self.scaler is None:
            raise RuntimeError("No scaler fitted; cannot persist scalers.json.")
        path = run_dir / "scalers.json"
        self.scaler.save(path)
        return path

    def persist_covariates(self, run_dir: Path) -> Path | None:
        """Write ``covariate_scalers.json`` (columns + train stats) when enabled."""
        if self.covariate_scaler is None:
            return None
        path = run_dir / COVARIATE_SCALER_FILE
        self.covariate_scaler.save(path)
        return path

    @classmethod
    def from_split_file(
        cls,
        config: Config,
        split_path: Path,
        *,
        raw_csv_pattern: str | None = None,
        refit_covariate_scaler: bool = False,
    ) -> "DataModule":
        """Rebuild a datamodule from a persisted split (used by eval).

        ``refit_covariate_scaler``: fit the covariate statistics afresh on the
        split's training animals instead of loading ``covariate_scalers.json``
        from the split file's run. Training uses this: the split may come from
        another arm that used a different covariate set (e.g. all 38 columns vs
        the six indices). Eval keeps the default, so a run is always scored with
        the exact statistics it was trained with, and a column mismatch fails.
        """
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        cut_strings = payload.pop("val_cut", None)
        ids = {k: set(v) for k, v in payload.items()}
        tailval = config.data.split_unit == "individual_kfold_tailval"
        if tailval != (cut_strings is not None):
            raise ValueError(
                f"{split_path}: split_unit={config.data.split_unit} but the split file "
                f"{'has no' if tailval else 'has'} per-animal validation cuts ('val_cut')."
            )
        if tailval:
            assert_tailval_split(ids["train"] | ids["val"], ids["test"], source=str(split_path))
        elif config.data.split_unit in DISJOINT_SPLIT_UNITS:
            assert_disjoint_ids(ids["train"], ids["val"], ids["test"], source=str(split_path))

        all_traj, cov_cols = _load(config, raw_csv_pattern)

        # An individual can yield multiple segments (gap splits); keep them all.
        by_id: dict[str, list[Trajectory]] = {}
        for t in all_traj:
            by_id.setdefault(t.individual_id, []).append(t)
        all_ids = set(by_id)
        missing = (ids["train"] | ids["val"] | ids["test"]) - all_ids
        if missing:
            raise ValueError(
                f"Split file references {len(missing)} individual(s) not found in the data "
                f"(data may have changed): {sorted(missing)[:5]}"
            )

        val_cuts = None
        if tailval:
            import pandas as pd

            val_cuts = {k: pd.Timestamp(v) for k, v in cut_strings.items()}
            pool = [seg for i in sorted(ids["train"] | ids["val"]) for seg in by_id[i]]
            train, val = apply_tail_cuts(pool, val_cuts)
        else:
            train = [seg for i in sorted(ids["train"]) for seg in by_id[i]]
            val = [seg for i in sorted(ids["val"]) for seg in by_id[i]]
        test = [seg for i in sorted(ids["test"]) for seg in by_id[i]]
        train_windows, val_windows, test_windows = _windows(config, train, val, test, cov_cols)

        scaler_path = split_path.parent / "scalers.json"
        scaler = FeatureScaler.load(scaler_path) if (scaler_path.exists() and config.transforms.scale) else None

        cov_scaler = None
        if cov_cols:
            cov_path = split_path.parent / COVARIATE_SCALER_FILE
            if cov_path.exists() and not refit_covariate_scaler:
                cov_scaler = CovariateScaler.load(cov_path, expected_columns=cov_cols)
            else:
                cov_scaler = fit_covariate_scaler(train, cov_cols, clip=config.covariates.clip)

        return cls(
            config=config,
            trajectories=all_traj,
            train_windows=train_windows,
            val_windows=val_windows,
            test_windows=test_windows,
            scaler=scaler,
            covariate_columns=cov_cols,
            covariate_scaler=cov_scaler,
            val_cuts=val_cuts,
        )
