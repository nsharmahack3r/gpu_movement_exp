"""Per-fix environmental covariates and time context for the model stage.

This is the loader half of ``covariate_plan.md`` §6: it reads the GEE-induced
per-fix CSV from ``GEE_DATASET_PATH`` (produced separately, see
``scripts/gee_export.py``), joins it onto the raw fixes, and turns each window's
covariate rows into model-ready tensors.

Design rules, all from the plan:

- **Per-fix, never per-window** (§2.4): every observed fix carries its own
  covariate vector, so along-path variation reaches the model.
- **Missingness is first-class** (§2.5): a value that is NaN (cloud, QA mask,
  no scene in the composite window) becomes 0 after standardisation *and* a
  per-variable ``missing`` flag the model sees. Nothing is silently imputed.
- **Train-only statistics**: :class:`CovariateScaler` is fit on the training
  animals' fixes only, persisted to ``covariate_scalers.json``, and reloaded at
  eval. Its column list is part of the file, so a changed covariate CSV fails
  loudly rather than feeding columns in a different order.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from math import pi
from pathlib import Path

import numpy as np
import pandas as pd

from movement.config import CovariatesConfig

logger = logging.getLogger(__name__)

KEY_COLUMNS = ("study_id", "individual_id", "timestamp")
COVARIATE_SCALER_FILE = "covariate_scalers.json"
# Local-solar hour sin/cos + year-phase sin/cos.
N_TIME_FEATURES = 4
_SECONDS_PER_DAY = 86_400.0
_TROPICAL_YEAR_DAYS = 365.2422


# ---------------------------------------------------------------------------
# Joining the GEE CSV onto raw fixes
# ---------------------------------------------------------------------------
def covariate_file(spec: CovariatesConfig, raw_csv: Path, source: str) -> Path:
    """Where the covariate CSV for one raw movement CSV and one source lives."""
    if spec.path is None:
        raise ValueError(
            "covariates.enabled=true but covariates.path is unset. Point it at "
            "${GEE_DATASET_PATH} in the model config."
        )
    return Path(spec.path) / spec.file_pattern.format(stem=Path(raw_csv).stem, source=source)


def _read_covariate_csv(path: Path, spec: CovariatesConfig) -> tuple[pd.DataFrame, list[str]]:
    """Read one covariate CSV → (keyed frame, covariate column names)."""
    from movement.data.loading import _coerce_timestamp, resolve_column

    if not path.exists():
        raise FileNotFoundError(
            f"Covariate file not found: {path}. Expected one per raw CSV at "
            f"<GEE_DATASET_PATH>/{spec.file_pattern}."
        )
    # Read only the columns that can be used (key, date and selected covariate
    # columns): multi-hundred-MB covariate files load several times faster.
    header = list(pd.read_csv(path, nrows=0).columns)
    inc = [re.compile(p) for p in spec.include]
    exc = [re.compile(p) for p in spec.exclude]
    key_like = {c for c in header for slot in KEY_COLUMNS if resolve_column(pd.DataFrame(columns=[c]), slot)}
    use = [c for c in header if c in key_like or c in spec.date_columns
           or (any(p.search(c) for p in inc) and not any(p.search(c) for p in exc))]
    df = pd.read_csv(path, low_memory=False, usecols=use)
    keys = {}
    for slot in KEY_COLUMNS:
        col = resolve_column(df, slot)
        if col is None:
            raise ValueError(f"{path.name}: no column for key {slot!r}; columns: {list(df.columns)[:12]}...")
        keys[slot] = col
    out = pd.DataFrame(
        {
            "study_id": df[keys["study_id"]].astype(str),
            "individual_id": df[keys["individual_id"]].astype(str),
            "timestamp": _coerce_timestamp(df[keys["timestamp"]]),
        }
    )

    # Date columns → signed age in days (fix time minus observation date).
    for col in spec.date_columns:
        if col in df.columns:
            obs = _coerce_timestamp(df[col])
            if getattr(obs.dt, "tz", None) is not None and getattr(out["timestamp"].dt, "tz", None) is None:
                obs = obs.dt.tz_localize(None)
            out[f"{col}_age_days"] = (out["timestamp"] - obs).dt.total_seconds() / _SECONDS_PER_DAY

    include = [re.compile(p) for p in spec.include]
    exclude = [re.compile(p) for p in spec.exclude]
    date_cols = set(spec.date_columns)
    candidates = [
        c for c in df.columns
        if c not in date_cols and c not in keys.values()
        and any(p.search(c) for p in include) and not any(p.search(c) for p in exclude)
    ]
    dropped = []
    for c in candidates:
        values = pd.to_numeric(df[c], errors="coerce")
        if values.notna().sum() == 0 and df[c].notna().sum() > 0:
            dropped.append(c)  # non-numeric column (e.g. a label string)
            continue
        if any(re.search(p, c) for p in spec.bounded_index_columns):
            n_out = int(((values < -1) | (values > 1)).sum())
            if n_out:
                logger.info("%s: clipped %d value(s) of %s to [-1, 1].", path.name, n_out, c)
            values = values.clip(-1.0, 1.0)
        out[c] = values.astype("float32")
    if dropped:
        logger.warning("%s: dropped non-numeric covariate column(s): %s", path.name, dropped)
    cols = [c for c in out.columns if c not in KEY_COLUMNS]
    age_cols = [c for c in cols if c.endswith("_age_days")]
    # Age columns are only kept if they also pass the include/exclude filters.
    cols = [c for c in cols if c not in age_cols or
            (any(p.search(c) for p in include) and not any(p.search(c) for p in exclude))]
    out = out[list(KEY_COLUMNS) + cols]
    dupes = out.duplicated(list(KEY_COLUMNS), keep="first")
    if dupes.any():
        logger.info("%s: dropped %d duplicate (study, individual, timestamp) rows.", path.name, int(dupes.sum()))
        out = out[~dupes]
    return out, cols


def attach_covariates(
    fixes: pd.DataFrame, raw_csv: Path, spec: CovariatesConfig
) -> tuple[pd.DataFrame, list[str]]:
    """Left-join every configured source's covariates onto one raw CSV's fixes.

    ``fixes`` is the normalised frame from :func:`movement.data.loading.load_raw_csv`.
    Row count is preserved exactly; fixes without a covariate row get NaN (and
    therefore a ``missing`` flag downstream). Coverage below
    ``spec.min_coverage`` fails loudly — it means the key columns disagree.
    """
    all_cols: list[str] = []
    out = fixes
    for source in spec.sources:
        path = covariate_file(spec, raw_csv, source)
        cov, cols = _read_covariate_csv(path, spec)
        clash = set(cols) & (set(all_cols) | set(out.columns))
        if clash:
            raise ValueError(f"{path.name}: covariate column(s) {sorted(clash)} already present from another source.")
        left = out.copy()
        # Align timestamp dtypes (naive vs tz-aware) before the join.
        lt, rt = left["timestamp"], cov["timestamp"]
        if getattr(lt.dt, "tz", None) is not None and getattr(rt.dt, "tz", None) is None:
            cov["timestamp"] = rt.dt.tz_localize(lt.dt.tz)
        elif getattr(lt.dt, "tz", None) is None and getattr(rt.dt, "tz", None) is not None:
            cov["timestamp"] = rt.dt.tz_convert(None)
        merged = left.merge(cov, on=list(KEY_COLUMNS), how="left", indicator="_cov_match", validate="many_to_one")
        if len(merged) != len(left):
            raise RuntimeError(f"{path.name}: join changed the row count ({len(left)} -> {len(merged)}).")
        coverage = float((merged["_cov_match"] == "both").mean())
        merged = merged.drop(columns="_cov_match")
        logger.info(
            "Covariates %s: %d column(s), %.1f%% of %d fixes matched; value missing rate "
            "median %.1f%% (max %.1f%%).",
            path.name, len(cols), coverage * 100, len(merged),
            float(merged[cols].isna().mean().median() * 100) if cols else 0.0,
            float(merged[cols].isna().mean().max() * 100) if cols else 0.0,
        )
        if coverage < spec.min_coverage:
            raise ValueError(
                f"{path.name}: only {coverage:.1%} of fixes matched a covariate row "
                f"(min_coverage={spec.min_coverage:.0%}). The (study_id, individual_id, "
                f"timestamp) keys disagree with {raw_csv.name}."
            )
        out = merged
        all_cols += cols
    return out, all_cols


# ---------------------------------------------------------------------------
# Train-only scaling with explicit missingness
# ---------------------------------------------------------------------------
@dataclass
class CovariateScaler:
    """NaN-aware per-column standardisation fit on training fixes only."""

    columns: list[str]
    mean: np.ndarray
    std: np.ndarray
    clip: float = 8.0

    @classmethod
    def fit(cls, rows: np.ndarray, columns: list[str], *, clip: float) -> "CovariateScaler":
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(columns):
            raise ValueError(f"Expected (n, {len(columns)}) covariate rows, got {rows.shape}")
        with np.errstate(invalid="ignore"), _quiet_nan_warnings():
            mean = np.nanmean(rows, axis=0)
            std = np.nanstd(rows, axis=0)
        empty = ~np.isfinite(mean)
        if empty.any():
            logger.warning(
                "Covariate column(s) with no observed value in the training split "
                "(always 'missing'): %s", [c for c, e in zip(columns, empty) if e],
            )
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
        return cls(columns=list(columns), mean=mean.astype(np.float32), std=std.astype(np.float32), clip=clip)

    def transform(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(T, C)`` raw values with NaN → (standardised & imputed, missing flags)."""
        missing = ~np.isfinite(values)
        z = (np.where(missing, self.mean, values) - self.mean) / self.std
        z = np.clip(z, -self.clip, self.clip)
        return z.astype(np.float32), missing.astype(np.float32)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {"columns": self.columns, "mean": self.mean.tolist(), "std": self.std.tolist(), "clip": self.clip},
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, *, expected_columns: list[str]) -> "CovariateScaler":
        data = json.loads(path.read_text(encoding="utf-8"))
        if list(data["columns"]) != list(expected_columns):
            raise ValueError(
                f"{path} was fit on different covariate columns than the data now provides "
                f"({len(data['columns'])} vs {len(expected_columns)}; first mismatch: "
                f"{next((a, b) for a, b in zip(data['columns'] + [None] * 999, expected_columns + [None] * 999) if a != b)}). "
                "Re-train, or restore the covariate CSV the run was trained with."
            )
        return cls(
            columns=list(data["columns"]),
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            clip=float(data["clip"]),
        )


class _quiet_nan_warnings:
    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


# ---------------------------------------------------------------------------
# Time context
# ---------------------------------------------------------------------------
def time_features(epoch_seconds: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """``(T,)`` epoch seconds + longitudes → ``(T, 4)`` cyclical time features.

    Local *mean solar* hour (UTC hour + lon/15), so the diel phase means the same
    thing at every study site — Satter et al. (2025) found pig behaviour switching
    on a solar-day schedule. Year phase is days since the epoch modulo the
    tropical year (< 1 day from calendar day-of-year, and vectorised).
    """
    t = np.asarray(epoch_seconds, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    solar_hour = ((t % _SECONDS_PER_DAY) / 3600.0 + lon / 15.0) % 24.0
    year_phase = ((t / _SECONDS_PER_DAY) % _TROPICAL_YEAR_DAYS) / _TROPICAL_YEAR_DAYS
    return np.stack(
        [
            np.sin(2 * pi * solar_hour / 24.0),
            np.cos(2 * pi * solar_hour / 24.0),
            np.sin(2 * pi * year_phase),
            np.cos(2 * pi * year_phase),
        ],
        axis=1,
    ).astype(np.float32)


def window_time_context(
    timestamps: np.ndarray, lon: np.ndarray, *, horizon: int, nominal_dt_hours: float
) -> tuple[np.ndarray, np.ndarray]:
    """Time features for the observed fixes and for the *nominal* future fixes.

    Future fix times are ``last + k * nominal_dt`` — the collar schedule, known
    in advance — never the actual target timestamps, which would leak gaps.
    """
    if timestamps is None:
        raise ValueError("time_context requires per-fix timestamps on every window.")
    obs = time_features(timestamps, lon)
    step = nominal_dt_hours * 3600.0
    future_t = float(timestamps[-1]) + step * np.arange(1, horizon + 1, dtype=np.float64)
    fut = time_features(future_t, np.full(horizon, float(lon[-1])))
    return obs, fut
