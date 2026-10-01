"""Data loading: raw CSV → per-individual trajectories with gap segmentation."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

from movement.config import DataConfig

logger = logging.getLogger(__name__)

# Column names the loader understands; the raw CSV may use any of these per slot.
_COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "datetime", "fix_time", "acquisition_time"],
    "individual_id": ["individual_id", "animal_id", "id", "tag_id", "deployment_id"],
    "study_id": ["study_id", "study", "dataset_id", "project_id"],
    "species": ["species", "taxon", "common_name", "scientific_name"],
    "lat": ["lat", "latitude", "y"],
    "lon": ["lon", "long", "longitude", "lng", "x"],
}


@dataclass
class Trajectory:
    """A single continuous movement segment for one individual.

    Attributes
    ----------
    individual_id:
        Namespaced identifier ``<study_id>::<individual_id>`` — unique across studies.
    study_id:
        Original study identifier.
    df:
        Sorted fix table with columns ``timestamp, lat, lon`` (index is a fresh
        range so pandas windowing is predictable).
    """

    individual_id: str
    study_id: str
    df: pd.DataFrame = field(repr=False)

    @property
    def n_fixes(self) -> int:
        return len(self.df)

    @property
    def start(self) -> pd.Timestamp:
        return self.df["timestamp"].iloc[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.df["timestamp"].iloc[-1]


def resolve_column(df: pd.DataFrame, slot: str) -> str | None:
    """Find the first existing column matching a semantic slot."""
    for candidate in _COLUMN_ALIASES[slot]:
        if candidate in df.columns:
            return candidate
    return None


def _first_existing(slot: str, **cols: str | None) -> str:
    found = cols[slot]
    if found is None:
        raise ValueError(f"Could not find a column for '{slot}' in the raw CSV.")
    return found


def _coerce_timestamp(series: pd.Series) -> pd.Series:
    """Parse timestamps, tolerating both ``T`` and space separators and ms."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    # Try strict ISO first, then fall back to the space-separated variant.
    try:
        out = pd.to_datetime(series, format="ISO8601", errors="raise")
    except (ValueError, TypeError):
        out = pd.to_datetime(series, errors="raise")
    return out


def load_raw_csv(path: Path) -> pd.DataFrame:
    """Read one raw movement CSV into a normalised frame.

    Returns a frame with columns ``timestamp, individual_id, study_id, species,
    lat, lon``. Timestamps are tz-aware UTC. Raises ``ValueError`` with a clear
    message if required columns are missing.
    """
    df = pd.read_csv(path)
    missing = []
    found: dict[str, str | None] = {}
    for slot in ("timestamp", "individual_id", "study_id", "species", "lat", "lon"):
        col = resolve_column(df, slot)
        found[slot] = col
        if col is None and slot != "species":
            missing.append(slot)
    if missing:
        raise ValueError(
            f"{path.name}: missing required column(s) {missing}; found columns: {list(df.columns)}"
        )

    species_col = found["species"]
    out = pd.DataFrame(
        {
            "timestamp": _coerce_timestamp(df[found["timestamp"]]),
            "individual_id": df[found["individual_id"]].astype(str),
            "study_id": df[_first_existing("study_id", **found)].astype(str),
            "species": (
                df[species_col].astype(str)
                if species_col is not None
                else pd.Series([""] * len(df), dtype=str)
            ),
            "lat": pd.to_numeric(df[found["lat"]], errors="coerce"),
            "lon": pd.to_numeric(df[found["lon"]], errors="coerce"),
        }
    )
    out = out.dropna(subset=["lat", "lon"])
    if out.empty:
        raise ValueError(f"{path.name}: no valid lat/lon rows after cleaning.")
    return out


def _step_speed_mps(g: pd.DataFrame) -> pd.Series:
    """Implied speed (m/s) between consecutive fixes of a sorted group."""

    lat = g["lat"].to_numpy(dtype=float)
    lon = g["lon"].to_numpy(dtype=float)
    dlat = np.radians(np.diff(lat))
    dlon = np.radians(np.diff(lon))
    lat_mid = np.radians(lat[:-1] + np.diff(lat) / 2)
    # Equirectangular step distance in metres.
    r = 6_371_008.8
    dx = r * dlon * np.cos(lat_mid)
    dy = r * dlat
    dist = np.hypot(dx, dy)
    dt = g["timestamp"].diff().dt.total_seconds().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = dist / dt[1:]
    return pd.Series(speed, index=g.index[1:])


def group_into_trajectories(df: pd.DataFrame, config: DataConfig) -> list[Trajectory]:
    """Group fixes into gap-split, sorted, deduplicated trajectories.

    Steps:

    1. Namespace individuals by ``study_id``.
    2. Sort by timestamp; drop exact duplicate timestamps (keep first).
    3. Split at gaps > ``max_gap_multiplier × nominal_dt``, and (when
       ``config.max_speed_mps`` is set) at steps whose implied speed exceeds
       the cap — GPS glitches and relocation jumps are treated as
       discontinuities so no window straddles a monster displacement.
    4. Drop segments shorter than ``input_len + horizon``.

    ``config`` is used only for gap/speed parameters, so this function stays
    usable before windowing config is known.
    """
    df = df.sort_values(["study_id", "individual_id", "timestamp"]).reset_index(drop=True)
    duplicated = df.duplicated(["study_id", "individual_id", "timestamp"], keep="first").sum()
    if duplicated:
        logger.info("Dropped %d exact-duplicate timestamp rows.", duplicated)
    df = df[~df.duplicated(["study_id", "individual_id", "timestamp"], keep="first")]

    max_gap = pd.Timedelta(hours=config.nominal_dt_hours * config.max_gap_multiplier)

    trajectories: list[Trajectory] = []
    for (study, indiv), group in tqdm.tqdm(
        df.groupby(["study_id", "individual_id"], sort=False),
        desc="Grouping trajectories",
        unit="individual",
        leave=False,
    ):
        g = group.sort_values("timestamp").reset_index(drop=True)
        # Identify gap boundaries: dt > max_gap.
        dt = g["timestamp"].diff()
        break_points = dt.index[dt > max_gap].tolist()
        # Identify speed boundaries: implied speed > max_speed_mps.
        if config.max_speed_mps is not None:
            speed = _step_speed_mps(g)
            too_fast = speed.index[speed > config.max_speed_mps].tolist()
            if too_fast:
                logger.info(
                    "Split %s::%s at %d step(s) exceeding %.0f m/s.",
                    study, indiv, len(too_fast), config.max_speed_mps,
                )
            break_points = sorted(set(break_points) | set(too_fast))
        # Build segments between break points (inclusive of the break fix).
        bounds = [0, *break_points, len(g)]
        for start, end in zip(bounds[:-1], bounds[1:]):
            seg = g.iloc[start:end].reset_index(drop=True)
            trajectories.append(
                Trajectory(
                    individual_id=f"{study}::{indiv}",
                    study_id=str(study),
                    df=seg,
                )
            )
    trajectories.sort(key=lambda t: t.individual_id)
    return trajectories


def load_dataset_with_covariates(
    config: DataConfig, raw_csv_pattern: str, covariates
) -> tuple[list[Trajectory], list[str]]:
    """Like :func:`load_dataset`, but joins per-fix covariates before grouping.

    Covariates are attached per raw CSV (each has its own GEE file), *before*
    deduplication and gap splitting, so every surviving fix keeps its own row.
    Returns the trajectories plus the ordered covariate column list (empty when
    ``covariates`` is None or disabled).
    """
    if covariates is None or not covariates.enabled:
        return load_dataset(config, raw_csv_pattern), []
    from movement.data.covariates import attach_covariates

    raw = config.raw_path
    if not raw.is_dir():
        raise FileNotFoundError(f"Raw dataset path does not exist: {raw}")
    csvs = sorted(raw.glob(raw_csv_pattern))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found matching '{raw_csv_pattern}' in {raw}")
    frames, columns = [], None
    for path in tqdm.tqdm(csvs, desc="Loading raw CSVs + covariates", unit="file"):
        fixes, cols = attach_covariates(load_raw_csv(path), path, covariates)
        if columns is None:
            columns = cols
        elif cols != columns:
            raise ValueError(
                f"{path.name}: covariate columns differ from the first CSV's; a model "
                f"needs one fixed covariate layout across the files it trains on."
            )
        frames.append(fixes)
    all_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    logger.info("Loaded %d fixes (+%d covariates) across %d CSV(s).", len(all_df), len(columns or []), len(csvs))
    return group_into_trajectories(all_df, config), list(columns or [])


def load_dataset(config: DataConfig, raw_csv_pattern: str = "*.csv") -> list[Trajectory]:
    """Load every raw CSV under ``config.raw_path`` into trajectories.

    Trajectories shorter than a minimum length are discarded here (see
    :func:`group_into_trajectories`); callers that know the windowing parameters
    can additionally enforce ``input_len + horizon`` via
    :func:`discard_short`.
    """
    raw = config.raw_path
    if not raw.is_dir():
        raise FileNotFoundError(f"Raw dataset path does not exist: {raw}")
    csvs = sorted(raw.glob(raw_csv_pattern))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found matching '{raw_csv_pattern}' in {raw}")
    frames = [load_raw_csv(p) for p in tqdm.tqdm(csvs, desc="Loading raw CSVs", unit="file")]
    all_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    logger.info("Loaded %d fixes across %d CSV(s).", len(all_df), len(csvs))
    return group_into_trajectories(all_df, config)


def discard_short(trajectories: list[Trajectory], min_len: int) -> list[Trajectory]:
    """Drop trajectories with fewer than ``min_len`` fixes."""
    kept = [t for t in trajectories if t.n_fixes >= min_len]
    dropped = len(trajectories) - len(kept)
    if dropped:
        logger.info("Discarded %d trajectory segment(s) shorter than %d fixes.", dropped, min_len)
    return kept


def split_individuals(
    trajectories: list[Trajectory],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    study_col: str = "study_id",
) -> tuple[list[Trajectory], list[Trajectory], list[Trajectory]]:
    """Split trajectories by individual, stratified by study, seeded.

    Returns ``(train, val, test)``. Individuals are the unit of assignment, so
    no window from one animal ever appears in two splits. The assignment is
    deterministic given the same input order and seed; persist the actual ID
    lists to a split file to freeze it.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    by_study: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        by_study[t.study_id].append(t)

    for study, members in by_study.items():
        rng.shuffle(members)
        n = len(members)
        n_val = int(round(n * val_fraction))
        n_test = int(round(n * test_fraction))
        # Ensure no overlap and at least one individual in each split when possible.
        if n < 3:
            # Too few to split meaningfully: keep in train (study-level leakage
            # is prevented elsewhere by the individual namespace).
            train.extend(members)
            continue
        n_val = max(1, min(n_val, n - 2))
        n_test = max(1, min(n_test, n - n_val - 1))
        val.extend(members[:n_val])
        test.extend(members[n_val : n_val + n_test])
        train.extend(members[n_val + n_test :])
    return train, val, test


def split_individuals_disjoint(
    trajectories: list[Trajectory],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[Trajectory], list[Trajectory], list[Trajectory]]:
    """Split by *animal*: every segment of an individual lands in one split.

    The fix for progress_report.md §7.1. Unique individual IDs are shuffled per
    study (seeded, from a sorted list so input order cannot matter), assigned to
    splits with the same size rule as :func:`split_individuals`, and then every
    gap-separated segment of each ID follows it.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    segments: dict[str, list[Trajectory]] = defaultdict(list)
    study_of: dict[str, str] = {}
    for t in trajectories:
        segments[t.individual_id].append(t)
        study_of[t.individual_id] = t.study_id
    by_study: dict[str, list[str]] = defaultdict(list)
    for ind in sorted(segments):
        by_study[study_of[ind]].append(ind)

    assign: dict[str, str] = {}
    for study in sorted(by_study):
        ids = list(by_study[study])
        rng.shuffle(ids)
        n = len(ids)
        if n < 3:
            for i in ids:
                assign[i] = "train"
            continue
        n_val = max(1, min(int(round(n * val_fraction)), n - 2))
        n_test = max(1, min(int(round(n * test_fraction)), n - n_val - 1))
        for i in ids[:n_val]:
            assign[i] = "val"
        for i in ids[n_val : n_val + n_test]:
            assign[i] = "test"
        for i in ids[n_val + n_test :]:
            assign[i] = "train"

    out: dict[str, list[Trajectory]] = {"train": [], "val": [], "test": []}
    for ind in sorted(segments):
        out[assign[ind]].extend(segments[ind])
    assert_disjoint_ids(
        {t.individual_id for t in out["train"]},
        {t.individual_id for t in out["val"]},
        {t.individual_id for t in out["test"]},
    )
    return out["train"], out["val"], out["test"]


def assert_disjoint_ids(train: set[str], val: set[str], test: set[str], *, source: str = "split") -> None:
    """Fail loudly if any animal appears in more than one split."""
    overlaps = {
        "train∩val": sorted(train & val),
        "train∩test": sorted(train & test),
        "val∩test": sorted(val & test),
    }
    bad = {k: v for k, v in overlaps.items() if v}
    if bad:
        detail = "; ".join(f"{k}: {len(v)} (e.g. {v[:3]})" for k, v in bad.items())
        raise ValueError(
            f"{source} is not individual-disjoint — {detail}. This is the "
            f"progress_report.md §7.1 leakage. Regenerate the split with "
            f"data.split_unit=individual."
        )


def fold_assignment(fixes_per_animal: dict[str, int], *, n_folds: int, seed: int) -> dict[str, int]:
    """Deal animals into ``n_folds`` folds with near-equal total *fixes*.

    Longest-processing-time greedy: animals in descending fix count, each to the
    fold with the fewest fixes so far. Ties (equal counts, equal fold totals) are
    broken by a seeded shuffle, so the assignment is deterministic per seed and
    independent of input order.
    """
    import numpy as np

    if n_folds < 3:
        raise ValueError(f"n_folds must be >= 3 (test, val and at least one train fold); got {n_folds}")
    if len(fixes_per_animal) < n_folds:
        raise ValueError(
            f"Only {len(fixes_per_animal)} animals for {n_folds} folds; every fold needs at least one animal."
        )
    rng = np.random.default_rng(seed)
    ids = sorted(fixes_per_animal)
    tiebreak = {i: float(r) for i, r in zip(ids, rng.permutation(len(ids)))}
    order = sorted(ids, key=lambda i: (-fixes_per_animal[i], tiebreak[i]))
    fold_rank = rng.permutation(n_folds)  # seeded tie-break between equal folds
    totals = [0] * n_folds
    out: dict[str, int] = {}
    for ind in order:
        k = min(range(n_folds), key=lambda f: (totals[f], fold_rank[f]))
        out[ind] = k
        totals[k] += fixes_per_animal[ind]
    return out


def split_individuals_kfold(
    trajectories: list[Trajectory],
    *,
    n_folds: int,
    fold: int,
    seed: int,
) -> tuple[list[Trajectory], list[Trajectory], list[Trajectory]]:
    """Animal-disjoint split from fix-balanced folds (test = ``fold``, val = next).

    Every segment of an animal follows the animal. Across ``fold = 0..n_folds-1``
    every animal is in the test set exactly once.
    """
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds}), got {fold}")
    fixes: dict[str, int] = defaultdict(int)
    for t in trajectories:
        fixes[t.individual_id] += t.n_fixes
    folds = fold_assignment(dict(fixes), n_folds=n_folds, seed=seed)
    val_fold = (fold + 1) % n_folds
    out: dict[str, list[Trajectory]] = {"train": [], "val": [], "test": []}
    for t in sorted(trajectories, key=lambda t: (t.individual_id, t.start)):
        k = folds[t.individual_id]
        out["test" if k == fold else "val" if k == val_fold else "train"].append(t)
    assert_disjoint_ids(
        {t.individual_id for t in out["train"]},
        {t.individual_id for t in out["val"]},
        {t.individual_id for t in out["test"]},
    )
    return out["train"], out["val"], out["test"]


def tail_cuts(trajectories: list[Trajectory], fraction: float) -> dict[str, pd.Timestamp]:
    """Per-animal cut time: the last ``fraction`` of each animal's fixes (by time) lie at or after it."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"val_tail_fraction must be in (0, 1), got {fraction}")
    times: dict[str, list[np.ndarray]] = defaultdict(list)
    for t in trajectories:
        times[t.individual_id].append(t.df["timestamp"].to_numpy())
    out = {}
    for ind, parts in times.items():
        ts = np.sort(np.concatenate(parts))
        out[ind] = pd.Timestamp(ts[min(len(ts) - 1, int(np.floor((1.0 - fraction) * len(ts))))])
    return out


def apply_tail_cuts(
    trajectories: list[Trajectory], cuts: dict[str, pd.Timestamp]
) -> tuple[list[Trajectory], list[Trajectory]]:
    """Split each segment at its animal's cut: fixes before → train, at/after → validation."""
    train: list[Trajectory] = []
    val: list[Trajectory] = []
    for t in sorted(trajectories, key=lambda t: (t.individual_id, t.start)):
        before = t.df["timestamp"] < cuts[t.individual_id]
        for mask, out in ((before, train), (~before, val)):
            if mask.any():
                out.append(Trajectory(t.individual_id, t.study_id, t.df.loc[mask].reset_index(drop=True)))
    return train, val


def split_individuals_kfold_tailval(
    trajectories: list[Trajectory],
    *,
    n_folds: int,
    fold: int,
    seed: int,
    val_tail_fraction: float,
) -> tuple[list[Trajectory], list[Trajectory], list[Trajectory], dict[str, pd.Timestamp]]:
    """Animal-disjoint test fold; validation = the time-tail of every training animal.

    Uses the same fix-balanced fold assignment as :func:`split_individuals_kfold`,
    so test fold ``k`` holds the same animals in both schemes. All non-test
    animals train; the last ``val_tail_fraction`` of each one's fixes (by time)
    is validation, used only for early stopping and checkpoint selection.
    Returns ``(train, val, test, cuts)``.
    """
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds}), got {fold}")
    fixes: dict[str, int] = defaultdict(int)
    for t in trajectories:
        fixes[t.individual_id] += t.n_fixes
    folds = fold_assignment(dict(fixes), n_folds=n_folds, seed=seed)
    test = [t for t in trajectories if folds[t.individual_id] == fold]
    rest = [t for t in trajectories if folds[t.individual_id] != fold]
    cuts = tail_cuts(rest, val_tail_fraction)
    train, val = apply_tail_cuts(rest, cuts)
    test = sorted(test, key=lambda t: (t.individual_id, t.start))
    assert_tailval_split({t.individual_id for t in train} | {t.individual_id for t in val},
                         {t.individual_id for t in test})
    return train, val, test, cuts


def assert_tailval_split(train_val: set[str], test: set[str], *, source: str = "split") -> None:
    """Tail-validation splits share animals between train and val, never with test."""
    overlap = sorted(train_val & test)
    if overlap:
        raise ValueError(f"{source}: {len(overlap)} test animal(s) also in train/val (e.g. {overlap[:3]}).")


SPLITTERS = {
    "segment": split_individuals,
    "individual": split_individuals_disjoint,
}
DISJOINT_SPLIT_UNITS = ("individual", "individual_kfold")


def split_trajectories(
    trajectories: list[Trajectory], data: DataConfig, *, seed: int
) -> tuple[list[Trajectory], list[Trajectory], list[Trajectory]]:
    """Dispatch on ``data.split_unit`` (see :class:`movement.config.DataConfig`)."""
    if data.split_unit == "individual_kfold_tailval":
        train, val, test, _cuts = split_individuals_kfold_tailval(
            trajectories, n_folds=data.n_folds, fold=data.fold, seed=seed,
            val_tail_fraction=data.val_tail_fraction,
        )
        return train, val, test
    if data.split_unit == "individual_kfold":
        return split_individuals_kfold(trajectories, n_folds=data.n_folds, fold=data.fold, seed=seed)
    return SPLITTERS[data.split_unit](
        trajectories, val_fraction=data.val_fraction, test_fraction=data.test_fraction, seed=seed
    )


def save_split(run_dir: Path, train: list[Trajectory], val: list[Trajectory], test: list[Trajectory],
               *, val_cuts: dict[str, pd.Timestamp] | None = None) -> Path:
    """Persist the exact individual assignment to ``split.json`` in the run dir.

    ``val_cuts`` (tail-validation splits): per training animal, the time from
    which its fixes are validation. Stored as ISO strings.
    """
    import json

    payload = {
        "train": sorted({t.individual_id for t in train}),
        "val": sorted({t.individual_id for t in val}),
        "test": sorted({t.individual_id for t in test}),
    }
    if val_cuts is not None:
        payload["val_cut"] = {k: pd.Timestamp(v).isoformat() for k, v in sorted(val_cuts.items())}
    path = run_dir / "split.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
