"""Join-back: completed export output -> one Parquet row per fix.

Reads the completed chunks' output (from GCS in normal operation, or from a local
directory for dry runs and tests), maps ``key_index`` back through the
``fix_id -> key_index`` table, and writes
``results/covariates/<dataset>/<source>.parquet`` in the schema of
``covariate_plan.md`` §5.4.

The final assert is the point of this module: **every fix appears exactly once** —
no drops, no duplicates. A covariate table that quietly loses or double-counts
fixes would silently misalign the whole feature matrix downstream.

``_age_hours`` is emitted once per source (all parameters of a source share one
observation time), and ``<param>_missing`` once per parameter. The buffer column
shares the point column's mask — both are sampled from the same ``updateMask``'d
image — so a single indicator per parameter covers both, as documented in the
delivered README.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import tqdm

from movement.covariates.ledger import Chunk, ledger_path, load_ledger
from movement.covariates.sources import (
    BUFFER_SUFFIX,
    SCENES_BAND,
    ExportConfig,
    Source,
    age_hours,
)

logger = logging.getLogger(__name__)

KEY_COLUMN = "key_index"
AGE_COLUMN = "_age_hours"
SCENES_COLUMN = SCENES_BAND

# Materialises one chunk's exported output as local file paths.
FetchFn = Callable[[Chunk], list[Path]]


# ---------------------------------------------------------------------------
# Chunk output collection
# ---------------------------------------------------------------------------
def read_chunk_csv(path: Path, source: Source) -> pd.DataFrame:
    """Read one chunk's exported CSV; keeps ``key_index`` + value columns."""
    frame = pd.read_csv(path)
    if KEY_COLUMN not in frame.columns:
        raise ValueError(
            f"{path.name}: exported chunk has no {KEY_COLUMN!r} column "
            f"(found {list(frame.columns)}) — the export must carry the join key."
        )
    return frame


def _download_from_gcs(client: Any, bucket: str, prefix: str, dest: Path) -> list[Path]:
    """Download every object whose name starts with ``prefix``."""
    blobs = list(client.list_blobs(bucket, prefix=prefix))
    if not blobs:
        raise FileNotFoundError(f"No GCS objects under gs://{bucket}/{prefix}")
    dest.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for blob in blobs:
        local = dest / Path(blob.name).name
        blob.download_to_filename(str(local))
        paths.append(local)
    return paths


def collect_chunk_outputs(
    export: ExportConfig,
    source: Source,
    chunks: Iterable[Chunk],
    fetch: "FetchFn",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Concatenate every completed chunk's sampled values.

    Returns ``(merged, rows_per_chunk)``. ``fetch`` is a :data:`FetchFn` that
    materialises one chunk's output as local paths, so local directories, Cloud
    Storage and Drive differ only in which fetcher the caller supplies.
    """
    chunks = list(chunks)
    frames: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    for chunk in tqdm.tqdm(chunks, desc=f"Collect {source.name}", unit="chunk"):
        paths = fetch(chunk)
        if not paths:
            raise FileNotFoundError(
                f"Chunk {chunk.chunk_id} marked complete but no output was found "
                f"for it ({source.name}/{chunk.dataset})."
            )
        frame = pd.concat([read_chunk_csv(p, source) for p in paths], ignore_index=True)
        counts[chunk.chunk_id] = len(frame)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No chunk output collected for source {source.name!r}")
    merged = pd.concat(frames, ignore_index=True)
    if merged[KEY_COLUMN].duplicated().any():
        dupes = int(merged[KEY_COLUMN].duplicated().sum())
        raise ValueError(
            f"{source.name}: {dupes} duplicated key_index in the collected chunk "
            f"output — a key was exported by more than one chunk."
        )
    return merged, counts


def local_fetcher(local_dir: Path | str) -> "FetchFn":
    """Read chunk output from a local directory (dry runs, tests, offline joins)."""
    def fetch(chunk: Chunk) -> list[Path]:
        paths = sorted(Path(local_dir).glob(f"{chunk.chunk_id}*"))
        if not paths:
            raise FileNotFoundError(
                f"Chunk {chunk.chunk_id} marked complete but no local output "
                f"matching {chunk.chunk_id!r} in {local_dir}"
            )
        return paths

    return fetch


def gcs_fetcher(
    export: ExportConfig, client: Any, bucket: str, scratch: Path | str
) -> "FetchFn":
    """Download chunk output from Cloud Storage into ``scratch``."""
    def fetch(chunk: Chunk) -> list[Path]:
        dest = Path(scratch) / chunk.dataset / chunk.chunk_id
        return _download_from_gcs(client, bucket, chunk.object_path(export), dest)

    return fetch


def drive_fetcher(client: Any, folder: str, scratch: Path | str) -> "FetchFn":
    """Download chunk output from a Drive folder into ``scratch``."""
    def fetch(chunk: Chunk) -> list[Path]:
        # The Drive file is `{source}_{dataset}_{chunk_id}.csv`; the chunk id is a
        # hash, so it identifies the file (and any shards) unambiguously.
        dest = Path(scratch) / chunk.dataset / chunk.chunk_id
        return _download_from_drive(client, folder, chunk.chunk_id, dest)

    return fetch


def _download_from_drive(
    client: Any, folder: str, contains: str, dest: Path
) -> list[Path]:
    """Materialise the Drive files matching ``contains`` under ``dest``."""
    files = client.fetch(folder, contains)
    dest.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, content in files:
        path = dest / name
        path.write_bytes(content)
        paths.append(path)
    return sorted(paths)


# ---------------------------------------------------------------------------
# Per-fix table
# ---------------------------------------------------------------------------
def _value_columns(source: Source, sampled: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(point value columns, buffer columns) present in the export output."""
    names = [c for c in source.sampled_bands if c in sampled.columns]
    buf = [f"{c}{BUFFER_SUFFIX}" for c in names if f"{c}{BUFFER_SUFFIX}" in sampled.columns]
    return names, buf


def build_fix_table(
    source: Source,
    fixes: pd.DataFrame,
    fixmap: pd.DataFrame,
    sampled: pd.DataFrame,
    *,
    dataset: str,
) -> pd.DataFrame:
    """Expand sampled key values back to one row per fix.

    Columns: ``fix_id, dataset, timestamp`` + one per parameter (+ ``_buf30``),
    ``<param>_missing``, ``_age_hours``, and ``n_scenes`` for composite sources.
    """
    point_cols, buf_cols = _value_columns(source, sampled)
    if not point_cols:
        raise ValueError(
            f"{source.name}: none of the declared bands {source.sampled_bands} "
            f"appear in the export output {list(sampled.columns)}"
        )

    joined = fixmap.merge(sampled, on=KEY_COLUMN, how="left", validate="many_to_one")
    if joined[KEY_COLUMN].isna().any():
        missing = int(joined[KEY_COLUMN].isna().sum())
        raise ValueError(
            f"{dataset}/{source.name}: {missing} fixes mapped to a key with no "
            f"exported value — refusing to write a partial covariate table."
        )

    fix_meta = fixes[["fix_id", "timestamp"]]
    joined = joined.merge(fix_meta, on="fix_id", how="left", validate="many_to_one")

    out = pd.DataFrame({"fix_id": joined["fix_id"], "dataset": dataset,
                        "timestamp": joined["timestamp"]})
    for col in point_cols + buf_cols:
        out[col] = pd.to_numeric(joined[col], errors="coerce")
    # Missingness is per parameter, and covers the buffer column too (same mask).
    for col in point_cols:
        out[f"{col}_missing"] = out[col].isna()
    if SCENES_COLUMN in joined.columns:
        out[SCENES_COLUMN] = pd.to_numeric(joined[SCENES_COLUMN], errors="coerce")
    out[AGE_COLUMN] = [age_hours(source, ts) for ts in tqdm.tqdm(
        joined["timestamp"], desc=f"Age ({source.name})", leave=False)]
    return out


def assert_one_row_per_fix(out: pd.DataFrame, fixes: pd.DataFrame, *, dataset: str, source: str) -> None:
    """The join's contract: exactly one output row per input fix."""
    if len(out) != len(fixes):
        raise AssertionError(
            f"{dataset}/{source}: {len(out)} output rows for {len(fixes)} fixes."
        )
    if out["fix_id"].duplicated().any():
        dupes = int(out["fix_id"].duplicated().sum())
        raise AssertionError(f"{dataset}/{source}: {dupes} duplicated fix_id in output.")
    if set(out["fix_id"]) != set(fixes["fix_id"]):
        missing = len(set(fixes["fix_id"]) - set(out["fix_id"]))
        extra = len(set(out["fix_id"]) - set(fixes["fix_id"]))
        raise AssertionError(
            f"{dataset}/{source}: fix_id set mismatch ({missing} missing, {extra} extra)."
        )


def output_path(export: ExportConfig, dataset: str, source: str) -> Path:
    return Path(export.results_root) / dataset / f"{source}.parquet"


def join_source(
    export: ExportConfig,
    source: Source,
    dataset: str,
    fixes: pd.DataFrame,
    fixmap: pd.DataFrame,
    chunks: list[Chunk],
    fetch: "FetchFn",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collect, expand and write one (dataset, source) covariate Parquet."""
    sampled, counts = collect_chunk_outputs(export, source, chunks, fetch)
    out = build_fix_table(source, fixes, fixmap, sampled, dataset=dataset)
    assert_one_row_per_fix(out, fixes, dataset=dataset, source=source.name)
    path = output_path(export, dataset, source.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    logger.info("Wrote %s (%d rows, %d columns)", path, len(out), len(out.columns))
    return out, counts


def completed_chunks(export: ExportConfig, source: str, expected: dict[str, Chunk]) -> list[Chunk]:
    """Chunks the ledger records as complete, in ledger order."""
    ledger = load_ledger(ledger_path(export, source))
    out = [expected[cid] for cid, row in ledger.items()
           if row.get("status") == "complete" and cid in expected]
    if not out:
        raise RuntimeError(
            f"{source}: no chunks recorded complete in {ledger_path(export, source)} — "
            f"run `submit` and `poll` first."
        )
    return out


def gcs_client(auth: dict | None = None) -> Any:  # pragma: no cover - credentials/network
    """Authenticated GCS client, reusing the Earth Engine credentials.

    Works for both auth modes: a service-account key, or the interactive OAuth
    login (whose stored scopes include ``devstorage.full_control``), so `join` does
    not need separate GCS credentials.
    """
    from movement.utils.env import gee_auth

    auth = auth or gee_auth()
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-storage is required to download export output. Install "
            "the GEE extra: `uv sync --extra gee`."
        ) from exc
    return storage.Client(project=auth["project"], credentials=auth["credentials"])


# ---------------------------------------------------------------------------
# Wide per-dataset CSV
# ---------------------------------------------------------------------------
# Covariate columns are prefixed with their source name in the wide CSV. Two
# sources can legitimately emit the same parameter name (`confidence` appears in
# both Annual NLCD and CDL) and every source emits `_age_hours` / `n_scenes`, so
# prefixing is the only collision-free way to concatenate them without dropping
# information. `fix_id`, `dataset` and `timestamp` stay unprefixed.
IDENTITY_COLUMNS = ("fix_id", "dataset", "timestamp")
# Compact but ample for every native unit in the registry (reflectances to 1e-4,
# elevations to metres, LST to hundredths of a kelvin).
WIDE_CSV_FLOAT_FORMAT = "%.6g"


def wide_csv_path(out_dir: Path | str, dataset: str) -> Path:
    """``<GEE_DATASET_PATH>/<dataset>.csv``."""
    return Path(out_dir) / f"{dataset}.csv"


def prefixed_column(source: str, column: str) -> str:
    """Wide-CSV name for a source's covariate column.

    The leading underscore of the tool's own columns (`_age_hours`) is dropped so
    the result reads ``era5_land_age_hours`` rather than ``era5_land__age_hours``.
    """
    return f"{source}_{column.lstrip('_')}"


def load_source_tables(export: ExportConfig, dataset: str) -> dict[str, pd.DataFrame]:
    """Every per-source Parquet present on disk for a dataset, keyed by source."""
    root = Path(export.results_root) / dataset
    if not root.is_dir():
        return {}
    return {path.stem: pd.read_parquet(path) for path in sorted(root.glob("*.parquet"))}


def combine_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join per-source fix tables into one wide table, one row per fix.

    Sources are merged on ``fix_id`` in alphabetical order. The fix universes must
    agree exactly: every source is derived from the same movement CSV, so a
    mismatch means a stale or truncated table and must fail loudly rather than
    silently produce a short column.
    """
    if not tables:
        raise ValueError("combine_tables() needs at least one source table")
    names = sorted(tables)
    base: pd.DataFrame | None = None
    for name in names:
        frame = tables[name]
        missing = [c for c in IDENTITY_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{name}: fix table is missing identity column(s) {missing}")
        covariates = [c for c in frame.columns if c not in IDENTITY_COLUMNS]
        renamed = frame[["fix_id", *covariates]].rename(
            columns={c: prefixed_column(name, c) for c in covariates}
        )
        if base is None:
            base = renamed
            continue
        if set(frame["fix_id"]) != set(base["fix_id"]):
            only_left = len(set(frame["fix_id"]) - set(base["fix_id"]))
            only_right = len(set(base["fix_id"]) - set(frame["fix_id"]))
            raise ValueError(
                f"{name}: fix universe disagrees with the sources already combined "
                f"({only_left} fixes only in {name}, {only_right} only in the others). "
                f"Refusing to write a wide table with a short column."
            )
        base = base.merge(renamed, on="fix_id", how="inner", validate="one_to_one")

    reference = tables[names[0]]
    identity = reference[list(IDENTITY_COLUMNS)]
    out = identity.merge(base, on="fix_id", how="inner", validate="one_to_one")
    assert_one_row_per_fix(out, reference, dataset=str(out["dataset"].iloc[0]),
                           source="+".join(names))
    return out.sort_values(["timestamp", "fix_id"]).reset_index(drop=True)


def write_wide_csv(
    export: ExportConfig, dataset: str, out_dir: Path | str, tables: dict[str, pd.DataFrame]
) -> Path:
    """Write ``<GEE_DATASET_PATH>/<dataset>.csv`` from the per-source tables."""
    combined = combine_tables(tables)
    path = wide_csv_path(out_dir, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False, float_format=WIDE_CSV_FLOAT_FORMAT)
    size_mb = path.stat().st_size / 1024**2
    logger.info(
        "Wrote %s (%d rows x %d columns, %.1f MB) from %d source(s)",
        path, len(combined), len(combined.columns), size_mb, len(tables),
    )
    if len(combined.columns) > 200:
        logger.warning(
            "%s has %d covariate columns — the wide CSV is a convenience view. The "
            "per-source Parquet under %s is the compact archive; prefer it for "
            "loading in the model pipeline.",
            path.name, len(combined.columns), Path(export.results_root) / dataset,
        )
    return path


def combine_dataset(
    export: ExportConfig, dataset: str, out_dir: Path | str
) -> Path | None:
    """Combine whatever per-source Parquet exists for a dataset into one CSV.

    Returns ``None`` (with a logged warning) when no source has been joined yet;
    this is an additive view over the Parquet archive, not a replacement for it.
    """
    tables = load_source_tables(export, dataset)
    if not tables:
        logger.warning(
            "No per-source Parquet for %s under %s — nothing to combine into %s.csv. "
            "Run `join` first.",
            dataset, Path(export.results_root) / dataset, dataset,
        )
        return None
    return write_wide_csv(export, dataset, out_dir, tables)
