"""Chunk ledger: export units, their identity, and resumable state.

Mirrors ``scripts/sweep.py``'s resumable-ledger pattern: the ledger is
**append-only** (``results/covariates/<source>/chunks.csv``), the last row per
``chunk_id`` wins, and a partially submitted run always leaves a valid file.
That is what makes killing the process mid-run safe — completed chunks are never
redone.

``chunk_id`` is a hash of the chunk's **resolved spec**: the source's full YAML
config plus a digest of its key list. An unchanged chunk therefore keeps its id
(and is skipped on resume), while editing a source's config or grid invalidates
exactly that source's chunks.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from movement.covariates.sources import ExportConfig, Source

logger = logging.getLogger(__name__)

LEDGER_COLUMNS = [
    "chunk_id", "dataset", "source", "zone", "time_bucket", "n_features",
    "status", "task_id", "gcs_uri", "submitted_at", "finished_at",
    "failure_reason", "n_rows_out",
]

# Lifecycle: pending -> submitted -> running -> complete | failed | skipped.
STATUSES = ("pending", "submitted", "running", "complete", "failed", "skipped")
# Anything not in this set is re-submitted on resume.
SETTLED = ("complete",)

KEY_COLUMNS = ["zone", "px_row", "px_col", "t0"]


def canonical_json(obj: Any) -> str:
    """Deterministic JSON for hashing (sorted keys, compact separators)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def chunk_id_for(source: Source, dataset: str, zone: int, bucket: str, key_digest: str) -> str:
    """Stable identity of an export unit.

    Includes the source's *resolved spec*, so a grid/band/QA change invalidates
    the chunks that depended on it and nothing else.
    """
    payload = canonical_json(
        {
            "source": source.resolved_spec(),
            "dataset": dataset,
            "zone": int(zone),
            "bucket": bucket,
            "keys": key_digest,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def key_digest(keys: pd.DataFrame) -> str:
    """Digest of a chunk's key list, independent of row order."""
    if keys.empty:
        return hashlib.sha256(b"").hexdigest()[:16]
    ordered = keys[KEY_COLUMNS].sort_values(KEY_COLUMNS)
    hashed = pd.util.hash_pandas_object(ordered, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


@dataclass
class Chunk:
    """One export unit: a bounded set of unique keys, one queue task."""

    chunk_id: str
    dataset: str
    source: str
    zone: int
    time_bucket: str
    keys: pd.DataFrame

    @property
    def n_features(self) -> int:
        return len(self.keys)

    def object_path(self, export: ExportConfig) -> str:
        """Bucket-relative object path for ``cloud_storage`` (what ``join`` reads)."""
        ext = export.file_format.lower()
        return f"{export.gcs_prefix}/{self.source}/{self.dataset}/{self.chunk_id}.{ext}"

    def drive_prefix(self, export: ExportConfig) -> str:
        """Drive ``fileNamePrefix`` (Earth Engine appends the format's extension).

        Drive exports land in a single folder, so the source and dataset are folded
        into the name rather than nested as directories.
        """
        return f"{self.source}_{self.dataset}_{self.chunk_id}"

    def output_name(self, export: ExportConfig) -> str:
        """Flat file name Drive will produce, for matching on read-back."""
        return f"{self.drive_prefix(export)}.{export.file_format.lower()}"

    def location(self, export: ExportConfig, destination: str, target: str) -> str:
        """Where this chunk's output lives, as recorded in the ledger.

        The ledger column keeps the spec's ``gcs_uri`` name; it holds a ``gs://``
        URI for Cloud Storage and a ``drive://`` URI for Drive.
        """
        if destination == "drive":
            return f"drive://{target}/{self.output_name(export)}"
        return f"gs://{target}/{self.object_path(export)}"


def build_chunks(
    source: Source,
    export: ExportConfig,
    dataset: str,
    keys: pd.DataFrame,
) -> list[Chunk]:
    """Split unique keys into export units, favouring fewer, larger tasks.

    Keys are ordered by ``(pixel-zone, time)`` and grouped into consecutive
    time buckets. Buckets are then **merged** until the task would exceed
    ``export.target_features_per_task`` or touch more than
    ``export.max_time_groups_per_task`` distinct images. Merging is what keeps a
    daily source from degenerating into one task per day: thousands of tiny tasks
    running two at a time is slower than tens of large ones.

    ``chunk_id`` depends on the resulting key list, so changing
    ``target_features_per_task`` re-partitions chunks and invalidates their ids.
    Fix the target from the ``plan`` estimate before the first submission.
    """
    if keys.empty:
        return []
    target = max(1, int(export.target_features_per_task))
    max_groups = max(1, int(export.max_time_groups_per_task))
    chunks: list[Chunk] = []

    for zone, zone_keys in keys.groupby("zone", sort=True):
        # Order buckets by their numeric time key (bucket strings are not
        # lexicographically ordered for composite windows: "c1000" < "c999").
        bucket_order = (
            zone_keys.groupby("bucket", sort=False)["t0"].min().sort_values(kind="stable")
        )
        buffer: list[pd.DataFrame] = []
        buckets: list[str] = []
        n_features = 0
        for bucket in bucket_order.index:
            group = zone_keys[zone_keys["bucket"] == bucket].sort_values(KEY_COLUMNS)
            # A single bucket bigger than the target is split, not emitted whole.
            if len(group) > target:
                if buffer:
                    chunks.append(_make_chunk(source, dataset, int(zone), buffer, buckets))
                    buffer, buckets, n_features = [], [], 0
                for start in range(0, len(group), target):
                    piece = group.iloc[start : start + target]
                    chunks.append(_make_chunk(source, dataset, int(zone), [piece], [str(bucket)]))
                continue
            if buffer and (n_features + len(group) > target or len(buckets) >= max_groups):
                chunks.append(_make_chunk(source, dataset, int(zone), buffer, buckets))
                buffer, buckets, n_features = [], [], 0
            buffer.append(group)
            buckets.append(str(bucket))
            n_features += len(group)
        if buffer:
            chunks.append(_make_chunk(source, dataset, int(zone), buffer, buckets))
    return chunks


def _make_chunk(
    source: Source,
    dataset: str,
    zone: int,
    frames: list[pd.DataFrame],
    buckets: list[str],
) -> Chunk:
    piece = pd.concat(frames, ignore_index=True).sort_values(KEY_COLUMNS).reset_index(drop=True)
    label = buckets[0] if len(buckets) == 1 else f"{buckets[0]}..{buckets[-1]}"
    return Chunk(
        chunk_id=chunk_id_for(source, dataset, zone, label, key_digest(piece)),
        dataset=dataset,
        source=source.name,
        zone=zone,
        time_bucket=label,
        keys=piece,
    )


# ---------------------------------------------------------------------------
# Ledger I/O
# ---------------------------------------------------------------------------
def ledger_path(export: ExportConfig, source: str) -> Path:
    return Path(export.results_root) / source / "chunks.csv"


def load_ledger(path: Path) -> dict[str, dict[str, str]]:
    """Last row per ``chunk_id`` (append-only ledger, last write wins)."""
    if not path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        import csv

        for row in csv.DictReader(f):
            if row.get("chunk_id"):
                rows[row["chunk_id"]] = row
    return rows


def record_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Append ledger rows (creating the header on first write).

    Append-only by design: an interrupted run leaves a valid ledger containing
    every state transition that actually happened.
    """
    rows = list(rows)
    if not rows:
        return
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LEDGER_COLUMNS})


def row_for(chunk: Chunk, *, status: str, **extra: Any) -> dict[str, Any]:
    """Build a ledger row for a chunk."""
    if status not in STATUSES:
        raise ValueError(f"Unknown chunk status {status!r}; expected one of {STATUSES}")
    row = {
        "chunk_id": chunk.chunk_id,
        "dataset": chunk.dataset,
        "source": chunk.source,
        "zone": chunk.zone,
        "time_bucket": chunk.time_bucket,
        "n_features": chunk.n_features,
        "status": status,
        "task_id": "",
        "gcs_uri": "",
        "submitted_at": "",
        "finished_at": "",
        "failure_reason": "",
        "n_rows_out": "",
    }
    row.update(extra)
    return row


def resumable_chunks(
    chunks: Iterable[Chunk],
    ledger: dict[str, dict[str, str]],
    *,
    retry_failed: bool = False,
) -> list[Chunk]:
    """Chunks that still need work.

    ``complete`` chunks are always skipped. ``failed`` chunks are re-attempted
    only with ``retry_failed=True``. Everything else (pending/submitted/running)
    is actionable — a ``submitted`` row whose task died is exactly what resume is
    for.
    """
    out: list[Chunk] = []
    for chunk in chunks:
        row = ledger.get(chunk.chunk_id)
        if row is None:
            out.append(chunk)
            continue
        status = row.get("status", "")
        if status in SETTLED:
            continue
        if status == "failed" and not retry_failed:
            continue
        out.append(chunk)
    return out


def ledger_status_counts(ledger: dict[str, dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {s: 0 for s in STATUSES}
    for row in ledger.values():
        status = row.get("status", "")
        counts[status] = counts.get(status, 0) + 1
    return counts
