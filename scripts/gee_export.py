"""GEE covariate export tool: plan | dedup | submit | poll | join | status.

Extracts the satellite covariates in ``covariate_plan.md`` §3 for the 8 CONUS
movement datasets via Google Earth Engine **batch exports**, and joins the results
back to the raw movement CSVs as per-fix Parquet tables.

The tool is resumable and quota-aware by construction:

- Everything goes through a batch export (``Export.table.toDrive`` or
  ``toCloudStorage``, per ``export.destination``) — never ``getInfo()`` or
  ``sampleRegions().getInfo()`` on anything large. The one ``getInfo()`` is in
  ``coverage``, which fetches a short list of monthly scene counts.
- Chunks are keyed by a hash of their resolved spec, so an unchanged chunk is
  never re-exported (``plan`` -> ``dedup`` -> ``submit`` -> ``poll`` -> ``join``).
- Never more than ``export.max_tasks_in_queue`` tasks in flight; the loop waits
  for a slot as tasks finish.
- The ledger is append-only and Ctrl-C safe: killing the process mid-run leaves
  a valid ledger, and re-running redoes no completed chunk.

Usage:
    uv run python scripts/gee_export.py plan   --dataset all --source all
    uv run python scripts/gee_export.py dedup  --dataset all --source all
    uv run python scripts/gee_export.py submit --dataset all --source all [--max-tasks N]
    uv run python scripts/gee_export.py poll   [--watch]
    uv run python scripts/gee_export.py join   --dataset all --source all
    uv run python scripts/gee_export.py status
    uv run python scripts/gee_export.py coverage --dataset mule_deer --source sentinel2

``plan`` writes nothing and performs no Earth Engine computation — run it first.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from movement.covariates.dedup import (
    DedupResult,
    check_reduction,
    dedup_dataset,
    dedup_summary,
    read_fixes,
)
from movement.covariates.drive import drive_client
from movement.covariates.join import (
    combine_dataset,
    completed_chunks,
    drive_fetcher,
    gcs_client,
    gcs_fetcher,
    join_source,
    local_fetcher,
)
from movement.covariates.ledger import (
    Chunk,
    build_chunks,
    ledger_path,
    ledger_status_counts,
    load_ledger,
    record_rows,
    resumable_chunks,
    row_for,
)
from movement.covariates.sources import (
    ExportConfig,
    Registry,
    Source,
    build_task_table,
    ee_module,
    handler_for,
    load_registry,
    monthly_scene_counts,
    task_selectors,
)
from movement.utils.env import (
    GEE_AUTH_MODES,
    GEE_SERVICE_ACCOUNT_VARS,
    GEE_TARGET_VARS,
    gee_auth,
    gee_dataset_path,
    raw_dataset_path,
    repo_root,
    resolve_auth_mode,
)

logger = logging.getLogger("gee_export")

REPO_ROOT = repo_root()
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "covariates" / "sources.yaml"

# Earth Engine task states -> ledger statuses.
_STATE_TO_STATUS = {
    "READY": "submitted",
    "RUNNING": "running",
    "COMPLETED": "complete",
    "FAILED": "failed",
    "CANCELLED": "failed",
    "CANCEL_REQUESTED": "running",
}
# Substrings that mark a *transient* failure worth retrying; anything else is
# recorded as a permanent failure with its reason and the loop moves on.
_TRANSIENT_MARKERS = (
    "rate limit", "quota", "too many requests", "429", "500", "502", "503", "504",
    "deadline exceeded", "timeout", "timed out", "backend error", "internal error",
    "unavailable", "connection reset", "temporarily", "try again",
)


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pairs(registry: Registry, args: argparse.Namespace) -> list[tuple[str, Source]]:
    datasets = registry.select_datasets(args.dataset)
    sources = registry.select_sources(args.source)
    return [(d, registry.source(s)) for d in datasets for s in sources]


def keys_path(registry: Registry, source: str, dataset: str) -> Path:
    return Path(registry.export.results_root) / source / "keys" / f"{dataset}.parquet"


def fixmap_path(registry: Registry, source: str, dataset: str) -> Path:
    return Path(registry.export.results_root) / source / "fixmap" / f"{dataset}.parquet"


def read_dataset_fixes(registry: Registry, dataset: str, raw_path: Path | None) -> pd.DataFrame:
    """Raw CSV -> normalised fix table (handles duplicated columns loudly)."""
    spec = registry.dataset(dataset)
    raw_dir = raw_path or raw_dataset_path()
    return read_fixes(raw_dir / spec.csv, duplicate_columns=spec.duplicate_columns)


def load_or_build_dedup(
    registry: Registry,
    dataset: str,
    source: Source,
    *,
    raw_path: Path | None = None,
    rebuild: bool = False,
    write: bool = False,
) -> DedupResult:
    """Dedup a (dataset, source) pair, reusing the on-disk key tables when present."""
    kpath, mpath = keys_path(registry, source.name, dataset), fixmap_path(registry, source.name, dataset)
    if not rebuild and kpath.exists() and mpath.exists():
        keys = pd.read_parquet(kpath)
        fixmap = pd.read_parquet(mpath)
        n_fixes = int(keys["n_fixes"].sum())
        return DedupResult(dataset=dataset, source=source.name, keys=keys,
                           fixmap=fixmap, n_fixes=n_fixes)

    fixes = read_dataset_fixes(registry, dataset, raw_path)
    result = dedup_dataset(fixes, source, dataset)
    check_reduction(source, registry.export, result)
    if write:
        kpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        result.keys.to_parquet(kpath, index=False)
        result.fixmap.to_parquet(mpath, index=False)
        logger.info("Wrote %s and %s", kpath, mpath)
    return result


def _init_ee(ee: object, auth: dict) -> None:
    """Initialise Earth Engine for the resolved auth mode, or fail loudly."""
    project = auth["project"]
    try:
        if auth["mode"] == "service_account":
            ee.Initialize(auth["credentials"], project=project)  # type: ignore[attr-defined]
        else:
            # No-op when the cached credentials are still valid (the usual case);
            # otherwise it opens the default browser and captures the OAuth
            # callback on http://localhost:8085.
            ee.Authenticate()  # type: ignore[attr-defined]
            ee.Initialize(project=project)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim, never swallowed
        raise RuntimeError(
            f"ee.Initialize() failed for project {project!r} (auth mode "
            f"{auth['mode']}): {exc}. Earth Engine requires a *registered* Cloud "
            f"project with the Earth Engine API enabled — an interactive login does "
            f"not create one. See the README's covariate section."
        ) from exc


def _interactive_account(credentials: object) -> str | None:
    """Best-effort: which Google account the interactive credentials belong to.

    Purely informational (used to make the fallback banner specific), so every
    failure is swallowed and reported as "unknown".
    """
    try:
        from google.auth.transport.requests import Request

        credentials.refresh(Request())  # type: ignore[attr-defined]
        request = urllib.request.Request(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {credentials.token}"},  # type: ignore[attr-defined]
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode()).get("email")
    except Exception:  # noqa: BLE001 - informational only
        return None


def _announce_auth(auth: dict) -> None:
    """State which identity is about to spend quota, loudly for the fallback."""
    destination = auth["destination"]
    where = (f"Drive folder {auth['target']!r}" if destination == "drive"
             else f"bucket gs://{auth['target']}")
    if auth["mode"] == "service_account":
        logger.info("Earth Engine auth: service account %s (project %s) -> %s",
                    auth["service_account"], auth["project"], where)
        return
    account = _interactive_account(auth["credentials"]) or "cached Earth Engine login"
    banner = (
        f"Earth Engine auth: INTERACTIVE as {account}.\n"
        f"{' and '.join(GEE_SERVICE_ACCOUNT_VARS)} are not both set in .env, so this "
        f"run is NOT reproducible from config alone: it depends on a personal Google "
        f"account login (project {auth['project']}).\n"
        f"Export destination: {where}.\n"
        f"Set both variables for a service account, or pin GEE_AUTH=service_account "
        f"to turn this fallback into an error."
    )
    print(f"\n{'!' * 78}\n{banner}\n{'!' * 78}\n", file=sys.stderr)


def _print_auth_mode(registry: Registry, args: argparse.Namespace) -> None:
    """Report which identity and destination the Earth Engine commands would use."""
    export = registry.export
    print(f"\nExport destination: {export.destination} "
          f"(set by export.destination in the registry)")
    try:
        mode = resolve_auth_mode(args.auth)
    except RuntimeError as exc:
        print(f"Earth Engine auth: unresolved - {exc}")
        return
    if mode == "service_account":
        print("Earth Engine auth: service account (configured in .env)")
    else:
        print(
            "Earth Engine auth: interactive (personal Google login) - "
            f"{' and '.join(GEE_SERVICE_ACCOUNT_VARS)} not set, so submit/poll/join "
            "are not reproducible from config alone"
        )
    target_var = GEE_TARGET_VARS[export.destination]
    if not os.environ.get(target_var, "").strip():
        print(f"  {target_var} is not set in .env")


def _is_transient(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def cmd_plan(registry: Registry, args: argparse.Namespace) -> None:
    """Print the chunk plan; write nothing, touch no Earth Engine."""
    rows = []
    unverified: set[str] = set()
    for dataset, source in _pairs(registry, args):
        result = load_or_build_dedup(registry, dataset, source, raw_path=args.raw_path)
        chunks = build_chunks(source, registry.export, dataset, result.keys)
        summary = dedup_summary(result)
        summary["chunks"] = len(chunks)
        summary["tier"] = source.tier
        summary["cadence"] = source.cadence
        summary["scale_m"] = source.scale_m
        summary["mean_features"] = (
            sum(c.n_features for c in chunks) / len(chunks) if chunks else 0
        )
        rows.append(summary)
        if not source.grid.verified:
            unverified.add(source.name)

    frame = pd.DataFrame(rows)
    print(f"# Covariate export plan  ({len(registry.datasets)} datasets x "
          f"{len(registry.sources)} sources)")
    print()
    print("ratio  = unique keys / fixes   (low = dedup works; high is legitimate when")
    print("                                the source cadence matches the fix cadence)")
    print("px/r   = distinct pixels / fixes (THE snapping diagnostic; must be low for")
    print("                                a coarse source or the grid config is wrong)")
    print()
    print(f"{'dataset':<17} {'source':<19} {'tier':>4} {'cadence':<10} {'scale':>7} "
          f"{'fixes':>9} {'cells':>8} {'px/r':>6} {'unique':>9} {'ratio':>6} "
          f"{'buckets':>8} {'chunks':>7} {'f/chunk':>8}")
    print("-" * 140)
    for row in frame.sort_values(["dataset", "source"]).itertuples():
        print(f"{row.dataset:<17} {row.source:<19} {row.tier:>4} {row.cadence:<10} "
              f"{row.scale_m:>7.0f} {row.fixes:>9,} {row.pixels:>8,} "
              f"{row.pixel_ratio:>6.3f} {row.unique_keys:>9,} {row.reduction_ratio:>6.3f} "
              f"{row.buckets:>8,} {row.chunks:>7,} {row.mean_features:>8,.0f}")

    print()
    print(f"Total export tasks: {int(frame['chunks'].sum()):,}")
    print(f"Queue cap (max tasks in flight): {registry.export.max_tasks_in_queue:,}")
    print(f"Earth Engine READY-state ceiling: {registry.export.max_ready_tasks:,}")
    print(f"Target features per task: {registry.export.target_features_per_task:,}")
    small = frame[
        (frame["mean_features"] < registry.export.target_features_per_task * 0.2)
        & (frame["chunks"] > 5)
    ]
    if not small.empty:
        print()
        print("Sources whose tasks fall well below the feature target "
              "(consider raising max_time_groups_per_task or merging datasets):")
        for row in small.itertuples():
            print(f"  {row.dataset}/{row.source}: {row.mean_features:,.0f} features/chunk "
                  f"({row.chunks:,} chunks)")
    if unverified:
        print()
        print(f"WARNING: grid origin(s) marked unverified in the registry: {sorted(unverified)}. "
              f"Verify the product's projection with `ee.Image(...).projection()` before "
              f"trusting sub-pixel placement (the 30 m buffer hedges this).")
    print("(plan writes nothing and performs no Earth Engine computation)")


# ---------------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------------
def cmd_dedup(registry: Registry, args: argparse.Namespace) -> None:
    """Write the unique-key tables and the pending chunk ledger rows."""
    for dataset, source in _pairs(registry, args):
        result = load_or_build_dedup(
            registry, dataset, source, raw_path=args.raw_path,
            rebuild=args.rebuild, write=True,
        )
        chunks = build_chunks(source, registry.export, dataset, result.keys)
        path = ledger_path(registry.export, source.name)
        known = load_ledger(path)
        fresh = [c for c in chunks if c.chunk_id not in known]
        record_rows(path, [row_for(c, status="pending") for c in fresh])
        logger.info(
            "%s/%s: %d fixes -> %d unique keys (ratio %.3f), %d chunk(s), %d new pending",
            dataset, source.name, result.n_fixes, result.n_unique,
            result.reduction_ratio, len(chunks), len(fresh),
        )


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------
def _in_flight(registry: Registry, source: str) -> int:
    ledger = load_ledger(ledger_path(registry.export, source))
    return sum(1 for row in ledger.values() if row.get("status") in ("submitted", "running"))


def _to_cloud_storage(ee: object, auth: dict, export: ExportConfig, source: Source,
                      chunk: Chunk, table: object, description: str) -> object:
    """Start one batch export into Cloud Storage."""
    return ee.batch.Export.table.toCloudStorage(  # type: ignore[attr-defined]
        collection=table,
        description=description,
        bucket=auth["target"],
        fileNamePrefix=chunk.object_path(export),
        fileFormat=export.file_format,
        selectors=task_selectors(source, export),
    )


def _to_drive(ee: object, auth: dict, export: ExportConfig, source: Source,
              chunk: Chunk, table: object, description: str) -> object:
    """Start one batch export into a Drive folder.

    `fileNamePrefix` is the *stem*: Earth Engine appends the format's extension
    itself, so passing `.csv` here would produce `name.csv.csv`.
    """
    return ee.batch.Export.table.toDrive(  # type: ignore[attr-defined]
        collection=table,
        description=description,
        folder=auth["target"],
        fileNamePrefix=chunk.drive_prefix(export),
        fileFormat=export.file_format,
        selectors=task_selectors(source, export),
    )


# Destination -> exporter, resolved by dict lookup so `export.destination` in
# sources.yaml selects behaviour without an if/elif chain.
EXPORTERS: dict[str, object] = {
    "cloud_storage": _to_cloud_storage,
    "drive": _to_drive,
}


def _start_task(ee: object, auth: dict, registry: Registry, source: Source, chunk: Chunk) -> str:
    """Build and start one export task, retrying transient Earth Engine errors."""
    export = registry.export
    handler = handler_for(source, export, ee)
    table = build_task_table(handler, export, chunk.keys, chunk.zone)
    description = f"{source.name}-{chunk.dataset}-{chunk.chunk_id}"
    start_export = EXPORTERS[export.destination]
    last_error = ""
    for attempt in range(1, export.retry_max_attempts + 1):
        try:
            task = start_export(ee, auth, export, source, chunk, table, description)
            task.start()
            return str(task.id)
        except Exception as exc:  # noqa: BLE001 - classified below, never swallowed
            last_error = str(exc)
            if not _is_transient(last_error) or attempt == export.retry_max_attempts:
                raise
            delay = export.retry_backoff_seconds * (2 ** (attempt - 1)) + random.uniform(
                0, export.retry_jitter_seconds
            )
            logger.warning("Transient submit error (attempt %d): %s — retrying in %.1fs",
                           attempt, last_error, delay)
            time.sleep(delay)
    raise RuntimeError(f"unreachable: {last_error}")


def cmd_submit(registry: Registry, args: argparse.Namespace) -> None:
    """Submit resumable chunks, respecting the queue cap."""
    export = registry.export
    auth = gee_auth(args.auth, destination=export.destination)
    ee = ee_module()
    _announce_auth(auth)
    _init_ee(ee, auth)
    max_tasks = args.max_tasks if args.max_tasks is not None else export.max_tasks_in_queue
    submitted_this_run = 0

    for dataset, source in _pairs(registry, args):
        result = load_or_build_dedup(registry, dataset, source, raw_path=args.raw_path)
        chunks = build_chunks(source, export, dataset, result.keys)
        path = ledger_path(export, source.name)
        todo = resumable_chunks(chunks, load_ledger(path), retry_failed=args.retry_failed)
        logger.info("%s/%s: %d chunk(s) to submit (%d complete)",
                    dataset, source.name, len(todo), len(chunks) - len(todo))

        for chunk in todo:
            if submitted_this_run >= max_tasks:
                logger.info("Reached --max-tasks=%d this run; stopping (resume-safe).", max_tasks)
                return
            _wait_for_slot(registry, ee, source, args.watch_cap)
            try:
                task_id = _start_task(ee, auth, registry, source, chunk)
            except Exception as exc:  # noqa: BLE001 - permanent failure is recorded, not raised
                message = str(exc)
                if _is_transient(message):
                    logger.error("Chunk %s still failing transiently after retries: %s",
                                 chunk.chunk_id, message)
                    raise
                logger.error("Chunk %s failed permanently: %s", chunk.chunk_id, message)
                record_rows(path, [row_for(chunk, status="failed", failure_reason=message[:200])])
                continue
            record_rows(path, [row_for(
                chunk, status="submitted", task_id=task_id,
                gcs_uri=chunk.location(export, auth["destination"], auth["target"]),
                submitted_at=_now(),
            )])
            submitted_this_run += 1
            logger.info("Submitted %s (task %s, %d features)",
                        chunk.chunk_id, task_id, chunk.n_features)


def _wait_for_slot(
    registry: Registry, ee: object, source: Source, poll_cap: float | None
) -> None:
    """Block until the source has fewer than the queue cap in flight."""
    cap = registry.export.max_tasks_in_queue
    delay = registry.export.poll_initial_seconds
    deadline = None if poll_cap is None else time.time() + poll_cap
    while _in_flight(registry, source.name) >= cap:
        if deadline is not None and time.time() > deadline:
            raise TimeoutError(
                f"Queue cap ({cap}) never freed for source {source.name!r} within "
                f"--watch-cap={poll_cap}s. The sweep is safe to resume."
            )
        logger.info("Queue full for %s (%d in flight); polling and backing off %.0fs.",
                    source.name, cap, delay)
        _poll_source(registry, ee, source.name)
        time.sleep(delay)
        delay = min(delay * registry.export.poll_backoff, registry.export.poll_max_seconds)


# ---------------------------------------------------------------------------
# poll
# ---------------------------------------------------------------------------
def _task_states(ee: object, task_ids: list[str]) -> dict[str, dict]:
    """Current Earth Engine state for a batch of task ids."""
    if not task_ids:
        return {}
    raw = ee.data.getTaskStatus(task_ids)  # type: ignore[attr-defined]
    if isinstance(raw, dict):  # single-id form in some API versions
        raw = [raw]
    out: dict[str, dict] = {}
    for entry in raw:
        task_id = entry.get("id") or entry.get("name", "").split("/")[-1]
        if task_id:
            out[task_id] = entry
    return out


def _poll_source(registry: Registry, ee: object, source: str) -> int:
    """Refresh the ledger for one source; returns the number still in flight."""
    path = ledger_path(registry.export, source)
    ledger = load_ledger(path)
    active = {cid: row for cid, row in ledger.items()
              if row.get("status") in ("submitted", "running") and row.get("task_id")}
    if not active:
        return 0
    states = _task_states(ee, [row["task_id"] for row in active.values()])
    updates: list[dict] = []
    still_active = 0
    for chunk_id, row in active.items():
        state = states.get(row["task_id"], {}).get("state", "")
        status = _STATE_TO_STATUS.get(state, "")
        if not status:
            still_active += 1
            continue
        if status == row.get("status"):
            still_active += 1
            continue
        extra: dict = {"task_id": row["task_id"]}
        if status in ("complete", "failed"):
            extra["finished_at"] = _now()
        if status == "failed":
            extra["failure_reason"] = (states.get(row["task_id"], {}).get("error_message")
                                       or state)[:200]
            logger.error("Task for chunk %s failed: %s", chunk_id, extra["failure_reason"])
        else:
            logger.info("Chunk %s -> %s", chunk_id, status)
            still_active += 1
        updates.append(_ledger_row_from(row, status, extra))
    record_rows(path, updates)
    return still_active


def _ledger_row_from(row: dict, status: str, extra: dict) -> dict:
    """A new ledger row carrying forward the previous row's fields."""
    merged = dict(row)
    merged.update(extra)
    merged["status"] = status
    return merged


def cmd_poll(registry: Registry, args: argparse.Namespace) -> None:
    """Refresh task states, optionally watching until the queue drains."""
    auth = gee_auth(args.auth)
    ee = ee_module()
    _announce_auth(auth)
    _init_ee(ee, auth)
    sources = registry.select_sources(args.source)
    delay = registry.export.poll_initial_seconds
    while True:
        active = sum(_poll_source(registry, ee, s) for s in sources)
        logger.info("In flight across %d source(s): %d", len(sources), active)
        if not args.watch or active == 0:
            break
        time.sleep(delay)
        delay = min(delay * registry.export.poll_backoff, registry.export.poll_max_seconds)


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------
def _fetcher_for(auth: dict, export: ExportConfig, scratch: Path):
    """The chunk-output reader for the configured destination."""
    builders = {
        "cloud_storage": lambda: gcs_fetcher(
            export, gcs_client(auth), auth["target"], scratch),
        "drive": lambda: drive_fetcher(drive_client(auth), auth["target"], scratch),
    }
    return builders[export.destination]()


def cmd_join(registry: Registry, args: argparse.Namespace) -> None:
    """Build one Parquet per (dataset, source), then the wide per-dataset CSV."""
    export = registry.export
    local_dir = Path(args.from_local) if args.from_local else None
    if local_dir:
        # Offline path: no credentials, no Earth Engine, no cloud storage.
        fetch = local_fetcher(local_dir)
    else:
        auth = gee_auth(args.auth, destination=export.destination)
        _announce_auth(auth)
        fetch = _fetcher_for(auth, export, Path(export.results_root) / "_download")
    out_dir = _wide_csv_dir()
    joined: list[str] = []
    for dataset, source in _pairs(registry, args):
        result = load_or_build_dedup(registry, dataset, source, raw_path=args.raw_path)
        chunks = build_chunks(source, export, dataset, result.keys)
        done = completed_chunks(export, source.name, {c.chunk_id: c for c in chunks})
        fixes = read_dataset_fixes(registry, dataset, args.raw_path)
        out, counts = join_source(export, source, dataset, fixes, result.fixmap, done, fetch)
        path = ledger_path(export, source.name)
        ledger = load_ledger(path)
        record_rows(path, [
            _ledger_row_from(ledger[chunk_id], "complete", {"n_rows_out": n})
            for chunk_id, n in counts.items() if chunk_id in ledger
        ])
        logger.info("Joined %s/%s: %d rows x %d columns", dataset, source.name,
                    len(out), len(out.columns))
        if dataset not in joined:
            joined.append(dataset)

    # Refresh the wide CSV once per dataset, from every source joined so far.
    if out_dir is not None:
        for dataset in joined:
            combine_dataset(export, dataset, out_dir)


def _wide_csv_dir(*, required: bool = False) -> Path | None:
    """The wide-CSV destination, or ``None`` when GEE_DATASET_PATH is unset.

    The Parquet archive is the spec's deliverable, so an unset GEE_DATASET_PATH
    skips the CSV with a warning rather than failing `join`. `combine` exists
    solely to write that CSV, so for it the variable is required.
    """
    try:
        return gee_dataset_path()
    except RuntimeError as exc:
        if required:
            raise
        logger.warning("%s Skipping the wide CSV; per-source Parquet still written.", exc)
        return None


def cmd_combine(registry: Registry, args: argparse.Namespace) -> None:
    """Regenerate the wide per-dataset CSV(s) from the Parquet archive (no GEE)."""
    out_dir = _wide_csv_dir(required=True)
    written = 0
    for dataset in registry.select_datasets(args.dataset):
        if combine_dataset(registry.export, dataset, out_dir) is not None:
            written += 1
    if written == 0:
        raise SystemExit(
            f"No per-source Parquet found under {registry.export.results_root} — "
            f"run `join` first."
        )


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def cmd_status(registry: Registry, args: argparse.Namespace) -> None:
    """Ledger and output inventory across every source."""
    print(f"{'source':<19} {'pending':>8} {'submitted':>10} {'running':>8} "
          f"{'complete':>9} {'failed':>7} {'skipped':>8}")
    print("-" * 76)
    for name in registry.select_sources(args.source):
        counts = ledger_status_counts(load_ledger(ledger_path(registry.export, name)))
        print(f"{name:<19} {counts['pending']:>8,} {counts['submitted']:>10,} "
              f"{counts['running']:>8,} {counts['complete']:>9,} {counts['failed']:>7,} "
              f"{counts['skipped']:>8,}")
    print()
    joined = sorted(Path(registry.export.results_root).glob("*/*.parquet"))
    print(f"Joined covariate tables on disk: {len(joined)}")
    for path in joined:
        print(f"  {path.relative_to(registry.export.results_root)}")
    try:
        out_dir = gee_dataset_path()
    except RuntimeError:
        out_dir = None
        print("\nGEE_DATASET_PATH: not set (wide CSV disabled)")
    if out_dir is not None:
        wide = sorted(out_dir.glob("*.csv"))
        print(f"\nWide covariate CSVs in {out_dir}: {len(wide)}")
        for path in wide:
            print(f"  {path.name}  ({path.stat().st_size / 1024**2:.1f} MB)")
    _print_auth_mode(registry, args)


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------
def cmd_coverage(registry: Registry, args: argparse.Namespace) -> None:
    """Scene counts per month over each dataset's footprint, before submitting.

    Answers "will this source actually have data for my fixes?" — e.g. whether
    Sentinel-2 surface reflectance exists over the study area in 2018 — with one
    small ``getInfo()`` per (dataset, source): a list of monthly image counts.
    Static and single-image sources are skipped (they have no time axis).
    """
    auth = gee_auth(args.auth, destination=registry.export.destination)
    ee = ee_module()
    _announce_auth(auth)
    _init_ee(ee, auth)
    for dataset, source in _pairs(registry, args):
        if source.kind != "image_collection" or not source.time_varying:
            logger.info("%s: static or single-image source; no coverage to check.", source.name)
            continue
        fixes = read_dataset_fixes(registry, dataset, args.raw_path)
        ts = pd.to_datetime(fixes["timestamp"], utc=True).dt.tz_convert(None)
        months = pd.period_range(ts.min().to_period("M"), ts.max().to_period("M"), freq="M")
        bbox = [float(fixes["lon"].min()), float(fixes["lat"].min()),
                float(fixes["lon"].max()), float(fixes["lat"].max())]
        counts = monthly_scene_counts(ee, source, bbox, [str(m) for m in months]).getInfo()
        per_month = ts.dt.to_period("M").value_counts()
        print(f"\n# {dataset} / {source.name}  bbox={[round(v, 3) for v in bbox]}")
        print(f"{'month':<9} {'scenes':>7} {'fixes':>8}")
        uncovered = 0
        for month, n in zip(months, counts, strict=True):
            n_fix = int(per_month.get(month, 0))
            flag = "  <-- no scenes" if n == 0 and n_fix else ""
            if n == 0:
                uncovered += n_fix
            print(f"{str(month):<9} {int(n):>7,} {n_fix:>8,}{flag}")
        print(f"fixes in months with zero scenes: {uncovered:,} of {len(fixes):,} "
              f"({uncovered / max(len(fixes), 1):.1%})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GEE covariate export (plan | dedup | submit | poll | join | combine | status | coverage)."
    )
    # Shared options attach to every subcommand, so the documented form
    # `gee_export.py plan --dataset all --source all` works as written.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--registry", default=str(DEFAULT_REGISTRY),
                        help="Path to configs/covariates/sources.yaml.")
    common.add_argument("--dataset", default="all",
                        help="'all' or a comma-separated list of dataset names.")
    common.add_argument("--source", default="all",
                        help="'all', 'tier1'/'tier2'/'tier3', or comma-separated source names.")
    common.add_argument("--raw-path", default=None,
                        help="Override the raw dataset dir (default: RAW_DATASET_PATH).")
    common.add_argument(
        "--auth", default=None, choices=GEE_AUTH_MODES,
        help="Earth Engine auth: auto (default), service_account, or interactive. "
             "Overrides GEE_AUTH. Pinning service_account makes the interactive "
             "fallback an error.",
    )

    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("plan", parents=[common],
                   help="Print the chunk plan; writes nothing.")

    p_dedup = sub.add_parser("dedup", parents=[common],
                             help="Write unique-key tables + pending ledger rows.")
    p_dedup.add_argument("--rebuild", action="store_true",
                         help="Recompute key tables even if present on disk.")

    p_submit = sub.add_parser("submit", parents=[common],
                              help="Submit resumable chunks (quota-aware).")
    p_submit.add_argument("--max-tasks", type=int, default=None,
                          help="Stop after N submissions this run.")
    p_submit.add_argument("--retry-failed", action="store_true",
                          help="Re-submit chunks recorded failed.")
    p_submit.add_argument("--watch-cap", type=float, default=None,
                          help="Seconds to wait for a queue slot before erroring.")

    p_poll = sub.add_parser("poll", parents=[common], help="Refresh task states.")
    p_poll.add_argument("--watch", action="store_true",
                        help="Keep polling until nothing is in flight.")

    p_join = sub.add_parser("join", parents=[common],
                            help="Build the per-fix Parquet tables (+ wide CSV).")
    p_join.add_argument("--from-local", default=None,
                        help="Read chunk output from a local dir instead of GCS (dry runs).")

    sub.add_parser("combine", parents=[common],
                   help="Regenerate the wide per-dataset CSV from the Parquet archive.")

    sub.add_parser("status", parents=[common], help="Ledger + output inventory.")

    sub.add_parser("coverage", parents=[common],
                   help="Monthly scene counts over each dataset's footprint (needs auth).")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    registry = load_registry(args.registry)
    handlers = {
        "plan": cmd_plan,
        "dedup": cmd_dedup,
        "submit": cmd_submit,
        "poll": cmd_poll,
        "join": cmd_join,
        "combine": cmd_combine,
        "status": cmd_status,
        "coverage": cmd_coverage,
    }
    try:
        handlers[args.command](registry, args)
    except KeyboardInterrupt:
        print("\nInterrupted — the ledger is append-only, so re-running resumes.",
              file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
