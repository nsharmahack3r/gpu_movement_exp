"""Deduplication: pixel-grid snapping and unique-key reduction.

This runs *before* anything touches Earth Engine (``covariate_plan.md`` §5.2).
Never sample 2 M fixes: every fix is snapped to the source's native pixel grid
(reprojected to the product's CRS and floored to its lattice — not a naive
lat/lon round), given a cadence-appropriate time key, and reduced to the set of
unique keys. The ``fix_id -> key`` mapping is what later re-expands the sampled
values back to per-fix rows.

``fix_id = sha1(study_id | individual_id | timestamp_iso)`` is stable across
runs, which is what lets ``join`` assert exactly one output row per fix.

No Earth Engine import at module scope: this module is offline by construction.
"""

from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer

from movement.covariates.sources import Source, nominal_time

logger = logging.getLogger(__name__)

# Day-count anchor for composite window indices.
_EPOCH = pd.Timestamp("1970-01-01T00:00:00Z")

WGS84 = "EPSG:4326"

# Column slots this tool needs from a raw movement CSV.
_COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "datetime", "fix_time"],
    "individual_id": ["individual_id", "animal_id", "id", "tag_id"],
    "study_id": ["study_id", "study", "project_id"],
    "lat": ["lat", "latitude", "y"],
    "lon": ["lon", "long", "longitude", "lng", "x"],
}


# ---------------------------------------------------------------------------
# fix_id — the stable join key
# ---------------------------------------------------------------------------
def timestamp_iso(timestamp: Any) -> str:
    """Canonical UTC ISO-8601 (millisecond precision) for a fix timestamp.

    The raw CSVs use ``YYYY-MM-DD HH:MM:SS.mmm`` and are treated as UTC. The
    canonical form is ``YYYY-MM-DDTHH:MM:SS.mmmZ`` so the hash does not depend on
    the input's separator or on trailing zeros.
    """
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def fix_id(study_id: Any, individual_id: Any, timestamp: Any) -> str:
    """``sha1(study_id | individual_id | timestamp_iso)`` — stable per fix."""
    payload = f"{study_id}|{individual_id}|{timestamp_iso(timestamp)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Raw CSV reading (handles the duplicated individual_id column)
# ---------------------------------------------------------------------------
def _true_header(path: Path) -> list[str]:
    """Column names exactly as written, duplicates included."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f))


def _resolve(df: pd.DataFrame, slot: str, path: Path) -> str:
    for candidate in _COLUMN_ALIASES[slot]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"{path.name}: could not find a column for {slot!r} "
        f"(looked for {_COLUMN_ALIASES[slot]}); found {list(df.columns)}"
    )


def read_fixes(
    path: Path,
    *,
    duplicate_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Read one raw movement CSV into a normalised fix table.

    Handles the duplicated ``individual_id`` column shipped by
    ``black_bear_reshaped.csv`` (and any column named in ``duplicate_columns``):
    the first occurrence is kept, and the tool **fails loudly** if a duplicate
    disagrees with it rather than silently dropping a conflicting value.

    Returns columns ``fix_id, timestamp, individual_id, study_id, lon, lat``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw movement CSV not found: {path}")

    header = _true_header(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")

    for name in duplicate_columns:
        occurrences = [i for i, col in enumerate(header) if col == name]
        if len(occurrences) < 2:
            continue
        # pandas mangles the 2nd+ occurrence: name, name.1, name.2 ...
        positions = [name] + [f"{name}.{k}" for k in range(1, len(occurrences))]
        missing = [p for p in positions if p not in df.columns]
        if missing:
            raise ValueError(f"{path.name}: expected duplicated column(s) {missing} not found")
        values = df[positions]
        agree = (values.nunique(axis=1) == 1).all()
        if not agree:
            bad = values[values.nunique(axis=1) > 1].head(3).to_dict("records")
            raise ValueError(
                f"{path.name}: duplicated {name!r} column(s) disagree — refusing to "
                f"guess which is authoritative. First offenders: {bad}"
            )
        df = df.drop(columns=positions[1:])
        logger.warning(
            "%s: dropped %d duplicated %r column(s) (all values agreed).",
            path.name, len(positions) - 1, name,
        )

    timestamp_col = _resolve(df, "timestamp", path)
    individual_col = _resolve(df, "individual_id", path)
    study_col = _resolve(df, "study_id", path)
    lon_col = _resolve(df, "lon", path)
    lat_col = _resolve(df, "lat", path)

    timestamp = pd.to_datetime(df[timestamp_col], format="ISO8601", errors="raise")
    timestamp = timestamp.dt.tz_localize("UTC") if timestamp.dt.tz is None else timestamp.dt.tz_convert("UTC")
    out = pd.DataFrame(
        {
            "timestamp": timestamp,
            "individual_id": df[individual_col].astype(str),
            "study_id": df[study_col].astype(str),
            "lon": pd.to_numeric(df[lon_col], errors="raise"),
            "lat": pd.to_numeric(df[lat_col], errors="raise"),
        }
    )
    out["fix_id"] = [
        fix_id(s, i, t)
        for s, i, t in zip(out["study_id"], out["individual_id"], out["timestamp"], strict=True)
    ]
    if out["fix_id"].duplicated().any():
        dupes = int(out["fix_id"].duplicated().sum())
        raise ValueError(
            f"{path.name}: {dupes} duplicate fix_id(s) — two rows share "
            f"(study_id, individual_id, timestamp). Deduplicate at the source; "
            f"the join asserts one row per fix."
        )
    return out


# ---------------------------------------------------------------------------
# Pixel-grid snapping
# ---------------------------------------------------------------------------
class GridSnapper:
    """Snaps lon/lat fixes to a source's native pixel lattice.

    ``fixed`` grids reproject to one product CRS and floor to a single lattice
    anchored at ``grid.origin``. ``utm`` grids resolve the zone from longitude
    (the native tiling of the Sentinel products) and floor to a lattice anchored
    at the zone's false origin. Both return a pixel *centroid* in lon/lat, which
    is the geometry Earth Engine samples — so every fix sharing a pixel gets
    exactly the same value.
    """

    def __init__(self, source: Source) -> None:
        self.source = source
        self.grid = source.grid
        self._transformers: dict[str, tuple[Transformer, Transformer]] = {}

    # -- CRS plumbing -------------------------------------------------------
    def _transformer(self, crs: str) -> tuple[Transformer, Transformer]:
        if crs not in self._transformers:
            self._transformers[crs] = (
                Transformer.from_crs(WGS84, crs, always_xy=True),
                Transformer.from_crs(crs, WGS84, always_xy=True),
            )
        return self._transformers[crs]

    def _crs_and_anchor_for(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-point CRS string and lattice anchor (x, y)."""
        grid = self.grid
        if grid.mode == "fixed":
            crs = np.full(len(lon), str(grid.crs), dtype=object)
            return crs, np.full(len(lon), grid.origin_x), np.full(len(lon), grid.origin_y)
        # UTM: zone from longitude, hemisphere from latitude; anchor at the
        # zone's false origin (500 000 E and either 0 or 10 000 000 N).
        zone = np.floor((lon + 180.0) / 6.0).astype(int) + 1
        zone = np.clip(zone, 1, 60)
        north = lat >= 0
        epsg = np.where(north, 32600 + zone, 32700 + zone)
        crs = np.array([f"EPSG:{int(e)}" for e in epsg], dtype=object)
        anchor_y = np.where(north, 0.0, 10_000_000.0)
        return crs, np.full(len(lon), 500_000.0), anchor_y

    # -- snapping -----------------------------------------------------------
    def snap(self, lon: np.ndarray, lat: np.ndarray) -> pd.DataFrame:
        """Return ``zone,px_row,px_col,px_lon,px_lat`` (pixel centroid) per fix."""
        grid = self.grid
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        crs_per_point, anchor_x, anchor_y = self._crs_and_anchor_for(lon, lat)

        n = len(lon)
        zone = np.zeros(n, dtype=np.int64)
        row = np.empty(n, dtype=np.int64)
        col = np.empty(n, dtype=np.int64)
        px_lon = np.empty(n, dtype=float)
        px_lat = np.empty(n, dtype=float)

        for crs in pd.unique(crs_per_point):
            mask = crs_per_point == crs
            fwd, inv = self._transformer(str(crs))
            xs, ys = fwd.transform(lon[mask], lat[mask])
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)

            col[mask] = np.floor((xs - anchor_x[mask]) / grid.res_x).astype(np.int64)
            row[mask] = np.floor((anchor_y[mask] - ys) / grid.res_y).astype(np.int64)

            # Pixel centroid, back in lon/lat — the geometry Earth Engine samples,
            # so every fix sharing a pixel receives the same value.
            cx = anchor_x[mask] + (col[mask] + 0.5) * grid.res_x
            cy = anchor_y[mask] - (row[mask] + 0.5) * grid.res_y
            clon, clat = inv.transform(cx, cy)
            px_lon[mask] = clon
            px_lat[mask] = clat
            if grid.mode == "utm":
                zone[mask] = np.int64(str(crs).split(":")[-1])

        return pd.DataFrame(
            {
                "zone": zone,
                "px_row": row,
                "px_col": col,
                "px_lon": px_lon,
                "px_lat": px_lat,
            }
        )


def _time_key_array(source: Source, timestamps: pd.Series) -> np.ndarray:
    """Vectorised :func:`time_key` over a whole fix column."""
    cadence = source.cadence
    if cadence == "static":
        return np.zeros(len(timestamps), dtype=np.int64)
    d = timestamps.dt
    if cadence == "annual":
        return np.minimum(d.year.to_numpy(), int(source.last_year)).astype(np.int64)
    if cadence == "monthly":
        return (d.year.to_numpy() * 100 + d.month.to_numpy()).astype(np.int64)
    if cadence == "daily":
        return (d.year.to_numpy() * 10000 + d.month.to_numpy() * 100 + d.day.to_numpy()).astype(np.int64)
    if cadence == "subdaily":
        return (
            d.year.to_numpy() * 1_000_000
            + d.month.to_numpy() * 10_000
            + d.day.to_numpy() * 100
            + d.hour.to_numpy()
        ).astype(np.int64)
    if cadence == "composite":
        size = max(1, 2 * int(source.composite_days))
        days = (timestamps - _EPOCH).dt.days.to_numpy()
        return (days // size).astype(np.int64)
    raise ValueError(f"Unhandled cadence {cadence!r}")


def _bucket_array(source: Source, timestamps: pd.Series) -> np.ndarray:
    """Vectorised :func:`time_bucket` over a whole fix column."""
    bucket = source.chunk_bucket
    if bucket == "static":
        return np.full(len(timestamps), "static", dtype=object)
    d = timestamps.dt
    if bucket == "year":
        return np.minimum(d.year.to_numpy(), int(source.last_year)).astype(str)
    if bucket == "month":
        return d.strftime("%Y-%m").to_numpy()
    if bucket == "day":
        return d.strftime("%Y-%m-%d").to_numpy()
    if bucket == "hour":
        return d.strftime("%Y-%m-%dT%H").to_numpy()
    if bucket == "composite":
        return np.char.add("c", _time_key_array(source, timestamps).astype(str))
    raise ValueError(f"Unhandled chunk_bucket {bucket!r}")


def snap_fixes(fixes: pd.DataFrame, source: Source) -> pd.DataFrame:
    """Snap a fix table, adding ``zone, px_row, px_col, px_lon, px_lat, t0, bucket``."""
    snapper = GridSnapper(source)
    snapped = snapper.snap(fixes["lon"].to_numpy(), fixes["lat"].to_numpy())
    snapped["fix_id"] = fixes["fix_id"].to_numpy()
    snapped["timestamp"] = fixes["timestamp"].to_numpy()
    snapped["t0"] = _time_key_array(source, fixes["timestamp"])
    snapped["bucket"] = _bucket_array(source, fixes["timestamp"])
    return snapped


# ---------------------------------------------------------------------------
# Unique-key reduction
# ---------------------------------------------------------------------------
@dataclass
class DedupResult:
    """Unique keys plus the fix -> key mapping for one (dataset, source)."""

    dataset: str
    source: str
    keys: pd.DataFrame
    fixmap: pd.DataFrame
    n_fixes: int

    @property
    def n_unique(self) -> int:
        return len(self.keys)

    @property
    def n_pixels(self) -> int:
        """Distinct pixels the fixes occupy — the diagnostic of correct snapping."""
        return int(
            len(self.keys[["zone", "px_row", "px_col"]].drop_duplicates())
        )

    @property
    def reduction_ratio(self) -> float:
        """``unique / fixes`` — lower is better."""
        return self.n_unique / self.n_fixes if self.n_fixes else 0.0

    @property
    def pixel_reduction_ratio(self) -> float:
        """``distinct_pixels / fixes``. Near 1.0 means the snapping is wrong."""
        return self.n_pixels / self.n_fixes if self.n_fixes else 0.0


def dedup_dataset(fixes: pd.DataFrame, source: Source, dataset: str) -> DedupResult:
    """Reduce a fix table to unique keys + a ``fix_id -> key_index`` mapping."""
    snapped = snap_fixes(fixes, source)
    key_cols = ["zone", "px_row", "px_col", "t0"]

    # Key identity: the pixel centroid is the sampling geometry, so it must be
    # derived from the key, not from whichever fix happened to be first.
    grouped = snapped.groupby(key_cols, sort=False, as_index=False).agg(
        px_lon=("px_lon", "first"),
        px_lat=("px_lat", "first"),
        first_ts=("timestamp", "min"),
        bucket=("bucket", "first"),
        n_fixes=("fix_id", "size"),
    )
    grouped = grouped.sort_values(key_cols).reset_index(drop=True)
    grouped["key_index"] = np.arange(len(grouped), dtype=np.int64)
    if source.time_varying:
        grouped["t_ref"] = [nominal_time(source, ts) for ts in grouped["first_ts"]]
    else:
        grouped["t_ref"] = grouped["first_ts"]

    fixmap = snapped.merge(grouped[key_cols + ["key_index"]], on=key_cols, how="inner")
    fixmap = fixmap[["fix_id", "key_index"]].sort_values("fix_id").reset_index(drop=True)
    if len(fixmap) != len(fixes):
        raise RuntimeError(
            f"{dataset}/{source.name}: fix->key mapping has {len(fixmap)} rows for "
            f"{len(fixes)} fixes — dedup lost or duplicated fixes."
        )
    if fixmap["key_index"].isna().any():
        raise RuntimeError(f"{dataset}/{source.name}: some fixes mapped to no key.")

    keys = grouped[["key_index", "zone", "px_row", "px_col", "t0",
                    "px_lon", "px_lat", "bucket", "n_fixes", "t_ref"]]
    return DedupResult(dataset=dataset, source=source.name, keys=keys,
                       fixmap=fixmap, n_fixes=len(fixes))


def check_reduction(source: Source, export: Any, result: DedupResult) -> None:
    """Fail loudly when a coarse source refuses to collapse *spatially*.

    The test is on **distinct pixels**, not on total unique keys. A coarse product
    sampled at tens of thousands of fixes must occupy far fewer pixels than it has
    fixes; if it does not, the grid snapping is wrong and the tool would export
    one feature per fix (``covariate_plan.md`` §5.2 / §9).

    The unique-key count is *not* the invariant, because a source whose cadence
    matches the fix cadence (ERA5-Land hourly against hourly fixes) legitimately
    keeps a ``(pixel, hour)`` key per fix even when the snapping is perfect. That
    case is reported as a warning instead of a failure.
    """
    if source.scale_m > export.reduction_required_above_scale_m:
        if result.pixel_reduction_ratio > export.min_pixel_reduction_ratio:
            raise RuntimeError(
                f"{result.dataset}/{source.name}: {result.n_pixels:,} distinct pixels "
                f"from {result.n_fixes:,} fixes (ratio "
                f"{result.pixel_reduction_ratio:.3f}) — expected <= "
                f"{export.min_pixel_reduction_ratio:.2f} for a {source.scale_m:g} m "
                f"source. The pixel-grid snapping is wrong (check grid.crs/res/origin "
                f"in configs/covariates/sources.yaml); refusing to export one feature "
                f"per fix."
            )
    if result.reduction_ratio > export.warn_unique_ratio_above:
        logger.warning(
            "%s/%s: %d unique keys from %d fixes (ratio %.3f) — the time key is doing "
            "little work. Expected when the source's cadence matches the fix cadence; "
            "check it otherwise.",
            result.dataset, source.name, result.n_unique, result.n_fixes,
            result.reduction_ratio,
        )


def dedup_summary(result: DedupResult) -> dict[str, Any]:
    """One ``plan``/``status`` table row for a dedup result."""
    return {
        "dataset": result.dataset,
        "source": result.source,
        "fixes": result.n_fixes,
        "unique_keys": result.n_unique,
        "reduction_ratio": result.reduction_ratio,
        "pixels": result.n_pixels,
        "pixel_ratio": result.pixel_reduction_ratio,
        "buckets": int(result.keys["bucket"].nunique()),
    }
