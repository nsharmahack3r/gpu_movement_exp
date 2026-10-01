"""Covariate export tool tests — CPU-only, no network, Earth Engine mocked.

Covers ``prompts/gee_export.md`` §11. Nothing here touches ``ee`` for real: the
graph builders are driven by a recording stub, and every other test exercises the
offline half of the tool (snapping, dedup, chunking, the ledger, QA arithmetic,
temporal alignment, join-back).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from movement.covariates import sources as S
from movement.covariates.dedup import (
    GridSnapper,
    check_reduction,
    dedup_dataset,
    fix_id,
    read_fixes,
    timestamp_iso,
)
from movement.covariates.drive import DriveClient, DriveError
from movement.covariates.join import (
    assert_one_row_per_fix,
    build_fix_table,
    combine_dataset,
    combine_tables,
    drive_fetcher,
    local_fetcher,
    output_path,
    prefixed_column,
    wide_csv_path,
    write_wide_csv,
)
from movement.covariates.ledger import (
    Chunk,
    build_chunks,
    key_digest,
    ledger_path,
    ledger_status_counts,
    load_ledger,
    record_rows,
    resumable_chunks,
    row_for,
)
from movement.covariates.sources import (
    Grid,
    Registry,
    SamplingRef,
    Source,
    age_hours,
    build_task_table,
    ee_module,
    extract_bits,
    handler_for,
    load_registry,
    nominal_time,
    point_collection,
    sampling_crs,
    task_selectors,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def registry() -> Registry:
    return load_registry()


def grid_source(registry: Registry, grid: Grid, name: str = "test_src", **kw) -> Source:
    """A real registry source with its grid (and optionally more) replaced."""
    return dataclasses.replace(registry.source("csp_ghm"), name=name, grid=grid, **kw)


def degree_grid(res: float) -> Grid:
    return Grid(mode="fixed", crs="EPSG:4326", res_x=res, res_y=res,
                origin_x=-180.0, origin_y=90.0, verified=True)


def make_fixes(rows: list[tuple[float, float, str]]) -> pd.DataFrame:
    """Build a normalised fix table from ``(lon, lat, iso_timestamp)`` rows."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([r[2] for r in rows], utc=True),
            "individual_id": "ind_1",
            "study_id": "study_1",
            "lon": [r[0] for r in rows],
            "lat": [r[1] for r in rows],
        }
    )
    frame["fix_id"] = [
        fix_id(s, i, t)
        for s, i, t in zip(frame["study_id"], frame["individual_id"], frame["timestamp"], strict=True)
    ]
    return frame


def make_keys(n: int, *, bucket: str = "static", t0: int = 0, zone: int = 0) -> pd.DataFrame:
    """A minimal unique-key frame matching ``dedup_dataset``'s output shape."""
    return pd.DataFrame(
        {
            "key_index": range(n),
            "zone": zone,
            "px_row": [i // 10 for i in range(n)],
            "px_col": [i % 10 for i in range(n)],
            "t0": t0,
            "px_lon": [-100.0 + 0.001 * i for i in range(n)],
            "px_lat": [40.0 + 0.001 * i for i in range(n)],
            "bucket": bucket,
            "n_fixes": 1,
            "t_ref": pd.Timestamp("2020-01-01T12:00:00Z"),
        }
    )


class _Node:
    """Recording stub for the ``ee`` module: chainable, callable, records calls."""

    def __init__(self, calls: list, name: str = "ee") -> None:
        object.__setattr__(self, "_calls", calls)
        object.__setattr__(self, "_name", name)

    def __getattr__(self, item: str) -> _Node:
        return _Node(self._calls, f"{self._name}.{item}")

    def __call__(self, *args, **kwargs) -> _Node:
        name = f"{self._name}()"
        self._calls.append((name, args, kwargs))
        return _Node(self._calls, name)


class EEStub(_Node):
    """Stand-in for the ``ee`` module."""

    def __init__(self) -> None:
        calls: list = []
        super().__init__(calls)
        object.__setattr__(self, "calls", calls)

    def calls_named(self, suffix: str) -> list[tuple]:
        return [c for c in self.calls if c[0].endswith(suffix)]


# ---------------------------------------------------------------------------
# Pixel-grid snapping
# ---------------------------------------------------------------------------
def test_snap_floors_to_the_transform() -> None:
    """A 1-degree lattice with origin (-180, 90): floor, not a lat/lon round."""
    source = grid_source(load_registry(), degree_grid(1.0))
    snapped = GridSnapper(source).snap(
        lon=pd.Series([0.5, 0.999, 1.0, -179.2]).to_numpy(),
        lat=pd.Series([89.5, 89.001, 89.5, 89.9]).to_numpy(),
    )
    # Row 0 is the northernmost 1-degree band; col 180 starts at lon 0.
    assert snapped["px_row"].tolist() == [0, 0, 0, 0]
    assert snapped["px_col"].tolist() == [180, 180, 181, 0]
    # The sampling geometry is the pixel centroid, not the fix.
    assert snapped["px_lon"].iloc[0] == pytest.approx(0.5)
    assert snapped["px_lat"].iloc[0] == pytest.approx(89.5)


def test_snap_is_idempotent() -> None:
    """Snapping a pixel centroid returns the same pixel."""
    source = grid_source(load_registry(), degree_grid(0.001))
    snapper = GridSnapper(source)
    first = snapper.snap(pd.Series([-100.1234]).to_numpy(), pd.Series([40.5678]).to_numpy())
    second = snapper.snap(first["px_lon"].to_numpy(), first["px_lat"].to_numpy())
    assert second["px_row"].iloc[0] == first["px_row"].iloc[0]
    assert second["px_col"].iloc[0] == first["px_col"].iloc[0]
    assert second["px_lon"].iloc[0] == pytest.approx(first["px_lon"].iloc[0])
    assert second["px_lat"].iloc[0] == pytest.approx(first["px_lat"].iloc[0])


def test_utm_mode_resolves_a_zone_per_point(registry: Registry) -> None:
    """Sentinel products are tiled in UTM: zone from longitude, hemisphere from lat."""
    source = registry.source("sentinel2")
    snapped = GridSnapper(source).snap(
        pd.Series([-96.0, -96.0, 12.0]).to_numpy(),
        pd.Series([40.0, -33.0, 41.0]).to_numpy(),
    )
    assert snapped["zone"].tolist() == [32615, 32715, 32633]
    assert sampling_crs(source, 32615) == "EPSG:32615"


def test_sampling_crs_is_always_explicit(registry: Registry) -> None:
    assert sampling_crs(registry.source("era5_land"), 0) == "EPSG:4326"
    assert sampling_crs(registry.source("annual_nlcd"), 0) == "EPSG:5070"


# ---------------------------------------------------------------------------
# fix_id and raw CSV reading
# ---------------------------------------------------------------------------
def test_timestamp_iso_is_canonical() -> None:
    assert timestamp_iso("2019-03-07 18:00:00.000") == "2019-03-07T18:00:00.000Z"
    # A naive and an explicit-UTC spelling of the same instant must agree.
    assert timestamp_iso("2019-03-07T18:00:00") == timestamp_iso("2019-03-07 18:00:00+00:00")


def test_fix_id_is_stable_and_field_sensitive() -> None:
    value = fix_id("s", "i", "2019-03-07 18:00:00.000")
    assert value == fix_id("s", "i", "2019-03-07T18:00:00Z")  # stable across runs/spellings
    assert value != fix_id("s", "i", "2019-03-07 19:00:00.000")
    assert value != fix_id("s", "other", "2019-03-07 18:00:00.000")
    assert value != fix_id("other", "i", "2019-03-07 18:00:00.000")


def test_read_fixes_handles_duplicated_individual_id(tmp_path: Path) -> None:
    """black_bear_reshaped.csv ships a duplicated individual_id column."""
    path = tmp_path / "dupe.csv"
    path.write_text(
        "timestamp,lon,lat,species,individual_id,individual_id,study_id\n"
        "2019-03-07 18:00:00.000,-72.858,42.019,Ursus americanus,Percent_3,Percent_3,CT bear\n"
        "2019-03-07 19:00:00.000,-72.858,42.020,Ursus americanus,Percent_3,Percent_3,CT bear\n",
        encoding="utf-8",
    )
    frame = read_fixes(path, duplicate_columns=("individual_id",))
    assert len(frame) == 2
    assert set(frame["individual_id"]) == {"Percent_3"}
    assert set(frame["study_id"]) == {"CT bear"}
    assert list(frame.columns) == ["timestamp", "individual_id", "study_id", "lon", "lat", "fix_id"]


def test_read_fixes_fails_loudly_when_duplicate_columns_disagree(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "timestamp,lon,lat,individual_id,individual_id,study_id\n"
        "2019-03-07 18:00:00.000,-72.85,42.01,bear_A,bear_B,CT\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disagree"):
        read_fixes(path, duplicate_columns=("individual_id",))


def test_read_fixes_rejects_duplicate_fix_ids(tmp_path: Path) -> None:
    """Two rows sharing (study, individual, timestamp) would break the join."""
    path = tmp_path / "dupe_fix.csv"
    path.write_text(
        "timestamp,lon,lat,individual_id,study_id\n"
        "2019-03-07 18:00:00.000,-72.85,42.01,bear_A,CT\n"
        "2019-03-07 18:00:00.000,-72.86,42.02,bear_A,CT\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate fix_id"):
        read_fixes(path)


# ---------------------------------------------------------------------------
# Dedup reduction and round-trip
# ---------------------------------------------------------------------------
def test_dedup_collapses_fixes_sharing_a_pixel(registry: Registry) -> None:
    """18 fixes inside one 0.01-degree pixel collapse to a single key."""
    source = grid_source(registry, degree_grid(0.01), name="test_1km")
    rows = [(-100.00500 + 0.00005 * k, 40.00500 + 0.00005 * k, f"2020-01-01T{k:02d}:00:00Z")
            for k in range(18)]
    fixes = make_fixes(rows)
    result = dedup_dataset(fixes, source, "synthetic")

    assert result.n_fixes == 18
    assert result.n_unique == 1
    assert result.n_pixels == 1
    assert result.reduction_ratio == pytest.approx(1 / 18)

    # fix_id -> key mapping round-trips: every fix maps to the one key.
    assert len(result.fixmap) == 18
    assert result.fixmap["key_index"].nunique() == 1
    assert set(result.fixmap["fix_id"]) == set(fixes["fix_id"])

    # The sampling geometry is the pixel centroid, not any individual fix.
    centroid = result.keys.iloc[0]
    assert centroid["px_lon"] == pytest.approx(-100.005, abs=1e-6)
    assert centroid["px_lat"] == pytest.approx(40.005, abs=1e-6)


def test_dedup_keeps_distinct_pixels_apart(registry: Registry) -> None:
    source = grid_source(registry, degree_grid(0.0083333333))
    fixes = make_fixes([(-100.0, 40.0, "2020-01-01T00:00:00Z"),
                        (-100.0, 41.0, "2020-01-01T01:00:00Z")])
    result = dedup_dataset(fixes, source, "synthetic")
    assert result.n_unique == 2
    assert result.n_pixels == 2


def test_subdaily_key_collapses_within_the_hour_only(registry: Registry) -> None:
    """ERA5-Land: exact hour join — the same hour collapses, the next does not."""
    fixes = make_fixes([(-100.0, 40.0, "2020-01-01T00:10:00Z"),
                        (-100.0, 40.0, "2020-01-01T00:50:00Z"),
                        (-100.0, 40.0, "2020-01-01T01:10:00Z")])
    result = dedup_dataset(fixes, registry.source("era5_land"), "synthetic")
    assert result.n_unique == 2
    assert result.n_pixels == 1


def test_check_reduction_fails_on_broken_snapping(registry: Registry) -> None:
    """A coarse source that does not collapse spatially must fail loudly."""
    source = registry.source("csp_ghm")  # 1 km
    fixes = make_fixes([(-100.0 + 0.01 * k, 40.0, f"2020-01-01T{k:02d}:00:00Z") for k in range(24)])
    result = dedup_dataset(fixes, source, "synthetic")  # 24 fixes, 24 distinct 1 km pixels
    assert result.pixel_reduction_ratio == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match="pixel-grid snapping is wrong"):
        check_reduction(source, registry.export, result)


def test_check_reduction_tolerates_a_cadence_matched_source(registry: Registry) -> None:
    """ERA5-Land keeps ~one (cell, hour) key per fix — warn, do not fail."""
    fixes = make_fixes([(-100.0, 40.0, f"2020-01-01T{k:02d}:00:00Z") for k in range(24)])
    result = dedup_dataset(fixes, registry.source("era5_land"), "synthetic")
    assert result.reduction_ratio > 0.9          # the time key does no work
    assert result.pixel_reduction_ratio < 0.5    # but the pixels collapsed
    check_reduction(registry.source("era5_land"), registry.export, result)  # must not raise


# ---------------------------------------------------------------------------
# Chunking and chunk identity
# ---------------------------------------------------------------------------
def test_chunking_respects_the_feature_target(registry: Registry) -> None:
    source = grid_source(registry, degree_grid(0.01))
    export = dataclasses.replace(registry.export, target_features_per_task=100)
    chunks = build_chunks(source, export, "synthetic", make_keys(250))
    assert [c.n_features for c in chunks] == [100, 100, 50]
    assert sum(c.n_features for c in chunks) == 250
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunking_merges_buckets_up_to_the_time_group_cap(registry: Registry) -> None:
    """Many small daily buckets must merge, not become one task each."""
    source = registry.source("modis_lst")
    keys = pd.concat(
        [make_keys(10, bucket=f"2020-01-{d:02d}", t0=20200100 + d) for d in range(1, 21)],
        ignore_index=True,
    )
    keys["key_index"] = range(len(keys))
    export = dataclasses.replace(registry.export, target_features_per_task=100_000,
                                 max_time_groups_per_task=8)
    chunks = build_chunks(source, export, "synthetic", keys)
    assert len(chunks) == 3                      # ceil(20 buckets / 8 groups)
    assert [c.n_features for c in chunks] == [80, 80, 40]
    assert chunks[0].time_bucket == "2020-01-01..2020-01-08"


def test_chunking_never_exceeds_the_feature_target(registry: Registry) -> None:
    """A single oversized bucket is still split, not emitted whole."""
    source = grid_source(registry, degree_grid(0.01))
    export = dataclasses.replace(registry.export, target_features_per_task=10)
    chunks = build_chunks(source, export, "ds", make_keys(25))
    assert [c.n_features for c in chunks] == [10, 10, 5]


def test_chunk_id_tracks_source_config_and_key_list(registry: Registry) -> None:
    source = grid_source(registry, degree_grid(0.01))
    export = registry.export
    keys = make_keys(50)
    base = build_chunks(source, export, "ds", keys)[0].chunk_id

    # Unchanged inputs -> unchanged identity (resume must skip it).
    assert build_chunks(source, export, "ds", keys)[0].chunk_id == base
    # A different dataset is a different chunk.
    assert build_chunks(source, export, "other", keys)[0].chunk_id != base
    # A changed source config (the grid here) invalidates the chunk.
    moved = dataclasses.replace(source, grid=degree_grid(0.02))
    assert build_chunks(moved, export, "ds", keys)[0].chunk_id != base
    # A changed chunking parameter re-partitions, hence a new id.
    resized = dataclasses.replace(export, target_features_per_task=20)
    assert build_chunks(source, resized, "ds", keys)[0].chunk_id != base
    # A changed key list invalidates the chunk.
    assert build_chunks(source, export, "ds", make_keys(49))[0].chunk_id != base


def test_resolved_spec_ignores_cosmetic_edits(registry: Registry) -> None:
    """Chunk identity must not move when only prose changes."""
    source = registry.source("era5_land")
    assert source.resolved_spec() == dataclasses.replace(
        source, description="rewritten", tier=1
    ).resolved_spec()


def test_key_digest_is_order_independent() -> None:
    keys = make_keys(20)
    shuffled = keys.sample(frac=1.0, random_state=0).reset_index(drop=True)
    assert key_digest(keys) == key_digest(shuffled)


# ---------------------------------------------------------------------------
# Ledger resume
# ---------------------------------------------------------------------------
def test_ledger_resume_skips_complete_and_failed_unless_retried(registry: Registry, tmp_path: Path) -> None:
    source = grid_source(registry, degree_grid(0.01))
    export = dataclasses.replace(registry.export, results_root=tmp_path,
                                 target_features_per_task=10)
    chunks = build_chunks(source, export, "ds", make_keys(30))  # 3 chunks
    path = ledger_path(export, source.name)

    record_rows(path, [row_for(chunks[0], status="complete"),
                       row_for(chunks[1], status="failed", failure_reason="boom")])
    ledger = load_ledger(path)
    assert ledger[chunks[0].chunk_id]["status"] == "complete"

    assert [c.chunk_id for c in resumable_chunks(chunks, ledger)] == [chunks[2].chunk_id]
    assert [c.chunk_id for c in resumable_chunks(chunks, ledger, retry_failed=True)] == [
        chunks[1].chunk_id, chunks[2].chunk_id]

    # Append-only ledger: the last row per chunk wins on read.
    record_rows(path, [row_for(chunks[1], status="complete")])
    assert load_ledger(path)[chunks[1].chunk_id]["status"] == "complete"
    assert ledger_status_counts(load_ledger(path))["complete"] == 2


def test_ledger_survives_an_interrupted_run(registry: Registry, tmp_path: Path) -> None:
    """A partially submitted run leaves a valid ledger (Ctrl-C safety)."""
    source = grid_source(registry, degree_grid(0.01))
    export = dataclasses.replace(registry.export, results_root=tmp_path,
                                 target_features_per_task=10)
    chunks = build_chunks(source, export, "ds", make_keys(30))
    path = ledger_path(export, source.name)
    record_rows(path, [row_for(chunks[0], status="pending")])
    record_rows(path, [row_for(chunks[0], status="submitted", task_id="t1")])

    # Both transitions are on disk (append-only history) ...
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # header + pending + submitted
    # ... while the resolved state is the last row per chunk.
    ledger = load_ledger(path)
    assert ledger[chunks[0].chunk_id]["status"] == "submitted"
    assert ledger[chunks[0].chunk_id]["task_id"] == "t1"
    assert ledger_status_counts(ledger)["submitted"] == 1
    # A submitted chunk whose task died is exactly what resume picks up.
    assert len(resumable_chunks(chunks, ledger)) == 3


def test_ledger_row_rejects_an_unknown_status(registry: Registry) -> None:
    chunk = build_chunks(grid_source(registry, degree_grid(0.01)), registry.export, "ds", make_keys(1))[0]
    with pytest.raises(ValueError, match="Unknown chunk status"):
        row_for(chunk, status="bogus")


# ---------------------------------------------------------------------------
# QA bit arithmetic and temporal alignment
# ---------------------------------------------------------------------------
def test_extract_bits_matches_hand_computation() -> None:
    assert extract_bits(0b00000010, 0, 1) == 2   # MODIS mandatory QA
    assert extract_bits(0b00001100, 2, 3) == 3
    assert extract_bits(0b10100000, 5, 7) == 5   # FparLai_QC SCF_QC
    assert extract_bits(0b1000, 3, 3) == 1
    assert extract_bits(0b1000, 2, 2) == 0
    with pytest.raises(ValueError):
        extract_bits(1, 3, 1)


@pytest.mark.parametrize(
    "source_name,fix_ts,expected",
    [
        ("csp_ghm", "2020-06-15T13:37:00Z", 0.0),            # static: exactly 0
        ("era5_land", "2020-06-15T13:37:00Z", 37 / 60),      # subdaily: hour anchor
        ("modis_snow", "2020-06-15T13:00:00Z", 1.0),         # daily: noon anchor
        ("viirs_nightlights", "2020-06-01T12:00:00Z", 336.0),  # monthly: mid-month
        ("annual_nlcd", "2020-07-01T12:00:00Z", 0.0),        # annual: 1 July anchor
    ],
)
def test_age_hours_per_cadence(registry: Registry, source_name: str, fix_ts: str, expected: float) -> None:
    assert age_hours(registry.source(source_name), pd.Timestamp(fix_ts)) == pytest.approx(expected, abs=1e-6)


def test_annual_clamps_a_2026_fix_to_the_2024_max(registry: Registry) -> None:
    """NLCD/CDL end in 2024; a 2026 fix must clamp and show the staleness."""
    source = registry.source("annual_nlcd")
    assert source.last_year == 2024
    fix = pd.Timestamp("2026-03-19T06:00:00Z")
    assert nominal_time(source, fix) == pd.Timestamp("2024-07-01T12:00:00Z")
    # 2024-07-01T12:00 -> 2026-03-19T06:00 is 626 days minus 6 hours.
    assert age_hours(source, fix) == pytest.approx(626 * 24 - 6, abs=1e-6)
    # In-range years are not clamped.
    assert nominal_time(source, pd.Timestamp("2021-03-19T06:00:00Z")) == pd.Timestamp("2021-07-01T12:00:00Z")


def test_composite_age_stays_inside_the_half_window(registry: Registry) -> None:
    source = registry.source("sentinel2")  # composite_days = 8
    for day in range(1, 29):
        ts = pd.Timestamp(f"2020-03-{day:02d}T12:00:00Z")
        assert age_hours(source, ts) <= 8 * 24 + 1e-6


# ---------------------------------------------------------------------------
# Join-back
# ---------------------------------------------------------------------------
def _join_fixture(registry: Registry, *, lon_lat: list[tuple[float, float]] | None = None):
    """A tiny (source, fixes, dedup result, sampled) tuple for join tests."""
    source = grid_source(
        registry,
        degree_grid(0.0083333333),
        name="joinsrc",
        bands=("value",),
        derived=(),
        focal=(),
        terrain=(),
        post=(),
        categorical=(),
    )
    coords = lon_lat or [(-100.0, 40.0), (-100.0, 40.0), (-99.0, 41.0)]
    fixes = make_fixes([(lon, lat, f"2020-01-01T{h:02d}:00:00Z")
                        for h, (lon, lat) in enumerate(coords)])
    result = dedup_dataset(fixes, source, "synthetic")
    sampled = pd.DataFrame({
        "key_index": result.keys["key_index"],
        "value": [1.5] * len(result.keys),
        "value_buf30": [2.5] * len(result.keys),
    })
    return source, fixes, result, sampled


def test_join_produces_one_row_per_fix_with_missing_and_age(registry: Registry) -> None:
    source, fixes, result, sampled = _join_fixture(registry)
    out = build_fix_table(source, fixes, result.fixmap, sampled, dataset="synthetic")

    assert_one_row_per_fix(out, fixes, dataset="synthetic", source=source.name)
    assert len(out) == 3
    assert list(out.columns) == [
        "fix_id", "dataset", "timestamp", "value", "value_buf30",
        "value_missing", "_age_hours",
    ]
    assert out["value"].tolist() == [1.5, 1.5, 1.5]
    assert out["value_buf30"].tolist() == [2.5, 2.5, 2.5]
    assert out["value_missing"].tolist() == [False, False, False]
    assert out["_age_hours"].tolist() == [0.0, 0.0, 0.0]  # static source


def test_join_marks_missing_values_per_parameter(registry: Registry) -> None:
    # Two fixes in two distinct pixels, one key each.
    source, fixes, result, sampled = _join_fixture(registry, lon_lat=[(-100.0, 40.0), (-99.0, 41.0)])
    assert len(sampled) == 2
    sampled.loc[0, "value"] = None  # the product could not fill this pixel
    out = build_fix_table(source, fixes, result.fixmap, sampled, dataset="synthetic")
    assert out["value_missing"].sum() == 1
    assert out.loc[out["value_missing"], "value"].isna().all()
    assert out.loc[~out["value_missing"], "value"].notna().all()


def test_join_asserts_one_row_per_fix(registry: Registry) -> None:
    source, fixes, result, sampled = _join_fixture(registry)
    out = build_fix_table(source, fixes, result.fixmap, sampled, dataset="synthetic")

    # A dropped fix must fail on the row count.
    with pytest.raises(AssertionError, match="output rows"):
        assert_one_row_per_fix(out.iloc[:-1], fixes, dataset="d", source="s")

    # A duplicated fix_id (same row count) must fail on the duplicate check.
    duplicated = out.copy()
    duplicated.loc[1, "fix_id"] = duplicated.loc[0, "fix_id"]
    assert len(duplicated) == len(fixes)
    with pytest.raises(AssertionError, match="duplicated fix_id"):
        assert_one_row_per_fix(duplicated, fixes, dataset="d", source="s")

    # A fix the export never covered (same row count, no duplicates) must fail.
    substituted = out.copy()
    substituted.loc[1, "fix_id"] = "0" * 40
    with pytest.raises(AssertionError, match="set mismatch"):
        assert_one_row_per_fix(substituted, fixes, dataset="d", source="s")


def test_join_output_path(registry: Registry, tmp_path: Path) -> None:
    export = dataclasses.replace(registry.export, results_root=tmp_path)
    assert output_path(export, "wolf", "era5_land") == tmp_path / "wolf" / "era5_land.parquet"


# ---------------------------------------------------------------------------
# Wide per-dataset CSV (GEE_DATASET_PATH)
# ---------------------------------------------------------------------------
def per_source_frame(n: int = 3, *, start: str = "2020-01-01", **columns) -> pd.DataFrame:
    """A frame shaped like ``build_fix_table``'s output for one source."""
    frame = pd.DataFrame({
        "fix_id": [f"fix{i:02d}" for i in range(n)],
        "dataset": "mule_deer",
        "timestamp": pd.date_range(start, periods=n, freq="h", tz="UTC"),
    })
    for name, values in columns.items():
        frame[name] = values
    return frame


def test_prefixed_column_drops_the_leading_underscore() -> None:
    assert prefixed_column("era5_land", "_age_hours") == "era5_land_age_hours"
    assert prefixed_column("sentinel2", "ndvi") == "sentinel2_ndvi"
    assert prefixed_column("sentinel2", "ndvi_missing") == "sentinel2_ndvi_missing"


def test_combine_tables_prefixes_and_merges_on_fix_id() -> None:
    a = per_source_frame(3, ndvi=[0.1, 0.2, 0.3], ndvi_missing=[False] * 3, _age_hours=[1.0] * 3)
    b = per_source_frame(3, temperature_2m=[265.0] * 3, _age_hours=[0.0] * 3)
    out = combine_tables({"sentinel2": a, "era5_land": b})

    # Sources in alphabetical order; within a source, that source's own column
    # order (`_age_hours` last, as build_fix_table emits it).
    assert list(out.columns) == [
        "fix_id", "dataset", "timestamp",
        "era5_land_temperature_2m", "era5_land_age_hours",
        "sentinel2_ndvi", "sentinel2_ndvi_missing", "sentinel2_age_hours",
    ]
    assert len(out) == 3
    assert out["sentinel2_ndvi"].tolist() == [0.1, 0.2, 0.3]
    assert out["era5_land_age_hours"].tolist() == [0.0] * 3
    assert out["sentinel2_age_hours"].tolist() == [1.0] * 3


def test_combine_tables_keeps_same_named_params_from_different_sources() -> None:
    """`confidence` exists in both Annual NLCD and CDL; neither may be lost."""
    out = combine_tables({
        "annual_nlcd": per_source_frame(2, confidence=[0.9, 0.8], landcover=[41, 42]),
        "nass_cdl": per_source_frame(2, confidence=[10.0, 20.0], cropland=[1, 5]),
    })
    assert out["annual_nlcd_confidence"].tolist() == [0.9, 0.8]
    assert out["nass_cdl_confidence"].tolist() == [10.0, 20.0]
    assert out["annual_nlcd_landcover"].tolist() == [41, 42]
    assert out["nass_cdl_cropland"].tolist() == [1, 5]


def test_combine_tables_fails_on_disagreeing_fix_universes() -> None:
    """A short/duplicated column must fail loudly, not silently truncate."""
    with pytest.raises(ValueError, match="fix universe disagrees"):
        combine_tables({
            "sentinel2": per_source_frame(3, ndvi=[0.1, 0.2, 0.3]),
            "era5_land": per_source_frame(2, temperature_2m=[265.0, 266.0]),
        })


def test_combine_tables_sorts_chronologically() -> None:
    late = per_source_frame(2, start="2021-06-01", ndvi=[0.5, 0.6])
    early = per_source_frame(2, start="2019-01-01", ndvi=[0.1, 0.2])
    late["fix_id"] = ["late0", "late1"]
    early["fix_id"] = ["early0", "early1"]
    out = combine_tables({"sentinel2": pd.concat([late, early], ignore_index=True)})
    assert out["fix_id"].tolist() == ["early0", "early1", "late0", "late1"]


def test_combine_tables_requires_at_least_one_source() -> None:
    with pytest.raises(ValueError, match="at least one source"):
        combine_tables({})


def test_write_wide_csv_round_trips(registry: Registry, tmp_path: Path) -> None:
    export = dataclasses.replace(registry.export, results_root=tmp_path / "results")
    tables = {"sentinel2": per_source_frame(3, ndvi=[0.1, 0.2, 0.3])}
    path = write_wide_csv(export, "mule_deer", tmp_path / "gee_induced", tables)

    assert path == wide_csv_path(tmp_path / "gee_induced", "mule_deer")
    assert path.exists()
    back = pd.read_csv(path)
    assert list(back.columns) == ["fix_id", "dataset", "timestamp", "sentinel2_ndvi"]
    assert back["sentinel2_ndvi"].tolist() == [0.1, 0.2, 0.3]
    assert len(back) == 3


def test_combine_dataset_reads_the_parquet_archive(registry: Registry, tmp_path: Path) -> None:
    """The CSV is assembled from whatever per-source Parquet exists on disk."""
    export = dataclasses.replace(registry.export, results_root=tmp_path / "results")
    archive = export.results_root / "mule_deer"
    archive.mkdir(parents=True)
    per_source_frame(3, ndvi=[0.1, 0.2, 0.3]).to_parquet(archive / "sentinel2.parquet", index=False)
    per_source_frame(3, temperature_2m=[265.0] * 3).to_parquet(
        archive / "era5_land.parquet", index=False)

    path = combine_dataset(export, "mule_deer", tmp_path / "gee_induced")
    frame = pd.read_csv(path)
    assert len(frame) == 3
    assert {"sentinel2_ndvi", "era5_land_temperature_2m"} <= set(frame.columns)
    # A source that has not been joined yet is simply absent, not an error.
    assert not any(c.startswith("viirs") for c in frame.columns)


def test_combine_dataset_warns_when_nothing_is_on_disk(registry: Registry, tmp_path: Path) -> None:
    export = dataclasses.replace(registry.export, results_root=tmp_path / "empty")
    assert combine_dataset(export, "mule_deer", tmp_path / "gee_induced") is None


def test_gee_dataset_path_fails_loudly_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """`.env` would repopulate the variable, so bypass its loading."""
    from movement.utils import env as env_mod

    monkeypatch.setattr(env_mod, "load_env", lambda: None)
    monkeypatch.delenv("GEE_DATASET_PATH", raising=False)
    with pytest.raises(RuntimeError, match="GEE_DATASET_PATH is not set"):
        env_mod.gee_dataset_path()


def test_gee_dataset_path_creates_the_target_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from movement.utils import env as env_mod

    monkeypatch.setattr(env_mod, "load_env", lambda: None)
    target = tmp_path / "gee_induced"
    monkeypatch.setenv("GEE_DATASET_PATH", str(target))
    assert env_mod.gee_dataset_path() == target
    assert target.is_dir()


# ---------------------------------------------------------------------------
# Earth Engine auth resolution (offline: never touches credentials or network)
# ---------------------------------------------------------------------------
@pytest.fixture
def gee_env(monkeypatch: pytest.MonkeyPatch):
    """The env module with .env loading disabled and all GEE vars cleared."""
    from movement.utils import env as env_mod

    monkeypatch.setattr(env_mod, "load_env", lambda: None)
    for var in ("GEE_AUTH", "GEE_SERVICE_ACCOUNT", "GEE_KEY_FILE",
                "GEE_PROJECT", "GCS_BUCKET", "GEE_DRIVE_FOLDER"):
        monkeypatch.delenv(var, raising=False)
    return env_mod


def test_auth_mode_falls_back_to_interactive(gee_env) -> None:
    """No service account configured -> interactive, per the chosen default."""
    assert gee_env.resolve_auth_mode() == "interactive"
    assert gee_env.service_account_configured() is False


@pytest.mark.parametrize("explicit", [None, "auto"])
def test_auth_mode_prefers_a_configured_service_account(gee_env, monkeypatch, explicit) -> None:
    monkeypatch.setenv("GEE_SERVICE_ACCOUNT", "bot@project.iam.gserviceaccount.com")
    monkeypatch.setenv("GEE_KEY_FILE", "key.json")
    assert gee_env.service_account_configured() is True
    assert gee_env.resolve_auth_mode(explicit) == "service_account"


def test_auth_mode_needs_both_service_account_variables(gee_env, monkeypatch) -> None:
    """Half a service account is not a service account: falls back with a warning."""
    monkeypatch.setenv("GEE_SERVICE_ACCOUNT", "bot@project.iam.gserviceaccount.com")
    assert gee_env.resolve_auth_mode() == "interactive"


def test_service_account_pin_forbids_the_interactive_fallback(gee_env) -> None:
    """The reproducibility escape hatch must be an error, never a quiet downgrade."""
    with pytest.raises(RuntimeError, match="cannot silently fall"):
        gee_env.resolve_auth_mode("service_account")
    with pytest.raises(RuntimeError, match="cannot silently fall"):
        gee_env.resolve_auth_mode("service_account")
    # Setting one of the two is still not enough.
    gee_env.os.environ["GEE_SERVICE_ACCOUNT"] = "bot@project"
    with pytest.raises(RuntimeError, match="GEE_KEY_FILE"):
        gee_env.resolve_auth_mode("service_account")


def test_env_pin_beats_detection_and_explicit_beats_env(gee_env, monkeypatch) -> None:
    monkeypatch.setenv("GEE_AUTH", "interactive")
    monkeypatch.setenv("GEE_SERVICE_ACCOUNT", "bot@project")
    monkeypatch.setenv("GEE_KEY_FILE", "key.json")
    # GEE_AUTH overrides the presence of a service account ...
    assert gee_env.resolve_auth_mode() == "interactive"
    # ... and an explicit --auth value overrides GEE_AUTH.
    assert gee_env.resolve_auth_mode("service_account") == "service_account"


def test_unknown_auth_mode_is_rejected(gee_env, monkeypatch) -> None:
    monkeypatch.setenv("GEE_AUTH", "magic")
    with pytest.raises(RuntimeError, match="Unknown Earth Engine auth mode"):
        gee_env.resolve_auth_mode()
    with pytest.raises(RuntimeError, match="Unknown Earth Engine auth mode"):
        gee_env.resolve_auth_mode("nonsense")


def test_project_and_bucket_are_required_in_every_mode(gee_env, monkeypatch) -> None:
    """Interactive auth replaces only the two service-account variables."""
    with pytest.raises(RuntimeError, match="GEE_PROJECT is not set"):
        gee_env.gee_auth()
    monkeypatch.setenv("GEE_PROJECT", "my-project")
    with pytest.raises(RuntimeError, match="GCS_BUCKET is not set"):
        gee_env.gee_auth()
    # Pinning the service account does not relax the project/bucket requirement.
    monkeypatch.setenv("GEE_KEY_FILE", "key.json")
    monkeypatch.setenv("GEE_SERVICE_ACCOUNT", "bot@project")
    with pytest.raises(RuntimeError, match="GCS_BUCKET is not set"):
        gee_env.gee_auth("service_account")


def test_service_account_mode_rejects_an_unreadable_key_file(gee_env, monkeypatch) -> None:
    monkeypatch.setenv("GEE_PROJECT", "my-project")
    monkeypatch.setenv("GCS_BUCKET", "my-bucket")
    monkeypatch.setenv("GEE_SERVICE_ACCOUNT", "bot@project")
    monkeypatch.setenv("GEE_KEY_FILE", "does-not-exist.json")
    with pytest.raises(FileNotFoundError, match="GEE_KEY_FILE"):
        gee_env.gee_auth("service_account")


# ---------------------------------------------------------------------------
# Interactive credentials (offline: the cache file is stubbed, no token refresh)
# ---------------------------------------------------------------------------
def _stub_credentials_file(tmp_path: Path, monkeypatch, payload: dict):
    """Point `ee.oauth` at a temporary credentials file."""
    ee_oauth = pytest.importorskip("ee.oauth")
    from movement.utils import env as env_mod

    monkeypatch.setattr(env_mod, "load_env", lambda: None)
    path = tmp_path / "credentials"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(ee_oauth, "get_credentials_path", lambda: str(path))
    return env_mod


def test_interactive_credentials_supply_the_required_token_argument(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression: google-auth requires `token`, and `ee.oauth` does not return it.

    `Credentials(**ee.oauth.get_credentials_arguments())` raises
    "missing 1 required positional argument: 'token'". The library's own
    `ee.data.get_persistent_credentials` does `Credentials(None, **args)`.
    """
    from google.oauth2.credentials import Credentials

    env_mod = _stub_credentials_file(tmp_path, monkeypatch, {
        "refresh_token": "fake-refresh-token",
        "scopes": ["https://www.googleapis.com/auth/earthengine"],
        "redirect_uri": "http://localhost:8085",
    })
    kwargs = env_mod._interactive_credential_kwargs()

    assert kwargs["token"] is None
    assert kwargs["refresh_token"] == "fake-refresh-token"
    # The OAuth client is the Earth Engine one compiled into the library.
    assert kwargs["client_id"] and kwargs["client_secret"]
    assert kwargs["token_uri"] == "https://oauth2.googleapis.com/token"

    credentials = Credentials(**kwargs)   # this is the call that used to raise
    assert credentials.refresh_token == "fake-refresh-token"
    assert credentials.client_id == kwargs["client_id"]


def test_interactive_credentials_fail_loudly_without_a_refresh_token(
    tmp_path: Path, monkeypatch
) -> None:
    env_mod = _stub_credentials_file(tmp_path, monkeypatch, {"scopes": []})
    with pytest.raises(RuntimeError, match="no refresh_token"):
        env_mod._interactive_credential_kwargs()


def test_interactive_credentials_fail_loudly_when_the_file_is_absent(
    tmp_path: Path, monkeypatch
) -> None:
    ee_oauth = pytest.importorskip("ee.oauth")
    from movement.utils import env as env_mod

    monkeypatch.setattr(env_mod, "load_env", lambda: None)
    missing = tmp_path / "nope"
    monkeypatch.setattr(ee_oauth, "get_credentials_path", lambda: str(missing))
    with pytest.raises(RuntimeError, match="No interactive Earth Engine credentials"):
        env_mod._interactive_credential_kwargs()


def test_a_future_ee_version_supplying_token_wins(tmp_path: Path, monkeypatch) -> None:
    """If `ee` ever returns `token`, our `None` must not clobber it.

    `ee.oauth.get_credentials_arguments` builds its dict from a fixed key list, so
    a `token` field in the file never surfaces — the helper itself has to be
    stubbed to exercise the merge order.
    """
    ee_oauth = pytest.importorskip("ee.oauth")
    from movement.utils import env as env_mod

    monkeypatch.setattr(env_mod, "load_env", lambda: None)
    path = tmp_path / "credentials"
    path.write_text(json.dumps({"refresh_token": "r", "scopes": []}), encoding="utf-8")
    monkeypatch.setattr(ee_oauth, "get_credentials_path", lambda: str(path))
    monkeypatch.setattr(ee_oauth, "get_credentials_arguments", lambda: {
        "token": "already-have-one",
        "refresh_token": "r",
        "client_id": "client-id",
        "client_secret": "client-secret",
        "token_uri": "https://oauth2.googleapis.com/token",
        "scopes": [],
        "quota_project_id": None,
    })
    assert env_mod._interactive_credential_kwargs()["token"] == "already-have-one"


# ---------------------------------------------------------------------------
# Export destination (Drive vs Cloud Storage)
# ---------------------------------------------------------------------------
def test_destination_defaults_to_the_registry_setting(registry: Registry) -> None:
    """The destination is a config choice, so it comes from the YAML."""
    assert registry.export.destination in S.DESTINATIONS


def test_registry_rejects_an_unknown_destination(registry: Registry, tmp_path: Path) -> None:
    import yaml

    data = yaml.safe_load(S.DEFAULT_SOURCES.read_text(encoding="utf-8"))
    data["export"]["destination"] = "dropbox"
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="not one of"):
        load_registry(path)


def test_each_destination_requires_its_own_target_variable(gee_env, monkeypatch) -> None:
    """Drive needs GEE_DRIVE_FOLDER; Cloud Storage needs GCS_BUCKET — not both."""
    monkeypatch.setenv("GEE_PROJECT", "my-project")
    # Drive configured, no bucket anywhere.
    monkeypatch.setenv("GEE_DRIVE_FOLDER", "gee_covariates")
    with pytest.raises(RuntimeError, match="GCS_BUCKET is not set"):
        gee_env.gee_auth(destination="cloud_storage")
    # Bucket configured, no Drive folder.
    monkeypatch.delenv("GEE_DRIVE_FOLDER")
    with pytest.raises(RuntimeError, match="GEE_DRIVE_FOLDER is not set"):
        gee_env.gee_auth(destination="drive")


def test_target_variable_mapping_is_declared(gee_env) -> None:
    assert gee_env.GEE_TARGET_VARS == {
        "cloud_storage": "GCS_BUCKET",
        "drive": "GEE_DRIVE_FOLDER",
    }


def test_destination_selects_the_output_naming(registry: Registry) -> None:
    """Cloud Storage nests by source/dataset; Drive uses one flat file name."""
    export = registry.export
    chunk = build_chunks(grid_source(registry, degree_grid(0.01)), export,
                         "mule_deer", make_keys(1))[0]

    assert chunk.object_path(export) == (
        f"{export.gcs_prefix}/test_src/mule_deer/{chunk.chunk_id}.csv")
    # Earth Engine appends the extension itself, so the prefix is the stem.
    assert chunk.drive_prefix(export) == f"test_src_mule_deer_{chunk.chunk_id}"
    assert chunk.output_name(export) == f"test_src_mule_deer_{chunk.chunk_id}.csv"

    # The ledger column keeps the spec'd name but holds either kind of URI.
    assert chunk.location(export, "drive", "gee_covariates").startswith(
        "drive://gee_covariates/")
    assert chunk.location(export, "cloud_storage", "my-bucket").startswith(
        "gs://my-bucket/")


# ---------------------------------------------------------------------------
# Drive client (transport injected: no network, no credentials)
# ---------------------------------------------------------------------------
class FakeDrive:
    """Records Drive REST calls and serves canned JSON / file bytes."""

    def __init__(self, folder_id: str = "folder-1", files: list[tuple[str, str]] | None = None):
        self.folder_id = folder_id
        self.files = files or []          # (id, name)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method: str, url: str, headers: dict, body: bytes | None) -> bytes:
        self.calls.append((method, url))
        assert headers.get("Authorization", "").startswith("Bearer ")
        if url.startswith("https://www.googleapis.com/drive/v3/files/") and "alt=media" in url:
            file_id = url.split("/files/")[1].split("?")[0]
            return f"key_index,value\n{file_id},1\n".encode()
        if "mimeType" in url:  # folder lookup
            return json.dumps({"files": [{"id": self.folder_id, "name": "gee_covariates"}]}).encode()
        return json.dumps({"files": [{"id": i, "name": n} for i, n in self.files]}).encode()


class _Creds:
    token = "fake-token"
    expired = False


def test_drive_client_finds_folder_lists_and_downloads() -> None:
    transport = FakeDrive(files=[("f1", "sentinel2_mule_deer_abc123.csv")])
    client = DriveClient(_Creds(), transport=transport)

    contents = client.fetch("gee_covariates", "abc123")
    assert contents == [("sentinel2_mule_deer_abc123.csv", b"key_index,value\nf1,1\n")]

    folder_calls = [u for _, u in transport.calls if "mimeType" in u]
    assert len(folder_calls) == 1
    assert "folder" in folder_calls[0] and "gee_covariates" in folder_calls[0]
    # The child listing is scoped to the folder and matched by substring.
    list_calls = [u for _, u in transport.calls if "in+parents" in u or "in%20parents" in u]
    assert len(list_calls) == 1
    assert "abc123" in list_calls[0]


def test_drive_client_reports_a_missing_folder_loudly() -> None:
    class NoFolder(FakeDrive):
        def __call__(self, method, url, headers, body):
            self.calls.append((method, url))
            return json.dumps({"files": []}).encode()

    client = DriveClient(_Creds(), transport=NoFolder())
    with pytest.raises(DriveError, match="No Drive folder named"):
        client.find_folder("gee_covariates")


def test_drive_client_reports_missing_chunk_output_loudly() -> None:
    client = DriveClient(_Creds(), transport=FakeDrive(files=[]))
    with pytest.raises(DriveError, match="No file matching"):
        client.fetch("gee_covariates", "abc123")


def test_drive_fetcher_writes_chunk_output_locally(registry: Registry, tmp_path: Path) -> None:
    """The fetcher materialises Drive files so the join path is destination-agnostic."""
    export = registry.export
    chunk = build_chunks(grid_source(registry, degree_grid(0.01)), export,
                         "mule_deer", make_keys(1))[0]
    transport = FakeDrive(files=[("f1", chunk.output_name(export))])
    fetch = drive_fetcher(DriveClient(_Creds(), transport=transport), "gee_covariates",
                          tmp_path / "scratch")

    paths = fetch(chunk)
    assert len(paths) == 1
    assert paths[0].read_text().startswith("key_index,value")


def test_local_fetcher_is_unchanged_for_dry_runs(tmp_path: Path) -> None:
    """`--from-local` must keep working with no credentials at all."""
    chunk = Chunk(chunk_id="abc123", dataset="mule_deer", source="sentinel2",
                  zone=0, time_bucket="static", keys=make_keys(1))
    (tmp_path / "abc123.csv").write_text("key_index,value\n1,2\n", encoding="utf-8")
    assert local_fetcher(tmp_path)(chunk) == [tmp_path / "abc123.csv"]


# ---------------------------------------------------------------------------
# CLI exporter wiring (mocked ee: no credentials, no network)
# ---------------------------------------------------------------------------
def _cli_module():
    """Load scripts/gee_export.py so its wiring can be unit-tested."""
    import importlib.util

    path = S.DEFAULT_SOURCES.parents[2] / "scripts" / "gee_export.py"
    spec = importlib.util.spec_from_file_location("gee_export_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_chunk(registry: Registry) -> Chunk:
    export = registry.export
    source = grid_source(registry, degree_grid(0.01))
    return build_chunks(source, export, "mule_deer", make_keys(1))[0]


def test_exporters_are_registered_per_destination(registry: Registry) -> None:
    cli = _cli_module()
    assert set(cli.EXPORTERS) == set(S.DESTINATIONS)


def test_drive_exporter_builds_a_todrive_task(registry: Registry) -> None:
    cli = _cli_module()
    ee = EEStub()
    export = registry.export
    source = registry.source("csp_ghm")
    chunk = _stub_chunk(registry)

    cli._to_drive(ee, {"target": "gee_covariates"}, export, source, chunk,
                  table="TABLE", description="desc")

    calls = ee.calls_named("toDrive()")
    assert len(calls) == 1
    _, _, kwargs = calls[0]
    assert kwargs["folder"] == "gee_covariates"
    # The stem, not the file name: Earth Engine appends ".csv" itself.
    assert kwargs["fileNamePrefix"] == chunk.drive_prefix(export)
    assert not kwargs["fileNamePrefix"].endswith(".csv")
    assert kwargs["fileFormat"] == export.file_format
    assert kwargs["selectors"] == task_selectors(source, export)
    # No bucket parameter in the Drive path.
    assert "bucket" not in kwargs


def test_cloud_storage_exporter_builds_a_tocloudstorage_task(registry: Registry) -> None:
    """The Cloud Storage path must still work, so switching back is one YAML line."""
    cli = _cli_module()
    ee = EEStub()
    export = registry.export
    source = registry.source("csp_ghm")
    chunk = _stub_chunk(registry)

    cli._to_cloud_storage(ee, {"target": "my-bucket"}, export, source, chunk,
                          table="TABLE", description="desc")

    calls = ee.calls_named("toCloudStorage()")
    assert len(calls) == 1
    _, _, kwargs = calls[0]
    assert kwargs["bucket"] == "my-bucket"
    assert kwargs["fileNamePrefix"] == chunk.object_path(export)
    assert "folder" not in kwargs


# ---------------------------------------------------------------------------
# EE graph construction (mocked)
# ---------------------------------------------------------------------------
def test_task_table_uses_explicit_scale_and_crs(registry: Registry) -> None:
    """Never rely on Earth Engine's sampling defaults."""
    ee = EEStub()
    source = registry.source("csp_ghm")
    handler = handler_for(source, registry.export, ee)
    build_task_table(handler, registry.export, make_keys(3), zone=0)

    sampled = ee.calls_named("sampleRegions()")
    assert len(sampled) == 1
    _, _, kwargs = sampled[0]
    assert kwargs["scale"] == source.scale_m
    assert kwargs["projection"] == "EPSG:4326"
    assert kwargs["geometries"] is False


def test_task_table_never_calls_getinfo_or_samples_large_objects(registry: Registry) -> None:
    """The whole design rests on batch exports: no getInfo() on a collection."""
    ee = EEStub()
    handler = handler_for(registry.source("era5_land"), registry.export, ee)
    build_task_table(handler, registry.export, make_keys(4), zone=0)
    assert not ee.calls_named("getInfo()")
    assert not [c for c in ee.calls if c[0].endswith("aggregate_array()")]


def test_task_table_rejects_a_chunk_beyond_the_payload_bound(registry: Registry) -> None:
    ee = EEStub()
    source = registry.source("csp_ghm")
    export = dataclasses.replace(registry.export, max_points_per_task=2)
    handler = handler_for(source, export, ee)
    with pytest.raises(RuntimeError, match="max_points_per_task"):
        build_task_table(handler, export, make_keys(3), zone=0)


def test_points_are_zipped_arrays_not_per_point_features(registry: Registry) -> None:
    """Regression: one ee.Feature per point blew the 10 MiB request limit.

    `point_collection` must zip parallel coordinate/key arrays and expand them
    server-side. A per-point `ee.Feature` costs ~291 serialized bytes versus ~36
    zipped, so ~94k points went out as 26 MiB and Earth Engine rejected it.
    """
    ee = EEStub()
    source = registry.source("csp_ghm")
    handler = handler_for(source, registry.export, ee)
    build_task_table(handler, registry.export, make_keys(4), zone=0)

    assert ee.calls_named("zip()"), "points must be sent as zipped coordinate/key arrays"
    assert ee.calls_named("ee.List()"), "coordinates and keys must be passed as ee.List"
    # The stub never invokes the map callback, so any recorded Feature construction
    # would mean a per-point list is being built client-side again.
    assert not ee.calls_named("ee.Feature()"), (
        "per-point ee.Feature objects blow the request payload limit"
    )


def test_point_collection_pairs_coordinates_with_keys(registry: Registry) -> None:
    """Each (lon, lat) pair must be zipped with its own key_index."""
    ee = EEStub()
    keys = make_keys(3)
    point_collection(ee, keys)

    listed = [c for c in ee.calls_named("ee.List()")]
    coordinates = listed[0][1][0]
    key_values = listed[1][1][0]
    assert coordinates == [
        [float(keys["px_lon"][i]), float(keys["px_lat"][i])] for i in range(3)
    ]
    assert key_values == keys["key_index"].tolist()


def test_buffer_eligibility_follows_native_scale(registry: Registry) -> None:
    """>=30 m gets point + 30 m buffer; an 11 km source gets no buffer."""
    ee = EEStub()
    export = registry.export
    assert handler_for(registry.source("usgs_3dep"), export, ee).buffer_eligible
    assert not handler_for(registry.source("era5_land"), export, ee).buffer_eligible
    assert not handler_for(registry.source("modis_lst"), export, ee).buffer_eligible


def test_buffer_bands_split_continuous_from_categorical(registry: Registry) -> None:
    """Continuous -> mean, categorical -> mode."""
    ee = EEStub()
    continuous, categorical = handler_for(registry.source("nass_cdl"), registry.export, ee).buffer_bands()
    assert set(categorical) == {"cropland", "cultivated"}
    assert continuous == ["confidence"]


def test_task_selectors_match_buffer_eligibility(registry: Registry) -> None:
    """The projection must name exactly the columns the graph produces."""
    export = registry.export
    # 10 m source that declares a buffer reducer -> point + buffer for every band.
    source = registry.source("eth_canopy")
    assert set(task_selectors(source, export)) == {
        *source.sampled_bands, *(f"{b}_buf30" for b in source.sampled_bands)}
    # Too coarse a pixel: a 30 m buffer inside it returns the pixel itself.
    assert all("_buf30" not in s for s in task_selectors(registry.source("csp_ghm"), export))
    assert all("_buf30" not in s for s in task_selectors(registry.source("modis_lst"), export))
    # No buffer reducer declared.
    assert all("_buf30" not in s for s in task_selectors(registry.source("era5_land"), export))
    # n_scenes only for composite sources.
    assert "n_scenes" in task_selectors(registry.source("sentinel2"), export)
    assert "n_scenes" not in task_selectors(source, export)


def test_handler_builds_a_graph_for_every_cadence(registry: Registry) -> None:
    """Every declared source must produce an image without raising (stubbed)."""
    ref = SamplingRef(timestamp=pd.Timestamp("2020-06-15T13:00:00Z"))
    for name, source in registry.sources.items():
        ee = EEStub()
        handler = handler_for(source, registry.export, ee)
        handler.image(ref)
        assert ee.calls, f"{name} built an empty graph"


def _renamed_to(ee: EEStub, name: str) -> bool:
    return any(name in (c[1][0] if c[1] and isinstance(c[1][0], list) else list(c[1]))
               for c in ee.calls_named("rename()"))


def test_composite_sources_carry_a_per_pixel_scene_count(registry: Registry) -> None:
    """n_scenes counts valid observations per pixel (count), not scenes in the
    footprint (size) — the latter is one number for every point in the chunk."""
    ee = EEStub()
    handler = handler_for(registry.source("sentinel2"), registry.export, ee)
    handler.image(SamplingRef(timestamp=pd.Timestamp("2020-06-01T00:00:00Z")))
    assert _renamed_to(ee, "n_scenes")
    assert ee.calls_named("count()")
    assert not ee.calls_named("size()")
    # A non-composite source must not emit n_scenes.
    ee2 = EEStub()
    handler_for(registry.source("csp_ghm"), registry.export, ee2).image(
        SamplingRef(timestamp=pd.Timestamp("2020-06-01T00:00:00Z"))
    )
    assert not _renamed_to(ee2, "n_scenes")


# ---------------------------------------------------------------------------
# Graph-construction regressions (2026-09-23 fixes)
# ---------------------------------------------------------------------------
_REF = pd.Timestamp("2020-06-15T12:00:00Z")


def test_single_image_assets_are_loaded_as_images(registry: Registry) -> None:
    """ee.ImageCollection(<Image asset id>) fails server-side; load it as ee.Image."""
    ee = EEStub()
    source = registry.source("eth_canopy")
    handler_for(source, registry.export, ee).image(SamplingRef(timestamp=_REF))
    assert any(c[1] == (source.asset,) for c in ee.calls_named("ee.Image()"))
    assert not any(c[1] == (source.asset,) for c in ee.calls_named("ee.ImageCollection()"))


def test_tiled_static_collections_are_mosaicked_not_firsted(registry: Registry) -> None:
    """3DEP is 1x1 degree tiles: first() keeps one tile and masks the rest."""
    ee = EEStub()
    handler_for(registry.source("usgs_3dep"), registry.export, ee).image(SamplingRef(timestamp=_REF))
    assert ee.calls_named("mosaic()")
    assert not ee.calls_named("first()")
    # The reduction loses its projection; terrain/focal ops need one restored.
    assert ee.calls_named("setDefaultProjection()")


def test_task_table_bounds_every_collection_to_the_chunk_footprint(registry: Registry) -> None:
    """Scenes *and* the s2cloudless join partner are filterBounds()'d."""
    ee = EEStub()
    handler = handler_for(registry.source("sentinel2"), registry.export, ee)
    build_task_table(handler, registry.export, make_keys(4, bucket="c1", t0=1), zone=32612)
    rects = ee.calls_named("ee.Geometry.Rectangle()")
    assert len(rects) == 1
    west, south, east, north = rects[0][1][0]
    pad = registry.export.footprint_pad_deg
    assert west == pytest.approx(-100.0 - pad) and north == pytest.approx(40.003 + pad)
    # current window: scenes + probabilities; previous window (ndvi_rate): same again.
    assert len(ee.calls_named("filterBounds()")) >= 4
    probability_colls = [c for c in ee.calls_named("ee.ImageCollection()")
                         if c[1] == ("COPERNICUS/S2_CLOUD_PROBABILITY",)]
    assert probability_colls


def test_empty_windows_yield_masked_bands_not_errors(registry: Registry) -> None:
    """A window with no scenes must not break `select` for the whole task."""
    ee = EEStub()
    handler_for(registry.source("sentinel2"), registry.export, ee).image(SamplingRef(timestamp=_REF))
    placeholders = [c for c in ee.calls_named("ee.Image.constant()")
                    if c[1] and isinstance(c[1][0], list)]
    assert placeholders, "composite must merge a masked placeholder carrying its bands"
    assert ee.calls_named("merge()")
    assert ee.calls_named("median()")


def test_sentinel2_registry_fixes(registry: Registry) -> None:
    s2 = registry.source("sentinel2")
    bands = s2.sampled_bands
    # `irg` was a verbatim copy of the NDVI formula.
    assert "irg" not in bands
    exprs = [d["expr"] for d in s2.derived]
    assert len(exprs) == len(set(exprs)), "two derived bands share one expression"
    assert "ndvi_rate" in bands and "snow_fraction" in bands
    # Snow is kept (and reported as a fraction), not masked as if it were cloud.
    assert 11 not in s2.qa.params["scl_classes"]
    # SCL is not sampled, so declaring it categorical was dead config.
    assert "SCL" not in s2.categorical


def test_temporal_rate_reads_the_previous_window(registry: Registry) -> None:
    """ndvi_rate compares against the composite one window (16 days) earlier."""
    ee = EEStub()
    s2 = registry.source("sentinel2")
    handler_for(s2, registry.export, ee).image(SamplingRef(timestamp=_REF))
    dates = {c[1][0] for c in ee.calls_named("ee.Date()") if c[1]}
    lag = 2 * s2.composite_days
    assert _REF.date().isoformat() in dates
    assert (_REF - pd.Timedelta(days=lag)).date().isoformat() in dates
    assert ee.calls_named("subtract()") and ee.calls_named("divide()")


def _registry_with(tmp_path: Path, name: str, patch: dict) -> Path:
    import yaml
    data = yaml.safe_load(S.DEFAULT_SOURCES.read_text(encoding="utf-8"))
    data["sources"][name].update(patch)
    out = tmp_path / "sources.yaml"
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


def test_registry_rejects_temporal_rates_on_a_non_composite_source(tmp_path: Path) -> None:
    path = _registry_with(tmp_path, "modis_snow",
                          {"temporal_rates": [{"name": "r", "band": "NDSI_Snow_Cover"}]})
    with pytest.raises(ValueError, match="composite"):
        load_registry(path)


def test_registry_rejects_a_rate_of_an_unknown_band(tmp_path: Path) -> None:
    path = _registry_with(tmp_path, "sentinel2",
                          {"temporal_rates": [{"name": "r", "band": "nope"}]})
    with pytest.raises(ValueError, match="neither a native nor a derived band"):
        load_registry(path)


def test_registry_rejects_an_unknown_kind(tmp_path: Path) -> None:
    path = _registry_with(tmp_path, "csp_ghm", {"kind": "table"})
    with pytest.raises(ValueError, match="unknown kind"):
        load_registry(path)


def test_monthly_scene_counts_is_one_list_without_getinfo(registry: Registry) -> None:
    ee = EEStub()
    S.monthly_scene_counts(ee, registry.source("sentinel2"), [-110, 41, -109, 43],
                           ["2018-01", "2018-02", "2018-03"])
    assert len(ee.calls_named("size()")) == 3
    assert not ee.calls_named("getInfo()")


def test_ee_module_error_is_actionable_when_absent() -> None:
    if importlib.util.find_spec("ee") is not None:
        pytest.skip("earthengine-api is installed; the missing-extra path is not reachable")
    with pytest.raises(RuntimeError, match="uv sync --extra gee"):
        ee_module()


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------
def test_registry_loads_the_shipped_inventory(registry: Registry) -> None:
    assert len(registry.datasets) == 8
    assert "african_elephant" not in registry.datasets
    assert len(registry.sources) == 16
    for name, source in registry.sources.items():
        assert source.cadence in S.CADENCES, name
        assert source.chunk_bucket in S.BUCKETS, name
        assert source.sampled_bands, f"{name} declares no sampled bands"


def test_every_declared_mechanism_is_registered(registry: Registry) -> None:
    """The YAML may only reference QA kinds and ops that exist in a registry.

    This is what makes "adding a source is a YAML edit" safe: a typo in a QA kind
    or an op name fails here rather than at submission time.
    """
    for name, source in registry.sources.items():
        assert source.qa.kind in S.QA_MASKS, f"{name}: unknown qa.kind {source.qa.kind!r}"
        assert source.qa.kind in S.QA_COLLECTION_HOOKS, name
        assert source.cadence in S.CADENCE_FILTERS, name
        assert source.cadence in S.CADENCE_HANDLERS, name
        for spec in source.focal:
            assert spec["op"] in S.FOCAL_OPS, f"{name}: unknown focal op {spec['op']!r}"
        for spec in source.post:
            assert spec["op"] in S.POST_OPS, f"{name}: unknown post op {spec['op']!r}"
        for op in source.terrain:
            assert op in S.TERRAIN_OPS, f"{name}: unknown terrain op {op!r}"


def test_out_of_scope_dataset_fails_loudly(registry: Registry) -> None:
    with pytest.raises(ValueError, match="out of scope"):
        registry.dataset("african_elephant")
    with pytest.raises(KeyError):
        registry.dataset("nope")


def test_selectors_support_tiers_and_mixing(registry: Registry) -> None:
    assert set(registry.select_sources("tier1")) == {
        "dynamic_world", "sentinel2", "sentinel1", "eth_canopy", "usgs_3dep"}
    assert "era5_land" in registry.select_sources("tier1,era5_land")
    assert registry.select_datasets("wolf,mule_deer") == ["wolf", "mule_deer"]
    with pytest.raises(KeyError):
        registry.select_sources("not_a_source")
    with pytest.raises(KeyError):
        registry.select_sources("tier1,nope")


def test_black_bear_registry_declares_the_duplicate_column(registry: Registry) -> None:
    assert registry.dataset("black_bear").duplicate_columns == ("individual_id",)
    assert registry.dataset("black_bear").csv == "black_bear_reshaped.csv"


def test_shipped_csv_names_match_the_registry(registry: Registry) -> None:
    """Every registered dataset points at a *_reshaped.csv in the raw dir."""
    for name, dataset in registry.datasets.items():
        assert dataset.csv.endswith("_reshaped.csv"), name
        assert dataset.fixes > 0, name
    assert sum(d.fixes for d in registry.datasets.values()) == 1_992_383


def test_registry_rejects_a_reducer_list_without_first(registry: Registry, tmp_path: Path) -> None:
    """The point value must always be a first-at-native-scale sample."""
    import yaml

    data = yaml.safe_load(S.DEFAULT_SOURCES.read_text(encoding="utf-8"))
    data["sources"]["csp_ghm"]["reducers"] = ["mean"]
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="must include 'first'"):
        load_registry(path)


def test_registry_path_is_configurable(registry: Registry) -> None:
    """The CLI's --registry flag must be able to point anywhere."""
    assert load_registry(S.DEFAULT_SOURCES).datasets.keys() == registry.datasets.keys()
