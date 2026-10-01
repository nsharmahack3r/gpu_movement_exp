"""Covariate source registry.

``configs/covariates/sources.yaml`` is the single source of truth for asset IDs,
bands, scales, cadences, pixel grids and QA specs. This module turns that file
into typed objects and provides the *mechanisms* the YAML refers to, each held in
a name -> callable registry so adding a source, a QA kind or an image op is a
YAML edit plus (at most) one registry entry — never an ``if source == ...`` chain.

Nothing here imports ``earthengine-api`` at module scope: ``plan`` and ``dedup``
must run with no Earth Engine present and with no network (acceptance criteria 1
and 5). The ``ee`` module is resolved lazily by :func:`ee_module`, so tests can
inject a stub into ``sys.modules``.

The EE graph builders are deliberately structural: they are exercised by mocked
tests, not by live Earth Engine (see the delivery notes' pushback section).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Timestamp anchors used by the temporal-alignment conventions.
_EPOCH = pd.Timestamp("1970-01-01T00:00:00Z")
_HALF_DAY = pd.Timedelta(hours=12)


def _DAYS(n: int) -> pd.Timedelta:
    return pd.Timedelta(days=int(n))


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCES = REPO_ROOT / "configs" / "covariates" / "sources.yaml"

CADENCES = ("static", "annual", "monthly", "daily", "subdaily", "composite")
BUCKETS = ("static", "year", "month", "day", "hour", "composite")
# Export destinations. Earth Engine computes remotely and cannot write to a local
# path, so a batch export must land in one of Google's own stores.
DESTINATIONS = ("drive", "cloud_storage")

# Datasets that exist on disk but are explicitly out of scope. Asking for one
# must fail loudly rather than yield partial coverage (prompts/gee_export.md).
OUT_OF_SCOPE_DATASETS = {"african_elephant"}

# Band carrying the number of contributing scenes for composite sources.
SCENES_BAND = "n_scenes"
# Column suffix for the 30 m buffer reducer.
BUFFER_SUFFIX = "_buf30"


# ---------------------------------------------------------------------------
# Lazy ee import
# ---------------------------------------------------------------------------
def ee_module() -> Any:
    """Return the ``ee`` module, or fail loudly with an actionable message."""
    try:
        import ee  # noqa: PLC0415 — deliberate lazy import
    except ImportError as exc:  # pragma: no cover - depends on install state
        raise RuntimeError(
            "earthengine-api is not installed. The `plan` and `dedup` commands do "
            "not need it; `submit`, `poll` and `join` do. Install it with:\n"
            "    uv sync --extra gee"
        ) from exc
    return ee


# ---------------------------------------------------------------------------
# Typed registry objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Grid:
    """The pixel lattice a source's values are indexed by.

    ``mode="fixed"`` uses one CRS-wide lattice (``crs`` + ``origin``).
    ``mode="utm"`` resolves the lattice per fix from its longitude (UTM zone,
    anchored at the zone's false origin) — the native grid of the Sentinel
    products, which are tiled in UTM.
    """

    mode: str
    res_x: float
    res_y: float
    crs: str | None = None
    origin_x: float = 0.0
    origin_y: float = 0.0
    verified: bool = False


@dataclass(frozen=True)
class QASpec:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Source:
    """One covariate source, resolved from the YAML registry entry."""

    name: str
    asset: str
    kind: str  # image | image_collection
    bands: tuple[str, ...]
    cadence: str
    grid: Grid
    qa: QASpec
    tier: int = 3
    description: str = ""
    scale_m: float = 10.0
    composite_days: int = 0
    last_year: int | None = None
    scale_factors: dict[str, float] = field(default_factory=dict)
    derived: tuple[dict[str, str], ...] = ()
    focal: tuple[dict[str, Any], ...] = ()
    post: tuple[dict[str, Any], ...] = ()
    terrain: tuple[str, ...] = ()
    reducers: tuple[str, ...] = ("first",)
    categorical: tuple[str, ...] = ()
    chunk_bucket: str = "static"
    extra_images: dict[str, str] = field(default_factory=dict)
    extra_collections: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Per-pixel fraction of valid (unmasked) observations in the cadence window
    # whose class band falls in `classes` — e.g. Sentinel-2 snow from SCL == 11.
    class_fractions: tuple[dict[str, Any], ...] = ()
    # Finite-difference rate of a band between this cadence window and the one
    # `lag_windows` earlier, in band units per day (composite cadence only).
    temporal_rates: tuple[dict[str, Any], ...] = ()

    @property
    def time_varying(self) -> bool:
        return self.cadence != "static"

    @property
    def sampled_bands(self) -> tuple[str, ...]:
        """Covariate columns this source contributes, in output order."""
        names = list(self.bands)
        names += [d["name"] for d in self.derived]
        names += [d["name"] for d in self.focal]
        names += [t for t in self.terrain if t != "hillshade"]
        names += [p["name"] for p in self.post if "name" in p]
        names += list(self.extra_images)
        for prefix, spec in self.extra_collections.items():
            names += [f"{prefix}_{b}" for b in spec.get("bands", [])]
        names += [c["name"] for c in self.class_fractions]
        names += [r["name"] for r in self.temporal_rates]
        return tuple(names)

    def resolved_spec(self) -> dict[str, Any]:
        """Canonical, hashable spec — the input to a chunk's identity.

        Deliberately built from the *semantic* fields only: editing a source's
        ``description`` or ``tier`` must not invalidate exported chunks, while
        changing its grid, bands, cadence, QA or ops must.
        """
        return {
            "asset": self.asset,
            "kind": self.kind,
            "bands": list(self.bands),
            "cadence": self.cadence,
            "composite_days": self.composite_days,
            "last_year": self.last_year,
            "scale_m": self.scale_m,
            "scale_factors": self.scale_factors,
            "derived": [dict(d) for d in self.derived],
            "focal": [dict(d) for d in self.focal],
            "post": [dict(d) for d in self.post],
            "terrain": list(self.terrain),
            "reducers": list(self.reducers),
            "categorical": list(self.categorical),
            "chunk_bucket": self.chunk_bucket,
            "extra_images": self.extra_images,
            "extra_collections": self.extra_collections,
            "class_fractions": [dict(c) for c in self.class_fractions],
            "temporal_rates": [dict(r) for r in self.temporal_rates],
            "grid": {
                "mode": self.grid.mode, "crs": self.grid.crs,
                "res_x": self.grid.res_x, "res_y": self.grid.res_y,
                "origin_x": self.grid.origin_x, "origin_y": self.grid.origin_y,
            },
            "qa": {"kind": self.qa.kind, "params": self.qa.params},
        }


@dataclass(frozen=True)
class Dataset:
    name: str
    csv: str
    fixes: int
    duplicate_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExportConfig:
    destination: str
    target_features_per_task: int
    max_tasks_in_queue: int
    max_time_groups_per_task: int
    max_ready_tasks: int
    poll_initial_seconds: float
    poll_max_seconds: float
    poll_backoff: float
    retry_max_attempts: int
    retry_backoff_seconds: float
    retry_jitter_seconds: float
    file_format: str
    gcs_prefix: str
    results_root: Path
    buffer_radius_m: float
    buffer_max_scale_m: float
    min_pixel_reduction_ratio: float
    reduction_required_above_scale_m: float
    warn_unique_ratio_above: float
    max_points_per_task: int
    footprint_pad_deg: float = 0.02


@dataclass(frozen=True)
class Registry:
    export: ExportConfig
    datasets: dict[str, Dataset]
    sources: dict[str, Source]

    def dataset(self, name: str) -> Dataset:
        """Resolve a dataset by name, refusing out-of-scope datasets loudly."""
        if name in OUT_OF_SCOPE_DATASETS:
            raise ValueError(
                f"Dataset {name!r} is out of scope for this tool "
                f"(prompts/gee_export.md: it is not CONUS and precedes the "
                f"Sentinel-2 era). It is deliberately absent from the registry — "
                f"no partial coverage will be produced."
            )
        if name not in self.datasets:
            raise KeyError(
                f"Unknown dataset {name!r}. Known: {sorted(self.datasets)}"
            )
        return self.datasets[name]

    def source(self, name: str) -> Source:
        if name not in self.sources:
            raise KeyError(f"Unknown source {name!r}. Known: {sorted(self.sources)}")
        return self.sources[name]

    def select_datasets(self, spec: str) -> list[str]:
        return _select(spec, self.datasets, "dataset")

    def select_sources(self, spec: str) -> list[str]:
        return _select(spec, self.sources, "source")


def _select(spec: str, known: dict[str, Any], what: str) -> list[str]:
    """Resolve ``all`` / ``tierN`` / names, freely mixed in a comma-separated list.

    Tokens union in order (``tier1,era5_land`` = tier 1 plus ERA5-Land); the
    result is de-duplicated. Unknown tokens fail loudly.
    """
    tokens = [t.strip() for t in (spec or "all").split(",") if t.strip()]
    if not tokens:
        tokens = ["all"]
    picked: list[str] = []
    for token in tokens:
        if token == "all":
            picked += sorted(known)
        elif token.startswith("tier") and len(token) > 4 and token[4:].isdigit():
            tier = int(token[4:])
            matches = sorted(n for n, s in known.items() if getattr(s, "tier", None) == tier)
            if not matches:
                raise ValueError(f"No {what}s in {token!r}")
            picked += matches
        elif token in known:
            picked.append(token)
        else:
            raise KeyError(f"Unknown {what} {token!r}. Known: {sorted(known)}")
    return list(dict.fromkeys(picked))


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------
def load_registry(path: Path | str = DEFAULT_SOURCES) -> Registry:
    """Load and validate the source registry YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Covariate registry not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for key in ("export", "datasets", "sources"):
        if key not in data:
            raise ValueError(f"Registry {path} is missing required key: {key}")

    export = _build_export(data["export"])
    datasets = {
        name: Dataset(
            name=name,
            csv=str(spec["csv"]),
            fixes=int(spec["fixes"]),
            duplicate_columns=tuple(spec.get("duplicate_columns", ())),
        )
        for name, spec in data["datasets"].items()
    }
    sources = {name: _build_source(name, spec) for name, spec in data["sources"].items()}
    return Registry(export=export, datasets=datasets, sources=sources)


def _destination(spec: dict[str, Any]) -> str:
    """Validate ``export.destination`` — a config choice, never an if/elif on names."""
    destination = str(spec.get("destination", "cloud_storage"))
    if destination not in DESTINATIONS:
        raise ValueError(
            f"export.destination {destination!r} is not one of {DESTINATIONS}"
        )
    return destination


def _build_export(spec: dict[str, Any]) -> ExportConfig:
    root = Path(spec["results_root"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    return ExportConfig(
        destination=_destination(spec),
        target_features_per_task=int(spec["target_features_per_task"]),
        max_tasks_in_queue=int(spec["max_tasks_in_queue"]),
        max_time_groups_per_task=int(spec["max_time_groups_per_task"]),
        max_ready_tasks=int(spec["max_ready_tasks"]),
        poll_initial_seconds=float(spec["poll_initial_seconds"]),
        poll_max_seconds=float(spec["poll_max_seconds"]),
        poll_backoff=float(spec["poll_backoff"]),
        retry_max_attempts=int(spec["retry_max_attempts"]),
        retry_backoff_seconds=float(spec["retry_backoff_seconds"]),
        retry_jitter_seconds=float(spec["retry_jitter_seconds"]),
        file_format=str(spec["file_format"]),
        gcs_prefix=str(spec["gcs_prefix"]),
        results_root=root,
        buffer_radius_m=float(spec["buffer_radius_m"]),
        buffer_max_scale_m=float(spec["buffer_max_scale_m"]),
        min_pixel_reduction_ratio=float(spec["min_pixel_reduction_ratio"]),
        reduction_required_above_scale_m=float(spec["reduction_required_above_scale_m"]),
        warn_unique_ratio_above=float(spec["warn_unique_ratio_above"]),
        max_points_per_task=int(spec["max_points_per_task"]),
        footprint_pad_deg=float(spec.get("footprint_pad_deg", 0.02)),
    )


def _build_source(name: str, spec: dict[str, Any]) -> Source:
    cadence = str(spec["cadence"])
    if cadence not in CADENCES:
        raise ValueError(f"Source {name!r}: unknown cadence {cadence!r}; expected one of {CADENCES}")
    bucket = str(spec.get("chunk_bucket", "static"))
    if bucket not in BUCKETS:
        raise ValueError(f"Source {name!r}: unknown chunk_bucket {bucket!r}; expected one of {BUCKETS}")
    grid_spec = spec["grid"]
    res = grid_spec["res"]
    origin = grid_spec.get("origin", [0.0, 0.0])
    grid = Grid(
        mode=str(grid_spec["mode"]),
        crs=grid_spec.get("crs"),
        res_x=float(res[0]),
        res_y=float(res[1]),
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        verified=bool(grid_spec.get("verified", False)),
    )
    if grid.mode not in ("fixed", "utm"):
        raise ValueError(f"Source {name!r}: grid.mode must be 'fixed' or 'utm'")
    if grid.mode == "fixed" and not grid.crs:
        raise ValueError(f"Source {name!r}: grid.mode 'fixed' requires grid.crs")
    if cadence == "composite" and not spec.get("composite_days"):
        raise ValueError(f"Source {name!r}: composite cadence requires composite_days")
    if cadence == "annual" and not spec.get("last_year"):
        raise ValueError(f"Source {name!r}: annual cadence requires last_year (the clamp target)")
    reducers = tuple(spec.get("reducers", ("first",)))
    # The point value is always a first-at-native-scale sample; a reducer list that
    # drops it would silently change what the column means.
    if "first" not in reducers:
        raise ValueError(
            f"Source {name!r}: reducers must include 'first' (the native-scale point "
            f"value); got {list(reducers)}"
        )
    unknown_reducers = set(reducers) - {"first", "mean", "mode"}
    if unknown_reducers:
        raise ValueError(f"Source {name!r}: unknown reducer(s) {sorted(unknown_reducers)}")
    kind = str(spec["kind"])
    if kind not in PRIMARY_LOADERS:
        raise ValueError(f"Source {name!r}: unknown kind {kind!r}; expected one of {sorted(PRIMARY_LOADERS)}")

    class_fractions = tuple(dict(c) for c in spec.get("class_fractions", ()))
    for c in class_fractions:
        missing = {"name", "band", "classes"} - set(c)
        if missing:
            raise ValueError(f"Source {name!r}: class_fractions entry missing {sorted(missing)}")
    temporal_rates = tuple(dict(r) for r in spec.get("temporal_rates", ()))
    if temporal_rates and cadence != "composite":
        raise ValueError(
            f"Source {name!r}: temporal_rates need a composite cadence (the lag is "
            f"measured in composite windows); got {cadence!r}"
        )
    rateable = set(spec.get("bands", ())) | {d["name"] for d in spec.get("derived", ())}
    for r in temporal_rates:
        if r.get("band") not in rateable:
            raise ValueError(
                f"Source {name!r}: temporal rate {r.get('name')!r} refers to band "
                f"{r.get('band')!r}, which is neither a native nor a derived band"
            )
        if int(r.get("lag_windows", 1)) < 1:
            raise ValueError(f"Source {name!r}: temporal rate lag_windows must be >= 1")

    return Source(
        name=name,
        asset=str(spec["asset"]),
        kind=kind,
        bands=tuple(spec.get("bands", ())),
        cadence=cadence,
        grid=grid,
        qa=QASpec(kind=str(spec.get("qa", {}).get("kind", "none")),
                  params=dict(spec.get("qa", {}).get("params", {}))),
        tier=int(spec.get("tier", 3)),
        description=str(spec.get("description", "")),
        scale_m=float(spec.get("scale_m", 10.0)),
        composite_days=int(spec.get("composite_days", 0)),
        last_year=spec.get("last_year"),
        scale_factors={k: float(v) for k, v in spec.get("scale_factors", {}).items()},
        derived=tuple(spec.get("derived", ())),
        focal=tuple(spec.get("focal", ())),
        post=tuple(spec.get("post", ())),
        terrain=tuple(spec.get("terrain", ())),
        reducers=reducers,
        categorical=tuple(spec.get("categorical", ())),
        chunk_bucket=bucket,
        extra_images=dict(spec.get("extra_images", {})),
        extra_collections=dict(spec.get("extra_collections", {})),
        class_fractions=class_fractions,
        temporal_rates=temporal_rates,
    )


# ---------------------------------------------------------------------------
# Regenerable climatology helper (bit arithmetic lives in ONE tested helper)
# ---------------------------------------------------------------------------
def extract_bits(value: int, start: int, end: int) -> int:
    """Extract the inclusive bit range ``[start, end]`` from an integer.

    This is the single place the QA bit arithmetic is defined; every QA mask
    calls it (directly for local values, via ``_ee_bits`` for EE images).
    """
    if start < 0 or end < start:
        raise ValueError(f"bad bit range [{start}, {end}]")
    width = end - start + 1
    return (int(value) >> start) & ((1 << width) - 1)


def _ee_bits(ee: Any, image: Any, band: str, start: int, end: int) -> Any:
    """EE equivalent of :func:`extract_bits`, for an image band."""
    width = end - start + 1
    return image.select(band).rightShift(start).bitwiseAnd((1 << width) - 1)


# ---------------------------------------------------------------------------
# EE registries — every dispatch below is a dict lookup, never an if/elif chain
# ---------------------------------------------------------------------------
def _merge_images(images: list[Any]) -> Any:
    """Band-wise merge of several single-image composites (first wins per band)."""
    out = images[0]
    for img in images[1:]:
        out = out.addBands(img, overwrite=False)
    return out


def _tpi(ee: Any, band_img: Any, radius_px: int) -> Any:
    kernel = ee.Kernel.circle(radius=radius_px, units="pixels")
    mean = band_img.reduceNeighborhood(ee.Reducer.mean(), kernel)
    return band_img.subtract(mean)


def _tri(ee: Any, band_img: Any, radius_px: int) -> Any:
    kernel = ee.Kernel.circle(radius=radius_px, units="pixels")
    dev = band_img.subtract(band_img.reduceNeighborhood(ee.Reducer.mean(), kernel)).abs()
    return dev.reduceNeighborhood(ee.Reducer.mean(), kernel)


def _roughness(ee: Any, band_img: Any, radius_px: int) -> Any:
    kernel = ee.Kernel.circle(radius=radius_px, units="pixels")
    return (
        band_img.reduceNeighborhood(ee.Reducer.max(), kernel)
        .subtract(band_img.reduceNeighborhood(ee.Reducer.min(), kernel))
    )


def _focal_mean(ee: Any, band_img: Any, radius_px: int) -> Any:
    return band_img.reduceNeighborhood(
        ee.Reducer.mean(), ee.Kernel.circle(radius=radius_px, units="pixels")
    )


FOCAL_OPS: dict[str, Callable[[Any, Any, int], Any]] = {
    "tpi": _tpi,
    "tri": _tri,
    "roughness": _roughness,
    "mean": _focal_mean,
}

TERRAIN_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "slope": lambda ee, img: ee.Terrain.slope(img),
    "aspect": lambda ee, img: ee.Terrain.aspect(img),
    "hillshade": lambda ee, img: ee.Terrain.hillshade(img),
}


def _op_distance_to_water(ee: Any, img: Any, params: dict[str, Any], source: Source, ref: Any) -> Any:
    """Distance (m) to the nearest pixel above an occurrence threshold."""
    band = params["source_band"]
    threshold = float(params["threshold"])
    water = img.select(band).gte(threshold).rename("water")
    inverse = water.Not()
    # fastDistanceTransform returns squared pixel distance; sqrt + pixelArea
    # sqrt converts it to metres at the image's own projection.
    dist_px = inverse.fastDistanceTransform(1024).sqrt()
    return dist_px.multiply(ee.Image.pixelArea().sqrt()).rename(params.get("name", "distance_to_water"))


def _op_days_since_burn(ee: Any, img: Any, params: dict[str, Any], source: Source, ref: Any) -> Any:
    """Days between the reference date and the burn day-of-year (-1 if unburned)."""
    band = params["source_band"]
    ref_doy = int(ref.timestamp.dayofyear)
    burn = img.select(band)
    return (
        ee.Image(ref_doy).subtract(burn)
        .where(burn.lte(0), -1)
        .rename(params.get("name", "days_since_burn"))
    )


POST_OPS: dict[str, Callable[..., Any]] = {
    "distance_to_water": _op_distance_to_water,
    "days_since_burn": _op_days_since_burn,
}


# --- QA masks: image -> masked image ---------------------------------------
def _qa_none(ee: Any, img: Any, params: dict[str, Any], source: Source) -> Any:
    return img


def _qa_s2_scl(ee: Any, img: Any, params: dict[str, Any], source: Source) -> Any:
    """Mask Sentinel-2 by SCL class plus an s2cloudless probability threshold.

    SCL classes dropped: 0 no-data, 1 saturated/defective, 3 cloud shadow,
    8 cloud (medium), 9 cloud (high), 10 thin cirrus, 11 snow/ice. The
    cloud-probability threshold is 40 %, i.e. pixels whose s2cloudless
    probability exceeds 40 % are dropped.
    """
    classes = ee.List(params["scl_classes"])
    scl = img.select("SCL")
    keep = scl.remap(classes, ee.List([0] * len(params["scl_classes"])), 1)
    mask = keep.eq(1)
    cloud_prob = params.get("cloud_probability")
    if cloud_prob:
        band = cloud_prob["band"]
        prob_ok = img.select(band).unmask(100).lt(cloud_prob["threshold"])
        mask = mask.And(prob_ok)
    return img.updateMask(mask)


def _qa_modis_lst_qc(ee: Any, img: Any, params: dict[str, Any], source: Source) -> Any:
    """Mask MODIS LST on the mandatory-QA bits (0-1) of QC_Day and QC_Night."""
    limit = int(params.get("max_mandatory_qa", 0))
    day_ok = _ee_bits(ee, img, params["day_band"], 0, 1).lte(limit)
    night_ok = _ee_bits(ee, img, params["night_band"], 0, 1).lte(limit)
    return img.updateMask(day_ok.And(night_ok))


def _qa_ndsi_max_valid(ee: Any, img: Any, params: dict[str, Any], source: Source) -> Any:
    band = params["band"]
    return img.updateMask(img.select(band).lte(float(params["max_valid"])))


def _qa_fpar_lai_qc(ee: Any, img: Any, params: dict[str, Any], source: Source) -> Any:
    qc = _ee_bits(ee, img, params["band"], int(params["modland_qa_start"]), int(params["modland_qa_end"]))
    return img.updateMask(qc.lte(int(params["max_modland_qa"])))


def _qa_et_qc(ee: Any, img: Any, params: dict[str, Any], source: Source) -> Any:
    qc = _ee_bits(ee, img, params["band"], int(params["qa_start"]), int(params["qa_end"]))
    return img.updateMask(qc.lte(int(params["max_qa"])))


def _qa_viirs_cf_cvg(ee: Any, img: Any, params: dict[str, Any], source: Source) -> Any:
    return img.updateMask(img.select(params["band"]).gt(int(params["max_observations"])))


QA_MASKS: dict[str, Callable[..., Any]] = {
    "none": _qa_none,
    "s2_scl": _qa_s2_scl,
    "modis_lst_qc": _qa_modis_lst_qc,
    "ndsi_max_valid": _qa_ndsi_max_valid,
    "fpar_lai_qc": _qa_fpar_lai_qc,
    "et_qc": _qa_et_qc,
    "viirs_cf_cvg": _qa_viirs_cf_cvg,
    "s1_orbit": _qa_none,  # filtering happens at collection level
}


def _bounded(coll: Any, ref: Any) -> Any:
    """Restrict a collection to the images intersecting the chunk's footprint.

    Without this every date-filtered collection is global: medians, joins and
    scene counts then run over thousands of tiles nowhere near the animals.
    """
    region = getattr(ref, "region", None)
    return coll if region is None else coll.filterBounds(region)


# --- QA collection hooks: collection -> collection --------------------------
def _coll_none(ee: Any, coll: Any, params: dict[str, Any], source: Source, ref: Any) -> Any:
    return coll


def _coll_s1_orbit(ee: Any, coll: Any, params: dict[str, Any], source: Source, ref: Any) -> Any:
    for mode in params.get("instrument_modes", []):
        coll = coll.filter(ee.Filter.eq("instrumentMode", mode))
    for pol in params.get("polarizations", []):
        coll = coll.filter(ee.Filter.listContains("transmitterReceiverPolarisation", pol))
    return coll


def _coll_s2_cloud_probability(
    ee: Any, coll: Any, params: dict[str, Any], source: Source, ref: Any
) -> Any:
    """Join s2cloudless probability onto the S2 collection by scene index."""
    spec = params.get("cloud_probability")
    if not spec:
        return coll
    # Filter the probability collection by the same window and footprint as the
    # scenes. Joining against the unfiltered global collection makes the join a
    # metadata scan over every s2cloudless image ever produced.
    probs = CADENCE_FILTERS[source.cadence](ee, source, ee.ImageCollection(spec["asset"]), ref)
    probs = _bounded(probs, ref).select([spec["band"]])
    join = ee.Join.saveFirst(matchKey="_cloud_prob")
    condition = ee.Filter.equals(leftField="system:index", rightField="system:index")
    joined = join.apply(coll, probs, condition)

    def _attach(img: Any) -> Any:
        prob = ee.Image(img.get("_cloud_prob")).select([spec["band"]])
        return img.addBands(prob, overwrite=True)

    return ee.ImageCollection(joined).map(_attach)


QA_COLLECTION_HOOKS: dict[str, Callable[..., Any]] = {
    "none": _coll_none,
    "s2_scl": _coll_s2_cloud_probability,
    "s1_orbit": _coll_s1_orbit,
    "modis_lst_qc": _coll_none,
    "ndsi_max_valid": _coll_none,
    "fpar_lai_qc": _coll_none,
    "et_qc": _coll_none,
    "viirs_cf_cvg": _coll_none,
}


# --- Cadence filters: (ee, source, collection, ref) -> collection -----------
def _cadence_static(ee: Any, source: Source, coll: Any, ref: Any) -> Any:
    return coll


def _cadence_annual(ee: Any, source: Source, coll: Any, ref: Any) -> Any:
    year = min(int(ref.timestamp.year), int(source.last_year))
    start = ee.Date.fromYMD(year, 1, 1)
    return coll.filterDate(start, start.advance(1, "year"))


def _cadence_monthly(ee: Any, source: Source, coll: Any, ref: Any) -> Any:
    start = ee.Date.fromYMD(int(ref.timestamp.year), int(ref.timestamp.month), 1)
    return coll.filterDate(start, start.advance(1, "month"))


def _cadence_daily(ee: Any, source: Source, coll: Any, ref: Any) -> Any:
    start = ee.Date(str(ref.timestamp.date().isoformat()))
    return coll.filterDate(start, start.advance(1, "day"))


def _cadence_subdaily(ee: Any, source: Source, coll: Any, ref: Any) -> Any:
    start = ee.Date(str(ref.timestamp.floor("h").isoformat()))
    return coll.filterDate(start, start.advance(1, "hour"))


def _cadence_composite(ee: Any, source: Source, coll: Any, ref: Any) -> Any:
    days = int(source.composite_days)
    centre = ee.Date(str(ref.timestamp.date().isoformat()))
    return coll.filterDate(centre.advance(-days, "day"), centre.advance(days, "day"))


CADENCE_FILTERS: dict[str, Callable[..., Any]] = {
    "static": _cadence_static,
    "annual": _cadence_annual,
    "monthly": _cadence_monthly,
    "daily": _cadence_daily,
    "subdaily": _cadence_subdaily,
    "composite": _cadence_composite,
}


# ---------------------------------------------------------------------------
# Handlers — one per cadence class, resolved by dict lookup from the source
# ---------------------------------------------------------------------------
class SourceHandler:
    """Builds the masked, derived image a chunk's points are sampled from.

    Subclasses differ only in how the candidate collection is filtered, which is
    itself a registry lookup — so a new *source* is a YAML entry, and a new
    *cadence* is one filter function plus one registry entry.
    """

    cadence: str = "static"

    def __init__(self, source: Source, export: "ExportConfig", ee: Any) -> None:
        self.source = source
        self.export = export
        self.ee = ee

    # -- collection ---------------------------------------------------------
    def collection(self, ref: Any) -> Any:
        """Candidate images for a reference fix, filtered but not yet masked.

        Filtered by the cadence window *and* by the chunk footprint
        (``ref.region``) so nothing downstream touches off-site tiles.
        """
        ee, source = self.ee, self.source
        coll = ee.ImageCollection(source.asset)
        coll = _bounded(CADENCE_FILTERS[source.cadence](ee, source, coll, ref), ref)
        return QA_COLLECTION_HOOKS[source.qa.kind](
            ee, coll, source.qa.params, source, ref
        )

    def _own_coll(self, asset: str, ref: Any) -> Any:
        """Filter an *extra* collection with the same cadence rule and footprint."""
        ee, source = self.ee, self.source
        coll = CADENCE_FILTERS[source.cadence](ee, source, ee.ImageCollection(asset), ref)
        return _bounded(coll, ref)

    # -- image assembly -----------------------------------------------------
    def _masked(self, img: Any) -> Any:
        return QA_MASKS[self.source.qa.kind](self.ee, img, self.source.qa.params, self.source)

    def _compose(self, coll: Any, bands: list[str]) -> Any:
        """Reduce a filtered collection to a single image.

        Composite cadences take the cloud-masked median (the §5.3 policy).
        Static collections are *tiled* (3DEP ships one image per 1x1 degree tile),
        so they are mosaicked — ``first()`` would keep one arbitrary tile and mask
        everything else. Other cadences pin one observation per key, so ``first``
        is the exact match. Reductions lose the native projection (they default
        to 1-degree WGS84), which silently breaks neighbourhood and terrain ops;
        a fixed projection at the source's native scale is set as the default.

        An empty window (no scene in 16 days, a product gap) must yield *masked*
        bands, not an error: one failing ``select`` would fail the whole export
        task and every other key in it. A fully masked placeholder image carrying
        the requested bands guarantees they exist.
        """
        masked = coll.map(self._masked)
        return COMPOSERS[self.source.cadence](self, masked, list(bands))

    def _projection(self) -> Any:
        crs = self.source.grid.crs if self.source.grid.mode == "fixed" else "EPSG:4326"
        return self.ee.Projection(crs).atScale(self.source.scale_m)

    def _placeholder(self, bands: list[str]) -> Any:
        ee = self.ee
        return (
            ee.Image.constant([0] * len(bands)).rename(list(bands)).toFloat()
            .updateMask(ee.Image.constant(0))
        )

    def _padded(self, masked: Any, bands: list[str], *, placeholder_first: bool = False) -> Any:
        """``masked`` restricted to ``bands`` with a masked placeholder added."""
        ee = self.ee
        real = masked.map(lambda i, bands=list(bands): i.select(bands).toFloat())
        pad = ee.ImageCollection([self._placeholder(bands)])
        return pad.merge(real) if placeholder_first else real.merge(pad)

    def _primary(self, ref: Any) -> tuple[Any, Any]:
        """(composed primary image, masked collection or None for single images)."""
        return PRIMARY_LOADERS[self.source.kind](self, ref)

    def n_scenes(self, masked: Any) -> Any:
        """Per-pixel count of valid (unmasked) observations in the window.

        This is the number of scenes that actually contributed to *this pixel's*
        median. ``collection.size()`` would instead count every scene touching
        the footprint, cloudy or not, and give every point the same value.
        """
        band = self.source.bands[0]
        return (
            self._padded(masked, [band]).count().unmask(0)
            .rename(SCENES_BAND).toFloat()
        )

    def _base(self, ref: Any) -> tuple[Any, Any]:
        """Masked, composed, scaled image with derived bands; plus its collection."""
        ee, source = self.ee, self.source
        primary, masked = self._primary(ref)
        parts = [primary]

        for name, asset in source.extra_images.items():
            parts.append(ee.Image(asset).rename(name))
        for prefix, spec in source.extra_collections.items():
            bands = list(spec["bands"])
            composed = self._compose(self._own_coll(spec["asset"], ref), bands)
            parts.append(composed.select(bands).rename([f"{prefix}_{b}" for b in bands]))
        img = _merge_images(parts)

        # 1. scale to SI units.
        for band, factor in source.scale_factors.items():
            img = img.addBands(img.select(band).multiply(factor).rename(band), overwrite=True)

        # 2. derived expressions.
        for spec in source.derived:
            img = img.addBands(ee.Image(img.expression(spec["expr"])).rename(spec["name"]))
        return img, masked

    def image(self, ref: Any) -> Any:
        """The fully prepared image: masked, scaled, derived, focal, post-ops."""
        ee, source = self.ee, self.source
        img, masked = self._base(ref)

        # 3. terrain derivatives.
        for op in source.terrain:
            if op == "hillshade":
                continue  # display aid, not a covariate
            img = img.addBands(TERRAIN_OPS[op](ee, img.select("elevation")).rename(op))

        # 4. multi-window focal derivatives.
        for spec in source.focal:
            radius_px = max(1, int(round(spec["radius_m"] / source.scale_m)))
            value = FOCAL_OPS[spec["op"]](ee, img.select(spec["source_band"]), radius_px)
            img = img.addBands(value.rename(spec["name"]))

        # 5. named post ops.
        for spec in source.post:
            img = img.addBands(POST_OPS[spec["op"]](ee, img, spec, source, ref))

        # 6. class fractions over the window's valid observations.
        for spec in source.class_fractions:
            classes = [int(c) for c in spec["classes"]]
            band, name = spec["band"], spec["name"]
            frac = self._padded(
                masked.map(
                    lambda i, band=band, classes=classes, name=name: i.select([band])
                    .remap(classes, [1] * len(classes), 0)
                    .rename(name)
                ),
                [name],
            ).mean()
            img = img.addBands(frac)

        # 7. temporal rates against an earlier window (e.g. NDVI green-up).
        for spec in source.temporal_rates:
            lag_days = 2 * int(source.composite_days) * int(spec.get("lag_windows", 1))
            prev_ref = SamplingRef(
                timestamp=ref.timestamp - _DAYS(lag_days),
                region=getattr(ref, "region", None),
            )
            prev, _ = self._base(prev_ref)
            rate = (
                img.select(spec["band"]).subtract(prev.select(spec["band"]))
                .divide(float(lag_days))
                .rename(spec["name"])
            )
            img = img.addBands(rate)

        # 8. composite sources carry the per-pixel valid-observation count.
        if source.cadence == "composite":
            img = img.addBands(self.n_scenes(masked))

        return img

    # -- reducers -----------------------------------------------------------
    @property
    def buffer_eligible(self) -> bool:
        """Whether this source also gets a 30 m-radius buffer reducer."""
        return buffer_eligible(self.source, self.export)

    def buffer_bands(self) -> tuple[list[str], list[str]]:
        """(continuous, categorical) sampled bands — mean vs mode in the buffer."""
        return buffer_bands(self.source)


class StaticSourceHandler(SourceHandler):
    cadence = "static"
    kind = "image"


class AnnualSourceHandler(SourceHandler):
    cadence = "annual"


class MonthlySourceHandler(SourceHandler):
    cadence = "monthly"


class DailySourceHandler(SourceHandler):
    cadence = "daily"


class SubdailySourceHandler(SourceHandler):
    cadence = "subdaily"


class CompositeSourceHandler(SourceHandler):
    cadence = "composite"


CADENCE_HANDLERS: dict[str, type[SourceHandler]] = {
    "static": StaticSourceHandler,
    "annual": AnnualSourceHandler,
    "monthly": MonthlySourceHandler,
    "daily": DailySourceHandler,
    "subdaily": SubdailySourceHandler,
    "composite": CompositeSourceHandler,
}


def _compose_median(handler: SourceHandler, masked: Any, bands: list[str]) -> Any:
    return (
        handler._padded(masked, bands).median().rename(bands)
        .setDefaultProjection(handler._projection())
    )


def _compose_mosaic(handler: SourceHandler, masked: Any, bands: list[str]) -> Any:
    # mosaic() puts the *last* image on top: the placeholder goes first.
    return (
        handler._padded(masked, bands, placeholder_first=True).mosaic()
        .setDefaultProjection(handler._projection())
    )


def _compose_first(handler: SourceHandler, masked: Any, bands: list[str]) -> Any:
    # The placeholder is appended, so first() is a real image whenever one exists.
    return handler.ee.Image(handler._padded(masked, bands).first())


COMPOSERS: dict[str, Callable[[SourceHandler, Any, list[str]], Any]] = {
    "static": _compose_mosaic,
    "annual": _compose_first,
    "monthly": _compose_first,
    "daily": _compose_first,
    "subdaily": _compose_first,
    "composite": _compose_median,
}


def _load_image(handler: SourceHandler, ref: Any) -> tuple[Any, Any]:
    """A single ``ee.Image`` asset. ``ee.ImageCollection(<image id>)`` would fail."""
    return handler._masked(handler.ee.Image(handler.source.asset)), None


def _load_collection(handler: SourceHandler, ref: Any) -> tuple[Any, Any]:
    coll = handler.collection(ref)
    masked = coll.map(handler._masked)
    bands = list(handler.source.bands)
    return COMPOSERS[handler.source.cadence](handler, masked, bands), masked


PRIMARY_LOADERS: dict[str, Callable[[SourceHandler, Any], tuple[Any, Any]]] = {
    "image": _load_image,
    "image_collection": _load_collection,
}


def handler_for(source: Source, export: ExportConfig, ee: Any = None) -> SourceHandler:
    """Resolve the handler for a source (cadence class -> handler class)."""
    cls = CADENCE_HANDLERS[source.cadence]
    return cls(source, export, ee if ee is not None else ee_module())


# ---------------------------------------------------------------------------
# Temporal alignment (covariate_plan.md §5.3 / §5.4)
# ---------------------------------------------------------------------------
def time_key(source: Source, timestamp: Any) -> tuple:
    """The source's dedup time key for a fix timestamp."""
    cadence = source.cadence
    if cadence == "static":
        return ()
    if cadence == "annual":
        return (min(int(timestamp.year), int(source.last_year)),)
    if cadence == "monthly":
        return (int(timestamp.year) * 100 + int(timestamp.month),)
    if cadence == "daily":
        return (int(timestamp.strftime("%Y%m%d")),)
    if cadence == "subdaily":
        return (int(timestamp.strftime("%Y%m%d%H")),)
    if cadence == "composite":
        return (composite_window_index(source, timestamp),)
    raise ValueError(f"Unhandled cadence {cadence!r}")


def composite_window_index(source: Source, timestamp: Any) -> int:
    """Index of the ±``composite_days`` window a date falls in.

    ``composite_days`` is the *half*-window, matching covariate_plan.md §5.3's
    "±8-day window centred on the fix date": the full window is twice that, and
    a date maps to exactly one window.
    """
    size = max(1, 2 * int(source.composite_days))
    return int(timestamp.normalize().value // (86_400 * 10**9)) // size


def time_bucket(source: Source, timestamp: Any) -> str:
    """Chunking bucket: the granularity at which tasks stay image-bounded."""
    bucket = source.chunk_bucket
    if bucket == "static":
        return "static"
    if bucket == "year":
        return f"{min(int(timestamp.year), int(source.last_year))}"
    if bucket == "month":
        return timestamp.strftime("%Y-%m")
    if bucket == "day":
        return timestamp.strftime("%Y-%m-%d")
    if bucket == "hour":
        return timestamp.strftime("%Y-%m-%dT%H")
    if bucket == "composite":
        return f"c{composite_window_index(source, timestamp)}"
    raise ValueError(f"Unhandled chunk_bucket {bucket!r}")


def nominal_time(source: Source, timestamp: Any) -> Any:
    """The nominal acquisition time of the observation a fix joins to.

    This is the anchor ``_age_hours`` is measured from, and it is what makes an
    annual clamp visible: a 2026 fix against a 2024-max product joins to
    mid-2024 and carries ~1.5 years of age rather than silently reporting 0.
    The per-cadence anchors are documented in ``prompts/gee_export.md`` §4 and
    ``covariate_plan.md`` §5.3; they are conventions, not product metadata, and
    that is deliberate — it keeps dedup free of Earth Engine calls.
    """
    ts = timestamp
    cadence = source.cadence
    if cadence == "static":
        return ts
    if cadence == "subdaily":
        return ts.floor("h")
    if cadence == "daily":
        return _at_noon(ts.normalize())
    if cadence == "monthly":
        # Mid-month: 15th at 12:00 UTC.
        return _at_noon(ts.replace(day=1).normalize()) + _DAYS(14)
    if cadence == "annual":
        year = min(int(ts.year), int(source.last_year))
        return _at_noon(ts.replace(year=year, month=7, day=1).normalize())
    if cadence == "composite":
        size = max(1, 2 * int(source.composite_days))
        centre_day = composite_window_index(source, ts) * size + int(source.composite_days)
        return _at_noon(_EPOCH.normalize()) + _DAYS(centre_day)
    raise ValueError(f"Unhandled cadence {cadence!r}")


def _at_noon(ts: Any) -> Any:
    return ts + _HALF_DAY


def buffer_requested(source: Source) -> bool:
    """Whether the source's declared reducer list asks for a buffer reducer."""
    return any(r in ("mean", "mode") for r in source.reducers)


def buffer_eligible(source: Source, export: "ExportConfig") -> bool:
    """Whether a 30 m-radius buffer is both requested and meaningful.

    Two conditions: the source declares a buffer reducer, and its native pixel is
    at or below ``export.buffer_max_scale_m`` (sampling a 1 km pixel inside a 30 m
    buffer would just return the pixel itself).
    """
    return buffer_requested(source) and source.scale_m <= export.buffer_max_scale_m


def buffer_bands(source: Source) -> tuple[list[str], list[str]]:
    """(continuous, categorical) sampled bands — mean vs mode in the buffer.

    A class raster must be summarised with a mode, never averaged into a
    meaningless fractional class.
    """
    categorical = set(source.categorical)
    bands = list(source.sampled_bands)
    return ([b for b in bands if b not in categorical],
            [b for b in bands if b in categorical])


def task_selectors(source: Source, export: "ExportConfig") -> list[str]:
    """Columns an export task must carry: value bands + buffer + scene count.

    Kept in step with the task graph builder so the projection never references a
    column the graph does not produce.
    """
    selectors = list(source.sampled_bands)
    if buffer_eligible(source, export):
        selectors += [f"{b}{BUFFER_SUFFIX}" for b in source.sampled_bands]
    if source.cadence == "composite":
        selectors.append(SCENES_BAND)
    return selectors


def age_hours(source: Source, timestamp: Any) -> float:
    """|fix timestamp - nominal observation time| in hours (0.0 for static)."""
    if source.cadence == "static":
        return 0.0
    delta = abs(timestamp - nominal_time(source, timestamp))
    return float(delta.total_seconds()) / 3600.0


# ---------------------------------------------------------------------------
# EE task graph
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SamplingRef:
    """The reference time of one chunk bucket — what the image is built around."""

    timestamp: Any
    # Footprint the chunk's points fall in (an ee.Geometry). Collections are
    # filtered to it; None means unbounded (only for tests / tiny ad-hoc calls).
    region: Any = None


def sampling_crs(source: Source, zone: int) -> str:
    """The explicit CRS a chunk is sampled in (never left to the default).

    ``fixed`` grids use the product CRS; ``utm`` chunks are partitioned by zone
    during dedup, so each chunk has exactly one — which is why the zone is part
    of the pixel key and of the chunk partition.
    """
    if source.grid.mode == "utm":
        return f"EPSG:{int(zone)}"
    return str(source.grid.crs)


def point_collection(ee: Any, points: Any) -> Any:
    """A FeatureCollection of a chunk's pixel centroids, keyed by ``key_index``.

    Built by zipping parallel coordinate and ``key_index`` arrays and expanding
    them **server-side**, rather than sending one ``ee.Feature`` per point.

    The difference is not cosmetic. Measured with ``ee.serializer.encode`` on the
    installed client:

    ==========================================  =================
    construction                                bytes per point
    ==========================================  =================
    one ``ee.Feature`` per point                291
    zipped coordinate / key arrays (this one)    36
    ==========================================  =================

    An ``ee.Feature`` repeats its ``type``/``geometry``/``properties`` scaffolding
    for every point, so ~94k points serialises to ~26 MiB against a 10 MiB request
    limit — which is exactly how the first mule_deer submission failed. Zipping the
    arrays keeps the same chunk sizes inside the limit.
    """
    coordinates: list[list[float]] = []
    keys: list[int] = []
    for row in points.itertuples():
        coordinates.append([float(row.px_lon), float(row.px_lat)])
        keys.append(int(row.key_index))
    pairs = ee.List(coordinates).zip(ee.List(keys))
    return ee.FeatureCollection(pairs).map(
        lambda pair: ee.Feature(ee.Geometry.Point(pair.get(0)), {"key_index": pair.get(1)})
    )


def footprint(ee: Any, points: Any, *, pad_deg: float) -> Any:
    """Padded lon/lat bounding box of a point group, as an ``ee.Geometry``.

    Only used to *select* images (filterBounds), never to clip them, so a
    generous pad costs nothing and keeps neighbourhood ops (the 30 m buffer,
    990 m TPI) supplied with data at the edges.
    """
    lon = points["px_lon"].astype(float)
    lat = points["px_lat"].astype(float)
    return ee.Geometry.Rectangle(
        [float(lon.min()) - pad_deg, float(lat.min()) - pad_deg,
         float(lon.max()) + pad_deg, float(lat.max()) + pad_deg],
        "EPSG:4326",
        False,
    )


def build_task_table(
    handler: "SourceHandler",
    export: "ExportConfig",
    points: Any,
    zone: int,
) -> Any:
    """The FeatureCollection one export task writes.

    One ``sampleRegions`` call per distinct reference time in the chunk (that is
    what bounds the number of images a task touches), merged and flattened.
    Point values are sampled at the source's native scale with explicit
    ``scale``/``projection``; the 30 m buffer value comes from a neighbourhood
    reducer on the same masked image, so a masked pixel masks both.

    ``points`` needs columns ``key_index``, ``px_lon``, ``px_lat``, ``t_ref``.
    """
    ee = handler.ee
    source = handler.source
    if len(points) > export.max_points_per_task:
        raise RuntimeError(
            f"{source.name}: chunk has {len(points):,} points but "
            f"export.max_points_per_task is {export.max_points_per_task:,}. Points "
            f"are inlined into the task-creation request, which Earth Engine caps at "
            f"10 MiB (a zipped point costs ~36 bytes; see point_collection). Lower "
            f"export.target_features_per_task in configs/covariates/sources.yaml — "
            f"refusing to submit a task that would be rejected."
        )
    crs = sampling_crs(source, zone)
    scale = source.scale_m
    kernel = ee.Kernel.circle(export.buffer_radius_m, "meters")
    eligible = buffer_eligible(source, export)
    cont_bands, cat_bands = buffer_bands(source)

    parts = []
    for t_ref, group in points.groupby("t_ref", sort=True):
        region = footprint(ee, group, pad_deg=export.footprint_pad_deg)
        image = handler.image(SamplingRef(timestamp=t_ref, region=region))
        combined = image.select(list(source.sampled_bands))
        if eligible:
            if cont_bands:
                combined = combined.addBands(
                    image.select(cont_bands)
                    .reduceNeighborhood(ee.Reducer.mean(), kernel)
                    .rename([f"{b}{BUFFER_SUFFIX}" for b in cont_bands])
                )
            if cat_bands:
                combined = combined.addBands(
                    image.select(cat_bands)
                    .reduceNeighborhood(ee.Reducer.mode(), kernel)
                    .rename([f"{b}{BUFFER_SUFFIX}" for b in cat_bands])
                )
        points_fc = point_collection(ee, group)
        parts.append(
            combined.sampleRegions(
                collection=points_fc,
                scale=scale,
                projection=crs,
                tileScale=4,
                geometries=False,
            )
        )
    return ee.FeatureCollection(parts).flatten()


def monthly_scene_counts(ee: Any, source: Source, bbox: list[float], months: list[str]) -> Any:
    """``ee.List`` of image counts per ``YYYY-MM`` month intersecting ``bbox``.

    Counts raw images (before QA masking) — it answers whether the product
    exists over the footprint at all, not how cloudy it was. Built as one
    server-side list so the caller needs a single small ``getInfo()``.
    """
    region = ee.Geometry.Rectangle(list(bbox), "EPSG:4326", False)
    coll = ee.ImageCollection(source.asset).filterBounds(region)
    counts = []
    for month in months:
        start = ee.Date(f"{month}-01")
        counts.append(coll.filterDate(start, start.advance(1, "month")).size())
    return ee.List(counts)
