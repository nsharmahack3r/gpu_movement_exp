"""Measure a task's serialized request body, to keep it under Earth Engine's limit.

Earth Engine rejects a task-creation request whose body exceeds **10 MiB**
(10485760 bytes). The body inlines the chunk's whole point FeatureCollection, so
the encoding of that collection is what decides how many points fit in a task —
and the difference between constructions is dramatic.

Run it after changing ``point_collection`` or the sampling band set:

    uv run python scripts/measure_payload.py

It needs Earth Engine initialised (for the algorithm list) but creates no tasks
and consumes no export quota: it is a token refresh plus a read-only metadata call,
then purely client-side serialization.
"""

from __future__ import annotations

import inspect
import json
import sys

import ee
from ee import serializer

from movement.covariates.sources import load_registry
from movement.utils.env import gee_auth

# Earth Engine's documented request-payload ceiling.
PAYLOAD_LIMIT_BYTES = 10 * 1024 * 1024
# Points to measure over; the per-point cost is stable well above ~100.
SAMPLE_POINTS = 500
# Fraction of the limit the point collection alone may occupy, leaving the rest
# for the per-bucket image graph.
POINTS_BUDGET = 0.5


def _coordinates(n: int) -> list[list[float]]:
    """A plausible CONUS point set (cheap, deterministic)."""
    return [[-108.99 + 0.0001 * i, 41.81 + 0.0001 * i] for i in range(n)]


def _encode(obj: object) -> str:
    """The request-body form of an expression (Cloud API encoding)."""
    return serializer.encode(obj, for_cloud_api=True)  # type: ignore[arg-type]


def _body_bytes(obj: object) -> int:
    return len(json.dumps({"expression": _encode(obj)}))


def main() -> int:
    registry = load_registry()
    auth = gee_auth(destination=registry.export.destination)
    ee.Initialize(auth["credentials"], project=auth["project"])
    print(f"# project={auth['project']} destination={auth['destination']}")
    print(f"# encode signature: {inspect.signature(serializer.encode)}")

    coords = _coordinates(SAMPLE_POINTS)

    # Strategy A: one ee.Feature per point (the original, rejected construction).
    per_feature = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point(c), {"key_index": i}) for i, c in enumerate(coords)]
    )
    # Strategy B: zipped arrays, expanded server-side (what the tool uses now).
    zipped = ee.FeatureCollection(
        ee.List(coords).zip(ee.List(list(range(SAMPLE_POINTS))))
    ).map(lambda pair: ee.Feature(ee.Geometry.Point(pair.get(0)), {"key_index": pair.get(1)}))

    print()
    print(f"{'construction':<26} {'bytes/point':>11} {'points at 50%':>14} {'points at 80%':>14}")
    print("-" * 70)
    per_point: dict[str, float] = {}
    for label, obj in (("ee.Feature per point", per_feature), ("zipped arrays", zipped)):
        cost = _body_bytes(obj) / SAMPLE_POINTS
        per_point[label] = cost
        print(f"{label:<26} {cost:>11.1f} {int(PAYLOAD_LIMIT_BYTES * 0.5 / cost):>14,} "
              f"{int(PAYLOAD_LIMIT_BYTES * 0.8 / cost):>14,}")

    # Report against what the registry currently allows.
    cap = registry.export.max_points_per_task
    zipped_cost = per_point["zipped arrays"]
    used = cap * zipped_cost
    print()
    print(f"export.max_points_per_task = {cap:,}")
    print(f"  zipped arrays        -> {used / 1024**2:5.1f} MiB "
          f"({used / PAYLOAD_LIMIT_BYTES:.0%} of the limit)")
    print(f"  budget               -> {PAYLOAD_LIMIT_BYTES * POINTS_BUDGET / 1024**2:.1f} MiB "
          f"for points, rest for the image graph")

    if used > PAYLOAD_LIMIT_BYTES * POINTS_BUDGET:
        print(f"\nFAIL: the current cap exceeds the {POINTS_BUDGET:.0%} points budget "
              f"with the zipped construction.", file=sys.stderr)
        return 1
    print("\nOK: the current cap fits the zipped-array construction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
