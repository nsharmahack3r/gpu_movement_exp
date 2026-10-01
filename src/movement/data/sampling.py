"""Nominal sampling-interval detection for raw movement datasets.

Different datasets in ``RAW_DATASET_PATH`` may be sampled at different rates
(hourly deer fixes, 15-minute seal fixes, daily lion fixes, ...). To keep the
study comparable *within* a dataset while respecting its actual temporal
resolution, each dataset's nominal interval is detected and the standard
24-hour-in / 12-hour-out windowing is scaled to that interval:

- ``nominal_dt_hours`` = the detected sampling interval (hours)
- ``input_len`` = round(24h / nominal_dt) — one "day" of fixes
- ``horizon`` = round(12h / nominal_dt) — half a "day" out
- ``max_gap_multiplier`` stays 3.0 (a gap > 3× the nominal interval splits)

This keeps the *temporal span* of every window identical across datasets, which
is the defensible comparison: one day in, half a day out, regardless of the
sampling rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from movement.data.loading import load_raw_csv

logger = logging.getLogger(__name__)

# Window spans expressed as calendar time (hours); the *temporal protocol* of
# the study, independent of sampling rate.
WINDOW_INPUT_HOURS = 24.0
WINDOW_HORIZON_HOURS = 12.0
MAX_GAP_MULTIPLIER = 3.0
# Upper bound on the scaled input window (fixes). Very fine sampling would
# otherwise demand windows no trajectory segment can fill (see scale_windowing).
MAX_INPUT_LEN = 128
# Floor on the effective gap threshold (hours). With very fine sampling,
# 3 × nominal_dt becomes absurdly small (e.g. 6 minutes for 2-min fixes) and
# shreds trajectories on trivial hiccups; a real gap should still be a gap.
# The threshold is max(3 × nominal_dt, MIN_GAP_HOURS).
MIN_GAP_HOURS = 1.0
# Floor on the effective nominal sampling interval (hours). Datasets with
# burst GPS (10-second fixes) would otherwise scale the window protocol to an
# absurd resolution and generate millions of near-identical windows. Any
# interval finer than this is treated as MIN_DT_HOURS for windowing purposes.
MIN_DT_HOURS = 0.25  # 15 minutes


@dataclass(frozen=True)
class SamplingProfile:
    """Detected sampling characteristics of one dataset."""

    nominal_dt_hours: float
    median_dt_hours: float
    mode_dt_hours: float | None
    fraction_regular: float  # share of intervals within ±10% of the median


def _interval_hours(df: pd.DataFrame) -> np.ndarray:
    """Per-individual sorted timestamps → consecutive-interval hours.

    Intervals are computed within each individual's own sorted timeline so
    cross-individual jumps (which are meaningless) never enter the stats.
    """
    intervals: list[float] = []
    for _, g in df.groupby(["study_id", "individual_id"], sort=False):
        ts = g["timestamp"].sort_values()
        if len(ts) >= 2:
            dt = ts.diff().dt.total_seconds().dropna().to_numpy() / 3600.0
            intervals.extend(dt[dt > 0].tolist())  # drop exact-duplicate gaps (0)
    return np.asarray(intervals, dtype=float)


def _modal_interval(intervals: np.ndarray, *, bins: int = 200, max_hours: float = 72.0) -> float | None:
    """Modal interval via a histogram over the plausible range, or None if empty."""
    clipped = intervals[(intervals > 0) & (intervals <= max_hours)]
    if len(clipped) == 0:
        return None
    hist, edges = np.histogram(clipped, bins=bins)
    peak = int(np.argmax(hist))
    return float((edges[peak] + edges[peak + 1]) / 2.0)


def detect_sampling(csv_path: Path, *, n_rows: int | None = None) -> SamplingProfile:
    """Detect the nominal sampling interval of one raw CSV.

    The **median** interval is the robust estimate of the nominal rate (gaps and
    burst fixes skew the mean). The **mode** is reported alongside it; when the
    two disagree badly, the dataset is too irregular for the fixed-stride
    windowing to be meaningful and the caller decides whether to proceed.
    """
    df = load_raw_csv(csv_path)
    intervals = _interval_hours(df)
    if len(intervals) == 0:
        raise ValueError(f"{csv_path.name}: fewer than 2 fixes per individual; cannot detect sampling.")
    median = float(np.median(intervals))
    mode = _modal_interval(intervals)
    # Fraction of intervals within ±10% of the median (regularity check).
    lo, hi = median * 0.9, median * 1.1
    fraction = float(np.mean((intervals >= lo) & (intervals <= hi)))
    return SamplingProfile(
        nominal_dt_hours=median,
        median_dt_hours=median,
        mode_dt_hours=mode,
        fraction_regular=fraction,
    )


def scale_windowing(nominal_dt_hours: float) -> tuple[int, int, float]:
    """Scale the 24h/12h window protocol to a detected sampling interval.

    Returns ``(input_len, horizon, max_gap_multiplier)``. The input is at least
    2 fixes, the horizon at least 1. The max-gap multiplier is set so the
    effective gap threshold is ``max(3 × nominal_dt, MIN_GAP_HOURS)``: for
    normal sampling that is 3× nominal (the classic rule); for very fine
    sampling it is floored at 1 hour so small hiccups don't shred trajectories.

    The effective nominal interval is floored at ``MIN_DT_HOURS``: burst GPS
    (10-second fixes) would otherwise scale to an absurd resolution. Both
    caps are visible in the returned values and in ``runs/<dataset>/sampling.json``.
    """
    dt = max(nominal_dt_hours, MIN_DT_HOURS)
    input_len = max(2, int(round(WINDOW_INPUT_HOURS / dt)))
    horizon = max(1, int(round(WINDOW_HORIZON_HOURS / dt)))
    if input_len > MAX_INPUT_LEN:
        logger.warning(
            "Scaled input_len=%d exceeds cap %d (nominal_dt=%.4g h). "
            "Capping to input_len=%d, horizon=%d — the window now spans %.1fh "
            "in / %.1fh out instead of the 24h/12h protocol.",
            input_len, MAX_INPUT_LEN, nominal_dt_hours,
            MAX_INPUT_LEN, MAX_INPUT_LEN // 2,
            MAX_INPUT_LEN * dt, (MAX_INPUT_LEN // 2) * dt,
        )
        input_len = MAX_INPUT_LEN
        horizon = max(1, MAX_INPUT_LEN // 2)
    # Effective gap threshold = max(3 × dt, MIN_GAP_HOURS) hours, expressed
    # back as a multiplier of the *raw* nominal dt so the datamodule's
    # threshold (raw_dt × multiplier) equals the intended hours.
    gap_multiplier = max(MAX_GAP_MULTIPLIER, MIN_GAP_HOURS / dt) * (dt / max(nominal_dt_hours, 1e-9))
    return input_len, horizon, gap_multiplier
