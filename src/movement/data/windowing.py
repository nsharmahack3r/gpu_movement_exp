"""Sliding-window construction over trajectory segments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from movement.data.loading import Trajectory


@dataclass(frozen=True)
class Window:
    """One (input, target) window.

    ``features`` holds the raw per-fix coordinate data before feature
    engineering; the target holds raw (lat, lon) coordinates for loss/reporting.
    Transformers (projection, deltas, scaling) are applied by the Dataset at
    window time so scalers fit on train only.
    """

    individual_id: str
    study_id: str
    timestamp: pd.Timestamp
    features: np.ndarray  # (input_len, 2) raw [lat, lon]
    target: np.ndarray  # (horizon, 2) raw [lat, lon]
    # Per-fix epoch seconds of the input window (for time-aware positional
    # encoding); None falls back to nominal hourly sampling in the dataset.
    timestamps: np.ndarray | None = None  # (input_len,) int64 epoch seconds
    # Per-fix raw covariate values of the observed fixes, NaN where missing.
    # A view into the segment's array (no copy), so memory stays flat.
    covariates: np.ndarray | None = None  # (input_len, C) float32


def windows_from_segment(
    seg: pd.DataFrame,
    individual_id: str,
    study_id: str,
    *,
    input_len: int,
    horizon: int,
    stride: int,
    covariate_columns: list[str] | None = None,
) -> list[Window]:
    """Build sliding windows from one segment.

    The segment's ``lat``/``lon`` columns are used directly; gap handling is the
    caller's job (segments never straddle a detected gap by construction).
    With ``covariate_columns`` each window also carries its observed fixes'
    covariate rows (future fixes' covariates are never attached).
    """
    lats = seg["lat"].to_numpy(dtype=float)
    lons = seg["lon"].to_numpy(dtype=float)
    total = input_len + horizon
    out: list[Window] = []
    n = len(seg)
    # Converted once per segment (was once per window: O(n^2) on long segments).
    ts = seg["timestamp"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    cov = (
        np.ascontiguousarray(seg[list(covariate_columns)].to_numpy(dtype=np.float32))
        if covariate_columns
        else None
    )
    for start in range(0, n - total + 1, stride):
        end = start + total
        features = np.stack([lats[start:end], lons[start:end]], axis=1)[:input_len]
        target = np.stack([lats[start:end], lons[start:end]], axis=1)[input_len:]
        out.append(
            Window(
                individual_id=individual_id,
                study_id=study_id,
                timestamp=seg["timestamp"].iloc[start + input_len],
                features=features,
                target=target,
                timestamps=ts[start : start + input_len],
                covariates=None if cov is None else cov[start : start + input_len],
            )
        )
    return out


def build_windows(
    trajectories: list[Trajectory],
    *,
    input_len: int,
    horizon: int,
    stride: int,
    covariate_columns: list[str] | None = None,
) -> list[Window]:
    """Build all windows across trajectories with the given stride."""
    out: list[Window] = []
    for t in trajectories:
        out.extend(
            windows_from_segment(
                t.df,
                t.individual_id,
                t.study_id,
                input_len=input_len,
                horizon=horizon,
                stride=stride,
                covariate_columns=covariate_columns,
            )
        )
    return out
