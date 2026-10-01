"""Memory features: what the animal's own past says about where it goes next.

Many animals return to the same places at the same time of day (resting sites,
feeding patches) and stay inside a home range. A 24-hour input window shows
only one day of that. These features give the model the animal's longer
history, built **only from the same animal's fixes at or before the last
observed fix** (the forecast origin), so they are available at forecast time.

All positions are in the forecast frame: metres east/north of the last observed
fix (as the targets), reported in kilometres.

Per forecast step k (time ``t_k = t_last + k * nominal_dt``), for the previous
days d = 1..D the fix nearest to ``t_k − d·24 h`` (within half a sampling
interval, and not after ``t_last``) is looked up. Step features
(:data:`STEP_FEATURES`):

- ``d1_dx, d1_dy, d1_avail`` — yesterday at this clock time;
- ``mean_dx, mean_dy`` — mean over the available days;
- ``frac`` — fraction of the D days available;
- ``spread`` — RMS distance of those positions from their mean.

Window features (:data:`GLOBAL_FEATURES`) over the last ``lookback_days``:

- ``centre_dx, centre_dy, centre_dist`` — home-range centre (mean position);
- ``near_frac`` — share of past fixes within ``near_m`` of the current position;
- ``history_days`` — span of history available, as a fraction of the lookback.

Vector pairs that must rotate with rotation augmentation are listed in
:data:`STEP_VECTOR_PAIRS` / :data:`GLOBAL_VECTOR_PAIRS`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from movement.data.transforms import project_to_local

STEP_FEATURES = ("d1_dx", "d1_dy", "d1_avail", "mean_dx", "mean_dy", "frac", "spread")
GLOBAL_FEATURES = ("centre_dx", "centre_dy", "centre_dist", "near_frac", "history_days")
STEP_VECTOR_PAIRS = ((0, 1), (3, 4))
GLOBAL_VECTOR_PAIRS = ((0, 1),)
DAY_S = 86400.0
#: Positions further than this (km) are clipped (keeps rare relocations bounded).
CLIP_KM = 25.0


@dataclass
class AnimalHistory:
    """All fixes of one animal, sorted by time."""

    t: np.ndarray  # (N,) int64 epoch seconds
    lat: np.ndarray  # (N,)
    lon: np.ndarray  # (N,)


def build_histories(trajectories) -> dict[str, AnimalHistory]:
    """Per-animal fix history from every segment (any split: queries never look past t_last)."""
    parts: dict[str, list] = {}
    for tr in trajectories:
        df = tr.df
        parts.setdefault(tr.individual_id, []).append((
            df["timestamp"].to_numpy(dtype="datetime64[s]").astype(np.int64),
            df["lat"].to_numpy(dtype=float),
            df["lon"].to_numpy(dtype=float),
        ))
    out = {}
    for ind, chunks in parts.items():
        t = np.concatenate([c[0] for c in chunks])
        order = np.argsort(t, kind="stable")
        t = t[order]
        lat = np.concatenate([c[1] for c in chunks])[order]
        lon = np.concatenate([c[2] for c in chunks])[order]
        keep = np.concatenate([[True], np.diff(t) > 0])  # segments never overlap; drop duplicates anyway
        out[ind] = AnimalHistory(t[keep], lat[keep], lon[keep])
    return out


def memory_features(
    hist: AnimalHistory,
    t_last: int,
    origin: tuple[float, float],
    *,
    horizon: int,
    dt_s: float,
    days: int = 7,
    lookback_days: float = 14.0,
    near_m: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """``(step (horizon, 7), global (5,))`` float32 memory features for one window."""
    end = int(np.searchsorted(hist.t, t_last, side="right"))  # fixes at or before t_last
    t = hist.t[:end]
    step = np.zeros((horizon, len(STEP_FEATURES)), dtype=np.float32)
    glob = np.zeros(len(GLOBAL_FEATURES), dtype=np.float32)
    if end == 0:
        return step, glob

    def xy(i: np.ndarray) -> np.ndarray:  # km in the forecast frame
        x, y, _, _ = project_to_local(hist.lat[i], hist.lon[i], origin=origin)
        return np.clip(np.stack([x, y], axis=-1) / 1000.0, -CLIP_KM, CLIP_KM)

    # --- same clock time on previous days ------------------------------------
    k = np.arange(1, horizon + 1, dtype=np.float64)
    d = np.arange(1, days + 1, dtype=np.float64)
    q = t_last + k[:, None] * dt_s - d[None, :] * DAY_S  # (H, D)
    idx = np.searchsorted(t, q)
    lo = np.clip(idx - 1, 0, end - 1)
    hi = np.clip(idx, 0, end - 1)
    pick = np.where(np.abs(t[hi] - q) < np.abs(t[lo] - q), hi, lo)
    ok = (np.abs(t[pick] - q) <= 0.5 * dt_s) & (q <= t_last)
    pos = xy(pick)  # (H, D, 2)
    n = ok.sum(axis=1)  # (H,)
    w = ok[..., None].astype(np.float64)
    mean = (pos * w).sum(axis=1) / np.maximum(n, 1)[:, None]
    spread = np.sqrt((((pos - mean[:, None]) ** 2).sum(-1) * ok).sum(1) / np.maximum(n, 1))
    step[:, 0:2] = np.where(ok[:, :1], pos[:, 0], 0.0)
    step[:, 2] = ok[:, 0]
    step[:, 3:5] = mean
    step[:, 5] = n / days
    step[:, 6] = spread

    # --- home range over the lookback --------------------------------------
    start = int(np.searchsorted(t, t_last - lookback_days * DAY_S, side="left"))
    recent = xy(np.arange(start, end))
    centre = recent.mean(axis=0)
    glob[0:2] = centre
    glob[2] = float(np.hypot(*centre))
    glob[3] = float((np.hypot(recent[:, 0], recent[:, 1]) * 1000.0 <= near_m).mean())
    glob[4] = float(min(1.0, (t_last - t[start]) / (lookback_days * DAY_S)))
    return step, glob


def rotate_pairs(a: np.ndarray, rot: np.ndarray, pairs) -> np.ndarray:
    """Apply a 2x2 rotation/mirror to the (x, y) column pairs of ``a`` (last axis)."""
    a = a.copy()
    for i, j in pairs:
        a[..., [i, j]] = a[..., [i, j]] @ rot.T
    return a
