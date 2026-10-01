"""Coordinate transforms: local projection, delta encoding, feature engineering, scaling.

The transforms pipeline turns raw ``(lat, lon)`` windows into the model's input
and target tensors:

1. **Project** each window to a local metric frame **anchored at the last
   observed fix** (tangent-plane approximation). The frame is built from
   observed fixes only. Windows are short (a day of fixes), so the local frame
   error is negligible.
2. **Delta-encode** displacements ``(Δx, Δy)`` when configured: the input is
   the observed per-step displacements (first row zero), the target is the
   future per-step displacements.
3. Optionally append engineered features (step length, turning angle, speed, Δt,
   cyclical time) — all off by default.
4. **Standardise** per-feature using statistics fit on the *training split only*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import asin, cos, pi, radians, sin, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_EARTH_RADIUS_M = 6_371_008.8

#: Version tag of the window representation. Bump whenever the mapping from a
#: raw window to model tensors changes, so checkpoints trained under an older
#: representation are never evaluated under a newer one.
#:
#: - ``centroid_v1`` (legacy, LEAKY): frame origin = centroid of all
#:   ``input_len + horizon`` fixes and the last input row carried ``p_T - c``.
#:   Together with the observed displacements this gives the model the exact
#:   sum of the future positions (a target leak).
#: - ``anchor_last_v2``: frame origin = last observed fix ``p_T``; inputs use
#:   observed fixes only.
WINDOW_REPRESENTATION = "anchor_last_v2"
REPRESENTATION_FILE = "window_representation.txt"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_M * asin(sqrt(a))


def project_to_local(
    lat: np.ndarray,
    lon: np.ndarray,
    origin: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Project (lat, lon) arrays to local x/y metres about ``origin``.

    Uses a simple equirectangular approximation (valid for small windows):
    x = R·Δlon·cos(lat₀), y = R·Δlat.

    ``origin`` is ``(lat0, lon0)``. When omitted, the array's centroid is used —
    fine for plotting or inspecting a track, but **never** for building model
    windows from input + target fixes (the centroid would include the target).

    Returns
    -------
    ``(x_m, y_m, lat0, lon0)`` — projected coords plus the frame origin, so
    predictions can be inverted back to lat/lon.
    """
    if origin is None:
        lat0 = float(np.mean(lat))
        lon0 = float(np.mean(lon))
    else:
        lat0, lon0 = float(origin[0]), float(origin[1])
    x = _EARTH_RADIUS_M * np.radians(lon - lon0) * cos(radians(lat0))
    y = _EARTH_RADIUS_M * np.radians(lat - lat0)
    return x, y, lat0, lon0


def inverse_project(x: np.ndarray, y: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """Invert :func:`project_to_local` given the projection origin."""
    lat = lat0 + np.degrees(y / _EARTH_RADIUS_M)
    lon = lon0 + np.degrees(x / (_EARTH_RADIUS_M * cos(radians(lat0))))
    return lat, lon


def last_observed_origin(window_features: np.ndarray) -> tuple[float, float]:
    """Frame origin for a window: the last observed ``(lat, lon)`` fix."""
    return float(window_features[-1, 0]), float(window_features[-1, 1])


@dataclass
class WindowTransform:
    """Everything needed to convert one raw window into model tensors."""

    input_len: int
    horizon: int
    delta_encoding: bool = True
    step_length: bool = False
    turning_angle: bool = False
    speed: bool = False
    delta_t: bool = False
    cyclical_time: bool = False

    def __post_init__(self) -> None:
        self._n_extra = sum(
            [
                self.step_length,
                self.turning_angle,
                self.speed,
                self.delta_t,
                self.cyclical_time,
            ]
        )

    @property
    def input_features(self) -> int:
        return 2 + self._n_extra

    @property
    def output_features(self) -> int:
        return 2 if self.delta_encoding else 2

    def apply(self, window_features: np.ndarray, window_target: np.ndarray, timestamp: pd.Timestamp) -> dict[str, Any]:
        """Transform one window's raw ``(lat, lon)`` into model input/target.

        Parameters
        ----------
        window_features:
            ``(input_len, 2)`` raw (lat, lon) of the observed window.
        window_target:
            ``(horizon, 2)`` raw (lat, lon) of the forecast target.
        timestamp:
            Timestamp of the last observed fix (for cyclical features).

        Returns
        -------
        ``{"x": (input_len, F), "y": (horizon, 2)}`` where ``y`` holds
        displacement deltas (metres) from the last observed fix when
        ``delta_encoding`` is on, else absolute projected positions.
        """
        lat = np.concatenate([window_features[:, 0], window_target[:, 0]]).astype(float)
        lon = np.concatenate([window_features[:, 1], window_target[:, 1]]).astype(float)
        # Frame anchored at the last OBSERVED fix: nothing about the frame
        # depends on the target fixes (see WINDOW_REPRESENTATION).
        x, y, _, _ = project_to_local(lat, lon, origin=last_observed_origin(window_features))

        n = self.input_len
        # Per-fix displacements across the whole window (input + target).
        dx = np.diff(x)
        dy = np.diff(y)
        # Observed-input deltas: a zero first row, then the (n-1) observed
        # per-step displacements d_t = p_t - p_{t-1}, t = 1..n-1. Row n-1 is the
        # last observed step. No absolute position is passed.
        delta_input = np.concatenate([np.zeros((1, 2)), np.column_stack([dx[: n - 1], dy[: n - 1]])], axis=0)
        # Target: per-step displacements from the last observed fix — the model
        # predicts each step's (Δx, Δy) increment.
        target_delta = np.column_stack([dx[n - 1 :], dy[n - 1 :]])

        if not self.delta_encoding:
            x_in = np.column_stack([x[:n], y[:n]])
            y_out = np.column_stack([x[n:], y[n:]])
        else:
            x_in = delta_input
            y_out = target_delta

        cols = [x_in]
        if self.step_length:
            step = np.hypot(dx, dy)
            cols.append(np.concatenate([[step[0]], step[: n - 1]]))
        if self.turning_angle:
            angles = np.arctan2(dy, dx)
            # Turn = change in heading; wrap to [-pi, pi].
            turn = np.diff(angles, prepend=angles[0])
            turn = (turn + pi) % (2 * pi) - pi
            cols.append(np.concatenate([[turn[0]], turn[: n - 1]]))
        if self.speed:
            speed = np.hypot(dx, dy)  # metres per fix (Δt ≈ nominal)
            cols.append(np.concatenate([[speed[0]], speed[: n - 1]]))
        if self.delta_t:
            # Seconds between fixes (nominal = 3600).
            dts = np.full(n, 3600.0)
            cols.append(dts)
        if self.cyclical_time:
            t = pd.Timestamp(timestamp)
            hour = t.hour + t.minute / 60
            doy = float(t.dayofyear)
            cols.append(np.full(n, np.sin(2 * pi * hour / 24)))
            cols.append(np.full(n, np.cos(2 * pi * hour / 24)))
            cols.append(np.full(n, np.sin(2 * pi * doy / 365.25)))
            cols.append(np.full(n, np.cos(2 * pi * doy / 365.25)))

        # Engineered features are 1-D per fix; make every block a column block.
        x_in = np.concatenate([c.reshape(n, -1) for c in cols], axis=1)
        return {"x": x_in.astype(np.float32), "y": y_out.astype(np.float32)}


class FeatureScaler:
    """Per-feature mean/std standardisation, fit on train only.

    Persisted to JSON as ``{"mean": [...], "std": [...]}`` so eval can invert
    scaling identically.
    """

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = mean
        self.std = std

    @property
    def fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def fit(self, x: np.ndarray) -> "FeatureScaler":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)  # avoid div-by-zero on constant features
        self.mean = mean
        self.std = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FeatureScaler.transform called before fit.")
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FeatureScaler.inverse called before fit.")
        return x * self.std + self.mean

    def save(self, path: Path) -> None:
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted scaler.")
        path.write_text(
            json.dumps(
                {"mean": self.mean.tolist(), "std": self.std.tolist()},
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "FeatureScaler":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(mean=np.array(data["mean"], dtype=np.float32), std=np.array(data["std"], dtype=np.float32))
