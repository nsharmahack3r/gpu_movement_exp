"""Integration test: invert_predictions consistency on a synthetic window."""

from __future__ import annotations

import numpy as np
import pandas as pd

from movement.data.transforms import WindowTransform
from movement.data.windowing import Window
from movement.evaluation.evaluate import invert_predictions


def _make_windows(n: int = 4, input_len: int = 8, horizon: int = 4) -> list[Window]:
    """Windows with constant-velocity northward motion (~11 m per fix)."""
    windows = []
    for i in range(n):
        lat0 = 38.0 + 0.01 * i
        lon0 = -79.8 + 0.01 * i
        ts = pd.date_range("2025-01-01", periods=input_len + horizon, freq="1h")
        features = np.column_stack(
            [lat0 + 1e-4 * np.arange(input_len), np.full(input_len, lon0)]
        )
        target = np.column_stack(
            [lat0 + 1e-4 * np.arange(input_len, input_len + horizon), np.full(horizon, lon0)]
        )
        windows.append(
            Window(
                individual_id=f"s::w{i}",
                study_id="s",
                timestamp=ts[input_len - 1],
                features=features,
                target=target,
            )
        )
    return windows


def test_perfect_predictions_score_zero():
    """Feeding the exact GT deltas back must give ADE/FDE of zero."""
    windows = _make_windows()
    transform = WindowTransform(input_len=8, horizon=4, delta_encoding=True)
    gts = []
    preds = []
    for w in windows:
        out = transform.apply(w.features, w.target, w.timestamp)
        gts.append(out["y"])
        preds.append(out["y"].copy())
    pred_all = np.stack(preds)

    inv = invert_predictions(pred_all, windows)
    # inv.gt is cumulative; the transform's targets are per-step increments.
    assert np.allclose(np.cumsum(inv.gt, axis=1), np.cumsum(inv.pred, axis=1), atol=1e-3)
    assert np.allclose(inv.gt, inv.pred, atol=1e-3), "perfect predictions must give zero error"


def test_haversine_reconstruction_matches_input():
    """Reconstructed GT lat/lon must match the raw target lat/lon."""
    windows = _make_windows()
    transform = WindowTransform(input_len=8, horizon=4, delta_encoding=True)
    gts = [transform.apply(w.features, w.target, w.timestamp)["y"] for w in windows]
    inv = invert_predictions(np.stack(gts), windows)
    for i, w in enumerate(windows):
        assert np.allclose(inv.latlon_gt[i], w.target, atol=1e-9), (
            f"reconstructed GT lat/lon mismatch for window {i}"
        )
