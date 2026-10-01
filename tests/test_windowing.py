"""Windowing tests: no gap straddling, no split leakage, correct shapes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from movement.config import DataConfig
from movement.data.datamodule import DataModule
from movement.data.loading import group_into_trajectories, load_raw_csv
from movement.data.windowing import windows_from_segment


def _gap_split_df() -> pd.DataFrame:
    """40 hourly fixes with a 15-hour gap at index 25."""
    ts = pd.date_range("2025-01-01", periods=26, freq="1h")
    ts = ts.append(pd.date_range("2025-01-02 15:00", periods=14, freq="1h"))
    return pd.DataFrame(
        {
            "timestamp": ts,
            "lat": np.linspace(38.0, 38.01, len(ts)),
            "lon": np.linspace(-79.8, -79.79, len(ts)),
        }
    )


def test_windows_never_straddle_a_gap():
    """Window input+target timestamps are always hourly-contiguous."""
    df = _gap_split_df()
    seg1 = df.iloc[:26].reset_index(drop=True)
    seg2 = df.iloc[26:].reset_index(drop=True)
    for seg in (seg1, seg2):
        wins = windows_from_segment(seg, "s::a", "s", input_len=8, horizon=4, stride=1)
        for w in wins:
            first = np.where((seg["lat"].values == w.features[0, 0]) & (seg["lon"].values == w.features[0, 1]))[0]
            assert len(first) == 1, "first fix must be unique within the segment"
            start = int(first[0])
            window_ts = seg["timestamp"].iloc[start : start + 12]
            assert window_ts.diff().dropna().max() <= pd.Timedelta(hours=1), "window straddles a gap"


def test_grouping_splits_at_gaps():
    """group_into_trajectories splits at gaps > max_gap_multiplier × nominal_dt."""
    df = _gap_split_df()
    df["individual_id"] = "deer"
    df["study_id"] = "study"
    cfg = DataConfig(raw_path=__import__("pathlib").Path("."), processed_path=__import__("pathlib").Path("."))
    trajs = group_into_trajectories(df, cfg)  # nominal 1h, max gap 3h → split at 15h gap
    assert len(trajs) == 2
    assert {t.n_fixes for t in trajs} == {26, 14}


def test_no_window_in_two_splits(base_config):
    """Windows from one individual must never appear in two splits."""
    dm = DataModule.build(base_config)
    train_ids = {w.individual_id for w in dm.train_windows}
    val_ids = {w.individual_id for w in dm.val_windows}
    test_ids = {w.individual_id for w in dm.test_windows}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_no_individual_spans_splits(base_config):
    """The same individual_id never appears in two splits."""
    dm = DataModule.build(base_config)
    train_inds = {w.individual_id for w in dm.train_windows}
    val_inds = {w.individual_id for w in dm.val_windows}
    test_inds = {w.individual_id for w in dm.test_windows}
    assert train_inds.isdisjoint(val_inds) and train_inds.isdisjoint(test_inds) and val_inds.isdisjoint(test_inds)


def test_window_shapes(base_config):
    """Windows are (input_len, 2) inputs and (horizon, 2) targets."""
    dm = DataModule.build(base_config)
    x, y, dt = dm.dataset("train")[0]
    assert x.shape == (base_config.windowing.input_len, 2)
    assert y.shape == (base_config.windowing.horizon, 2)
    assert dt.shape == (base_config.windowing.input_len,)


def test_duplicate_timestamps_dropped(synthetic_csv):
    """Exact-duplicate timestamps are dropped, keeping the first."""
    df = load_raw_csv(synthetic_csv)
    dup = df.iloc[[0, 0, 1]].copy()
    from movement.data.loading import group_into_trajectories

    cfg = DataConfig(raw_path=synthetic_csv.parent, processed_path=synthetic_csv.parent)
    trajs = group_into_trajectories(dup, cfg)
    for t in trajs:
        assert t.df["timestamp"].is_unique
