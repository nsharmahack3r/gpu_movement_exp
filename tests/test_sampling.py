"""Sampling-detection tests: nominal dt, window scaling, regularity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from movement.data.sampling import detect_sampling, scale_windowing


def _write_csv(path: Path, dt_hours: float, n_individuals: int = 4, n_fixes: int = 100) -> Path:
    rows = []
    ts = pd.date_range("2025-01-01", periods=n_fixes, freq=f"{dt_hours}h")
    for i in range(n_individuals):
        lat0, lon0 = 38.0 + 0.01 * i, -79.8 + 0.01 * i
        for k in range(n_fixes):
            rows.append(
                {
                    "timestamp": ts[k],
                    "individual_id": f"ind_{i}",
                    "study_id": "study",
                    "species": "test",
                    "lat": lat0 + k * 1e-4,
                    "lon": lon0 + k * 1e-4,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def test_detect_hourly(tmp_path: Path):
    csv = _write_csv(tmp_path / "hourly.csv", dt_hours=1.0)
    profile = detect_sampling(csv)
    assert abs(profile.nominal_dt_hours - 1.0) < 1e-6
    assert profile.fraction_regular > 0.99


def test_detect_15min(tmp_path: Path):
    csv = _write_csv(tmp_path / "fifteen.csv", dt_hours=0.25)
    profile = detect_sampling(csv)
    assert abs(profile.nominal_dt_hours - 0.25) < 1e-6


def test_detect_daily(tmp_path: Path):
    csv = _write_csv(tmp_path / "daily.csv", dt_hours=24.0)
    profile = detect_sampling(csv)
    assert abs(profile.nominal_dt_hours - 24.0) < 1e-6


def test_scale_windowing_hourly():
    # 1h → the classic 24/12, gap threshold 3h.
    assert scale_windowing(1.0) == (24, 12, 3)


def test_scale_windowing_15min():
    # 0.25h → 24h/0.25 = 96 input, 12h/0.25 = 48 horizon.
    # Gap threshold floored at 1h → multiplier 1/0.25 = 4.
    il, hz, mg = scale_windowing(0.25)
    assert (il, hz) == (96, 48)
    assert mg == 4


def test_scale_windowing_daily():
    # 24h → 1 input... but the floor is 2, and horizon is 1.
    il, hz, mg = scale_windowing(24.0)
    assert il == 2
    assert hz == 1
    assert mg == 3


def test_scale_windowing_burst_gps():
    """10-second fixes are floored to the 15-min effective interval."""
    from movement.data.sampling import MIN_DT_HOURS

    dt = 10 / 3600  # 0.00278h
    il, hz, mg = scale_windowing(dt)
    # Eff dt floored to 0.25h → 24h/0.25 = 96 input, 12h/0.25 = 48 horizon.
    assert (il, hz) == (96, 48)
    # Gap threshold = max(3*0.25, 1h) = 1h, expressed as raw-dt multiplier.
    assert abs(mg - 1.0 / dt) < 1e-6
    assert MIN_DT_HOURS == 0.25


def test_scale_windowing_capped_at_max_input_len():
    """Very fine sampling below the dt floor never exceeds MAX_INPUT_LEN."""
    from movement.data.sampling import MAX_INPUT_LEN

    # 1-minute fixes → floored to 0.25h → 96/48, never the uncapped 1440.
    il, hz, _ = scale_windowing(1 / 60)
    assert il <= MAX_INPUT_LEN
    assert il == 96
    assert hz == 48
