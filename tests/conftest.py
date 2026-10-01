"""Shared pytest fixtures: tiny synthetic dataset + config."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from movement.config import Config
from movement.data.datamodule import DataModule

# A small but realistic synthetic dataset: 12 individuals, hourly fixes,
# a few gaps, constant-velocity motion so baselines score zero.
N_INDIVIDUALS = 12
FIXES_PER_INDIVIDUAL = 120
STUDY_IDS = ["study_a", "study_b"]


def make_synthetic_csv(tmp_path: Path, path: Path | None = None) -> Path:
    """Write a synthetic movement CSV; returns its path."""
    path = path or tmp_path / "synthetic.csv"
    rows: list[dict] = []
    ts = pd.date_range("2025-01-01", periods=FIXES_PER_INDIVIDUAL + 10, freq="1h")
    for i in range(N_INDIVIDUALS):
        study = STUDY_IDS[i % len(STUDY_IDS)]
        lat0, lon0 = 38.0 + 0.01 * i, -79.8 + 0.01 * i
        # Constant velocity: 10 m per hour north-east.
        for k in range(FIXES_PER_INDIVIDUAL):
            # ~10 m = 9e-5 deg lat, lon scaled by cos(lat).
            lat = lat0 + k * 9.0e-5
            lon = lon0 + k * 9.0e-5 / np.cos(np.radians(lat0))
            if k % 30 == 0:  # a small gap every 30 fixes (skip 2 hours)
                continue
            rows.append(
                {
                    "timestamp": ts[k],
                    "individual_id": f"deer_{i:02d}",
                    "study_id": study,
                    "species": "Odocoileus virginianus",
                    "lat": lat,
                    "lon": lon,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_synthetic_csv(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def base_config(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """A config pointing at the synthetic CSV with tiny windowing."""
    import yaml

    from movement.config import load_config

    raw_dir = tmp_path_factory.mktemp("raw")
    make_synthetic_csv(raw_dir)
    # Point the base config at the synthetic dir via env, then load with overrides.
    yaml_path = tmp_path_factory.mktemp("cfg") / "base.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "raw_path": str(raw_dir),
                    "processed_path": str(tmp_path_factory.mktemp("proc")),
                    "nominal_dt_hours": 1.0,
                    "max_gap_multiplier": 3.0,
                    "val_fraction": 0.2,
                    "test_fraction": 0.2,
                },
                "windowing": {"input_len": 8, "horizon": 4, "stride": 1, "eval_stride": 4},
                "transforms": {"delta_encoding": True, "scale": True},
                "model": {"name": "tcn", "channels": [16, 16, 16], "kernel_size": 3, "dropout": 0.1},
                "trainer": {"seed": 42, "batch_size": 32, "eval_batch_size": 64, "max_epochs": 3},
                "evaluation": {"n_plots": 2},
                "tracking": {"backend": "tensorboard", "offline": True},
            }
        ),
        encoding="utf-8",
    )
    return load_config(yaml_path)


@pytest.fixture(scope="session")
def datamodule(base_config: Config) -> DataModule:
    return DataModule.build(base_config)
