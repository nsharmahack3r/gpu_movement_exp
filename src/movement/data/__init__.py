"""Data pipeline: loading, windowing, transforms, datamodule."""

from .datamodule import DataModule, MovementDataset, fit_scaler
from .loading import (
    Trajectory,
    discard_short,
    group_into_trajectories,
    load_dataset,
    load_raw_csv,
    save_split,
    split_individuals,
)
from .sampling import (
    MAX_GAP_MULTIPLIER,
    MAX_INPUT_LEN,
    MIN_DT_HOURS,
    MIN_GAP_HOURS,
    WINDOW_HORIZON_HOURS,
    WINDOW_INPUT_HOURS,
    SamplingProfile,
    detect_sampling,
    scale_windowing,
)
from .transforms import FeatureScaler, WindowTransform, haversine_m, inverse_project, project_to_local
from .windowing import Window, build_windows, windows_from_segment

__all__ = [
    "MAX_GAP_MULTIPLIER",
    "MAX_INPUT_LEN",
    "MIN_DT_HOURS",
    "MIN_GAP_HOURS",
    "DataModule",
    "FeatureScaler",
    "MovementDataset",
    "SamplingProfile",
    "Trajectory",
    "WINDOW_HORIZON_HOURS",
    "WINDOW_INPUT_HOURS",
    "Window",
    "WindowTransform",
    "build_windows",
    "detect_sampling",
    "discard_short",
    "fit_scaler",
    "group_into_trajectories",
    "haversine_m",
    "inverse_project",
    "load_dataset",
    "load_raw_csv",
    "project_to_local",
    "save_split",
    "scale_windowing",
    "split_individuals",
    "windows_from_segment",
]
