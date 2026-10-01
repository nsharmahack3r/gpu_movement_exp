"""Forecast metrics, computed in metres after inverting scaling/projection.

All metrics operate on arrays of *displacement* or absolute-position errors in
metric units (metres). Naive baselines (constant-position, constant-velocity)
are computed on the same displacement space so every model row is comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from movement.data.transforms import haversine_m


def ade(displacement_error: np.ndarray) -> float:
    """Average displacement error over the horizon.

    Parameters
    ----------
    displacement_error:
        ``(B, horizon, 2)`` per-step, per-axis error in metres.
    """
    return float(np.mean(np.hypot(displacement_error[..., 0], displacement_error[..., 1])))


def fde(displacement_error: np.ndarray) -> float:
    """Final displacement error at the last horizon step."""
    return float(np.mean(np.hypot(displacement_error[:, -1, 0], displacement_error[:, -1, 1])))


def rmse_per_step(displacement_error: np.ndarray) -> np.ndarray:
    """Per-horizon-step RMSE in metres, shape ``(horizon,)``."""
    return np.sqrt(np.mean(displacement_error**2, axis=(0, 2)))


def mae_per_step(displacement_error: np.ndarray) -> np.ndarray:
    """Per-horizon-step MAE in metres, shape ``(horizon,)``."""
    return np.mean(np.hypot(displacement_error[..., 0], displacement_error[..., 1]), axis=0)


def haversine_ade(gt_latlon: np.ndarray, pred_latlon: np.ndarray) -> float:
    """Haversine ADE on reconstructed lat/lon (projection sanity check)."""
    n = gt_latlon.shape[0]
    total = 0.0
    for i in range(n):
        for h in range(gt_latlon.shape[1]):
            total += haversine_m(
                gt_latlon[i, h, 0],
                gt_latlon[i, h, 1],
                pred_latlon[i, h, 0],
                pred_latlon[i, h, 1],
            )
    return float(total / (n * gt_latlon.shape[1]))


def constant_position_error(gt: np.ndarray) -> np.ndarray:
    """Error of the constant-position baseline (repeat the last fix).

    The baseline predicts zero displacement, so in cumulative-position space its
    error is simply the ground-truth positions themselves.

    Parameters
    ----------
    gt:
        ``(B, horizon, 2)`` ground-truth cumulative positions (metres, relative
        to the last observed fix).

    Returns
    -------
    ``(B, horizon, 2)`` error array (== ``gt``).
    """
    return gt.copy()


def constant_velocity_error(gt: np.ndarray, last_displacement: np.ndarray) -> np.ndarray:
    """Error of the constant-velocity baseline (extrapolate the last step).

    In cumulative-position space the baseline predicts ``k * last_displacement``
    at horizon step ``k``.

    Parameters
    ----------
    gt:
        ``(B, horizon, 2)`` ground-truth cumulative positions (metres).
    last_displacement:
        ``(B, 2)`` the displacement between the last two observed fixes.

    Returns
    -------
    ``(B, horizon, 2)`` error array.
    """
    steps = np.arange(1, gt.shape[1] + 1, dtype=float)[None, :, None]
    return gt - steps * last_displacement[:, None, :]


def metrics_report(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    gt_constant_position: np.ndarray | None = None,
    gt_constant_velocity: np.ndarray | None = None,
) -> dict[str, float]:
    """Full metric dictionary for one model/baseline.

    Parameters
    ----------
    gt:
        ``(B, horizon, 2)`` ground-truth displacement targets.
    pred:
        ``(B, horizon, 2)`` model predictions (same space).
    gt_constant_position / gt_constant_velocity:
        Precomputed *error* arrays from :func:`constant_position_error` /
        :func:`constant_velocity_error`, or ``None`` to skip those rows.

    Returns
    -------
    ``{"ade": ..., "fde": ..., "rmse": ..., "mae": ..., "ade_cp": ...,
    "ade_cv": ...}`` where ``ade_cp``/``ade_cv`` are the baseline ADEs.
    """
    err = gt - pred
    rep = {
        "ade": ade(err),
        "fde": fde(err),
        "rmse_step": rmse_per_step(err).tolist(),
        "mae_step": mae_per_step(err).tolist(),
    }
    if gt_constant_position is not None:
        rep["ade_cp"] = ade(gt_constant_position)
        rep["fde_cp"] = fde(gt_constant_position)
    if gt_constant_velocity is not None:
        rep["ade_cv"] = ade(gt_constant_velocity)
        rep["fde_cv"] = fde(gt_constant_velocity)
    return rep


def per_horizon_dataframe(gt: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """Per-horizon-step error table: ADE, RMSE, MAE per t+k.

    Parameters
    ----------
    gt, pred:
        ``(B, horizon, 2)`` in metres.

    Returns
    -------
    DataFrame with columns ``step, ade, rmse, mae``.
    """
    err = gt - pred
    steps = np.arange(1, gt.shape[1] + 1)
    return pd.DataFrame(
        {
            "step": steps,
            "ade": np.hypot(err[..., 0], err[..., 1]).mean(axis=0),
            "rmse": rmse_per_step(err),
            "mae": mae_per_step(err),
        }
    )
