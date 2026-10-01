"""Evaluation: test-set evaluation + per-horizon breakdown + plotting."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from movement.data.datamodule import DataModule, unpack_batch
from movement.data.transforms import inverse_project, last_observed_origin, project_to_local
from movement.evaluation.metrics import (
    constant_position_error,
    constant_velocity_error,
    haversine_ade,
    metrics_report,
    per_horizon_dataframe,
)
from movement.models.base import ForecastModel

logger = logging.getLogger(__name__)


@dataclass
class PredictionBatch:
    """Inverted (metric) predictions + ground truth for one eval batch.

    ``pred`` and ``gt`` hold *cumulative positions* in metres relative to the
    last observed fix (i.e. absolute position error is ``gt - pred``), matching
    the conventional trajectory-forecasting ADE/FDE definition.
    """

    pred: np.ndarray  # (B, horizon, 2) predicted cumulative positions (metres)
    gt: np.ndarray  # (B, horizon, 2) true cumulative positions (metres)
    last_observed: np.ndarray  # (B, 2) metres — absolute position of the last observed fix
    last_displacement: np.ndarray  # (B, 2) metres — displacement of the last observed step
    latlon_gt: np.ndarray  # (B, horizon, 2) reconstructed lat/lon
    latlon_pred: np.ndarray  # (B, horizon, 2) reconstructed lat/lon
    windows: list  # raw Window objects for plotting


def invert_predictions(
    pred_delta: np.ndarray,
    windows: list,
) -> PredictionBatch:
    """Invert model predictions from delta space to metres and lat/lon.

    The model outputs displacement deltas directly in metres (targets are
    raw metres; the scaler only normalises inputs), so no un-normalisation is
    needed here.

    Parameters
    ----------
    pred_delta:
        ``(B, horizon, 2)`` model output in metres.
    windows:
        Source :class:`Window` objects aligned with the batch (for lat/lon and
        plot metadata).
    """
    B, H, _ = pred_delta.shape
    pred = pred_delta.astype(float)

    last_observed = np.zeros((B, 2))
    last_displacement = np.zeros((B, 2))
    gt = np.zeros((B, H, 2))
    latlon_gt = np.zeros((B, H, 2))
    latlon_pred = np.zeros((B, H, 2))

    for i, w in enumerate(windows):
        # Project the FULL window (input + target) in ONE local frame so the
        # displacement targets are consistent with the input's metric space.
        # The frame is anchored at the last observed fix, exactly as in
        # WindowTransform.apply, so last_observed is the origin (0, 0).
        lat_full = np.concatenate([w.features[:, 0], w.target[:, 0]]).astype(float)
        lon_full = np.concatenate([w.features[:, 1], w.target[:, 1]]).astype(float)
        x_full, y_full, lat0, lon0 = project_to_local(lat_full, lon_full, origin=last_observed_origin(w.features))
        n = len(w.features)
        x_obs, y_obs = x_full[:n], y_full[:n]
        last_observed[i] = [x_obs[-1], y_obs[-1]]
        last_displacement[i] = [x_obs[-1] - x_obs[-2], y_obs[-1] - y_obs[-2]]

        # Per-step increment targets, matching WindowTransform's delta encoding
        # (dx[n-1:], dy[n-1:]); cumsum to absolute (cumulative) positions.
        dx_full = np.diff(x_full)
        dy_full = np.diff(y_full)
        step_gt = np.column_stack([dx_full[n - 1 :], dy_full[n - 1 :]])
        gt[i] = np.cumsum(step_gt, axis=0)
        pred[i] = np.cumsum(pred[i], axis=0)
        latlon_gt[i] = np.column_stack(
            inverse_project(last_observed[i, 0] + gt[i, :, 0], last_observed[i, 1] + gt[i, :, 1], lat0, lon0)
        )
        latlon_pred[i] = np.column_stack(
            inverse_project(last_observed[i, 0] + pred[i, :, 0], last_observed[i, 1] + pred[i, :, 1], lat0, lon0)
        )

    return PredictionBatch(
        pred=pred,
        gt=gt,
        last_observed=last_observed,
        last_displacement=last_displacement,
        latlon_gt=latlon_gt,
        latlon_pred=latlon_pred,
        windows=windows,
    )


N_BASELINE_SAMPLES = 64
# Destination grid used to score probabilistic arms without a hex head (H3 res-9 size).
HEX_RINGS = 10
HEX_EDGE_M = 174.0
MAX_CLIMATOLOGY_WINDOWS = 20000


def predict_split(
    model: ForecastModel,
    datamodule: DataModule,
    device: torch.device,
    split: str = "test",
    *,
    return_hex: bool = False,
):
    """Run the model over a split (loader order = ``dataset(split).windows``).

    Returns ``(point_delta, samples_pos)``: ``point_delta`` is ``(B, H, 2)``
    per-step displacements in metres. For a probabilistic model it is the
    per-step spatial median of the samples, and ``samples_pos`` holds the
    ``(B, M, H, 2)`` sampled *cumulative* positions; for a point model
    ``samples_pos`` is ``None``. With ``return_hex`` a third element holds the
    destination-hexagon head's ``(B, n_classes)`` probabilities (``None`` without a head).
    """
    from movement.evaluation.scoring import spatial_median

    model.eval()
    probabilistic = bool(getattr(model, "is_probabilistic", False))
    points: list[np.ndarray] = []
    samples: list[np.ndarray] = []
    hex_probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in datamodule.dataloader(split, shuffle=False):
            x, _y, dt_seconds, extras = unpack_batch(batch, device)
            pred = model(x, context={"stage": "test", "dt_seconds": dt_seconds, **extras})
            if getattr(model, "hex_rings", 0) > 0:
                hex_probs.append(torch.softmax(model.last_hex_logits.double(), dim=-1).cpu().numpy())
            if probabilistic:
                pos = torch.cumsum(pred.double(), dim=2)
                med = spatial_median(pos)
                points.append(torch.diff(med, dim=1, prepend=torch.zeros_like(med[:, :1])).cpu().numpy())
                samples.append(pos.cpu().numpy())
            else:
                points.append(pred.detach().double().cpu().numpy())
    point = np.concatenate(points, axis=0)
    samp = np.concatenate(samples, axis=0) if probabilistic else None
    if return_hex:
        return point, samp, (np.concatenate(hex_probs, axis=0) if hex_probs else None)
    return point, samp


def _hour(w) -> int:
    return int(pd.Timestamp(w.timestamp).hour)


def climatology_baselines(datamodule: DataModule, gt: np.ndarray, windows: list, *, seed: int = 0,
                          hex_grid=None) -> dict:
    """Energy scores of the no-skill probabilistic baselines on ``gt``.

    Both resample training-window futures and rotate each draw at random, so
    they carry no information about where the test animal is heading:

    - ``es_clim_all``: any training window;
    - ``es_clim_hour``: training windows whose last fix has the same clock hour
      (the diel activity rhythm — how far animals move at this time of day).

    A model whose energy score beats ``es_clim_hour`` knows more than the time
    of day. Train split only; seeded for reproducibility.
    """
    from movement.data.datamodule import _window_transform
    from movement.evaluation.scoring import (
        COVERAGE_LEVELS,
        CURVE_LEVELS,
        climatology_samples,
        energy_score_np,
        region_calibration,
    )

    train = datamodule.train_windows
    if not train:
        return {}
    idx = np.linspace(0, len(train) - 1, num=min(MAX_CLIMATOLOGY_WINDOWS, len(train))).astype(int)
    transform = _window_transform(datamodule.config)
    fut = np.stack([np.cumsum(transform.apply(train[i].features, train[i].target, train[i].timestamp)["y"], axis=0)
                    for i in idx]).astype(np.float64)
    tr_hours = np.array([_hour(train[i]) for i in idx])
    te_hours = np.array([_hour(w) for w in windows])
    out = {}
    for key, by_hour in (("es_clim_all", False), ("es_clim_hour", True)):
        rng = np.random.default_rng(seed)
        samp = climatology_samples(fut, tr_hours, te_hours, N_BASELINE_SAMPLES, rng, by_hour=by_hour)
        es = energy_score_np(samp, gt)
        out[key] = float(es.mean())
        out[f"{key}_step"] = es.mean(axis=0).tolist()
        if by_hour:  # calibration reference: a no-skill forecaster that knows the diel rhythm
            out["calibration_clim_hour"] = region_calibration(samp, gt, COVERAGE_LEVELS)
            out["calibration_curve_clim_hour"] = region_calibration(samp, gt, CURVE_LEVELS)["coverage"]
            if hex_grid is not None:
                from movement.evaluation.hexgrid import hex_scores

                sc = hex_scores(hex_grid.kde_probs(samp[:, :, -1]), hex_grid.assign(gt[:, -1]))
                sc.pop("nll_per_window")
                out["hex_clim_hour"] = sc
    return out


def evaluate_model(
    model: ForecastModel,
    datamodule: DataModule,
    device: torch.device,
    *,
    split: str = "test",
    per_horizon: bool = True,
    return_details: bool = False,
):
    """Run inference over a split and return the full metric dictionary.

    Returns ``{"ade": ..., "fde": ..., "rmse_step": [...], "mae_step": [...],
    "ade_cp": ..., "fde_cp": ..., "ade_cv": ..., "fde_cv": ...,
    "haversine_ade": ..., "es": ..., "es_step": [...], "es_clim_all": ...,
    "es_clim_hour": ...}`` plus ``"per_horizon"`` when requested.

    ``es`` is the step-averaged energy score (metres). For a point model it
    equals ``ade`` exactly (the energy score of a single path is its error);
    ``es_cp`` (stay put) likewise equals ``ade_cp``. With ``return_details``
    the result is ``(report, inv, samples_pos)`` for plotting.
    """
    from movement.evaluation.scoring import energy_score_np

    ds = datamodule.dataset(split)
    windows_all = ds.windows  # same order as the non-shuffled loader
    point, samples, head_probs = predict_split(model, datamodule, device, split, return_hex=True)

    inv = invert_predictions(point, windows_all)

    report = metrics_report(
        inv.gt,
        inv.pred,
        gt_constant_position=constant_position_error(inv.gt),
        gt_constant_velocity=constant_velocity_error(inv.gt, inv.last_displacement),
    )
    report["haversine_ade"] = haversine_ade(inv.latlon_gt, inv.latlon_pred)
    report["n_windows"] = int(len(point))

    if samples is not None:
        from movement.evaluation.scoring import COVERAGE_LEVELS, CURVE_LEVELS, region_calibration

        es = energy_score_np(samples, inv.gt)
        report["probabilistic"] = True
        report["n_samples"] = int(samples.shape[1])
        # Coverage of centre-outward sample regions (see scoring.region_calibration).
        report["calibration"] = region_calibration(samples, inv.gt, COVERAGE_LEVELS)
        report["calibration_curve"] = region_calibration(samples, inv.gt, CURVE_LEVELS)["coverage"]
    else:
        es = np.hypot(*(inv.gt - inv.pred).transpose(2, 0, 1))  # (B, H): point ES = error
        report["probabilistic"] = False
    report["es"] = float(es.mean())
    report["es_step"] = es.mean(axis=0).tolist()
    report["es_cp"] = report["ade_cp"]
    hex_grid = None
    if samples is not None:
        from movement.evaluation.hexgrid import HexGrid

        own = int(getattr(model, "hex_rings", 0)) > 0  # score on the head's own grid when it has one
        hex_grid = HexGrid(model.hex_rings, model.hex_edge_m) if own else HexGrid(HEX_RINGS, HEX_EDGE_M)
    report.update(climatology_baselines(datamodule, inv.gt, windows_all, hex_grid=hex_grid))
    hex_nll = {}
    if hex_grid is not None:
        from movement.evaluation.hexgrid import hex_scores

        true_cell = hex_grid.assign(inv.gt[:, -1])
        report["hex"] = {"rings": hex_grid.rings, "edge_m": hex_grid.edge, "n_cells": hex_grid.n_cells}
        # Stay put = all mass on the centre cell: only its hit rate is meaningful.
        centre = int(hex_grid.assign(np.zeros((1, 2)))[0])
        report["hex"]["stay_put"] = {"top1": float((true_cell == centre).mean())}
        sources = {"samples": hex_grid.kde_probs(samples[:, :, -1])}
        if head_probs is not None:
            sources["head"] = head_probs
        for name, probs in sources.items():
            sc = hex_scores(probs, true_cell)
            hex_nll[f"hex_nll_{name}"] = sc.pop("nll_per_window")
            report["hex"][name] = sc
        if "hex_clim_hour" in report:
            report["hex"]["clim_hour"] = report.pop("hex_clim_hour")

    if per_horizon:
        frame = per_horizon_dataframe(inv.gt, inv.pred)
        frame["es"] = report["es_step"]
        if "es_clim_hour_step" in report:
            frame["es_clim_hour"] = report["es_clim_hour_step"]
        report["per_horizon"] = frame
    err = np.hypot(*(inv.gt - inv.pred).transpose(2, 0, 1))  # (B, H) point-forecast error
    report["per_window"] = pd.DataFrame({
        "individual_id": [w.individual_id for w in windows_all],
        "t_origin": [pd.Timestamp(w.timestamp) for w in windows_all],
        "es": es.mean(axis=1),
        "ade": err.mean(axis=1),
        "fde": err[:, -1],
        "ade_cp": np.hypot(inv.gt[..., 0], inv.gt[..., 1]).mean(axis=1),
        **hex_nll,
    })
    if return_details:
        return report, inv, samples
    return report


def save_plots(
    inv: PredictionBatch,
    run_dir: Path,
    *,
    n_plots: int = 4,
    seed: int = 0,
    samples: np.ndarray | None = None,
    n_sample_paths: int = 20,
) -> list[Path]:
    """Save ``n_plots`` predicted-vs-actual trajectory figures to ``run_dir/figures``.

    With ``samples`` (``(B, M, H, 2)`` cumulative positions from a probabilistic
    model), up to ``n_sample_paths`` sampled paths are drawn faintly behind the
    median forecast.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(inv.windows), size=min(n_plots, len(inv.windows)), replace=False)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, j in enumerate(idx):
        w = inv.windows[j]
        fig, ax = plt.subplots(figsize=(6, 5))
        lat_in, lon_in = w.features[:, 0].astype(float), w.features[:, 1].astype(float)
        # Same frame as inv.gt / inv.pred: origin at the last observed fix.
        x_m, y_m, _, _ = project_to_local(lat_in, lon_in, origin=last_observed_origin(w.features))
        ax.plot(x_m, y_m, "o-", color="tab:blue", label="observed", ms=3, lw=1)
        if samples is not None:
            for m in range(min(n_sample_paths, samples.shape[1])):
                ax.plot(np.r_[0.0, samples[j, m, :, 0]], np.r_[0.0, samples[j, m, :, 1]], "-",
                        color="tab:red", alpha=0.15, lw=0.8, label="samples" if m == 0 else None)
        ax.plot(inv.gt[j, :, 0], inv.gt[j, :, 1], "o--", color="tab:green", label="ground truth", ms=3, lw=1)
        ax.plot(inv.pred[j, :, 0], inv.pred[j, :, 1], "o--", color="tab:red",
                label="prediction" if samples is None else "median", ms=3, lw=1)
        ax.set_title(f"{w.individual_id} · window {i}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = fig_dir / f"pred_vs_actual_{i:02d}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)
    return paths


def write_eval_outputs(
    run_dir: Path,
    report: dict,
    *,
    model_name: str,
    split: str,
) -> None:
    """Write ``metrics.json``, ``per_horizon.csv`` and ``per_window.csv`` into the run dir."""
    out: dict = {k: v for k, v in report.items() if k not in ("per_horizon", "per_window")}
    if "per_window" in report:
        report["per_window"].to_csv(run_dir / "per_window.csv", index=False)
        out["per_window_file"] = "per_window.csv"
    if "per_horizon" in report:
        report["per_horizon"].to_csv(run_dir / "per_horizon.csv", index=False)
        out["per_horizon_file"] = "per_horizon.csv"
    out["model"] = model_name
    out["split"] = split
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("Wrote metrics.json and per_horizon.csv to %s", run_dir)
