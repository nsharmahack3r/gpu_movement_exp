"""Metrics tests: hand-computed ADE/FDE, perfect scores zero, CV baseline exact."""

from __future__ import annotations

import numpy as np

from movement.evaluation.metrics import (
    ade,
    constant_position_error,
    constant_velocity_error,
    fde,
    haversine_ade,
    metrics_report,
)


def test_ade_fde_hand_computed():
    """One sample, two steps: hand-compute expected ADE/FDE."""
    gt = np.array([[[0.0, 0.0], [0.0, 10.0]]])  # (1, 2, 2)
    pred = np.array([[[3.0, 4.0], [0.0, 10.0]]])  # error step1 = 5, step2 = 0
    assert ade(gt - pred) == 2.5  # (5 + 0) / 2
    assert fde(gt - pred) == 0.0


def test_perfect_prediction_scores_zero():
    gt = np.random.default_rng(1).normal(size=(8, 5, 2))
    rep = metrics_report(gt, gt.copy())
    assert rep["ade"] == 0.0
    assert rep["fde"] == 0.0
    assert np.allclose(rep["rmse_step"], 0.0)
    assert np.allclose(rep["mae_step"], 0.0)


def test_constant_velocity_baseline_exact_on_straight_line():
    """Constant-velocity extrapolation errors are zero on linear constant-speed data."""
    horizon = 6
    n = 5
    last_disp = np.tile(np.array([10.0, 0.0]), (n, 1))
    # Truth follows constant velocity → cumulative positions (h+1)·last_disp.
    gt_cum = np.stack([(h + 1) * last_disp for h in range(horizon)], axis=1)
    err = constant_velocity_error(gt_cum, last_disp)
    assert np.allclose(err, 0.0)
    assert ade(err) == 0.0


def test_constant_position_baseline():
    """Constant-position error equals the true cumulative displacement."""
    gt = np.array([[[10.0, 0.0], [20.0, 0.0]]])
    err = constant_position_error(gt)
    assert np.allclose(err, gt)


def test_haversine_ade_zero_for_identical():
    gt = np.array([[[38.0, -79.8], [38.0001, -79.8]]])
    assert haversine_ade(gt, gt.copy()) == 0.0
    # ~11 m for 0.0001 deg latitude.
    assert abs(haversine_ade(gt, gt.copy()) - 0.0) < 1e-6
