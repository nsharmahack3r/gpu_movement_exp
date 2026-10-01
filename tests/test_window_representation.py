"""The window representation must not leak the forecast target.

Legacy ``centroid_v1`` anchored each window's frame at the centroid of all
input + target fixes and passed ``p_T - c`` as the last input row, which made
the mean future position an exact function of the inputs. ``anchor_last_v2``
anchors at the last observed fix and builds inputs from observed fixes only.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest

from movement.cli.eval import check_representation
from movement.data.transforms import (
    REPRESENTATION_FILE,
    WINDOW_REPRESENTATION,
    WindowTransform,
    project_to_local,
)

N_IN, H = 8, 4


def _track(seed: int, n: int = N_IN + H) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lat = 34.0 + np.cumsum(rng.normal(0, 2e-3, n))
    lon = -97.2 + np.cumsum(rng.normal(0, 2e-3, n))
    return np.column_stack([lat, lon])


@pytest.mark.parametrize("delta", [True, False])
@pytest.mark.parametrize("extras", [False, True])
def test_inputs_do_not_depend_on_target(delta: bool, extras: bool):
    t = WindowTransform(input_len=N_IN, horizon=H, delta_encoding=delta,
                        step_length=extras, turning_angle=extras, speed=extras)
    track = _track(0)
    feats, target = track[:N_IN], track[N_IN:]
    other_target = _track(1)[N_IN:] + 0.05  # a completely different future
    ts = pd.Timestamp("2019-06-01 12:00")
    x_a = t.apply(feats, target, ts)["x"]
    x_b = t.apply(feats, other_target, ts)["x"]
    np.testing.assert_array_equal(x_a, x_b)


def test_delta_input_rows_are_observed_displacements():
    t = WindowTransform(input_len=N_IN, horizon=H, delta_encoding=True)
    track = _track(2)
    out = t.apply(track[:N_IN], track[N_IN:], pd.Timestamp("2019-06-01"))
    x, y, _, _ = project_to_local(track[:, 0], track[:, 1], origin=tuple(track[N_IN - 1]))
    assert np.allclose(out["x"][0], 0.0)
    np.testing.assert_allclose(out["x"][1:, 0], np.diff(x[:N_IN]), atol=1e-3)
    np.testing.assert_allclose(out["x"][1:, 1], np.diff(y[:N_IN]), atol=1e-3)
    # Targets: per-step future displacements, first one from p_T.
    np.testing.assert_allclose(out["y"][:, 0], np.diff(x[N_IN - 1:]), atol=1e-3)


def test_mean_future_position_not_recoverable_from_inputs():
    """The legacy identity sum(p_{T+k}-p_T) = -(T+H)(p_T-c) - sum_{t<T}(p_t-p_T)
    must no longer be computable: two windows with identical inputs but different
    futures produce identical inputs, so no function of x can recover it."""
    t = WindowTransform(input_len=N_IN, horizon=H, delta_encoding=True)
    track = _track(3)
    fut_a, fut_b = track[N_IN:], track[N_IN:][::-1] + 0.01
    ts = pd.Timestamp("2019-06-01")
    oa, ob = t.apply(track[:N_IN], fut_a, ts), t.apply(track[:N_IN], fut_b, ts)
    np.testing.assert_array_equal(oa["x"], ob["x"])
    assert not np.allclose(oa["y"].cumsum(0).sum(0), ob["y"].cumsum(0).sum(0))


def test_project_origin_default_is_centroid_and_explicit_origin_is_zero():
    track = _track(4)
    _, _, lat0, lon0 = project_to_local(track[:, 0], track[:, 1])
    assert lat0 == pytest.approx(track[:, 0].mean())
    x, y, _, _ = project_to_local(track[:, 0], track[:, 1], origin=tuple(track[3]))
    assert x[3] == pytest.approx(0.0) and y[3] == pytest.approx(0.0)


def test_eval_refuses_legacy_runs(tmp_path):
    with pytest.raises(SystemExit, match="centroid_v1"):
        check_representation(tmp_path)  # no marker -> legacy
    (tmp_path / REPRESENTATION_FILE).write_text("centroid_v1\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        check_representation(tmp_path)
    (tmp_path / REPRESENTATION_FILE).write_text(WINDOW_REPRESENTATION + "\n", encoding="utf-8")
    check_representation(tmp_path)  # no error


def test_study_dirs_are_versioned_by_representation():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_covariate_study.py"
    spec = importlib.util.spec_from_file_location("run_covariate_study", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = argparse.Namespace(dataset="boar_reshaped", mode="kfold", folds=5, fold_seed=42, split_seeds="1,2")
    study_dir, tag, _ = mod.study_layout(args)
    assert tag == f"kfold5_seed42_{WINDOW_REPRESENTATION}"
    assert study_dir.name == tag
    args.mode = "random"
    study_dir, tag, _ = mod.study_layout(args)
    assert tag == f"random_{WINDOW_REPRESENTATION}" and study_dir.name == tag
