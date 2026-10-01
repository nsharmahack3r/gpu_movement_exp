"""Tail-validation split, memory features and the destination-hexagon head."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from test_cov_transformer import _config, _write_raw_and_covariates

from movement.data.datamodule import DataModule
from movement.data.loading import Trajectory, split_individuals_kfold, split_individuals_kfold_tailval
from movement.data.memory import (
    GLOBAL_FEATURES,
    STEP_FEATURES,
    AnimalHistory,
    memory_features,
    rotate_pairs,
)
from movement.data.transforms import project_to_local
from movement.evaluation.hexgrid import HexGrid, hex_scores
from movement.models import build_model


def _cfg(tmp_path, *, memory=False, tailval=False, **model):
    raw_dir, gee_dir = _write_raw_and_covariates(tmp_path)
    cfg = _config(tmp_path, raw_dir, gee_dir, use_covariates=False, **model)
    cfg.covariates.enabled = False
    cfg.transforms.memory = memory
    if tailval:
        cfg.data.split_unit = "individual_kfold_tailval"
        cfg.data.n_folds = 5
        cfg.data.fold = 1
    return cfg


# ---------------------------------------------------------------------------
# Tail-validation split
# ---------------------------------------------------------------------------
def _trajs(n_animals=10, n=200):
    out = []
    for i in range(n_animals):
        t = pd.date_range("2021-01-01", periods=n, freq="1h")
        df = pd.DataFrame({"timestamp": t, "lat": 50 + np.arange(n) * 1e-4, "lon": 8.0 + i * 0.01})
        out += [Trajectory(f"a{i}", "s", df.iloc[:120].reset_index(drop=True)),
                Trajectory(f"a{i}", "s", df.iloc[120:].reset_index(drop=True))]
    return out


def test_tail_split_tests_the_same_unseen_animals_and_validates_on_the_track_tail():
    trajs = _trajs()
    train, val, test, cuts = split_individuals_kfold_tailval(trajs, n_folds=5, fold=2, seed=42,
                                                             val_tail_fraction=0.15)
    _, _, test_ref = split_individuals_kfold(trajs, n_folds=5, fold=2, seed=42)
    test_ids = {t.individual_id for t in test}
    assert test_ids == {t.individual_id for t in test_ref}
    assert not test_ids & ({t.individual_id for t in train} | {t.individual_id for t in val})
    assert {t.individual_id for t in train} == {t.individual_id for t in val} == set(cuts)
    assert len(cuts) == 8  # 80% of animals train (vs 60% with a validation fold)
    for t in train:
        assert (t.df["timestamp"] < cuts[t.individual_id]).all()
    for t in val:
        assert (t.df["timestamp"] >= cuts[t.individual_id]).all()
    n_val = sum(t.n_fixes for t in val)
    n_train = sum(t.n_fixes for t in train)
    assert n_val / (n_val + n_train) == pytest.approx(0.15, abs=0.01)


def test_tail_split_file_round_trips(tmp_path):
    cfg = _cfg(tmp_path, tailval=True)
    dm = DataModule.build(cfg)
    assert dm.val_cuts and dm.val_windows
    run = tmp_path / "run"
    run.mkdir()
    path = dm.persist_split(run)
    payload = json.loads(path.read_text())
    assert set(payload["val_cut"]) >= set(payload["train"])
    dm2 = DataModule.from_split_file(cfg, path)
    for a, b in ((dm.train_windows, dm2.train_windows), (dm.val_windows, dm2.val_windows),
                 (dm.test_windows, dm2.test_windows)):
        assert [(w.individual_id, w.timestamp) for w in a] == [(w.individual_id, w.timestamp) for w in b]
    cfg.data.split_unit = "individual_kfold"
    with pytest.raises(ValueError, match="val_cut"):
        DataModule.from_split_file(cfg, path)


# ---------------------------------------------------------------------------
# Memory features
# ---------------------------------------------------------------------------
def _commuter(days=10, dt_h=1.0):
    """An animal that is at a fixed site each hour of the day, every day."""
    t0 = pd.Timestamp("2021-05-01").value // 10**9
    t = t0 + (np.arange(int(days * 24 / dt_h)) * dt_h * 3600).astype(np.int64)
    hour = ((t - t0) // 3600) % 24
    lat = 50.0 + 0.001 * np.sin(2 * np.pi * hour / 24)
    lon = 8.0 + 0.001 * np.cos(2 * np.pi * hour / 24)
    return AnimalHistory(t, lat, lon)


def test_memory_recovers_yesterdays_position_at_each_forecast_clock_time():
    hist = _commuter()
    i_last = 8 * 24 + 5
    t_last = int(hist.t[i_last])
    origin = (hist.lat[i_last], hist.lon[i_last])
    step, glob = memory_features(hist, t_last, origin, horizon=6, dt_s=3600.0)
    assert step.shape == (6, len(STEP_FEATURES)) and glob.shape == (len(GLOBAL_FEATURES),)
    true_future = hist.t[i_last + 1 : i_last + 7]
    j = [int(np.flatnonzero(hist.t == tt)[0]) for tt in true_future]
    x, y, _, _ = project_to_local(hist.lat[j], hist.lon[j], origin=origin)
    np.testing.assert_allclose(step[:, 0], x / 1000, atol=1e-6)  # site repeats daily
    np.testing.assert_allclose(step[:, 1], y / 1000, atol=1e-6)
    np.testing.assert_allclose(step[:, 3:5], step[:, 0:2], atol=1e-6)
    assert (step[:, 2] == 1).all() and (step[:, 5] == 1).all()
    assert np.abs(step[:, 6]).max() < 1e-6
    assert glob[4] == pytest.approx(i_last / 24 / 14, abs=1e-3)  # 8.2 of 14 lookback days
    assert glob[3] > 0  # back at this site at this hour on earlier days


def test_memory_never_reads_fixes_after_the_forecast_origin():
    hist = _commuter()
    i_last = 6 * 24 + 3
    t_last = int(hist.t[i_last])
    origin = (hist.lat[i_last], hist.lon[i_last])
    a = memory_features(hist, t_last, origin, horizon=12, dt_s=3600.0)
    cut = AnimalHistory(hist.t[: i_last + 1], hist.lat[: i_last + 1], hist.lon[: i_last + 1])
    b = memory_features(cut, t_last, origin, horizon=12, dt_s=3600.0)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)
    # With a 30 h horizon, steps beyond 24 h would need "yesterday" = the future: unavailable.
    step, _ = memory_features(hist, t_last, origin, horizon=30, dt_s=3600.0)
    assert (step[24:, 2] == 0).all() and (step[:24, 2] == 1).all()


def test_memory_without_history_is_zero():
    hist = _commuter(days=1)
    step, glob = memory_features(hist, int(hist.t[0]) - 10, (50.0, 8.0), horizon=4, dt_s=3600.0)
    assert not step.any() and not glob.any()


def test_rotation_augmentation_rotates_memory_vectors_with_the_targets(tmp_path):
    cfg = _cfg(tmp_path, memory=True, probabilistic=True)
    cfg.trainer.augment_rotation = True
    dm = DataModule.build(cfg)
    plain = dm.dataset("val")
    aug = dm.dataset("train")
    aug.windows = plain.windows
    np.random.seed(3)
    _, y0, _, e0 = plain[5]
    _, y1, _, e1 = aug[5]
    # Recover the rotation from the target path and check the memory used the same one.
    rot, *_ = np.linalg.lstsq(y0.numpy().astype(np.float64), y1.numpy().astype(np.float64), rcond=None)
    rot = rot.T
    np.testing.assert_allclose(e1["memory_step"].numpy(),
                               rotate_pairs(e0["memory_step"].numpy(), rot.astype(np.float32), ((0, 1), (3, 4))),
                               atol=1e-4)
    np.testing.assert_allclose(e1["memory_global"][2], e0["memory_global"][2], atol=1e-6)  # distance invariant


# ---------------------------------------------------------------------------
# Hex grid
# ---------------------------------------------------------------------------
def test_hex_grid_geometry():
    g = HexGrid(10, 174.0)
    assert g.n_cells == 331 and g.n_classes == 332
    np.testing.assert_array_equal(g.assign(g.centres), np.arange(331))
    # Points just inside a cell (within the inscribed circle) stay in it.
    jitter = np.random.default_rng(0).uniform(-1, 1, size=(331, 2)) * 0.6 * 174 * np.sqrt(3) / 2 / np.sqrt(2)
    np.testing.assert_array_equal(g.assign(g.centres + jitter), np.arange(331))
    assert g.assign(np.array([[5000.0, 0.0], [0.0, -4000.0]])).tolist() == [g.outside, g.outside]
    assert g.assign(np.zeros((1, 2)))[0] == int(np.flatnonzero((g.q == 0) & (g.r == 0))[0])


def test_kde_cell_probabilities_and_scores():
    g = HexGrid(4, 174.0)
    rng = np.random.default_rng(1)
    tight = g.centres[7] + rng.normal(size=(50, 64, 2)) * 20
    probs = g.kde_probs(tight)
    np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-9)
    assert (probs.argmax(1) == 7).all()
    sc = hex_scores(probs, np.full(50, 7))
    assert sc["top1"] == 1.0 and sc["nll"] < 0.5
    far = g.kde_probs(np.full((3, 64, 2), 9000.0))
    assert (far[:, g.outside] > 0.99).all()
    uniform = np.full((10, g.n_classes), 1.0 / g.n_classes)
    assert hex_scores(uniform, np.zeros(10, dtype=int))["nll"] == pytest.approx(np.log(g.n_classes), rel=1e-6)


# ---------------------------------------------------------------------------
# Model + training + evaluation
# ---------------------------------------------------------------------------
def test_memory_hex_model_trains_and_reports_hex_scores(tmp_path):
    from movement.evaluation import evaluate_model
    from movement.training import Trainer
    from movement.utils.tracking import Tracker, make_run_dir

    cfg = _cfg(tmp_path, memory=True, tailval=True, probabilistic=True, n_samples_train=4,
               n_samples_eval=8, hex_rings=3)
    cfg.trainer.max_epochs = 1
    dm = DataModule.build(cfg)
    spec = dm.model_data_spec()
    assert spec["n_memory_step"] == len(STEP_FEATURES) and spec["n_memory_global"] == len(GLOBAL_FEATURES)
    model = build_model(cfg.model, cfg.windowing, cfg.transforms, data_spec=spec)
    x, y, dt, extras = next(iter(dm.dataloader("train")))
    out = model(x, context={"dt_seconds": dt, **extras})
    assert out.shape == (x.shape[0], 4, cfg.windowing.horizon, 2)
    assert model.last_hex_logits.shape == (x.shape[0], 3 * 3 * 4 + 2)
    with pytest.raises(ValueError, match="memory_step"):
        model(x, context={"dt_seconds": dt, **{k: v for k, v in extras.items() if k != "memory_step"}})

    run_dir = make_run_dir(cfg)
    dm.persist_split(run_dir)
    dm.persist_scaler(run_dir)
    Trainer(model, dm, cfg, Tracker(cfg, run_dir), run_dir, device=torch.device("cpu")).fit()
    dm = DataModule.from_split_file(cfg, run_dir / "split.json")  # eval path rebuilds the tail split
    report = evaluate_model(model, dm, torch.device("cpu"), split="test")
    hx = report["hex"]
    assert hx["rings"] == 3 and {"samples", "head", "clim_hour"} <= set(hx)
    for src in ("samples", "head", "clim_hour"):
        assert np.isfinite(hx[src]["nll"]) and 0 <= hx[src]["top5"] <= 1
    assert {"hex_nll_samples", "hex_nll_head"} <= set(report["per_window"].columns)


def test_probabilistic_arm_without_head_is_scored_on_the_default_grid(tmp_path):
    from movement.evaluation import evaluate_model

    cfg = _cfg(tmp_path, probabilistic=True, n_samples_train=4, n_samples_eval=8)
    dm = DataModule.build(cfg)
    model = build_model(cfg.model, cfg.windowing, cfg.transforms, data_spec=dm.model_data_spec())
    report = evaluate_model(model, dm, torch.device("cpu"), split="test")
    assert report["hex"]["rings"] == 10 and "head" not in report["hex"] and "samples" in report["hex"]


def test_study_arms_and_tail_layout():
    from test_cov_transformer import _study_module

    mod = _study_module()
    assert {"pmem", "pmem_hex", "phex"} <= mod.PROBABILISTIC_ARMS
    assert "transforms.memory=true" in mod.ARMS["pmem_hex"][1] and "model.hex_rings=10" in mod.ARMS["pmem_hex"][1]
    args = mod.build_parser().parse_args(["--dataset", "boar_reshaped", "--val", "tail"])
    study_dir, tag, units = mod.study_layout(args)
    assert "_tailval_" in tag and "data.split_unit=individual_kfold_tailval" in units[0][1]
    args = mod.build_parser().parse_args(["--dataset", "boar_reshaped"])
    assert "_tailval" not in mod.study_layout(args)[1]


def test_report_hex_section(tmp_path, monkeypatch):
    from test_calibration import _fake
    from test_cov_transformer import _study_module

    mod = _study_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    study = tmp_path / "study"
    for unit in ("fold0", "fold1"):
        for arm, head in (("pnocov", False), ("pmem_hex", True)):
            run = _fake(study, unit, arm, {"0.5": 0.5, "0.8": 0.8, "0.9": 0.9, "0.95": 0.95})
            m = json.loads((run / "metrics.json").read_text())
            m["hex"] = {"rings": 10, "samples": {"nll": 4.0, "top1": 0.1, "top5": 0.3, "outside": 0.05},
                        "clim_hour": {"nll": 5.0, "top1": 0.12, "top5": 0.25, "outside": 0.05},
                        "stay_put": {"top1": 0.2}}
            if head:
                m["hex"]["head"] = {"nll": 3.5, "top1": 0.15, "top5": 0.4, "outside": 0.05}
            (run / "metrics.json").write_text(json.dumps(m))
    text = mod.write_report("boar", study, "kfold9_tailval_seed42", "fold").read_text(encoding="utf-8")
    assert "## Destination hexagon" in text and "tail validation" in text
    assert "| pmem_hex | head | 3.500 | 15.0% | 40.0% |" in text
    assert "| clim (hour) | samples | 5.000 | 12.0% | 25.0% |" in text
    assert "| stay put | centre cell | — | 20.0% | — |" in text
    assert "| fold0 | 3.500 | 4.000 |" in text  # per-unit best source
