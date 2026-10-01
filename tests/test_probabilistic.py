"""Probabilistic head, energy score, climatology baselines and rotation augmentation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_cov_transformer import _config, _context, _model, _study_module, _write_raw_and_covariates

from movement.config import load_config
from movement.data.datamodule import DataModule
from movement.evaluation.scoring import (
    climatology_samples,
    energy_score,
    energy_score_np,
    random_rotations,
    spatial_median,
)
from movement.models import build_model


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def test_energy_score_of_identical_samples_is_the_euclidean_error():
    torch.manual_seed(0)
    point = torch.randn(5, 4, 2) * 100
    target = torch.randn(5, 4, 2) * 100
    samples = point.unsqueeze(1).expand(5, 8, 4, 2).double()
    es = energy_score(samples, target.double())
    err = torch.linalg.norm(point - target, dim=-1).double()
    assert torch.allclose(es, err, atol=1e-3)
    # M = 1 is the plain error too.
    assert torch.allclose(energy_score(point.unsqueeze(1).double(), target.double()), err, atol=1e-3)


def test_energy_score_rewards_calibrated_spread_over_stay_put():
    """Targets on a ring of radius r around 0: samples spread on that ring beat the point at 0."""
    rng = np.random.default_rng(0)
    n, m, r = 2000, 64, 100.0
    ang_t = rng.uniform(0, 2 * np.pi, n)
    target = np.stack([r * np.cos(ang_t), r * np.sin(ang_t)], -1)[:, None, :]  # (n, 1, 2)
    ang_s = rng.uniform(0, 2 * np.pi, (n, m))
    ring = np.stack([r * np.cos(ang_s), r * np.sin(ang_s)], -1)[:, :, None, :]  # (n, m, 1, 2)
    stay = np.zeros((n, 1, 1, 2))
    es_ring = energy_score_np(ring, target).mean()
    es_stay = energy_score_np(stay, target).mean()
    assert es_stay == pytest.approx(r, rel=1e-6)
    assert es_ring < 0.75 * es_stay  # theory: 4r/pi - ... ≈ 0.64 r


def test_energy_score_gradient_is_finite_with_coincident_samples():
    samples = torch.zeros(2, 4, 3, 2, requires_grad=True)
    target = torch.ones(2, 3, 2)
    energy_score(samples, target).mean().backward()
    assert torch.isfinite(samples.grad).all()


def test_spatial_median_is_robust_to_an_outlier():
    pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1000.0, 1000.0]])
    med = spatial_median(pts[None, :, None, :].double())[0, 0]
    assert torch.linalg.norm(med - torch.tensor([0.5, 0.5]).double()) < 1.0


def test_random_rotations_are_orthonormal_and_mirror_half_the_time():
    rot = random_rotations(4000, np.random.default_rng(1))
    eye = np.einsum("nij,nkj->nik", rot, rot)
    assert np.allclose(eye, np.eye(2), atol=1e-12)
    det = np.linalg.det(rot)
    assert np.allclose(np.abs(det), 1.0) and 0.45 < (det < 0).mean() < 0.55


def test_climatology_samples_draw_same_hour_futures_and_preserve_distance():
    rng = np.random.default_rng(0)
    fut = np.zeros((200, 3, 2))
    hours = np.repeat([2, 14], 100)
    fut[hours == 2] = [[10.0, 0.0]] * 3   # night: 10 m
    fut[hours == 14] = [[1.0, 0.0]] * 3   # day: 1 m
    out = climatology_samples(fut, hours, np.array([2, 14, 14]), 16, rng, by_hour=True)
    assert out.shape == (3, 16, 3, 2)
    dist = np.linalg.norm(out, axis=-1)
    assert np.allclose(dist[0], 10.0) and np.allclose(dist[1:], 1.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def test_probabilistic_model_returns_distinct_samples_and_trains():
    torch.manual_seed(0)
    m = _model(probabilistic=True, n_samples_train=6, n_samples_eval=10)
    ctx = _context()
    out = m(torch.randn(3, 12, 2), context=ctx)
    assert out.shape == (3, 6, 4, 2)
    assert out[:, 0].sub(out[:, 1]).abs().max() > 0  # samples differ
    loss = energy_score(out.cumsum(2), torch.randn(3, 4, 2).cumsum(1)).mean()
    loss.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    m.eval()
    assert m(torch.randn(3, 12, 2), context=ctx).shape == (3, 10, 4, 2)
    assert m(torch.randn(3, 12, 2), context={**ctx, "n_samples": 33}).shape == (3, 33, 4, 2)  # > chunk


def test_point_mode_is_unchanged_by_the_new_options():
    m = _model()
    assert not m.is_probabilistic and not hasattr(m, "noise_query")
    assert m(torch.randn(2, 12, 2), context=_context(B=2)).shape == (2, 4, 2)


def test_probabilistic_model_needs_two_samples():
    with pytest.raises(ValueError, match="at least 2"):
        _model(probabilistic=True, n_samples_train=1)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
@pytest.fixture()
def aug_setup(tmp_path):
    raw_dir, gee_dir = _write_raw_and_covariates(tmp_path)
    cfg = _config(tmp_path, raw_dir, gee_dir)
    cfg = cfg.model_copy(update={"trainer": cfg.trainer.model_copy(update={"augment_rotation": True})})
    return cfg, DataModule.build(cfg)


def test_augmentation_rotates_train_only_and_preserves_step_lengths(aug_setup):
    cfg, dm = aug_setup
    plain = DataModule.dataset.__get__(dm)("train", scaled=False)
    assert plain.augment_rotation  # train split
    assert not dm.dataset("val").augment_rotation and not dm.dataset("test").augment_rotation
    np.random.seed(0)
    item = plain[0]  # one draw: every __getitem__ call re-randomises
    x_aug, y_aug = item[0].numpy(), item[1].numpy()
    raw = plain.transform.apply(plain.windows[0].features, plain.windows[0].target, plain.windows[0].timestamp)
    assert np.allclose(np.linalg.norm(x_aug[:, :2], axis=1), np.linalg.norm(raw["x"][:, :2], axis=1), atol=1e-3)
    assert np.allclose(np.linalg.norm(y_aug, axis=1), np.linalg.norm(raw["y"], axis=1), atol=1e-3)
    assert not np.allclose(y_aug, raw["y"])  # actually rotated
    # The same rotation is applied to inputs and targets: angle between the last
    # observed step and the first target step is preserved (up to a mirror).
    def ang(a, b):
        return np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1))
    assert ang(x_aug[-1, :2], y_aug[0]) == pytest.approx(ang(raw["x"][-1, :2], raw["y"][0]), abs=1e-3)


def test_augmentation_refuses_turning_angle(aug_setup):
    cfg, dm = aug_setup
    from movement.data.datamodule import MovementDataset
    from movement.data.transforms import WindowTransform

    with pytest.raises(ValueError, match="turning_angle"):
        MovementDataset(windows=dm.train_windows, augment_rotation=True,
                        transform=WindowTransform(input_len=8, horizon=4, turning_angle=True))


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def test_probabilistic_train_eval_round_trip(tmp_path):
    from movement.evaluation import evaluate_model, save_plots
    from movement.training import Trainer
    from movement.utils.tracking import Tracker, make_run_dir

    raw_dir, gee_dir = _write_raw_and_covariates(tmp_path)
    cfg = _config(tmp_path, raw_dir, gee_dir, probabilistic=True, n_samples_train=4, n_samples_eval=8)
    cfg = cfg.model_copy(update={"trainer": cfg.trainer.model_copy(update={"augment_rotation": True})})
    dm = DataModule.build(cfg)
    model = build_model(cfg.model, cfg.windowing, cfg.transforms, data_spec=dm.model_data_spec())
    run_dir = make_run_dir(cfg)
    tracker = Tracker(cfg, run_dir)
    summary = Trainer(model, dm, cfg, tracker, run_dir, device=torch.device("cpu")).fit()
    tracker.close()
    assert summary["selection_metric"] == "val_es" and summary["best_val_metric"] is not None

    report, inv, samples = evaluate_model(model, dm, torch.device("cpu"), split="test", return_details=True)
    assert report["probabilistic"] and samples.shape[1:] == (8, 4, 2)
    for key in ("es", "es_cp", "es_clim_all", "es_clim_hour", "ade"):
        assert np.isfinite(report[key])
    assert report["es_cp"] == report["ade_cp"]
    assert "es" in report["per_horizon"].columns
    assert save_plots(inv, run_dir, n_plots=1, samples=samples)


def test_point_model_energy_score_equals_ade(tmp_path):
    from movement.evaluation import evaluate_model

    raw_dir, gee_dir = _write_raw_and_covariates(tmp_path)
    cfg = _config(tmp_path, raw_dir, gee_dir)
    dm = DataModule.build(cfg)
    model = build_model(cfg.model, cfg.windowing, cfg.transforms, data_spec=dm.model_data_spec())
    report = evaluate_model(model, dm, torch.device("cpu"), split="test")
    assert not report["probabilistic"]
    assert report["es"] == pytest.approx(report["ade"], rel=1e-9)


# ---------------------------------------------------------------------------
# Study script
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arm,cov,prob,aug", [("pcov", True, True, True), ("pnocov", False, True, True),
                                               ("nocov_aug", False, False, True), ("pcov_idx", True, True, True),
                                               ("cov_idx", True, False, False)])
def test_study_script_probabilistic_arms_resolve(arm, cov, prob, aug, monkeypatch, tmp_path):
    mod = _study_module()
    for var in ("RAW_DATASET_PATH", "PROCESSED_DATASET_PATH", "GEE_DATASET_PATH"):
        monkeypatch.setenv(var, str(tmp_path))
    args = mod.build_parser().parse_args(["--dataset", "boar_reshaped"])
    _, _, units = mod.study_layout(args)
    config, extra = mod.ARMS[arm]
    cfg = load_config(config, [*units[0][1], *extra, f"trainer.run_dir={tmp_path}"])
    assert cfg.covariates.enabled is cov and cfg.model.use_covariates is cov
    assert cfg.model.probabilistic is prob and cfg.trainer.augment_rotation is aug
    assert {"pcov", "pnocov", "pcov_idx"} <= set(mod.DEFAULT_ARMS)
    if arm.endswith("_idx"):
        import re

        cols = ["s2_B4", "s2_ndvi", "s2_evi", "s2_savi", "s2_ndwi", "s2_ndmi", "s2_nbr", "s2_ndvi_buf30",
                "s2_ndvi_rate", "s2_n_obs", "s2_snow_fraction", "s2_composite_age_days"]
        pats = [re.compile(p) for p in cfg.covariates.include]
        kept = [c for c in cols if any(p.search(c) for p in pats)]
        assert kept == ["s2_ndvi", "s2_evi", "s2_savi", "s2_ndwi", "s2_ndmi", "s2_nbr"]


def _fake(root: Path, unit: str, arm: str, *, ade: float, es: float | None, clim: float | None, n: int = 100):
    run = root / unit / arm / "20260101-000000-x-s42"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "best.pt").write_bytes(b"")
    m = {"ade": ade, "fde": ade, "ade_cp": 500.0, "ade_cv": 900.0, "n_windows": n}
    if es is not None:
        m.update(es=es, es_clim_all=clim + 20, es_clim_hour=clim, probabilistic=arm.startswith("p"))
    (run / "metrics.json").write_text(json.dumps(m))
    (run / "manifest.json").write_text(json.dumps({"parameter_count": 10, "train_metrics": {}}))
    (run / "split.json").write_text(json.dumps({"train": ["a"], "val": ["b"], "test": ["c"]}))


def test_report_energy_section(tmp_path, monkeypatch):
    mod = _study_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    study = tmp_path / "study"
    for unit, clim in (("fold0", 400.0), ("fold1", 380.0)):
        _fake(study, unit, "nocov", ade=495.0, es=None, clim=None)  # legacy point run: no ES keys
        _fake(study, unit, "pnocov", ade=499.0, es=clim - 10, clim=clim)
        _fake(study, unit, "pcov", ade=499.0, es=clim - 5, clim=clim)
    df = mod.collect(study, "fold")
    assert not df.loc[df["arm"] == "pnocov", "collapsed"].any()  # judged on ES, not the ADE ratio
    assert (df.loc[df["arm"] == "nocov", "es"] == 495.0).all()  # point ES == ADE
    assert (df.loc[df["arm"] == "nocov", "es_clim_hour"].to_numpy() == [400.0, 380.0]).all()
    text = mod.write_report("boar", study, "kfold2_seed42", "fold").read_text(encoding="utf-8")
    assert "## Energy score" in text
    assert "| `pnocov` − clim (hour) | -10.0 ± 0.0 |" in text
    assert "| `pcov` − `pnocov` | +5.0 ± 0.0 |" in text
    assert "2/2" in text


def test_training_can_reuse_a_split_from_an_arm_with_other_covariates(tmp_path):
    """pcov_idx reuses the cov arm's split.json, whose covariate_scalers.json has other columns."""
    raw_dir, gee_dir = _write_raw_and_covariates(tmp_path)
    cfg_all = _config(tmp_path, raw_dir, gee_dir)
    dm_all = DataModule.build(cfg_all)
    run = tmp_path / "cov_run"
    run.mkdir()
    dm_all.persist_split(run)
    dm_all.persist_scaler(run)
    dm_all.persist_covariates(run)
    first = dm_all.covariate_columns[0]
    cfg_one = cfg_all.model_copy(update={"covariates": cfg_all.covariates.model_copy(
        update={"include": [f"^{first}$"]})})
    with pytest.raises(ValueError, match="different covariate columns"):
        DataModule.from_split_file(cfg_one, run / "split.json")  # eval path: strict
    dm_one = DataModule.from_split_file(cfg_one, run / "split.json", refit_covariate_scaler=True)
    assert dm_one.covariate_columns == [first]
    assert dm_one.covariate_scaler.columns == [first]
