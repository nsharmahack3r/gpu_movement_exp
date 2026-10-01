"""Calibration of probabilistic forecasts: region coverage, eval output, study report, re-eval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_cov_transformer import _config, _study_module, _write_raw_and_covariates

from movement.data.datamodule import DataModule
from movement.evaluation.scoring import COVERAGE_LEVELS, CURVE_LEVELS, region_calibration
from movement.models import build_model


def test_calibrated_samples_cover_at_the_nominal_rate():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(4000, 400, 3, 2)) * 80
    target = rng.normal(size=(4000, 3, 2)) * 80
    cal = region_calibration(samples, target)
    for a in COVERAGE_LEVELS:
        assert cal["coverage"][f"{a:g}"] == pytest.approx(a, abs=0.02)
        assert len(cal["coverage_step"][f"{a:g}"]) == 3
    # Rayleigh median radius for sigma = 80 m: 80 * sqrt(2 ln 2) ≈ 94 m.
    assert cal["radius"]["0.5"] == pytest.approx(94.2, rel=0.05)


def test_overconfident_and_underconfident_samples_are_detected():
    rng = np.random.default_rng(1)
    target = rng.normal(size=(3000, 2, 2)) * 100
    narrow = region_calibration(rng.normal(size=(3000, 200, 2, 2)) * 40, target)
    wide = region_calibration(rng.normal(size=(3000, 200, 2, 2)) * 250, target)
    assert narrow["coverage"]["0.9"] < 0.6  # overconfident
    assert wide["coverage"]["0.5"] > 0.85  # underconfident


def test_probabilistic_eval_reports_calibration_for_model_and_climatology(tmp_path):
    from movement.evaluation import evaluate_model

    raw_dir, gee_dir = _write_raw_and_covariates(tmp_path)
    cfg = _config(tmp_path, raw_dir, gee_dir, probabilistic=True, n_samples_train=4, n_samples_eval=16)
    dm = DataModule.build(cfg)
    model = build_model(cfg.model, cfg.windowing, cfg.transforms, data_spec=dm.model_data_spec())
    report = evaluate_model(model, dm, torch.device("cpu"), split="test")
    for key in ("calibration", "calibration_clim_hour"):
        cov = report[key]["coverage"]
        assert set(cov) == {f"{a:g}" for a in COVERAGE_LEVELS}
        assert all(0.0 <= v <= 1.0 for v in cov.values())
    assert len(report["calibration_curve"]) == len(CURVE_LEVELS)
    json.dumps(report["calibration"])  # serialisable for metrics.json


def test_point_eval_has_no_model_calibration(tmp_path):
    from movement.evaluation import evaluate_model

    raw_dir, gee_dir = _write_raw_and_covariates(tmp_path)
    cfg = _config(tmp_path, raw_dir, gee_dir)
    dm = DataModule.build(cfg)
    model = build_model(cfg.model, cfg.windowing, cfg.transforms, data_spec=dm.model_data_spec())
    report = evaluate_model(model, dm, torch.device("cpu"), split="test")
    assert "calibration" not in report and "calibration_clim_hour" in report


def _fake(root: Path, unit: str, arm: str, cov: dict | None, n: int = 100):
    run = root / unit / arm / "20260101-000000-x-s42"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "best.pt").write_bytes(b"")
    m = {"ade": 450.0, "fde": 600.0, "ade_cp": 500.0, "ade_cv": 900.0, "n_windows": n,
         "es": 390.0, "es_clim_all": 450.0, "es_clim_hour": 440.0, "probabilistic": cov is not None}
    if cov is not None:
        m["calibration"] = {"coverage": cov, "radius": {"0.5": 200.0, "0.9": 500.0}}
        m["calibration_curve"] = {f"{a:g}": a for a in CURVE_LEVELS}
        m["calibration_clim_hour"] = {"coverage": {k: v - 0.1 for k, v in cov.items()},
                                      "radius": {"0.5": 250.0, "0.9": 600.0}}
        m["calibration_curve_clim_hour"] = {f"{a:g}": a * 0.9 for a in CURVE_LEVELS}
    (run / "metrics.json").write_text(json.dumps(m))
    (run / "manifest.json").write_text(json.dumps({"parameter_count": 10, "train_metrics": {}}))
    (run / "split.json").write_text(json.dumps({"train": ["a"], "val": ["b"], "test": ["c"]}))
    return run


def test_report_calibration_section_and_figure(tmp_path, monkeypatch):
    mod = _study_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    study = tmp_path / "study"
    cov = {"0.5": 0.48, "0.8": 0.79, "0.9": 0.88, "0.95": 0.94}
    for unit in ("fold0", "fold1"):
        _fake(study, unit, "pnocov", cov)
        _fake(study, unit, "nocov", None)
    out = mod.write_report("boar", study, "kfold2_seed42", "fold")
    text = out.read_text(encoding="utf-8")
    assert "## Calibration (probabilistic arms)" in text
    assert "| pnocov | 48.0% | 79.0% | 88.0% | 94.0% | 200 | 500 |" in text
    assert "| clim (hour) | 38.0% | 69.0% | 78.0% | 84.0% | 250 | 600 |" in text
    assert "| nocov |" not in text.split("## Calibration")[1].split("\n\n")[2]  # point arm not listed
    assert (out.parent / "figures" / "boar_kfold2_seed42_calibration.png").exists()


def test_reeval_runs_eval_on_the_latest_run_of_each_requested_arm(tmp_path, monkeypatch):
    mod = _study_module()
    study = tmp_path / "study"
    runs = [_fake(study, u, "pnocov", {"0.5": 0.5}) for u in ("fold0", "fold1")]
    _fake(study, "fold0", "nocov", None)
    calls = []
    monkeypatch.setattr(mod, "_run", lambda cmd, dry: calls.append(cmd))
    mod.reeval(study, "fold", "pnocov", dry=False)
    assert [c[-1] for c in calls] == [str(r) for r in runs]
    assert all(c[2] == "movement.cli.eval" for c in calls)
    calls.clear()
    mod.reeval(study, "fold", "prob", dry=False)  # every probabilistic arm present
    assert len(calls) == 2
    with pytest.raises(SystemExit):
        mod.reeval(study, "fold", "bogus", dry=False)


def test_first_arm_defines_the_split_and_later_arms_reuse_it(tmp_path, monkeypatch):
    mod = _study_module()
    study = tmp_path / "study"
    calls = []

    def fake_run(cmd, dry):
        calls.append(cmd)
        if cmd[2] == "movement.cli.train":
            arm_dir = Path(next(o.split("=", 1)[1] for o in cmd if o.startswith("trainer.run_dir=")))
            run = arm_dir / "20260101-000000-x-s42"
            (run / "checkpoints").mkdir(parents=True)
            (run / "checkpoints" / "best.pt").write_bytes(b"")
            (run / "split.json").write_text("{}")
        else:
            (Path(cmd[-1]) / "metrics.json").write_text("{}")

    monkeypatch.setattr(mod, "_run", fake_run)
    mod.run_unit("fold0", [], ["pnocov", "faunaformer"], [], study, dry=False)
    train_calls = [c for c in calls if c[2] == "movement.cli.train"]
    assert "--split-file" not in train_calls[0]  # first arm builds the split
    i = train_calls[1].index("--split-file")
    assert "pnocov" in train_calls[1][i + 1]  # second arm reuses it
    # A later invocation with a new arm reuses the existing split too.
    calls.clear()
    mod.run_unit("fold0", [], ["pcov_idx"], [], study, dry=False)
    c = [c for c in calls if c[2] == "movement.cli.train"][0]
    assert "--split-file" in c and "/fold0/" in c[c.index("--split-file") + 1].replace("\\", "/")
