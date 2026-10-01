"""Green-wave experiment: imagery embargo, per-window outputs, study arms and the pre-registered analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from movement.config import load_config
from movement.data.datamodule import DataModule
from movement.models import build_model

from test_cov_transformer import _config, _study_module, _write_raw_and_covariates

ROOT = Path(__file__).resolve().parents[1]


def _gw_module():
    spec = importlib.util.spec_from_file_location("gw", ROOT / "scripts" / "green_wave_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_embargo_hides_recent_covariates(tmp_path):
    raw_dir, gee_dir = _write_raw_and_covariates(tmp_path)
    cfg = _config(tmp_path, raw_dir, gee_dir)
    cfg = cfg.model_copy(update={"covariates": cfg.covariates.model_copy(update={"embargo_days": 3 / 24})})
    dm = DataModule.build(cfg)
    ds = dm.dataset("test")
    assert ds.covariate_embargo_s == pytest.approx(3 * 3600)
    extras = ds[0][3]
    # 8 hourly fixes: the last 3 (offsets 0, 1, 2 h before the origin) are hidden.
    assert torch.all(extras["covariate_missing"][-3:] == 1)
    assert torch.all(extras["covariates"][-3:] == 0)
    plain = DataModule.build(_config(tmp_path, raw_dir, gee_dir)).dataset("test")[0][3]
    assert torch.equal(extras["covariate_missing"][:-3], plain["covariate_missing"][:-3])


def test_eval_writes_per_window_scores(tmp_path):
    from movement.evaluation import evaluate_model, write_eval_outputs

    raw_dir, gee_dir = _write_raw_and_covariates(tmp_path)
    cfg = _config(tmp_path, raw_dir, gee_dir, probabilistic=True, n_samples_train=4, n_samples_eval=8)
    dm = DataModule.build(cfg)
    model = build_model(cfg.model, cfg.windowing, cfg.transforms, data_spec=dm.model_data_spec())
    report = evaluate_model(model, dm, torch.device("cpu"), split="test")
    pw = report["per_window"]
    assert list(pw.columns)[:6] == ["individual_id", "t_origin", "es", "ade", "fde", "ade_cp"]
    assert len(pw) == report["n_windows"]
    assert pw["es"].mean() == pytest.approx(report["es"], rel=1e-9)
    assert pw["ade"].mean() == pytest.approx(report["ade"], rel=1e-6)
    write_eval_outputs(tmp_path, report, model_name="x", split="test")
    assert (tmp_path / "per_window.csv").exists()


@pytest.mark.parametrize("arm,cov,embargo", [("gw_nocov", False, 0), ("gw_cov", True, 9), ("gw_cov_centred", True, 0)])
def test_green_wave_arms_resolve(arm, cov, embargo, monkeypatch, tmp_path):
    import re

    mod = _study_module()
    for var in ("RAW_DATASET_PATH", "PROCESSED_DATASET_PATH", "GEE_DATASET_PATH"):
        monkeypatch.setenv(var, str(tmp_path))
    args = mod.build_parser().parse_args(["--dataset", "mule_deer_6h", "--input-len", "56", "--horizon", "12"])
    _, _, units = mod.study_layout(args)
    config, extra = mod.ARMS[arm]
    cfg = load_config(config, [*units[0][1], *extra, f"trainer.run_dir={tmp_path}"])
    assert cfg.model.probabilistic and cfg.trainer.augment_rotation
    assert cfg.covariates.enabled is cov and cfg.covariates.embargo_days == embargo
    if cov:
        assert cfg.model.covariate_features == "both"
        pats = [re.compile(p) for p in cfg.covariates.include]
        cols = ["s2_ndvi", "s2_ndvi_rate", "s2_snow_fraction", "s2_ndvi_buf30", "s2_evi", "s2_n_obs"]
        assert [c for c in cols if any(p.search(c) for p in pats)] == ["s2_ndvi", "s2_ndvi_rate", "s2_snow_fraction"]


def test_dataset_overrides_accept_custom_windows(tmp_path):
    mod = _study_module()
    csv = tmp_path / "d.csv"
    ts = pd.date_range("2019-01-01", periods=200, freq="6h")
    pd.DataFrame({"timestamp": ts, "lon": 0.0, "lat": 0.0, "individual_id": "a", "study_id": "s"}).to_csv(csv, index=False)
    out = mod.dataset_overrides(csv, input_len=56, horizon=12, stride=2)
    assert "windowing.input_len=56" in out and "windowing.horizon=12" in out
    assert "windowing.stride=2" in out and "windowing.eval_stride=12" in out


def test_seasons_follow_the_preregistration():
    gw = _gw_module()
    t = pd.Series(pd.to_datetime(["2019-04-01", "2019-06-30 23:00", "2019-07-01", "2018-12-01", "2019-03-15",
                                  "2019-03-16", "2019-11-30"], format="mixed"))
    assert gw.season(t).tolist() == ["spring migration", "spring migration", "summer/autumn", "winter",
                                     "winter", "summer/autumn", "summer/autumn"]


def _fake_study(root: Path, spring_gain: float, winter_gain: float):
    rng = np.random.default_rng(1)
    for f in range(5):
        n = 200
        t = pd.to_datetime(rng.choice(pd.date_range("2018-01-01", "2019-09-30", freq="6h"), n))
        animals = [f"a{f}_{i % 6}" for i in range(n)]
        base = 400 + rng.normal(0, 20, n)
        s = pd.Series(t).map(lambda x: x.month * 100 + x.day)
        gain = np.where((s >= 401) & (s <= 630), spring_gain, np.where((s >= 1201) | (s <= 315), winter_gain, 0.0))
        for arm, es in (("gw_nocov", base), ("gw_cov", base - gain + rng.normal(0, 0.5, n)),
                        ("gw_cov_centred", base - gain)):
            run = root / f"fold{f}" / arm / "20260101-000000-x"
            run.mkdir(parents=True)
            pd.DataFrame({"individual_id": animals, "t_origin": t, "es": es, "ade": es * 1.3, "fde": es * 1.8,
                          "ade_cp": 600.0}).to_csv(run / "per_window.csv", index=False)


def test_analysis_decision_rules(tmp_path, monkeypatch):
    import json

    gw = _gw_module()
    monkeypatch.setattr(gw, "REPO_ROOT", tmp_path)
    study = tmp_path / "runs" / "covariate_study" / "fake" / "tag"
    _fake_study(study, spring_gain=10.0, winter_gain=0.0)
    out = gw.main(["--dataset", "fake", "--tag", "tag", "--reference", "gw_nocov", "--arms", "gw_cov,gw_cov_centred"])
    v = json.loads(out.with_suffix(".json").read_text())
    assert v["gw_cov"]["H1_spring"] and v["gw_cov"]["H2_spring_vs_winter"]
    text = out.read_text(encoding="utf-8")
    assert "H1 (spring) **supported**" in text


def test_analysis_rejects_no_effect(tmp_path, monkeypatch):
    import json

    gw = _gw_module()
    monkeypatch.setattr(gw, "REPO_ROOT", tmp_path)
    study = tmp_path / "runs" / "covariate_study" / "fake" / "tag"
    _fake_study(study, spring_gain=0.0, winter_gain=0.0)
    out = gw.main(["--dataset", "fake", "--tag", "tag", "--reference", "gw_nocov", "--arms", "gw_cov,gw_cov_centred"])
    v = json.loads(out.with_suffix(".json").read_text())
    assert not v["gw_cov"]["H1_spring"] and not v["gw_cov"]["H2_spring_vs_winter"]


def test_exploratory_report_does_not_overwrite_the_preregistered_one(tmp_path, monkeypatch):
    gw = _gw_module()
    monkeypatch.setattr(gw, "REPO_ROOT", tmp_path)
    study = tmp_path / "runs" / "covariate_study" / "fake" / "tag"
    _fake_study(study, spring_gain=0.0, winter_gain=0.0)
    main = gw.main(["--dataset", "fake", "--tag", "tag", "--reference", "gw_nocov", "--arms", "gw_cov"])
    expl = gw.main(["--dataset", "fake", "--tag", "tag", "--reference", "gw_nocov", "--arms", "gw_cov",
                    "--exploratory", "gw_cov"])
    assert main.name == "fake_green_wave.md" and expl.name == "fake_green_wave_gw_cov.md"
    assert "EXPLORATORY" in expl.read_text(encoding="utf-8") and "EXPLORATORY" not in main.read_text(encoding="utf-8")
