"""FaunaFormer: covariate change features, late gated fusion, config, registry, study arm."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_cov_transformer import _config, _context, _model, _study_module, _write_raw_and_covariates

from movement.config import load_config
from movement.data.datamodule import DataModule
from movement.models import build_model
from movement.models.cov_transformer import (
    CovariateTransformer,
    FaunaFormer,
    covariate_changes,
    covariate_feature_names,
)


def _ff(n_cov=5, **kw) -> FaunaFormer:
    defaults = dict(input_len=12, horizon=4, n_features=2, n_covariates=n_cov, n_time_features=4,
                    d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1,
                    dim_feedforward=32, dropout=0.0, var_embed_dim=4, n_samples_train=4, n_samples_eval=6)
    defaults.update(kw)
    return FaunaFormer(**defaults)


# ---------------------------------------------------------------------------
# Change features
# ---------------------------------------------------------------------------
def test_covariate_changes_anomaly_and_step():
    v = torch.tensor([[[1.0], [2.0], [0.0], [5.0]]])  # (1, 4, 1)
    miss = torch.tensor([[[0.0], [0.0], [1.0], [0.0]]])  # fix 2 missing
    chg, chg_miss = covariate_changes(v, miss)
    anomaly, step = chg[0, :, 0], chg[0, :, 1]
    # mean over observed = (1 + 2 + 5) / 3
    m = 8.0 / 3.0
    assert torch.allclose(anomaly, torch.tensor([1 - m, 2 - m, 0.0, 5 - m]))
    assert torch.allclose(step, torch.tensor([0.0, 1.0, 0.0, 0.0]))  # steps touching the gap undefined
    assert chg_miss[0, :, 0].tolist() == [0, 0, 1, 0]
    assert chg_miss[0, :, 1].tolist() == [1, 0, 1, 1]


def test_covariate_changes_are_invariant_to_the_location_level():
    """Adding a constant (a greener *place*) changes levels but not the change features."""
    torch.manual_seed(0)
    v = torch.randn(3, 10, 4)
    miss = (torch.rand(3, 10, 4) < 0.2).float()
    v = v * (1 - miss)
    a, _ = covariate_changes(v, miss)
    b, _ = covariate_changes((v + 7.0) * (1 - miss), miss)
    assert torch.allclose(a, b, atol=1e-5)


def test_hidden_prefix_does_not_leak_into_the_window_mean():
    v = torch.tensor([[[100.0], [1.0], [3.0]]])
    miss = torch.zeros_like(v)
    hidden = torch.tensor([[True, False, False]])
    chg, chg_miss = covariate_changes(v, miss, hidden)
    assert torch.allclose(chg[0, 1:, 0], torch.tensor([-1.0, 1.0]))  # mean of fixes 1-2 only
    assert chg_miss[0, 0, 0] == 1


def test_feature_names():
    assert covariate_feature_names(["a", "b"], "changes") == ["a:anomaly", "b:anomaly", "a:step", "b:step"]
    assert covariate_feature_names(["a"], "both") == ["a", "a:anomaly", "a:step"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def test_faunaformer_defaults_and_shapes():
    torch.manual_seed(0)
    m = _ff()
    assert isinstance(m, CovariateTransformer)
    assert m.is_probabilistic and m.covariate_fusion == "late" and m.covariate_features == "changes"
    assert m.n_covariate_inputs == 10
    out = m(torch.randn(3, 12, 2), context=_context())
    assert out.shape == (3, 4, 4, 2)
    out.pow(2).mean().backward()
    assert m.fusion_gate.grad is not None and torch.isfinite(m.fusion_gate.grad).all()
    assert m.fusion_gate_value() == pytest.approx(1 / (1 + np.exp(4)), rel=1e-4)


def test_late_fusion_keeps_covariates_out_of_the_encoder():
    """With late fusion, the encoder memory must not depend on covariates."""
    torch.manual_seed(0)
    m = _ff(probabilistic=False).eval()
    ctx = _context(B=2)
    seen = {}
    m.encoder.register_forward_hook(lambda mod, inp, out: seen.setdefault("mem", []).append(out.detach()))
    x = torch.randn(2, 12, 2)
    out_a = m(x, context=ctx)
    ctx_b = {**ctx, "covariates": ctx["covariates"] + torch.randn_like(ctx["covariates"])}
    out_b = m(x, context=ctx_b)
    assert torch.allclose(seen["mem"][0], seen["mem"][1])
    assert not torch.allclose(out_a, out_b)  # ...but covariates still reach the output via the gate


def test_closed_gate_reduces_to_the_movement_only_decoder():
    torch.manual_seed(0)
    m = _ff(probabilistic=False, fusion_gate_init=-50.0).eval()
    ctx = _context(B=2)
    x = torch.randn(2, 12, 2)
    ctx_b = {**ctx, "covariates": torch.randn_like(ctx["covariates"]) * 10}
    assert torch.allclose(m(x, context=ctx), m(x, context=ctx_b), atol=1e-6)


def test_early_levels_path_is_unchanged_and_has_no_new_parameters():
    base = _model()
    assert base.covariate_fusion == "early" and base.covariate_features == "levels"
    names = {n for n, _ in base.named_parameters()} | {n for n, _ in base.named_buffers()}
    assert not any(k.startswith(("cov_attn", "fusion_gate", "cov_norm", "covariate_change_scale")) for k in names)


def test_change_scale_is_applied_and_checked():
    m = _ff(covariate_change_scale=[2.0] * 10)
    assert torch.allclose(m.covariate_change_scale, torch.full((10,), 2.0))
    with pytest.raises(ValueError, match="needs 10 values"):
        _ff(covariate_change_scale=[1.0] * 3)


# ---------------------------------------------------------------------------
# Config, registry, data spec, end to end
# ---------------------------------------------------------------------------
def test_faunaformer_yaml_resolves(monkeypatch, tmp_path):
    for var in ("RAW_DATASET_PATH", "PROCESSED_DATASET_PATH", "GEE_DATASET_PATH"):
        monkeypatch.setenv(var, str(tmp_path))
    cfg = load_config("configs/model/faunaformer.yaml")
    assert cfg.model.name == "faunaformer" and cfg.model.probabilistic
    assert cfg.model.covariate_fusion == "late" and cfg.model.covariate_features == "changes"
    assert cfg.trainer.augment_rotation and cfg.covariates.include == ["^s2_(ndvi|evi|savi|ndwi|ndmi|nbr)$"]


def test_faunaformer_train_eval_round_trip(tmp_path):
    from movement.evaluation import evaluate_model
    from movement.training import Trainer
    from movement.utils.tracking import Tracker, make_run_dir

    raw_dir, gee_dir = _write_raw_and_covariates(tmp_path)
    cfg = _config(tmp_path, raw_dir, gee_dir, name="faunaformer", n_samples_train=4, n_samples_eval=6)
    assert type(cfg.model).__name__ == "FaunaFormerModelConfig"
    dm = DataModule.build(cfg)
    spec = dm.model_data_spec()
    scale = spec["covariate_change_scale"]
    assert len(scale) == 2 * dm.n_covariates and all(v > 0 for v in scale)
    model = build_model(cfg.model, cfg.windowing, cfg.transforms, data_spec=spec)
    assert isinstance(model, FaunaFormer)
    run_dir = make_run_dir(cfg)
    tracker = Tracker(cfg, run_dir)
    summary = Trainer(model, dm, cfg, tracker, run_dir, device=torch.device("cpu")).fit()
    tracker.close()
    assert summary["selection_metric"] == "val_es"
    report = evaluate_model(model, dm, torch.device("cpu"), split="test")
    assert np.isfinite(report["es"]) and report["probabilistic"]
    sel = model.covariate_selection_summary(dm, torch.device("cpu"), split="test")
    assert len(sel) == 2 * dm.n_covariates and sel["covariate"].str.contains(":anomaly").any()


def test_study_arms(monkeypatch, tmp_path):
    mod = _study_module()
    for var in ("RAW_DATASET_PATH", "PROCESSED_DATASET_PATH", "GEE_DATASET_PATH"):
        monkeypatch.setenv(var, str(tmp_path))
    args = mod.build_parser().parse_args(["--dataset", "boar_reshaped"])
    _, _, units = mod.study_layout(args)
    assert "faunaformer" in mod.DEFAULT_ARMS
    for arm, fusion, feats in (("faunaformer", "late", "changes"), ("ff_levels", "late", "levels"),
                               ("ff_early", "early", "changes")):
        config, extra = mod.ARMS[arm]
        cfg = load_config(config, [*units[0][1], *extra, f"trainer.run_dir={tmp_path}"])
        assert cfg.model.name == "faunaformer"
        assert (cfg.model.covariate_fusion, cfg.model.covariate_features) == (fusion, feats)
        assert cfg.data.split_unit == "individual_kfold"


def test_fusion_gate_is_excluded_from_weight_decay(tmp_path):
    from movement.training import Trainer
    from movement.utils.tracking import Tracker, make_run_dir

    raw_dir, gee_dir = _write_raw_and_covariates(tmp_path)
    cfg = _config(tmp_path, raw_dir, gee_dir, name="faunaformer", n_samples_train=4, n_samples_eval=6)
    dm = DataModule.build(cfg)
    model = build_model(cfg.model, cfg.windowing, cfg.transforms, data_spec=dm.model_data_spec())
    run_dir = make_run_dir(cfg)
    tr = Trainer(model, dm, cfg, Tracker(cfg, run_dir), run_dir, device=torch.device("cpu"))
    groups = tr.optimizer.param_groups
    assert len(groups) == 2 and groups[1]["weight_decay"] == 0.0
    assert any(p is model.fusion_gate for p in groups[1]["params"])
    assert not any(p is model.fusion_gate for p in groups[0]["params"])


def test_ff_no_nbr_arm_keeps_five_indices_and_drops_nbr(monkeypatch, tmp_path):
    import re

    mod = _study_module()
    for var in ("RAW_DATASET_PATH", "PROCESSED_DATASET_PATH", "GEE_DATASET_PATH"):
        monkeypatch.setenv(var, str(tmp_path))
    args = mod.build_parser().parse_args(["--dataset", "boar_reshaped"])
    _, _, units = mod.study_layout(args)
    config, extra = mod.ARMS["ff_no_nbr"]
    cfg = load_config(config, [*units[0][1], *extra, f"trainer.run_dir={tmp_path}"])
    # Everything but the covariate list is FaunaFormer's.
    assert cfg.model.name == "faunaformer" and cfg.model.probabilistic and cfg.trainer.augment_rotation
    assert (cfg.model.covariate_fusion, cfg.model.covariate_features) == ("late", "changes")
    cols = ["s2_ndvi", "s2_evi", "s2_savi", "s2_ndwi", "s2_ndmi", "s2_nbr", "s2_ndvi_buf30", "s2_nbr_buf30",
            "s2_B4", "s2_ndvi_rate", "s2_n_obs"]
    pats = [re.compile(p) for p in cfg.covariates.include]
    kept = [c for c in cols if any(p.search(c) for p in pats)]
    assert kept == ["s2_ndvi", "s2_evi", "s2_savi", "s2_ndwi", "s2_ndmi"]
    assert "ff_no_nbr" in mod.PROBABILISTIC_ARMS and "ff_no_nbr" not in mod.DEFAULT_ARMS


def test_bounded_indices_are_clipped_to_unit_range(tmp_path):
    import pandas as pd

    from movement.config import CovariatesConfig
    from movement.data.covariates import _read_covariate_csv as read_cov
    from movement.data.covariates import attach_covariates  # noqa: F401  (import path check)

    f = tmp_path / "x_sentinel2.csv"
    pd.DataFrame({
        "timestamp": ["2019-01-01 00:00:00", "2019-01-01 01:00:00", "2019-01-01 02:00:00"],
        "individual_id": ["a", "a", "a"], "study_id": ["s", "s", "s"],
        "s2_composite_date": ["2019-01-01"] * 3,
        "s2_evi": [0.3, -390.0, 275.0], "s2_ndvi_buf30": [0.1, 2.0, -3.0], "s2_B4": [0.2, 5.0, 0.1],
        "unused_col": [1, 2, 3],
    }).to_csv(f, index=False)
    out, cols = read_cov(f, CovariatesConfig(enabled=True, path=tmp_path))
    assert out["s2_evi"].tolist() == pytest.approx([0.3, -1.0, 1.0])
    assert out["s2_ndvi_buf30"].tolist() == pytest.approx([0.1, 1.0, -1.0])
    assert out["s2_B4"].tolist() == pytest.approx([0.2, 5.0, 0.1])  # reflectance is not clipped
    assert "unused_col" not in out.columns
