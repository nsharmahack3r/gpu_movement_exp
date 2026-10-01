"""Covariate-aware Transformer: split fix, covariate join/scaling, time context, model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from movement.config import CovariatesConfig, load_config
from movement.data.covariates import (
    CovariateScaler,
    attach_covariates,
    time_features,
    window_time_context,
)
from movement.data.datamodule import DataModule, unpack_batch
from movement.data.loading import (
    Trajectory,
    assert_disjoint_ids,
    split_individuals,
    split_individuals_disjoint,
)
from movement.models import build_model
from movement.models.cov_transformer import CovariateTransformer

COV_COLS = ["s2_B4", "s2_B8", "s2_ndvi"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _multi_segment_trajectories(n_ind: int = 6, n_seg: int = 3) -> list[Trajectory]:
    """Individuals that genuinely span several gap-separated segments.

    This is the condition the original fixture could not express, which is why
    the §7.1 leak survived two split tests.
    """
    out = []
    for i in range(n_ind):
        for s in range(n_seg):
            df = pd.DataFrame({"timestamp": pd.date_range(f"2020-0{s + 1}-01", periods=40, freq="1h"),
                               "lat": 0.0, "lon": 0.0})
            out.append(Trajectory(individual_id=f"study::{chr(65 + i)}", study_id="study", df=df))
    return out


def _write_raw_and_covariates(root: Path, *, drop_every: int = 7, nan_every: int = 5) -> tuple[Path, Path]:
    """Synthetic raw CSV (with a real multi-segment gap) + matching covariate CSV."""
    raw_dir, gee_dir = root / "raw", root / "gee"
    raw_dir.mkdir(parents=True, exist_ok=True)
    gee_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(0)
    for i in range(10):
        start = pd.Timestamp("2020-06-01")
        times = list(pd.date_range(start, periods=60, freq="1h"))
        # A 10 h gap splits each animal into two segments (threshold is 3 h).
        times += list(pd.date_range(times[-1] + pd.Timedelta(hours=10), periods=60, freq="1h"))
        lat, lon = 35.0 + 0.01 * i, -97.0 + 0.01 * i
        for k, t in enumerate(times):
            lat += rng.normal(0, 5e-4)
            lon += rng.normal(0, 5e-4)
            rows.append({"timestamp": t, "individual_id": f"pig_{i:02d}", "study_id": "okla",
                         "species": "Sus scrofa", "lat": lat, "lon": lon})
    raw = pd.DataFrame(rows)
    raw_path = raw_dir / "pigs.csv"
    raw.to_csv(raw_path, index=False)

    cov = raw.copy()
    cov["s2_composite_date"] = (cov["timestamp"] - pd.Timedelta(days=3)).dt.strftime("%Y-%m-%d")
    cov["s2_B4"] = rng.uniform(0.02, 0.2, len(cov))
    cov["s2_B8"] = rng.uniform(0.1, 0.5, len(cov))
    cov["s2_ndvi"] = (cov["s2_B8"] - cov["s2_B4"]) / (cov["s2_B8"] + cov["s2_B4"])
    cov["s2_label"] = "grass"  # non-numeric: must be dropped, not crash
    cov.loc[cov.index % nan_every == 0, ["s2_B4", "s2_B8", "s2_ndvi"]] = np.nan
    cov = cov[cov.index % drop_every != 3]  # some fixes have no covariate row at all
    cov.to_csv(gee_dir / "pigs_sentinel2.csv", index=False)
    return raw_dir, gee_dir


def _config(tmp_path: Path, raw_dir: Path, gee_dir: Path, **model_overrides):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({
        "data": {"raw_path": str(raw_dir), "processed_path": str(tmp_path / "proc"),
                 "raw_csv": "pigs.csv", "nominal_dt_hours": 1.0, "max_gap_multiplier": 3.0,
                 "val_fraction": 0.2, "test_fraction": 0.2, "split_unit": "individual"},
        "windowing": {"input_len": 8, "horizon": 4, "stride": 1, "eval_stride": 4},
        "transforms": {"delta_encoding": True, "scale": True, "time_context": True},
        "covariates": {"enabled": True, "path": str(gee_dir), "include": ["^s2_"],
                       "date_columns": ["s2_composite_date"]},
        "model": {"name": "cov_transformer", "d_model": 16, "nhead": 2, "num_encoder_layers": 1,
                  "num_decoder_layers": 1, "dim_feedforward": 32, "var_embed_dim": 4,
                  "dropout": 0.0, **model_overrides},
        "trainer": {"seed": 42, "batch_size": 16, "eval_batch_size": 32, "max_epochs": 2, "amp": False,
                    "run_dir": str(tmp_path / "runs")},
        "evaluation": {"n_plots": 1},
        "tracking": {"backend": "tensorboard", "offline": True},
    }), encoding="utf-8")
    return load_config(base)


@pytest.fixture()
def cov_setup(tmp_path):
    raw_dir, gee_dir = _write_raw_and_covariates(tmp_path)
    cfg = _config(tmp_path, raw_dir, gee_dir)
    return cfg, DataModule.build(cfg)


# ---------------------------------------------------------------------------
# Split fix (progress_report.md §7.1)
# ---------------------------------------------------------------------------
def test_disjoint_split_keeps_every_segment_of_an_animal_together():
    trajs = _multi_segment_trajectories()
    train, val, test = split_individuals_disjoint(trajs, val_fraction=0.2, test_fraction=0.2, seed=42)
    ids = [{t.individual_id for t in s} for s in (train, val, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert all(len(s) > 0 for s in ids)
    # Every segment is assigned exactly once.
    assert len(train) + len(val) + len(test) == len(trajs)
    for split, id_set in zip((train, val, test), ids):
        for ind in id_set:
            assert sum(t.individual_id == ind for t in split) == 3


def test_legacy_segment_split_leaks_on_multi_segment_animals():
    """Documents the defect the new unit fixes (and why the default is kept only for reproduction)."""
    trajs = _multi_segment_trajectories()
    train, val, test = split_individuals(trajs, val_fraction=0.2, test_fraction=0.2, seed=42)
    ids = [{t.individual_id for t in s} for s in (train, val, test)]
    assert ids[0] & (ids[1] | ids[2]), "legacy splitter no longer leaks — revisit the default"


def test_disjoint_split_is_order_independent():
    trajs = _multi_segment_trajectories()
    a = split_individuals_disjoint(trajs, val_fraction=0.2, test_fraction=0.2, seed=7)
    b = split_individuals_disjoint(list(reversed(trajs)), val_fraction=0.2, test_fraction=0.2, seed=7)
    for sa, sb in zip(a, b):
        assert {t.individual_id for t in sa} == {t.individual_id for t in sb}


def test_assert_disjoint_ids_raises_on_overlap():
    with pytest.raises(ValueError, match="not individual-disjoint"):
        assert_disjoint_ids({"a", "b"}, {"b"}, {"c"})


def test_from_split_file_rejects_a_leaky_split_when_individual(cov_setup, tmp_path):
    cfg, dm = cov_setup
    ids = sorted({w.individual_id for w in dm.train_windows})
    leaky = tmp_path / "leaky" / "split.json"
    leaky.parent.mkdir()
    leaky.write_text(json.dumps({"train": ids, "val": ids[:1], "test": ids[1:2]}))
    with pytest.raises(ValueError, match="not individual-disjoint"):
        DataModule.from_split_file(cfg, leaky)


def test_individual_split_datamodule_is_disjoint(cov_setup):
    _, dm = cov_setup
    s = [{w.individual_id for w in ws} for ws in (dm.train_windows, dm.val_windows, dm.test_windows)]
    assert not (s[0] & s[1] or s[0] & s[2] or s[1] & s[2])


# ---------------------------------------------------------------------------
# Covariate join + scaling
# ---------------------------------------------------------------------------
def test_attach_covariates_preserves_rows_and_derives_age(tmp_path):
    from movement.data.loading import load_raw_csv

    raw_dir, gee_dir = _write_raw_and_covariates(tmp_path)
    fixes = load_raw_csv(raw_dir / "pigs.csv")
    spec = CovariatesConfig(enabled=True, path=gee_dir)
    joined, cols = attach_covariates(fixes, raw_dir / "pigs.csv", spec)
    assert len(joined) == len(fixes)
    assert cols == ["s2_composite_date_age_days", *COV_COLS]  # label dropped, age added
    matched = joined["s2_composite_date_age_days"].notna()
    assert np.allclose(joined.loc[matched, "s2_composite_date_age_days"], 3.0, atol=1.0)
    # Rows dropped from the covariate CSV come back as NaN, not as dropped fixes.
    assert joined["s2_B4"].isna().sum() > 0


def test_attach_covariates_fails_loudly_on_mismatched_keys(tmp_path):
    from movement.data.loading import load_raw_csv

    raw_dir, gee_dir = _write_raw_and_covariates(tmp_path)
    cov_path = gee_dir / "pigs_sentinel2.csv"
    cov = pd.read_csv(cov_path)
    cov["individual_id"] = "someone_else"
    cov.to_csv(cov_path, index=False)
    with pytest.raises(ValueError, match="matched a covariate row"):
        attach_covariates(load_raw_csv(raw_dir / "pigs.csv"), raw_dir / "pigs.csv",
                          CovariatesConfig(enabled=True, path=gee_dir))


def test_attach_covariates_requires_the_file(tmp_path):
    from movement.data.loading import load_raw_csv

    raw_dir, gee_dir = _write_raw_and_covariates(tmp_path)
    (gee_dir / "pigs_sentinel2.csv").unlink()
    with pytest.raises(FileNotFoundError, match="Covariate file not found"):
        attach_covariates(load_raw_csv(raw_dir / "pigs.csv"), raw_dir / "pigs.csv",
                          CovariatesConfig(enabled=True, path=gee_dir))


def test_covariate_scaler_is_nan_aware_and_flags_missing(tmp_path):
    rows = np.array([[1.0, np.nan], [3.0, 10.0], [np.nan, 20.0]])
    sc = CovariateScaler.fit(rows, ["a", "b"], clip=8.0)
    assert sc.mean.tolist() == [2.0, 15.0]
    z, miss = sc.transform(np.array([[np.nan, 15.0], [2.0, np.nan]], dtype=np.float32))
    assert miss.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert np.allclose(z, 0.0)  # missing -> mean -> 0; observed at mean -> 0
    path = tmp_path / "c.json"
    sc.save(path)
    assert np.allclose(CovariateScaler.load(path, expected_columns=["a", "b"]).std, sc.std)
    with pytest.raises(ValueError, match="different covariate columns"):
        CovariateScaler.load(path, expected_columns=["b", "a"])


def test_covariate_scaler_clips_outliers():
    sc = CovariateScaler.fit(np.array([[0.0], [1.0], [0.0], [1.0]]), ["a"], clip=3.0)
    z, _ = sc.transform(np.array([[1e6]], dtype=np.float32))
    assert z.item() == 3.0


def test_covariate_scaler_is_fit_on_train_animals_only(cov_setup):
    cfg, dm = cov_setup
    train_ids = {w.individual_id for w in dm.train_windows}
    train_rows = np.concatenate([t.df[dm.covariate_columns].to_numpy(float)
                                 for t in dm.trajectories if t.individual_id in train_ids])
    assert np.allclose(dm.covariate_scaler.mean, np.nanmean(train_rows, axis=0), atol=1e-5)


# ---------------------------------------------------------------------------
# Time context
# ---------------------------------------------------------------------------
def test_time_features_use_local_solar_hour():
    noon_utc = pd.Timestamp("2020-03-20T12:00:00").value // 10**9
    six_pm_utc = pd.Timestamp("2020-03-20T18:00:00").value // 10**9
    f = time_features(np.array([noon_utc, six_pm_utc]), np.array([0.0, -90.0]))
    # Both are local solar noon: sin(pi) = 0, cos(pi) = -1.
    assert np.allclose(f[:, 0], 0.0, atol=1e-5)
    assert np.allclose(f[:, 1], -1.0, atol=1e-5)


def test_future_time_context_uses_nominal_not_actual_times():
    ts = np.arange(4) * 3600 + 1_600_000_000
    obs, fut = window_time_context(ts, np.zeros(4), horizon=3, nominal_dt_hours=2.0)
    expected = time_features(ts[-1] + np.array([2, 4, 6]) * 3600, np.zeros(3))
    assert obs.shape == (4, 4) and fut.shape == (3, 4)
    assert np.allclose(fut, expected)


# ---------------------------------------------------------------------------
# Datamodule extras
# ---------------------------------------------------------------------------
def test_batches_carry_covariates_for_observed_fixes_only(cov_setup):
    cfg, dm = cov_setup
    assert dm.covariate_columns == ["s2_composite_date_age_days", *COV_COLS]
    assert all(w.covariates.shape == (cfg.windowing.input_len, 4) for w in dm.train_windows[:20])
    batch = next(iter(dm.dataloader("train")))
    x, y, dt, extras = unpack_batch(batch, torch.device("cpu"))
    B, T, H = x.shape[0], cfg.windowing.input_len, cfg.windowing.horizon
    assert extras["covariates"].shape == (B, T, 4)
    assert extras["covariate_missing"].shape == (B, T, 4)
    assert extras["time_feats"].shape == (B, T, 4)
    assert extras["future_time_feats"].shape == (B, H, 4)
    assert torch.isfinite(extras["covariates"]).all()
    assert extras["covariate_missing"].sum() > 0  # the synthetic NaNs survive as flags


def test_plain_batches_are_unchanged_without_extras(datamodule):
    """The original arms keep their exact (x, y, dt) batches."""
    item = datamodule.dataset("train")[0]
    assert isinstance(item, tuple) and len(item) == 3


def test_windows_share_memory_with_the_segment(cov_setup):
    _, dm = cov_setup
    a, b = dm.train_windows[0], dm.train_windows[1]
    if a.individual_id == b.individual_id:
        assert np.shares_memory(a.covariates, b.covariates)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _model(n_cov=5, **kw) -> CovariateTransformer:
    defaults = dict(input_len=12, horizon=4, n_features=2, n_covariates=n_cov, n_time_features=4,
                    d_model=16, nhead=2, num_encoder_layers=2, num_decoder_layers=1,
                    dim_feedforward=32, dropout=0.0, var_embed_dim=4)
    defaults.update(kw)
    return CovariateTransformer(**defaults)


def _context(B=3, T=12, H=4, C=5):
    return {"time_feats": torch.randn(B, T, 4), "future_time_feats": torch.randn(B, H, 4),
            "covariates": torch.randn(B, T, C), "covariate_missing": (torch.rand(B, T, C) < 0.2).float()}


def test_output_shape_and_finite_backward():
    torch.manual_seed(0)
    m = _model()
    out = m(torch.randn(3, 12, 2), context=_context())
    assert out.shape == (3, 4, 2)
    loss = out.pow(2).mean()
    loss.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_default_config_is_larger_than_every_existing_arm():
    m = CovariateTransformer(input_len=24, horizon=12, n_features=2, n_covariates=38, n_time_features=4)
    assert m.count_parameters() > 417_432  # the existing Transformer arm
    assert 1_000_000 <= m.count_parameters() <= 1_500_000


def test_no_covariate_ablation_drops_only_the_covariate_branch():
    with_cov, without = _model(n_cov=5), _model(n_cov=0)
    names_with = {n for n, _ in with_cov.named_parameters()}
    names_without = {n for n, _ in without.named_parameters()}
    assert names_without < names_with
    assert all(n.startswith(("covariates.", "fuse_gate.")) for n in names_with - names_without)
    out = without(torch.randn(2, 12, 2), context={k: v[:2] for k, v in _context().items()
                                                   if k in ("time_feats", "future_time_feats")})
    assert out.shape == (2, 4, 2)


def test_missing_context_fails_loudly():
    with pytest.raises(ValueError, match="covariates"):
        _model().eval()(torch.randn(3, 12, 2), context={k: v for k, v in _context().items() if k != "covariates"})
    with pytest.raises(ValueError, match="time_feats"):
        _model(n_cov=0)(torch.randn(3, 12, 2), context={})


def test_covariates_actually_reach_the_output():
    torch.manual_seed(0)
    m = _model().eval()
    ctx = _context()
    x = torch.randn(3, 12, 2)
    a = m(x, context=ctx)
    b = m(x, context=dict(ctx, covariates=ctx["covariates"] + 3.0))
    assert not torch.allclose(a, b)


def test_eval_mode_is_deterministic_and_ignores_training_regularisers():
    m = _model(covariate_dropout=1.0, context_dropout=1.0).eval()
    x, ctx = torch.randn(3, 12, 2), _context()
    assert torch.equal(m(x, context=ctx), m(x, context=ctx))
    assert m._context_mask(3, torch.device("cpu")) is None


def test_covariate_dropout_marks_everything_missing():
    m = _model(covariate_dropout=1.0).train()
    ctx = _context()
    cov, missing = m._covariate_inputs(ctx, batch=3)
    assert torch.all(cov == 0) and torch.all(missing == 1)


def test_context_mask_always_keeps_min_context_fixes():
    torch.manual_seed(0)
    m = _model(context_dropout=1.0, min_context=4).train()
    mask = m._context_mask(500, torch.device("cpu"))
    visible = (~mask).sum(dim=1)
    assert visible.min().item() >= 4 and visible.max().item() <= 11
    assert not mask[:, -1].any()  # the last observed fix is never hidden


def test_masked_prefix_cannot_influence_the_forecast():
    torch.manual_seed(0)
    m = _model()
    m.eval()
    x, ctx = torch.randn(1, 12, 2), {k: v[:1] for k, v in _context().items()}
    pad = torch.zeros(1, 12, dtype=torch.bool)
    pad[:, :5] = True
    m._context_mask = lambda b, d: pad  # force the variable-context mask in eval
    a = m(x, context=ctx)
    x2 = x.clone()
    x2[:, :5] += 100.0
    ctx2 = dict(ctx, covariates=ctx["covariates"].clone())
    ctx2["covariates"][:, :5] += 100.0
    b = m(x2, context=ctx2)
    assert torch.allclose(a, b, atol=1e-5)


def test_registry_builds_with_data_spec_and_refuses_without(cov_setup):
    cfg, dm = cov_setup
    m = build_model(cfg.model, cfg.windowing, cfg.transforms, data_spec=dm.model_data_spec())
    assert m.n_covariates == 4
    with pytest.raises(ValueError, match="time context"):
        build_model(cfg.model, cfg.windowing, cfg.transforms)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def test_train_checkpoint_reload_and_eval(cov_setup, tmp_path):
    from movement.evaluation import evaluate_model
    from movement.training import Trainer
    from movement.utils.tracking import Tracker, make_run_dir

    cfg, dm = cov_setup
    model = build_model(cfg.model, cfg.windowing, cfg.transforms, data_spec=dm.model_data_spec())
    run_dir = make_run_dir(cfg)
    dm.persist_split(run_dir)
    dm.persist_scaler(run_dir)
    assert dm.persist_covariates(run_dir).exists()
    tracker = Tracker(cfg, run_dir)
    summary = Trainer(model, dm, cfg, tracker, run_dir, device=torch.device("cpu")).fit()
    tracker.close()
    assert summary["best_val_ade"] is not None

    # Rebuild exactly as the eval CLI does: persisted split + both scalers.
    dm2 = DataModule.from_split_file(cfg, run_dir / "split.json")
    assert np.allclose(dm2.covariate_scaler.mean, dm.covariate_scaler.mean)
    model2 = build_model(cfg.model, cfg.windowing, cfg.transforms, data_spec=dm2.model_data_spec())
    state = torch.load(run_dir / cfg.trainer.checkpoint_dir / "best.pt", map_location="cpu", weights_only=False)
    model2.load_state_dict(state["model_state"])
    report = evaluate_model(model2, dm2, torch.device("cpu"), split="test")
    assert np.isfinite(report["ade"]) and np.isfinite(report["ade_cp"])

    sel = model2.covariate_selection_summary(dm2, torch.device("cpu"), split="test")
    assert list(sel["covariate"].sort_values()) == sorted(dm.covariate_columns)
    assert sel["mean_weight"].sum() == pytest.approx(1.0, abs=1e-4)


def _study_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_covariate_study", Path(__file__).resolve().parents[1] / "scripts" / "run_covariate_study.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("mode", ["kfold", "random"])
@pytest.mark.parametrize("arm", ["cov", "nocov", "transformer", "tcn", "lstm"])
def test_study_script_overrides_resolve_for_every_arm(arm, mode, monkeypatch, tmp_path):
    """Every override the covariate-study runner passes must exist in the merged YAML
    (apply_overrides rejects unknown keys) and validate."""
    mod = _study_module()
    monkeypatch.setenv("RAW_DATASET_PATH", str(tmp_path))
    monkeypatch.setenv("PROCESSED_DATASET_PATH", str(tmp_path))
    monkeypatch.setenv("GEE_DATASET_PATH", str(tmp_path))
    args = mod.build_parser().parse_args(["--dataset", "boar_reshaped", "--mode", mode])
    _, _, units = mod.study_layout(args)
    config, extra = mod.ARMS[arm]
    label, unit_overrides = units[1]
    cfg = load_config(config, [f"trainer.max_epochs={args.epochs}",
                               f"trainer.early_stopping_patience={args.patience}",
                               *unit_overrides, *extra, f"trainer.run_dir={tmp_path}"])
    expected = "individual_kfold" if mode == "kfold" else "individual"
    assert cfg.data.split_unit == expected
    if mode == "kfold":
        assert cfg.data.fold == 1 and cfg.data.n_folds == 5 and label == "fold1"
    assert cfg.covariates.enabled is (arm == "cov")
    assert cfg.trainer.max_epochs == 200 and cfg.trainer.early_stopping_patience == 30


# ---------------------------------------------------------------------------
# Fix-balanced k-fold (evaluation protocol v2)
# ---------------------------------------------------------------------------
def _uneven_trajectories() -> list[Trajectory]:
    """18 animals whose sizes span two orders of magnitude, like boar."""
    sizes = [56, 199, 236, 398, 418, 616, 637, 642, 725, 1285, 2339, 2970, 3284, 3844, 5640, 6097, 6488, 6588]
    out = []
    for i, n in enumerate(sizes):
        df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=n, freq="1h"), "lat": 0.0, "lon": 0.0})
        out.append(Trajectory(individual_id=f"s::{i:02d}", study_id="s", df=df))
    return out


def test_folds_are_balanced_by_fixes_not_animals():
    from movement.data.loading import fold_assignment

    trajs = _uneven_trajectories()
    fixes = {t.individual_id: t.n_fixes for t in trajs}
    folds = fold_assignment(fixes, n_folds=5, seed=42)
    totals = [sum(n for i, n in fixes.items() if folds[i] == k) for k in range(5)]
    assert max(totals) / min(totals) < 1.05  # random 70/15/15 draws gave ~15x on these sizes
    assert set(folds.values()) == set(range(5))


def test_fold_assignment_is_deterministic_and_seeded():
    from movement.data.loading import fold_assignment

    fixes = {t.individual_id: t.n_fixes for t in _uneven_trajectories()}
    assert fold_assignment(fixes, n_folds=5, seed=1) == fold_assignment(dict(reversed(fixes.items())), n_folds=5, seed=1)
    assert any(fold_assignment(fixes, n_folds=5, seed=s) != fold_assignment(fixes, n_folds=5, seed=1)
               for s in range(2, 8))


def test_kfold_tests_every_animal_exactly_once_and_rotates_val():
    from movement.data.loading import split_individuals_kfold

    trajs = _uneven_trajectories()
    all_ids = {t.individual_id for t in trajs}
    tested = []
    for k in range(5):
        train, val, test = split_individuals_kfold(trajs, n_folds=5, fold=k, seed=42)
        ids = [{t.individual_id for t in s} for s in (train, val, test)]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        assert ids[0] | ids[1] | ids[2] == all_ids
        _, _, next_test = split_individuals_kfold(trajs, n_folds=5, fold=(k + 1) % 5, seed=42)
        assert ids[1] == {t.individual_id for t in next_test}  # val of fold k = test of fold k+1
        tested += sorted(ids[2])
    assert sorted(tested) == sorted(all_ids)


def test_kfold_keeps_multi_segment_animals_together():
    from movement.data.loading import split_individuals_kfold

    train, val, test = split_individuals_kfold(_multi_segment_trajectories(n_ind=6), n_folds=3, fold=0, seed=0)
    for split in (train, val, test):
        for ind in {t.individual_id for t in split}:
            assert sum(t.individual_id == ind for t in split) == 3


def test_kfold_rejects_bad_arguments():
    from movement.data.loading import fold_assignment, split_individuals_kfold

    with pytest.raises(ValueError, match="n_folds must be >= 3"):
        fold_assignment({"a": 1, "b": 2}, n_folds=2, seed=0)
    with pytest.raises(ValueError, match="animals for"):
        fold_assignment({"a": 1, "b": 2, "c": 3}, n_folds=4, seed=0)
    with pytest.raises(ValueError, match="fold must be in"):
        split_individuals_kfold(_uneven_trajectories(), n_folds=5, fold=5, seed=0)


def test_kfold_datamodule_and_split_file_round_trip(tmp_path):
    raw_dir, gee_dir = _write_raw_and_covariates(tmp_path)
    cfg = _config(tmp_path, raw_dir, gee_dir)
    cfg.data.split_unit = "individual_kfold"
    cfg.data.n_folds = 5
    cfg.data.fold = 2
    dm = DataModule.build(cfg)
    ids = [{w.individual_id for w in ws} for ws in (dm.train_windows, dm.val_windows, dm.test_windows)]
    assert all(ids) and not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    dm.persist_split(run_dir)
    dm2 = DataModule.from_split_file(cfg, run_dir / "split.json")
    assert {w.individual_id for w in dm2.test_windows} == ids[2]


def test_trainer_records_best_epoch(cov_setup):
    from movement.training import Trainer
    from movement.utils.tracking import Tracker, make_run_dir

    cfg, dm = cov_setup
    model = build_model(cfg.model, cfg.windowing, cfg.transforms, data_spec=dm.model_data_spec())
    run_dir = make_run_dir(cfg)
    tracker = Tracker(cfg, run_dir)
    summary = Trainer(model, dm, cfg, tracker, run_dir, device=torch.device("cpu")).fit()
    tracker.close()
    assert summary["epochs_run"] == cfg.trainer.max_epochs
    assert 0 <= summary["best_epoch"] < summary["epochs_run"]
    assert summary["max_epochs"] == cfg.trainer.max_epochs


# ---------------------------------------------------------------------------
# Study report
# ---------------------------------------------------------------------------
def _fake_run(root: Path, unit: str, arm: str, ade: float, cp: float, *, n: int = 100,
              best: int = 10, epochs: int = 50, max_epochs: int = 200, stopped: bool = True) -> None:
    run = root / unit / arm / "20260101-000000-x-s42"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "best.pt").write_bytes(b"")
    (run / "metrics.json").write_text(json.dumps(
        {"ade": ade, "fde": ade * 1.4, "ade_cp": cp, "ade_cv": cp * 2, "n_windows": n}))
    (run / "manifest.json").write_text(json.dumps({"parameter_count": 1000, "train_metrics": {
        "best_epoch": best, "epochs_run": epochs, "max_epochs": max_epochs, "stopped_early": stopped,
        "n_val_windows": 700}}))
    (run / "split.json").write_text(json.dumps({"train": ["a"], "val": ["b"], "test": ["c", "d"]}))


def test_report_flags_collapse_and_excludes_it_from_paired_means(tmp_path, monkeypatch):
    mod = _study_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    study = tmp_path / "study"
    # fold0: nocov collapsed (ratio 0.99) — must not create a fake covariate "win".
    _fake_run(study, "fold0", "cov", 500, 700)
    _fake_run(study, "fold0", "nocov", 693, 700)
    _fake_run(study, "fold1", "cov", 310, 450, n=300)
    _fake_run(study, "fold1", "nocov", 300, 450, n=300)
    # fold2: best epoch at the very end, no early stop -> still improving.
    _fake_run(study, "fold2", "cov", 320, 460, best=198, epochs=200, stopped=False)
    _fake_run(study, "fold2", "nocov", 330, 460)
    df = mod.collect(study, "fold")
    flags = df.set_index(["unit", "arm"])
    assert flags.loc[("fold0", "nocov"), "collapsed"] and not flags.loc[("fold0", "cov"), "collapsed"]
    assert flags.loc[("fold2", "cov"), "still_improving"] and not flags.loc[("fold1", "cov"), "still_improving"]

    text = mod.write_report("boar", study, "kfold3_seed42", "fold").read_text(encoding="utf-8")
    assert "excluded: nocov collapsed" in text
    # Paired mean uses fold1 (+10) and fold2 (-10) only: 0.0, not the collapse-driven -193.
    assert "**+0.0 ± 14.1**" in text
    # Pooled cov ADE weights windows: (500*100 + 310*300 + 320*100) / 500 = 350.0
    assert "| 350.0 |" in text
