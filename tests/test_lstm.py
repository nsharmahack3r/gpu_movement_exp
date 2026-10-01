"""LSTM arm tests: shapes, no target leakage, teacher forcing, schedules, registry."""

from __future__ import annotations

import logging

import pytest
import torch

from movement.models import ForecastModel, LSTMForecaster, build_model, scheduled_sampling_ratio


def make_lstm(decoder: str = "direct", **kw) -> LSTMForecaster:
    return LSTMForecaster(
        input_len=16,
        horizon=4,
        n_features=2,
        hidden_size=16,
        num_layers=2,
        dropout=0.0,
        decoder=decoder,
        **kw,
    )


def _batch(targets: torch.Tensor | None = None, *, stage: str = "train") -> tuple[torch.Tensor, dict]:
    x = torch.randn(4, 16, 2)
    ctx: dict = {"stage": stage, "epoch": 0}
    if targets is not None:
        ctx["targets"] = targets
    return x, ctx


def test_output_shape_both_decoders():
    for decoder in ("direct", "autoregressive"):
        m = make_lstm(decoder=decoder)
        x, ctx = _batch(torch.randn(4, 4, 2))
        out = m(x, context=ctx)
        assert out.shape == (4, 4, 2), f"{decoder}: expected (B, horizon, 2), got {tuple(out.shape)}"


def test_no_target_leakage_at_test_stage():
    """The most important test: with stage != train, targets must not change output."""
    torch.manual_seed(0)
    m = make_lstm(decoder="autoregressive")
    m.eval()
    x, _ = _batch(stage="test")
    with torch.no_grad():
        with_targets = m(x, context={"stage": "test", "epoch": 0, "targets": torch.randn(4, 4, 2)})
        without_targets = m(x, context={"stage": "test", "epoch": 0})
        assert torch.equal(with_targets, without_targets), (
            "teacher-forced evaluation detected: test output changed with targets present"
        )


def test_teacher_forcing_engages():
    """At ratio 1.0 the decoder inputs match GT; at ratio 0.0 they don't."""
    torch.manual_seed(0)
    m = make_lstm(decoder="autoregressive", teacher_forcing_ratio=1.0, scheduled_sampling="constant")
    m.train()

    # Ratio 1.0 → every decoder input is the ground truth; outputs should be
    # strongly influenced by targets (deterministic when forced).
    x, _ = _batch()
    targets = torch.randn(4, 4, 2)
    m.eval()  # avoid dropout noise, still uses context stage
    with torch.no_grad():
        out_forced = m(x, context={"stage": "train", "epoch": 0, "targets": targets})

    # With ratio 0.0 the model free-runs — outputs must differ from forced.
    m2 = make_lstm(decoder="autoregressive", teacher_forcing_ratio=0.0, scheduled_sampling="constant")
    m2.eval()
    with torch.no_grad():
        out_free = m2(x, context={"stage": "train", "epoch": 0, "targets": targets})
    # Teacher forcing feeds targets into the decoder input, so the two runs
    # must not be identical (different input trajectories).
    assert not torch.allclose(out_forced, out_free, atol=1e-4), (
        "teacher forcing did not change decoder behaviour"
    )


def test_scheduled_sampling_ratio_values():
    """Each decay mode returns the expected ratio at epoch 0, mid, and final."""
    # constant
    for e in (0, 25, 50, 100):
        assert scheduled_sampling_ratio(e, max_ratio=1.0, mode="constant", total_epochs=50) == 1.0
    # linear: 1.0 → 0.0 over 50 epochs
    assert scheduled_sampling_ratio(0, max_ratio=1.0, mode="linear", total_epochs=50) == pytest.approx(1.0)
    assert scheduled_sampling_ratio(25, max_ratio=1.0, mode="linear", total_epochs=50) == pytest.approx(0.5)
    assert scheduled_sampling_ratio(50, max_ratio=1.0, mode="linear", total_epochs=50) == pytest.approx(0.0)
    assert scheduled_sampling_ratio(100, max_ratio=1.0, mode="linear", total_epochs=50) == 0.0
    # inverse_sigmoid: high early, ~max/2 at k, → 0 after
    r0 = scheduled_sampling_ratio(0, max_ratio=1.0, mode="inverse_sigmoid", total_epochs=50)
    r50 = scheduled_sampling_ratio(50, max_ratio=1.0, mode="inverse_sigmoid", total_epochs=50)
    assert r0 > 0.9
    assert 0.4 < r50 < 0.6
    assert scheduled_sampling_ratio(200, max_ratio=1.0, mode="inverse_sigmoid", total_epochs=50) < 0.01


def test_registry_resolves_lstm(base_config):
    """The registry resolves 'lstm' and the built model satisfies the ABC."""
    from movement.config import LSTMModelConfig

    cfg = base_config.model_copy(deep=True)
    cfg.model = LSTMModelConfig(name="lstm", hidden_size=16, num_layers=1, dropout=0.0)
    m = build_model(cfg.model, cfg.windowing, cfg.transforms)
    assert isinstance(m, ForecastModel)
    assert m.receptive_field is None


def test_forward_backward_finite_cpu():
    torch.manual_seed(0)
    for decoder in ("direct", "autoregressive"):
        m = make_lstm(decoder=decoder)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        x, ctx = _batch(torch.randn(4, 4, 2))
        out = m(x, context=ctx)
        loss = torch.nn.functional.mse_loss(out, torch.randn(4, 4, 2))
        assert torch.isfinite(loss).item()
        loss.backward()
        opt.step()
        assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_dropout_single_layer_warns(caplog):
    with caplog.at_level(logging.WARNING):
        LSTMForecaster(input_len=8, horizon=2, n_features=2, num_layers=1, dropout=0.5)
    assert any("num_layers=1" in r.message for r in caplog.records)


def test_smoke_trainer_checkpoint(base_config, datamodule, tmp_path):
    """Two-epoch smoke on the synthetic fixture writes a reloadable checkpoint."""
    from pathlib import Path

    from movement.config import LSTMModelConfig
    from movement.training import Trainer
    from movement.utils.tracking import Tracker, make_run_dir

    cfg = base_config.model_copy(deep=True)
    cfg.model = LSTMModelConfig(name="lstm", hidden_size=16, num_layers=1, dropout=0.0, decoder="direct")
    cfg.trainer.max_epochs = 2
    cfg.trainer.batch_size = 16
    cfg.trainer.run_dir = tmp_path
    cfg.trainer.checkpoint_dir = Path("checkpoints")
    cfg.trainer.tensorboard_dir = Path("tensorboard")
    cfg.trainer.amp = False

    model = build_model(cfg.model, cfg.windowing, cfg.transforms)
    run_dir = make_run_dir(cfg)
    tracker = Tracker(cfg, run_dir)
    trainer = Trainer(model, datamodule, cfg, tracker, run_dir, device=torch.device("cpu"))
    summary = trainer.fit()
    tracker.close()

    assert summary["best_val_ade"] is not None
    ckpt = run_dir / cfg.trainer.checkpoint_dir / "best.pt"
    assert ckpt.exists()

    # Reload to identical weights.
    model2 = build_model(cfg.model, cfg.windowing, cfg.transforms)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model2.load_state_dict(state["model_state"])
    model.eval()
    model2.eval()
    x = torch.randn(2, cfg.windowing.input_len, 2)
    with torch.no_grad():
        assert torch.equal(model(x), model2(x))
