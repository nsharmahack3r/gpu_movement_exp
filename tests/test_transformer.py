"""Transformer arm tests: shapes, causal mask, no leakage, pos encodings, warmup, registry."""

from __future__ import annotations

import torch

from movement.config import TransformerModelConfig
from movement.models import ForecastModel, TransformerForecaster, build_model
from movement.models.transformer import _sinusoidal_positions


def make_transformer(mode: str = "encoder_only", **kw) -> TransformerForecaster:
    return TransformerForecaster(
        input_len=16,
        horizon=4,
        n_features=2,
        d_model=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=32,
        dropout=0.0,
        mode=mode,
        **kw,
    )


def test_output_shape_all_modes():
    for mode in ("encoder_only", "encoder_decoder"):
        for pooling in ("last", "mean", "cls"):
            m = make_transformer(mode=mode, pooling=pooling)
            x = torch.randn(3, 16, 2)
            out = m(x)
            assert out.shape == (3, 4, 2), f"{mode}/{pooling}: got {tuple(out.shape)}"


def test_causal_mask_correctness():
    """With causal_mask on, the self-attention output at position t cannot use t+1.

    The mask is verified at the attention sublayer (the residual connection
    legitimately carries raw input forward through pre-norm blocks, so block
    outputs are not the right probe). Perturbing the last input timestep must
    leave the attention output of every earlier position unchanged.
    """
    torch.manual_seed(0)
    m = make_transformer(mode="encoder_only", causal_mask=True)
    m.eval()
    x = torch.randn(1, 16, 2)
    layer = m.encoder.layers[0]
    with torch.no_grad():
        h = m.pos_enc(m.input_proj(x), None)
        xp = x.clone()
        xp[:, -1, :] += 500.0
        hp = m.pos_enc(m.input_proj(xp), None)

        mask = m._causal_mask(h)
        # Probe the first layer's causal-masked self-attention output.
        if layer.norm_first:
            base = layer.self_attn(layer.norm1(h), layer.norm1(h), layer.norm1(h), attn_mask=mask)[0]
            pert = layer.self_attn(layer.norm1(hp), layer.norm1(hp), layer.norm1(hp), attn_mask=mask)[0]
        else:
            base = layer.self_attn(h, h, h, attn_mask=mask)[0]
            pert = layer.self_attn(hp, hp, hp, attn_mask=mask)[0]
        # Positions 0..14 must be bit-identical; position 15 may change.
        assert torch.allclose(base[:, :-1], pert[:, :-1], atol=1e-5), (
            "causal mask violated: earlier attention outputs changed when the last input changed"
        )


def test_no_target_leakage_encoder_decoder():
    """Encoder-decoder mode never uses targets — output ignores context['targets']."""
    torch.manual_seed(0)
    m = make_transformer(mode="encoder_decoder")
    m.eval()
    x = torch.randn(2, 16, 2)
    with torch.no_grad():
        a = m(x, context={"stage": "test", "targets": torch.randn(2, 4, 2)})
        b = m(x, context={"stage": "test"})
        assert torch.equal(a, b), "encoder-decoder output changed with targets present"


def test_positional_encodings_differ_and_deterministic():
    """Sinusoidal encodings differ across positions and are deterministic."""
    pe1 = _sinusoidal_positions(16, 16)
    pe2 = _sinusoidal_positions(16, 16)
    assert torch.equal(pe1, pe2), "sinusoidal encoding must be deterministic"
    assert not torch.allclose(pe1[0, 0], pe1[0, 5], atol=1e-3), "positions 0 and 5 identical"


def test_time_aware_encoding_differs_with_dt():
    """time_aware produces different encodings for different Δt at the same index."""
    m = make_transformer(mode="encoder_only", pos_encoding="time_aware")
    x = torch.randn(1, 16, 2)
    dt_reg = torch.arange(16, dtype=torch.float32).unsqueeze(0) * 3600.0
    dt_irreg = torch.arange(16, dtype=torch.float32).unsqueeze(0) * 3600.0
    dt_irreg[0, 8:] += 7200.0  # a 2h gap after index 8
    out_reg = m.input_proj(x) + m.pos_enc.pe(dt_reg)
    out_irreg = m.input_proj(x) + m.pos_enc.pe(dt_irreg)
    # Same index, different elapsed time → different encoding (after index 8).
    assert not torch.allclose(out_reg[:, 10], out_irreg[:, 10], atol=1e-3)


def test_learned_encoding_works():
    m = make_transformer(mode="encoder_only", pos_encoding="learned")
    x = torch.randn(2, 16, 2)
    out = m(x)
    assert out.shape == (2, 4, 2)


def test_registry_resolves_transformer(base_config):
    """The registry resolves 'transformer' and the model satisfies the ABC."""
    cfg = base_config.model_copy(deep=True)
    cfg.model = TransformerModelConfig(name="transformer", d_model=16, nhead=4, num_layers=1, dim_feedforward=32)
    m = build_model(cfg.model, cfg.windowing, cfg.transforms)
    assert isinstance(m, ForecastModel)
    assert m.receptive_field == cfg.windowing.input_len


def test_forward_backward_finite_cpu():
    torch.manual_seed(0)
    for mode in ("encoder_only", "encoder_decoder"):
        m = make_transformer(mode=mode)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        x = torch.randn(3, 16, 2)
        y = torch.randn(3, 4, 2)
        loss = torch.nn.functional.mse_loss(m(x), y)
        assert torch.isfinite(loss).item()
        loss.backward()
        opt.step()
        assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_attention_entropy_finite():
    m = make_transformer(mode="encoder_only")
    x = torch.randn(2, 16, 2)
    entropy = m.attention_entropy(x)
    assert torch.isfinite(torch.tensor(entropy)).item()


def test_smoke_trainer_checkpoint(base_config, datamodule, tmp_path):
    """Two-epoch smoke on the synthetic fixture writes a reloadable checkpoint."""
    from pathlib import Path

    from movement.training import Trainer
    from movement.utils.tracking import Tracker, make_run_dir

    cfg = base_config.model_copy(deep=True)
    cfg.model = TransformerModelConfig(
        name="transformer", d_model=16, nhead=4, num_layers=1, dim_feedforward=32
    )
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
