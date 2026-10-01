"""TCN tests: output shape, causality, receptive field, forward/backward on CPU."""

from __future__ import annotations

import torch

from movement.models.tcn import TCN


def make_tcn(input_len: int = 16, horizon: int = 4, **kw) -> TCN:
    # RF for 3 blocks, k=3: 1 + 2*(1+2+4) = 15 < 16, so add a 4th block.
    channels = kw.pop("channels", [8, 8, 8, 8])
    return TCN(
        input_len=input_len,
        horizon=horizon,
        n_features=2,
        channels=channels,
        kernel_size=3,
        dropout=0.0,
        **kw,
    )


def test_output_shape():
    m = make_tcn()
    x = torch.randn(5, 16, 2)
    out = m(x)
    assert out.shape == (5, 4, 2)


def test_receptive_field_covers_input():
    m = make_tcn(input_len=24, horizon=6)
    assert m.receptive_field >= 24
    m2 = make_tcn(input_len=64, horizon=12, channels=[8, 8, 8, 8, 8, 8])  # RF = 1+2*(1+2+4+8+16+32)=127
    assert m2.receptive_field >= 64


def test_causality():
    """Perturbation sensitivity: with RF >= input_len, every input matters.

    Causality itself is structural (causal convs, verified separately in
    ``test_causal_conv_no_future_leak``); here we confirm the model actually
    uses the whole window — both the newest and the oldest timestep influence
    the forecast.
    """
    torch.manual_seed(0)
    m = make_tcn(input_len=24, horizon=4, channels=[8, 8, 8, 8, 8])  # RF = 1+2*(1+2+4+8+16) = 63
    m.eval()
    x = torch.randn(1, 24, 2)
    with torch.no_grad():
        base = m(x)
        # Perturb the last timestep → output must change.
        x_new = x.clone()
        x_new[:, -1, :] += 500.0
        assert not torch.equal(m(x_new), base), "newest input did not affect the output"

        # Perturb the oldest timestep → output must change (RF covers it).
        x_old = x.clone()
        x_old[:, 0, :] += 500.0
        assert not torch.equal(m(x_old), base), "oldest input did not affect the output"


def test_causal_conv_no_future_leak():
    """Verify causal padding: the conv at position t cannot see t+1.

    Concretely, with kernel_size=3 and dilation=1, output[t] must equal
    output[t] computed with input[t+1:] zeroed — causal padding only pads left.
    """
    torch.manual_seed(0)
    from movement.models.tcn import CausalConv1d

    conv = CausalConv1d(2, 4, kernel_size=3, dilation=1)
    x = torch.randn(1, 2, 10)
    with torch.no_grad():
        full = conv(x)
        # Zero everything after position t and re-run: outputs at <= t unchanged.
        for t in range(10):
            x_masked = x.clone()
            x_masked[:, :, t + 1 :] = 0.0
            masked = conv(x_masked)
            assert torch.allclose(full[:, :, : t + 1], masked[:, :, : t + 1], atol=1e-6), (
                f"future input at t={t} leaked into earlier outputs"
            )


def test_forward_backward_cpu():
    torch.manual_seed(0)
    m = make_tcn()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = torch.randn(4, 16, 2)
    y = torch.randn(4, 4, 2)
    loss = torch.nn.functional.mse_loss(m(x), y)
    loss.backward()
    opt.step()
    assert loss.item() > 0.0
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_autoregressive_head_runs():
    m = make_tcn(head="autoregressive")
    x = torch.randn(3, 16, 2)
    out = m(x)
    assert out.shape == (3, 4, 2)
