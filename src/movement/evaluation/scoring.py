"""Sample-based probabilistic scoring: energy score and spatial median.

All functions take *cumulative positions* (metres, frame anchored at the last
observed fix), not per-step displacements.

Energy score (Gneiting & Raftery 2007), per forecast step h, for samples
``X_1..X_M`` of a 2-D position and the observed position ``y``:

    ES_h = mean_m ||X_m - y|| - 1/(2 M (M-1)) * sum_{m != m'} ||X_m - X_m'||

This is the unbiased ("fair") estimator. It is a proper scoring rule, in metres,
and for a deterministic forecast (all samples equal) it reduces to the Euclidean
error ``||x - y||`` — so the step-averaged ES of a point forecast *is* its ADE,
and ES values of probabilistic and point forecasts are directly comparable.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

_EPS = 1e-9


def _norm(v: Tensor) -> Tensor:
    # sqrt(x + eps): finite gradient at exactly-equal samples (the diagonal).
    return torch.sqrt((v * v).sum(dim=-1) + _EPS)


def energy_score(samples: Tensor, target: Tensor) -> Tensor:
    """Per-window, per-step energy score.

    Parameters
    ----------
    samples: ``(B, M, H, 2)`` sampled positions.
    target:  ``(B, H, 2)`` observed positions.

    Returns ``(B, H)``. With ``M == 1`` only the first term is defined, so the
    score is the plain Euclidean error.
    """
    if samples.dim() != 4 or target.dim() != 3:
        raise ValueError(f"expected samples (B,M,H,2) and target (B,H,2); got {tuple(samples.shape)}, {tuple(target.shape)}")
    m = samples.size(1)
    term1 = _norm(samples - target.unsqueeze(1)).mean(dim=1)  # (B, H)
    if m == 1:
        return term1
    s = samples.permute(0, 2, 1, 3)  # (B, H, M, 2)
    pair = _norm(s.unsqueeze(3) - s.unsqueeze(2))  # (B, H, M, M); diagonal ~ sqrt(eps)
    term2 = pair.sum(dim=(-1, -2)) / (2.0 * m * (m - 1))
    return term1 - term2


def spatial_median(samples: Tensor, iters: int = 30) -> Tensor:
    """Per-step geometric (spatial) median of ``(B, M, H, 2)`` samples → ``(B, H, 2)``.

    Weiszfeld iterations from the coordinate-wise mean. The spatial median
    minimises the expected Euclidean error, so it is the natural point forecast
    to score with ADE/FDE.
    """
    z = samples.mean(dim=1)
    for _ in range(iters):
        d = _norm(samples - z.unsqueeze(1)).clamp_min(1e-6)  # (B, M, H)
        w = 1.0 / d
        z = (w.unsqueeze(-1) * samples).sum(dim=1) / w.sum(dim=1).unsqueeze(-1)
    return z


def energy_score_np(samples: np.ndarray, target: np.ndarray, chunk: int = 256) -> np.ndarray:
    """NumPy wrapper of :func:`energy_score` (chunked); returns ``(B, H)``."""
    out = []
    with torch.no_grad():
        for i in range(0, len(samples), chunk):
            out.append(energy_score(torch.as_tensor(samples[i:i + chunk], dtype=torch.float64),
                                    torch.as_tensor(target[i:i + chunk], dtype=torch.float64)).numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, target.shape[1]))


def spatial_median_np(samples: np.ndarray, chunk: int = 256) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(samples), chunk):
            out.append(spatial_median(torch.as_tensor(samples[i:i + chunk], dtype=torch.float64)).numpy())
    return np.concatenate(out, axis=0)


def random_rotations(n: int, rng: np.random.Generator, *, mirror: bool = True) -> np.ndarray:
    """``(n, 2, 2)`` uniform random rotations, each mirrored (x → −x) with p = 0.5."""
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    if mirror:
        flip = np.where(rng.random(n) < 0.5, -1.0, 1.0)
        rot = rot * np.stack([flip, np.ones(n)], -1)[:, None, :]  # scale column 0 = mirror x first
    return rot


def climatology_samples(
    train_future: np.ndarray,
    train_hours: np.ndarray,
    test_hours: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    *,
    by_hour: bool,
    min_pool: int = 30,
) -> np.ndarray:
    """No-skill probabilistic baseline: resample training futures, randomly rotated.

    ``train_future``: ``(N, H, 2)`` cumulative future positions of training
    windows (anchored at their last observed fix). For each test window, draw
    ``n_samples`` of them — from windows whose last fix has the same clock hour
    when ``by_hour`` (falling back to all windows if fewer than ``min_pool``) —
    and apply a random rotation/mirror to each. The result knows how far animals
    move at that time of day but nothing about where *this* animal is heading.
    Returns ``(B_test, n_samples, H, 2)``.
    """
    pools = {}
    all_idx = np.arange(len(train_future))
    if by_hour:
        for h in np.unique(test_hours):
            idx = np.flatnonzero(train_hours == h)
            pools[int(h)] = idx if len(idx) >= min_pool else all_idx
    out = np.empty((len(test_hours), n_samples) + train_future.shape[1:], dtype=np.float64)
    for i, h in enumerate(test_hours):
        pool = pools[int(h)] if by_hour else all_idx
        pick = train_future[rng.choice(pool, size=n_samples, replace=True)]  # (M, H, 2)
        rot = random_rotations(n_samples, rng)
        out[i] = np.einsum("mij,mhj->mhi", rot, pick)
    return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
#: Nominal coverage levels reported in metrics.json and the study report.
COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)
#: Finer grid for calibration curves (figure).
CURVE_LEVELS = tuple(round(0.05 * k, 2) for k in range(1, 20))


def region_calibration(samples: np.ndarray, target: np.ndarray,
                       levels: tuple[float, ...] = COVERAGE_LEVELS) -> dict:
    """Coverage of centre-outward prediction regions built from samples.

    For each window and forecast step the α-region is the disc centred on the
    samples' spatial median whose radius is the α-quantile of the samples'
    distances to that median. A calibrated forecaster puts the true position
    inside the α-region a fraction α of the time; coverage below α means the
    predicted spread is too narrow (overconfident), above α too wide.

    Parameters
    ----------
    samples: ``(B, M, H, 2)`` sampled cumulative positions (metres).
    target:  ``(B, H, 2)`` true cumulative positions.

    Returns ``{"coverage": {α: float}, "coverage_step": {α: [H floats]},
    "radius": {α: float}}`` — radius is the mean region radius in metres (a
    sharpness measure: smaller is sharper at equal coverage).
    """
    med = spatial_median_np(samples)  # (B, H, 2)
    r_samp = np.linalg.norm(samples - med[:, None], axis=-1)  # (B, M, H)
    r_obs = np.linalg.norm(target - med, axis=-1)  # (B, H)
    out = {"coverage": {}, "coverage_step": {}, "radius": {}}
    for a in levels:
        rad = np.quantile(r_samp, a, axis=1)  # (B, H)
        inside = r_obs <= rad
        key = f"{a:g}"
        out["coverage"][key] = float(inside.mean())
        out["coverage_step"][key] = inside.mean(axis=0).tolist()
        out["radius"][key] = float(rad.mean())
    return out
