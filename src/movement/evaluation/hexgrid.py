"""Local hexagonal destination grid and hex-cell scoring.

The grid lives in the forecast frame (metres east/north of the last observed
fix): pointy-top hexagons with edge ``edge_m`` (174 m ≈ H3 resolution 9), all
cells within ``rings`` rings of the centre cell, plus one extra class for
"outside the grid". With rings = 10 there are 331 cells reaching ~3 km.

Why a local grid rather than global H3 cells: the forecast frame is centred on
the animal and (in training) randomly rotated, so a grid tied to that frame
gives every window the same classes. Cell sizes match H3 resolution 9.

Scores for a forecast that gives each window a probability over the
``n_cells + 1`` classes:

- ``nll``  — mean −log p(true cell) (lower is better);
- ``top1`` / ``top5`` — share of windows whose true cell is the most / one of
  the five most probable;
- ``outside`` — share of true destinations outside the grid.

Sample-based forecasts (the probabilistic arms, the climatology) get cell
probabilities from a Gaussian kernel density of their sampled destinations, so
every model is scored on the same classes.
"""

from __future__ import annotations

import numpy as np

SQRT3 = np.sqrt(3.0)
#: Probability mixed in uniformly over all classes so a single miss never costs ∞.
EPS = 1e-4


class HexGrid:
    def __init__(self, rings: int = 10, edge_m: float = 174.0):
        if rings < 0 or edge_m <= 0:
            raise ValueError("rings must be >= 0 and edge_m > 0")
        self.rings = int(rings)
        self.edge = float(edge_m)
        qs, rs = [], []
        for q in range(-rings, rings + 1):
            for r in range(max(-rings, -q - rings), min(rings, -q + rings) + 1):
                qs.append(q)
                rs.append(r)
        self.q = np.array(qs)
        self.r = np.array(rs)
        self.n_cells = len(qs)
        self.n_classes = self.n_cells + 1
        self.outside = self.n_cells
        self._lookup = np.full((2 * rings + 1, 2 * rings + 1), self.outside, dtype=np.int64)
        self._lookup[self.q + rings, self.r + rings] = np.arange(self.n_cells)
        self.centres = np.stack([self.edge * SQRT3 * (self.q + self.r / 2.0), self.edge * 1.5 * self.r], axis=1)
        self.cell_area = 1.5 * SQRT3 * self.edge**2

    def assign(self, xy: np.ndarray) -> np.ndarray:
        """Class index of each point ``(..., 2)`` in metres (``outside`` beyond the grid)."""
        x, y = xy[..., 0] / self.edge, xy[..., 1] / self.edge
        qf = SQRT3 / 3.0 * x - y / 3.0
        rf = 2.0 / 3.0 * y
        sf = -qf - rf
        q, r, s = np.round(qf), np.round(rf), np.round(sf)
        dq, dr, ds = np.abs(q - qf), np.abs(r - rf), np.abs(s - sf)
        fix_q = (dq > dr) & (dq > ds)
        fix_r = ~fix_q & (dr > ds)
        q = np.where(fix_q, -r - s, q)
        r = np.where(fix_r, -q - s, r)
        q, r = q.astype(np.int64), r.astype(np.int64)
        dist = np.maximum.reduce([np.abs(q), np.abs(r), np.abs(q + r)])
        inside = dist <= self.rings
        qi = np.clip(q + self.rings, 0, 2 * self.rings)
        ri = np.clip(r + self.rings, 0, 2 * self.rings)
        return np.where(inside, self._lookup[qi, ri], self.outside)

    def kde_probs(self, dest: np.ndarray, *, chunk: int = 256) -> np.ndarray:
        """Cell probabilities ``(B, n_classes)`` from sampled destinations ``(B, M, 2)``.

        Gaussian KDE per window (Scott bandwidth, at least half an edge), evaluated
        at cell centres × cell area; the outside class takes the remaining mass.
        """
        b, m, _ = dest.shape
        sd = np.sqrt(dest.var(axis=1).mean(axis=1))  # (B,)
        h = np.maximum(sd * m ** (-1.0 / 6.0), 0.5 * self.edge)
        out = np.empty((b, self.n_classes))
        for s in range(0, b, chunk):
            d = dest[s : s + chunk]  # (b, M, 2)
            hh = h[s : s + chunk, None, None]
            diff = d[:, :, None, :] - self.centres[None, None]  # (b, M, C, 2)
            dens = np.exp(-0.5 * (diff**2).sum(-1) / hh**2).mean(1) / (2 * np.pi * hh[:, :, 0] ** 2)
            inside = np.minimum(dens * self.cell_area, 1.0)
            tot = inside.sum(1, keepdims=True)
            inside = inside / np.maximum(tot, 1.0)  # renormalise only if the KDE overshoots
            out[s : s + chunk, : self.n_cells] = inside
            out[s : s + chunk, self.n_cells] = 1.0 - inside.sum(1)
        return out


def smooth(probs: np.ndarray) -> np.ndarray:
    """Mix ``EPS`` uniformly into class probabilities (rows sum to 1)."""
    probs = np.clip(probs, 0.0, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return (1.0 - EPS) * probs + EPS / probs.shape[1]


def hex_scores(probs: np.ndarray, true_cell: np.ndarray) -> dict:
    """``nll``, ``top1``, ``top5`` and ``outside`` for class probabilities ``(B, K)``."""
    p = smooth(probs)
    rows = np.arange(len(true_cell))
    nll = -np.log(p[rows, true_cell])
    top5 = np.argsort(-p, axis=1)[:, :5]
    return {
        "nll": float(nll.mean()),
        "top1": float((top5[:, 0] == true_cell).mean()),
        "top5": float((top5 == true_cell[:, None]).any(axis=1).mean()),
        "outside": float((true_cell == probs.shape[1] - 1).mean()),
        "nll_per_window": nll,
    }
