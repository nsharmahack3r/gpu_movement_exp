"""Covariate-aware Transformer (``cov_transformer``) for displacement forecasting.

A fourth Transformer design built to *use* the per-fix satellite covariates in
``GEE_DATASET_PATH``, with each component traceable to the reviewed literature
(``literature_review.md``, ``literature.md``):

1. **Per-fix gated fusion.** Every observed fix becomes one token that fuses its
   movement features, its time context and — through a gate — its covariate
   vector. The gate lets the model learn to ignore covariates wherever they do
   not help, which the literature says is the likely case: MoveFormer (Cífka et
   al. 2023) ranks the movement vector far above any environmental feature, and
   Forrest et al. (2026) found derived-covariate models overfit.
2. **Variable selection over covariates** (Temporal Fusion Transformer, Lim et
   al. 2021 — the architecture of the only multi-horizon animal forecaster in the
   review, the elephant-seal study). Each covariate gets its own embedding of
   ``(value, missing)``; a softmax over variables weights them per fix. The
   weights are exported at eval (``covariate_selection.csv``) as a built-in,
   per-variable importance read-out.
3. **Missingness as input** (``covariate_plan.md`` §2.5): the missing flag is
   part of every variable's embedding, so "no cloud-free scene" is information,
   not a zero.
4. **Modality dropout**: during training all covariates of a sample are hidden
   (marked missing) with probability ``covariate_dropout``, so the model stays
   usable when imagery is absent (cloud, pre-2019 Sentinel-2 gaps).
5. **Time as position** (MoveFormer): local-solar hour and year phase per fix are
   added to every token, next to a learned index embedding. Satter et al. (2025)
   show wild-pig state switching on a solar-diel schedule.
6. **Variable receptive field training** (MoveFormer's best ablation, ``VarCtx``):
   with probability ``context_dropout`` a random-length prefix of the window is
   masked out, so the model learns from contexts shorter than ``input_len``.
7. **Horizon queries with known future time** (TFT's "known future inputs"):
   ``horizon`` learned queries, each told the *nominal* clock time of the fix it
   forecasts, cross-attend to the encoded history and are decoded jointly in one
   non-autoregressive pass. No ground truth is ever fed.

8. **Output in units of the training displacement scale.** The head predicts
   per-step displacement divided by ``target_scale`` (the train-split RMS step,
   stored as a buffer so checkpoints carry it). Raw-metre targets of ±hundreds
   of metres make a freshly initialised head sit at "no movement" for thousands
   of Adam steps at Transformer learning rates — the flat-loss symptom seen in
   the first boar probe and a plausible cause of the wolf-LSTM collapse
   (progress_report.md §5.1f/§10).

9. **Optional probabilistic head** (``probabilistic=true``). A Gaussian noise
   vector is added to the horizon queries and to the decoder output, so each
   noise draw decodes to one plausible 12-step path. The model then returns
   ``(B, n_samples, horizon, 2)`` and is trained with the energy score over
   samples — the proper scoring rule MoveBench (2026) uses for animal movement,
   where single-path forecasts reward "predict no movement". The encoder runs
   once per window; only the one-layer decoder runs per sample.

10. **Optional memory features** (``n_memory_step`` / ``n_memory_global`` > 0,
   from ``transforms.memory``; see :mod:`movement.data.memory`): where the animal
   was at each forecast step's clock time on previous days is added to that
   step's horizon query, and a home-range summary to every token and query.

11. **Optional destination-hexagon head** (``hex_rings`` > 0): a classifier over
   a local hexagonal grid (see :mod:`movement.evaluation.hexgrid`) for the final
   horizon step, read from the noise-free decoder output. Its logits are left in
   ``last_hex_logits`` after every forward pass; the trainer adds their
   cross-entropy to the energy score.

Point mode (the default) is unchanged from the other arms: per-step (Δx, Δy)
displacements in metres, direct multi-step. With ``use_covariates=false`` the
identical model is built minus the covariate branch — the no-covariate ablation
that isolates what the satellite data adds.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from movement.models.base import ForecastModel

logger = logging.getLogger(__name__)

COVARIATE_FEATURE_MULTIPLIER = {"levels": 1, "changes": 2, "both": 3}


def covariate_changes(values: Tensor, missing: Tensor, hidden: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Within-window covariate changes: anomaly vs the window mean, and step change.

    Parameters
    ----------
    values, missing: ``(B, T, C)`` standardised values (0 where missing) and flags.
    hidden: optional ``(B, T)`` bool, True for fixes the model may not see (the
        masked prefix of variable-receptive-field training). Hidden fixes are
        treated as missing, so they never leak into the window mean.

    Returns ``(changes, changes_missing)``, both ``(B, T, 2C)``: first the
    anomaly ``v_t − mean_{observed s}(v_s)`` for every covariate, then the step
    ``v_t − v_{t−1}`` (defined only where both fixes are observed). Undefined
    entries are 0 and flagged missing.
    """
    obs = 1.0 - missing
    if hidden is not None:
        obs = obs * (~hidden).unsqueeze(-1).to(obs.dtype)
    n_obs = obs.sum(dim=1, keepdim=True)
    mean = (values * obs).sum(dim=1, keepdim=True) / n_obs.clamp_min(1.0)
    anomaly = (values - mean) * obs
    both = obs[:, 1:] * obs[:, :-1]
    step = torch.zeros_like(values)
    step[:, 1:] = (values[:, 1:] - values[:, :-1]) * both
    step_obs = torch.zeros_like(obs)
    step_obs[:, 1:] = both
    return torch.cat([anomaly, step], dim=-1), torch.cat([1.0 - obs, 1.0 - step_obs], dim=-1)


def covariate_feature_names(columns: list[str], features: str) -> list[str]:
    """Names of the derived covariate inputs, in model order."""
    out: list[str] = []
    if features in ("levels", "both"):
        out += list(columns)
    if features in ("changes", "both"):
        out += [f"{c}:anomaly" for c in columns] + [f"{c}:step" for c in columns]
    return out


class GatedResidualNetwork(nn.Module):
    """TFT gated residual network: ``LayerNorm(skip(a) + GLU(W2 ELU(W1 a)))``."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.glu = nn.Linear(d_hidden, 2 * d_out)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, a: Tensor) -> Tensor:
        h = F.elu(self.fc1(a))
        h = self.drop(self.fc2(h))
        h = F.glu(self.glu(h), dim=-1)
        return self.norm(self.skip(a) + h)


class CovariateSelection(nn.Module):
    """Per-variable ``(value, missing)`` embeddings + softmax variable selection.

    Input ``(B, T, C)`` values (standardised, 0 where missing) and ``(B, T, C)``
    missing flags. Output ``(B, T, d_model)`` plus the ``(B, T, C)`` weights.
    """

    def __init__(self, n_vars: int, d_var: int, d_model: int, dropout: float):
        super().__init__()
        self.n_vars = n_vars
        # Variable-specific affine embedding of (value, missing): C x 2 x d_var.
        self.embed_weight = nn.Parameter(torch.randn(n_vars, 2, d_var) * (1.0 / 2**0.5))
        self.embed_bias = nn.Parameter(torch.zeros(n_vars, d_var))
        # Shared non-linear transform of each variable's embedding.
        self.var_grn = GatedResidualNetwork(d_var, d_var, d_var, dropout)
        # Selection weights from the whole covariate vector at this fix.
        self.selector = GatedResidualNetwork(2 * n_vars, d_model, n_vars, dropout)
        self.out = nn.Linear(d_var, d_model)

    def forward(self, values: Tensor, missing: Tensor) -> tuple[Tensor, Tensor]:
        pair = torch.stack([values, missing], dim=-1)  # (B, T, C, 2)
        emb = torch.einsum("btci,cid->btcd", pair, self.embed_weight) + self.embed_bias
        emb = self.var_grn(emb)  # (B, T, C, d_var)
        weights = torch.softmax(self.selector(torch.cat([values, missing], dim=-1)), dim=-1)
        pooled = (weights.unsqueeze(-1) * emb).sum(dim=-2)  # (B, T, d_var)
        return self.out(pooled), weights


class CovariateTransformer(ForecastModel):
    """Covariate-aware encoder–decoder Transformer (see module docstring)."""

    def __init__(
        self,
        input_len: int,
        horizon: int,
        n_features: int,
        n_covariates: int,
        n_time_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        var_embed_dim: int = 16,
        covariate_dropout: float = 0.1,
        context_dropout: float = 0.2,
        min_context: int = 4,
        covariate_columns: list[str] | None = None,
        target_scale: float = 1.0,
        probabilistic: bool = False,
        noise_dim: int = 16,
        n_samples_train: int = 16,
        n_samples_eval: int = 64,
        covariate_fusion: str = "early",
        covariate_features: str = "levels",
        fusion_gate_init: float = -4.0,
        covariate_change_scale: list[float] | None = None,
        n_memory_step: int = 0,
        n_memory_global: int = 0,
        hex_rings: int = 0,
        hex_edge_m: float = 174.0,
    ):
        super().__init__()
        if n_time_features <= 0:
            raise ValueError("CovariateTransformer needs time features (transforms.time_context=true).")
        if not 1 <= min_context <= input_len:
            raise ValueError(f"min_context must be in [1, input_len={input_len}], got {min_context}")
        self.input_len = input_len
        self.horizon = horizon
        self.n_covariates = n_covariates
        self.use_covariates = n_covariates > 0
        self.covariate_dropout = covariate_dropout
        self.context_dropout = context_dropout
        self.min_context = min_context
        self.covariate_columns = list(covariate_columns or [])
        if covariate_fusion not in ("early", "late"):
            raise ValueError(f"covariate_fusion must be 'early' or 'late', got {covariate_fusion!r}")
        if covariate_features not in COVARIATE_FEATURE_MULTIPLIER:
            raise ValueError(f"covariate_features must be one of {sorted(COVARIATE_FEATURE_MULTIPLIER)}")
        self.covariate_fusion = covariate_fusion
        self.covariate_features = covariate_features
        self.n_covariate_inputs = n_covariates * COVARIATE_FEATURE_MULTIPLIER[covariate_features]
        self.is_probabilistic = bool(probabilistic)
        self.noise_dim = noise_dim
        self.n_samples_train = n_samples_train
        self.n_samples_eval = n_samples_eval
        if self.is_probabilistic and (n_samples_train < 2 or n_samples_eval < 2):
            raise ValueError("The energy score needs at least 2 samples (n_samples_train/eval >= 2).")

        # --- per-fix token -------------------------------------------------
        self.motion = GatedResidualNetwork(n_features, d_model, d_model, dropout)
        self.time_in = nn.Linear(n_time_features, d_model)
        self.index_embed = nn.Embedding(input_len, d_model)
        if self.use_covariates:
            self.covariates = CovariateSelection(self.n_covariate_inputs, var_embed_dim, d_model, dropout)
            if covariate_fusion == "early":
                self.fuse_gate = nn.Linear(2 * d_model, d_model)
            else:
                # Late fusion: a covariate memory (per fix, with its own time and
                # index context) that decoder outputs cross-attend to, gated.
                self.cov_norm = nn.LayerNorm(d_model)
                self.cov_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
                self.fusion_gate = nn.Parameter(torch.full((d_model,), float(fusion_gate_init)))
                # Excluded from weight decay (the trainer honours this flag): AdamW's
                # decay would pull the gate logit towards 0, i.e. *open*, whether or
                # not covariates help, and the gate's opening is reported as a result.
                self.fusion_gate.no_weight_decay = True
            if covariate_features != "levels":
                n_chg = 2 * n_covariates
                scale = torch.ones(n_chg) if covariate_change_scale is None else torch.tensor(
                    [float(v) if v and v > 1e-8 else 1.0 for v in covariate_change_scale])
                if scale.numel() != n_chg:
                    raise ValueError(f"covariate_change_scale needs {n_chg} values, got {scale.numel()}")
                # Train-split std of each anomaly/step input (checkpointed).
                self.register_buffer("covariate_change_scale", scale)
        self.token_norm = nn.LayerNorm(d_model)
        self.token_drop = nn.Dropout(dropout)

        # --- encoder -------------------------------------------------------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            activation="gelu", norm_first=True, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers, enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(d_model)

        # --- horizon decoder ----------------------------------------------
        self.horizon_query = nn.Parameter(torch.randn(horizon, d_model) * 0.02)
        self.future_time_in = nn.Linear(n_time_features, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            activation="gelu", norm_first=True, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        if self.is_probabilistic:
            self.noise_query = nn.Linear(noise_dim, d_model)
            self.noise_out = nn.Linear(noise_dim, d_model)

        # --- memory features (optional) -------------------------------------
        self.n_memory_step = int(n_memory_step)
        self.n_memory_global = int(n_memory_global)
        if self.n_memory_step:
            self.memory_step_in = GatedResidualNetwork(n_memory_step, d_model, d_model, dropout)
        if self.n_memory_global:
            self.memory_global_token = GatedResidualNetwork(n_memory_global, d_model, d_model, dropout)
            self.memory_global_query = nn.Linear(d_model, d_model)

        # --- destination-hexagon head (optional) ----------------------------
        self.hex_rings = int(hex_rings)
        self.hex_edge_m = float(hex_edge_m)
        self.last_hex_logits: Tensor | None = None
        if self.hex_rings > 0:
            n_classes = 3 * self.hex_rings * (self.hex_rings + 1) + 2  # cells + "outside"
            self.hex_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_classes))

        # Persistent buffer: saved in checkpoints, so eval uses the training value.
        self.register_buffer("target_scale", torch.tensor(float(target_scale)))

        self._last_selection: Tensor | None = None

    @property
    def receptive_field(self) -> int:
        return self.input_len

    # ------------------------------------------------------------------
    def _require(self, context: dict, key: str) -> Tensor:
        if key not in context:
            raise ValueError(
                f"CovariateTransformer needs context[{key!r}] — the datamodule adds it when "
                f"transforms.time_context / covariates.enabled are set."
            )
        return context[key]

    def _covariate_inputs(self, context: dict, batch: int) -> tuple[Tensor, Tensor]:
        cov = self._require(context, "covariates")
        missing = self._require(context, "covariate_missing")
        if cov.size(-1) != self.n_covariates:
            raise ValueError(f"Expected {self.n_covariates} covariates, got {cov.size(-1)}")
        if self.training and self.covariate_dropout > 0:
            drop = torch.rand(batch, 1, 1, device=cov.device) < self.covariate_dropout
            cov = cov.masked_fill(drop, 0.0)
            missing = torch.where(drop, torch.ones_like(missing), missing)
        return cov, missing

    def _context_mask(self, batch: int, device: torch.device) -> Tensor | None:
        """Key-padding mask hiding a random prefix (variable receptive field)."""
        if not (self.training and self.context_dropout > 0 and self.min_context < self.input_len):
            return None
        apply = torch.rand(batch, device=device) < self.context_dropout
        # Number of leading fixes to hide: 1 .. input_len - min_context.
        hide = torch.randint(1, self.input_len - self.min_context + 1, (batch,), device=device)
        hide = torch.where(apply, hide, torch.zeros_like(hide))
        positions = torch.arange(self.input_len, device=device).unsqueeze(0)
        return positions < hide.unsqueeze(1)  # True = masked

    def forward(self, x: Tensor, *, context: dict | None = None) -> Tensor:
        """``x: (B, input_len, F)`` + context extras → ``(B, horizon, 2)``."""
        if x.dim() != 3 or x.size(1) != self.input_len:
            raise ValueError(f"Expected x of shape (B, {self.input_len}, F), got {tuple(x.shape)}")
        context = context or {}
        batch = x.size(0)
        time_feats = self._require(context, "time_feats").to(x.dtype)
        future_time = self._require(context, "future_time_feats").to(x.dtype)

        positions = torch.arange(self.input_len, device=x.device)
        index = self.index_embed(positions).unsqueeze(0)
        time_emb = self.time_in(time_feats)
        h = self.motion(x) + time_emb + index
        mem_global = None
        if self.n_memory_global:
            mem_global = self.memory_global_token(self._require(context, "memory_global").to(x.dtype))  # (B, d)
            h = h + mem_global.unsqueeze(1)
        cov_memory = None
        if self.use_covariates:  # modality dropout draws first (same RNG order as before)
            cov, missing = self._covariate_inputs(context, batch)
        pad = self._context_mask(batch, x.device)
        if self.use_covariates:
            values, flags = self._covariate_features(cov.to(x.dtype), missing.to(x.dtype), pad)
            c, weights = self.covariates(values, flags)
            self._last_selection = weights.detach()
            self._last_missing = flags.detach()
            if self.covariate_fusion == "early":
                gate = torch.sigmoid(self.fuse_gate(torch.cat([h, c], dim=-1)))
                h = h + gate * c
            else:
                cov_memory = self.cov_norm(c + time_emb + index)
        h = self.token_drop(self.token_norm(h))

        memory = self.encoder_norm(self.encoder(h, src_key_padding_mask=pad))

        queries = self.horizon_query.unsqueeze(0) + self.future_time_in(future_time)  # (B, H, d)
        if self.n_memory_step:
            queries = queries + self.memory_step_in(self._require(context, "memory_step").to(x.dtype))
        if mem_global is not None:
            queries = queries + self.memory_global_query(mem_global).unsqueeze(1)
        if self.hex_rings > 0:
            # Noise-free decoder pass: the destination distribution over cells.
            self.last_hex_logits = self.hex_head(self._decode(queries, memory, pad, cov_memory)[:, -1])
        if self.is_probabilistic:
            n = int(context.get("n_samples") or (self.n_samples_train if self.training else self.n_samples_eval))
            return self._sample(queries, memory, pad, n, cov_memory)  # (B, M, H, 2) metres
        out = self._decode(queries, memory, pad, cov_memory)
        return self.head(out) * self.target_scale  # (B, H, 2) metres

    def _covariate_features(self, cov: Tensor, missing: Tensor, hidden: Tensor | None) -> tuple[Tensor, Tensor]:
        """Model inputs for the covariate branch: levels, changes, or both (see config)."""
        if self.covariate_features == "levels":
            return cov, missing
        chg, chg_missing = covariate_changes(cov, missing, hidden)
        chg = chg / self.covariate_change_scale.to(chg.dtype)
        if self.covariate_features == "changes":
            return chg, chg_missing
        return torch.cat([cov, chg], dim=-1), torch.cat([missing, chg_missing], dim=-1)

    def _decode(self, queries: Tensor, memory: Tensor, pad: Tensor | None, cov_memory: Tensor | None) -> Tensor:
        """Decoder + (late fusion) gated cross-attention to the covariate memory; normalised output."""
        out = self.decoder_norm(self.decoder(queries, memory, memory_key_padding_mask=pad))
        if cov_memory is not None:
            attn, _ = self.cov_attn(out, cov_memory, cov_memory, key_padding_mask=pad, need_weights=False)
            out = out + torch.sigmoid(self.fusion_gate) * attn
        return out

    def _sample(self, queries: Tensor, memory: Tensor, pad: Tensor | None, n: int,
                cov_memory: Tensor | None = None, chunk: int = 16) -> Tensor:
        """Decode ``n`` noise draws per window → ``(B, n, H, 2)``; ``chunk`` bounds memory."""
        b, h, d = queries.shape
        outs = []
        for start in range(0, n, chunk):
            m = min(chunk, n - start)
            z = torch.randn(b, m, self.noise_dim, device=queries.device, dtype=queries.dtype)
            q = (queries.unsqueeze(1) + self.noise_query(z).unsqueeze(2)).reshape(b * m, h, d)
            mem = memory.repeat_interleave(m, dim=0)
            kpm = pad.repeat_interleave(m, dim=0) if pad is not None else None
            cmem = cov_memory.repeat_interleave(m, dim=0) if cov_memory is not None else None
            dec = self._decode(q, mem, kpm, cmem).reshape(b, m, h, d)
            dec = dec + self.noise_out(z).unsqueeze(2)
            outs.append(self.head(dec))
        return torch.cat(outs, dim=1) * self.target_scale

    def fusion_gate_value(self) -> float | None:
        """Mean opening of the late-fusion gate (0 = covariates ignored, 1 = fully used)."""
        if not (self.use_covariates and self.covariate_fusion == "late"):
            return None
        return float(torch.sigmoid(self.fusion_gate.detach()).mean())

    # ------------------------------------------------------------------
    @torch.no_grad()
    def covariate_selection_summary(self, datamodule, device: torch.device, *, split: str = "test"):
        """Mean variable-selection weight per covariate over a split (eval read-out).

        Weights are averaged over every fix of every window, and also split by
        whether that covariate was observed or missing at the fix. They describe
        how the trained model *routes* covariate information; they are not a
        causal importance (use a permutation test for that).
        """
        import pandas as pd

        from movement.data.datamodule import unpack_batch

        if not self.use_covariates:
            return None
        was_training = self.training
        self.eval()
        total = torch.zeros(self.n_covariate_inputs, dtype=torch.float64)
        obs_sum = torch.zeros(self.n_covariate_inputs, dtype=torch.float64)
        obs_n = torch.zeros(self.n_covariate_inputs, dtype=torch.float64)
        n = 0
        for batch in datamodule.dataloader(split, shuffle=False):
            x, _y, dt, extras = unpack_batch(batch, device)
            self(x, context={"stage": split, "dt_seconds": dt, "n_samples": 2, **extras})
            w = self._last_selection.double().cpu()  # (B, T, C_in)
            observed = (1.0 - self._last_missing.double().cpu())
            total += w.sum(dim=(0, 1))
            obs_sum += (w * observed).sum(dim=(0, 1))
            obs_n += observed.sum(dim=(0, 1))
            n += w.shape[0] * w.shape[1]
        if was_training:
            self.train()
        base = self.covariate_columns or [f"cov_{i}" for i in range(self.n_covariates)]
        names = covariate_feature_names(base, self.covariate_features)
        frame = pd.DataFrame(
            {
                "covariate": names,
                "mean_weight": (total / max(n, 1)).numpy(),
                "mean_weight_when_observed": (obs_sum / obs_n.clamp_min(1)).numpy(),
                "observed_fraction": (obs_n / max(n, 1)).numpy(),
            }
        )
        return frame.sort_values("mean_weight", ascending=False).reset_index(drop=True)


class FaunaFormer(CovariateTransformer):
    """FaunaFormer — probabilistic movement forecaster with late, gated fusion of
    satellite-covariate *changes*.

    Built on the ``cov_transformer`` backbone (per-fix movement + time tokens,
    4-layer encoder, horizon queries with known future clock times, noise-driven
    decoder trained with the energy score). What is new is how covariates enter:

    - **Changes, not levels.** For each spectral index the model sees the value
      at a fix minus the window's observed mean, and the change since the
      previous fix (each scaled by its train-split std). This asks "is the animal
      moving onto greener / wetter ground?" instead of "which place is this?",
      the location fingerprint that made level covariates hurt on unseen animals.
    - **Late, gated fusion.** The movement encoder never sees covariates. The
      decoder cross-attends to a separate covariate memory through a per-channel
      gate initialised near-closed, so training starts from the movement-only
      model (the best arm so far) and the gate's final opening
      (``fusion_gate_value``) reports how much the covariates were used.
    - **Stronger modality dropout** (0.3) so forecasts stay usable without imagery.

    Defaults come from :class:`movement.config.FaunaFormerModelConfig`.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("probabilistic", True)
        kwargs.setdefault("covariate_fusion", "late")
        kwargs.setdefault("covariate_features", "changes")
        super().__init__(*args, **kwargs)
