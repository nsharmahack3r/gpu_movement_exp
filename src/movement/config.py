"""Typed configuration schema, YAML loading/merging, and CLI overrides.

Config resolution order (later wins):

1. ``configs/base.yaml`` — shared defaults for every model arm.
2. A model config file (e.g. ``configs/model/tcn.yaml``) — model-arm overrides.
3. CLI ``--key value`` overrides, flattened with dot notation (e.g.
   ``--model.hidden_dim 128``).
4. Environment variables read through :mod:`movement.utils.env` (``RAW_DATASET_PATH``,
   ``PROCESSED_DATASET_PATH``).

The full merged config is snapshotted into every run directory so runs are
self-describing and reproducible.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataConfig(BaseModel):
    """Raw/processed data locations and trajectory assembly parameters."""

    model_config = ConfigDict(extra="forbid")

    raw_path: Path
    processed_path: Path
    raw_csv: str = "*.csv"
    # Sampling-interval assumptions. nominal_dt is the expected interval between
    # fixes (hours); a gap larger than max_gap_multiplier * nominal_dt splits the
    # trajectory into separate segments.
    nominal_dt_hours: float = 1.0
    max_gap_multiplier: float = 3.0
    # Maximum biologically plausible speed between consecutive fixes (m/s).
    # Steps exceeding this (GPS glitches, relocations) split the trajectory the
    # same way a gap does, so no window straddles a monster jump. None = off.
    max_speed_mps: float | None = None
    # Fraction of a trajectory kept for the *training* split. The remaining tail is
    # used for val/test so every animal appears in exactly one split.
    # val_fraction is of the whole dataset (after dedup); test_fraction likewise.
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    # Unit of the train/val/test assignment.
    #   "segment"    — legacy behaviour of the 27-run study: gap-separated
    #                  *segments* are shuffled, so one animal's segments can land
    #                  in several splits (progress_report.md §7.1). Kept as the
    #                  default so existing runs and split files reproduce exactly.
    #   "individual" — unique animal IDs are shuffled and every segment of an
    #                  animal goes to the same split. Split files are asserted
    #                  pairwise-disjoint on load.
    #   "individual_kfold" — animals are dealt into `n_folds` folds balanced by
    #                  *fix count* (not animal count); fold `fold` is test, fold
    #                  `fold + 1` (mod n_folds) is validation, the rest train.
    #                  Running every fold tests each animal exactly once. Use this
    #                  for small datasets, where a random 70/15/15 draw of a few
    #                  animals makes val/test sizes swing by an order of magnitude.
    #   "individual_kfold_tailval" — as individual_kfold for the *test* fold
    #                  (animal-disjoint), but there is no validation fold: every
    #                  other animal trains, and the last `val_tail_fraction` of each
    #                  training animal's fixes (by time) is held out as validation
    #                  for early stopping / checkpoint selection. ~80% of animals
    #                  train instead of ~60% (5 folds); test stays on unseen animals.
    split_unit: Literal["segment", "individual", "individual_kfold", "individual_kfold_tailval"] = "segment"
    # Only for split_unit=individual_kfold(_tailval) (the fold assignment uses trainer.seed).
    n_folds: int = 5
    fold: int = 0
    # Only for split_unit=individual_kfold_tailval.
    val_tail_fraction: float = 0.15


class WindowingConfig(BaseModel):
    """Sliding-window construction parameters."""

    model_config = ConfigDict(extra="forbid")

    input_len: int = 24
    horizon: int = 12
    stride: int = 1
    # Eval stride is applied to the *non-train* splits to reduce overlap.
    eval_stride: int = 12


class TransformsConfig(BaseModel):
    """Feature engineering on top of raw (lat, lon, timestamp) fixes."""

    model_config = ConfigDict(extra="forbid")

    # Project each segment to a local metric frame about its centroid.
    projection: Literal["local_enu", "utm"] = "local_enu"
    # Model displacements (delta x/y) rather than absolute position.
    delta_encoding: bool = True
    # Extra input features (all off by default so the CNN baseline is clean).
    step_length: bool = False
    turning_angle: bool = False
    speed: bool = False
    delta_t: bool = False
    cyclical_time: bool = False
    # Standardisation scalers are fit on the training split only and persisted.
    scale: bool = True
    # Per-fix time context delivered to the model through ``context`` (not
    # appended to ``x``): local-solar hour and day-of-year as sin/cos for every
    # observed fix, plus the same for the *nominal* future fix times
    # (last fix + k * nominal_dt). Future clock times are known in advance (the
    # collar schedule), so they are legitimate decoder inputs; actual future
    # timestamps are never used. Arms that do not read it ignore it.
    time_context: bool = False
    # Memory features (``movement.data.memory``): where the animal was at the same
    # clock time on each of the previous ``memory_days`` days (one feature vector
    # per forecast step), and a home-range summary of the last
    # ``memory_lookback_days`` (one vector per window). Built strictly from the
    # same animal's fixes at or before the last observed fix; delivered through
    # ``context`` like the time features.
    memory: bool = False
    memory_days: int = 7
    memory_lookback_days: float = 14.0
    memory_near_m: float = 200.0


class CovariatesConfig(BaseModel):
    """Per-fix environmental covariates joined from ``GEE_DATASET_PATH``.

    Off by default: the clean no-covariate baseline stays byte-identical. The
    covariate file for a raw CSV ``<stem>.csv`` and source ``<source>`` is
    ``<path>/<file_pattern>`` (default ``<stem>_<source>.csv``), one row per fix,
    keyed on (study_id, individual_id, timestamp).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    path: Path | None = None
    sources: list[str] = ["sentinel2"]
    file_pattern: str = "{stem}_{source}.csv"
    # Regexes selecting covariate columns (after `date_columns` are converted).
    include: list[str] = ["^s2_"]
    exclude: list[str] = []
    # Date columns converted to a signed age feature `<col>_age_days`
    # (fix time minus the observation's date), then dropped.
    date_columns: list[str] = ["s2_composite_date"]
    # Standardised values are clipped to +/- this many train SDs.
    clip: float = 8.0
    # Fail loudly if fewer than this fraction of fixes find a covariate row.
    min_coverage: float = 0.5
    # Normalised-difference indices are physically bounded to [-1, 1]; columns
    # matching these regexes are clipped to that range when loaded. EVI's
    # denominator can approach 0 over bright snow, giving rare values of ±100s
    # (0.24% of mule-deer fixes) that would otherwise inflate its training SD
    # 7-fold and squash every ordinary value towards 0. Boar values all lie
    # inside [-1, 1], so boar runs are unaffected.
    bounded_index_columns: list[str] = ["^s2_(ndvi|evi|savi|ndwi|ndmi|nbr)(_buf30)?$"]
    # Imagery embargo (days). Covariates of observed fixes less than this many
    # days before the forecast origin (the last observed fix) are hidden
    # (marked missing). Sentinel-2 composites are centred on the fix date
    # (±8 days), so an embargo of 9 days guarantees that no imagery acquired
    # after the forecast origin reaches the model. 0 = off.
    embargo_days: float = 0.0


class ModelConfig(BaseModel):
    """CNN (TCN) hyperparameters — all in config, never hardcoded."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["tcn"] = "tcn"
    # Channel widths per residual block.
    channels: list[int] = [64, 64, 64, 64]
    kernel_size: int = 3
    dropout: float = 0.2
    activation: Literal["gelu", "relu"] = "gelu"
    norm: Literal["none", "layernorm"] = "layernorm"
    # "direct" multi-step head (default) or "autoregressive" ablation.
    head: Literal["direct", "autoregressive"] = "direct"


class LSTMModelConfig(BaseModel):
    """LSTM encoder–decoder hyperparameters (all in config, never hardcoded)."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["lstm"] = "lstm"
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    # "direct" multi-step head (default) or "autoregressive" decoder ablation.
    decoder: Literal["direct", "autoregressive"] = "direct"
    # Teacher forcing / scheduled sampling (autoregressive decoder only).
    teacher_forcing_ratio: float = 1.0
    # Decay of the teacher-forcing ratio over `scheduled_sampling_epochs`:
    # "constant", "linear", or "inverse_sigmoid".
    scheduled_sampling: Literal["constant", "linear", "inverse_sigmoid"] = "constant"
    scheduled_sampling_epochs: int = 50
    # Optional attention in the autoregressive decoder (ablation, default off).
    attention: Literal["none", "additive", "dot"] = "none"


class TransformerModelConfig(BaseModel):
    """Transformer encoder hyperparameters (all in config, never hardcoded)."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["transformer"] = "transformer"
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    # "encoder_only" (default, direct multi-step head) or "encoder_decoder" ablation.
    mode: Literal["encoder_only", "encoder_decoder"] = "encoder_only"
    # Positional encoding: "sinusoidal" (default), "learned", "time_aware".
    pos_encoding: Literal["sinusoidal", "learned", "time_aware"] = "sinusoidal"
    # Pooling for the encoder-only head: "last" (default), "mean", "cls".
    pooling: Literal["last", "mean", "cls"] = "last"
    # Causal attention mask (ablation to match the TCN's causal inductive bias).
    causal_mask: bool = False
    # Pre-norm transformer layers (modern default; post-norm needs careful warmup).
    norm_first: bool = True


class CovariateTransformerModelConfig(BaseModel):
    """Covariate-aware Transformer (``cov_transformer``) hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["cov_transformer"] = "cov_transformer"
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 1
    dim_feedforward: int = 512
    dropout: float = 0.1
    # Covariate branch. False builds the identical model minus the covariate
    # encoder: the no-covariate ablation of the same architecture.
    use_covariates: bool = True
    # Width of each covariate's own embedding before variable selection.
    var_embed_dim: int = 16
    # Modality dropout: per sample, hide *all* covariates (marked missing) with
    # this probability during training, so the model cannot become unusable
    # when imagery is absent (cloud, 2018 Sentinel-2 gaps).
    covariate_dropout: float = 0.1
    # Variable-receptive-field training (MoveFormer): per sample, with this
    # probability mask a random-length prefix of the input window so the model
    # learns from contexts shorter than input_len. Training only.
    context_dropout: float = 0.2
    min_context: int = 4
    # Probabilistic head: the decoder is driven by a Gaussian noise vector and
    # returns sample trajectories (B, n_samples, horizon, 2) instead of one path.
    # Trained with the energy score (a proper scoring rule over samples;
    # Gneiting & Raftery 2007, as in MoveBench); the point forecast used for
    # ADE/FDE is the per-step spatial median of the samples.
    probabilistic: bool = False
    noise_dim: int = 16
    n_samples_train: int = 16
    n_samples_eval: int = 64
    # How covariates enter the model.
    #   early — fused into every observed-fix token before the encoder (the
    #           original design; lets covariates act as a location fingerprint).
    #   late  — a separate covariate memory the *decoder* cross-attends to,
    #           behind a learnable per-channel gate initialised almost closed
    #           (sigmoid(fusion_gate_init) ≈ 0.018), so training starts from the
    #           movement-only model and opens the gate only if it helps.
    covariate_fusion: Literal["early", "late"] = "early"
    # What the model sees of each covariate at each fix.
    #   levels  — the standardised value itself (original design).
    #   changes — (a) the value minus the window's observed mean and (b) the
    #             change since the previous fix: "is this spot greener/wetter
    #             than where the animal has been?" rather than "which place is
    #             this?". Each is divided by its train-split std.
    #   both    — levels and changes.
    covariate_features: Literal["levels", "changes", "both"] = "levels"
    fusion_gate_init: float = -4.0
    # Destination-hexagon head (0 = off). A classifier over a hexagonal grid in
    # the forecast frame (centred on the last observed fix): `hex_rings` rings of
    # pointy-top cells with edge `hex_edge_m` (174 m ≈ H3 resolution 9) plus one
    # "outside the grid" class. It predicts which cell holds the position at the
    # final horizon step; trained jointly as energy score + hex_loss_weight * CE.
    hex_rings: int = 0
    hex_edge_m: float = 174.0
    hex_loss_weight: float = 0.5


class FaunaFormerModelConfig(CovariateTransformerModelConfig):
    """FaunaFormer: the probabilistic covariate Transformer with late, gated fusion of
    covariate *changes* (see ``movement.models.cov_transformer.FaunaFormer``).

    Same architecture family and hyperparameters as ``cov_transformer``; only the
    defaults below differ. Its config (``configs/model/faunaformer.yaml``) also
    restricts covariates to the six spectral indices and turns on rotation
    augmentation.
    """

    name: Literal["faunaformer"] = "faunaformer"
    probabilistic: bool = True
    covariate_fusion: Literal["early", "late"] = "late"
    covariate_features: Literal["levels", "changes", "both"] = "changes"
    covariate_dropout: float = 0.3


class TrainerConfig(BaseModel):
    """Training-loop parameters (model-agnostic)."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    deterministic: bool = False
    batch_size: int = 256
    eval_batch_size: int = 512
    gradient_accumulation_steps: int = 1
    max_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # Cosine schedule with linear warmup (fraction of total steps).
    warmup_fraction: float = 0.05
    # Gradient clipping norm.
    clip_grad_norm: float = 10.0
    amp: bool = True
    early_stopping_patience: int = 15
    # Train-split augmentation: rotate every training window by a uniform random
    # angle and mirror it with probability 0.5 (inputs and targets together).
    # Movement direction is not identifiable from the track on unseen animals, so
    # this asks the model to learn step length, timing and turning instead of
    # memorised headings. Val/test are never augmented.
    augment_rotation: bool = False
    # Paths.
    run_dir: Path = Field(default=Path("runs"))
    checkpoint_dir: Path = Field(default=Path("checkpoints"))
    tensorboard_dir: Path = Field(default=Path("tensorboard"))
    log_level: str = "INFO"


class EvaluationConfig(BaseModel):
    """Test-set evaluation parameters."""

    model_config = ConfigDict(extra="forbid")

    # Evaluate the per-horizon error breakdown on the test set.
    per_horizon: bool = True
    # Number of predicted-vs-actual figures to save.
    n_plots: int = 4
    # Random seed for figure selection.
    plot_seed: int = 0


class TrackingConfig(BaseModel):
    """Run tracking: TensorBoard (default) and optional W&B."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["tensorboard", "wandb"] = "tensorboard"
    project: str = "movement-forecasting"
    entity: str | None = None
    offline: bool = True


class Config(BaseModel):
    """Top-level configuration tree.

    ``model`` is a discriminated union on ``model.name``: the CNN and LSTM arms
    have disjoint schemas, and the matching one is selected automatically.
    """

    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    windowing: WindowingConfig
    transforms: TransformsConfig
    model: Annotated[
        ModelConfig | LSTMModelConfig | TransformerModelConfig | CovariateTransformerModelConfig
        | FaunaFormerModelConfig,
        Field(discriminator="name"),
    ]
    trainer: TrainerConfig
    evaluation: EvaluationConfig
    tracking: TrackingConfig
    # Optional; absent from older run snapshots, which therefore load with
    # covariates disabled — exactly what those runs trained with.
    covariates: CovariatesConfig = Field(default_factory=CovariatesConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _interpolate_env(node: Any) -> Any:
    """Recursively substitute ``${ENV_VAR}`` strings from the environment.

    Missing variables raise a clear error instead of silently passing through
    (the prompt demands no silent fallbacks).
    """
    if isinstance(node, str) and node.startswith("${") and node.endswith("}"):
        name = node[2:-1]
        if name not in os.environ:
            raise RuntimeError(
                f"Config references environment variable {name!r} but it is not set "
                f"(checked in {name!r}). Ensure .env defines it."
            )
        return os.environ[name]
    if isinstance(node, dict):
        return {k: _interpolate_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate_env(v) for v in node]
    return node


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` into ``base`` (mutation-free).

    The ``"model"`` section is special-cased to **replace** rather than merge:
    each arm config declares the full set of its own keys, and inheriting the
    other arm's keys (e.g. TCN channels into an LSTM config) would fail schema
    validation. Everything else merges recursively.
    """
    out = copy.deepcopy(base)
    if isinstance(override, dict) and "model" in override and isinstance(override["model"], dict):
        out["model"] = copy.deepcopy(override["model"])
    for key, value in override.items():
        if key == "model":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dict(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``--key value`` CLI overrides using dot notation.

    Raises a clear error for unknown keys rather than silently ignoring them.
    """
    if not overrides:
        return cfg
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected --key=value")
        key, value = item.split("=", 1)
        parts = key.split(".")
        node = out
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"Unknown config key: {key}")
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key: {key}")
        node[parts[-1]] = _coerce(value)
    return out


def _coerce(value: str) -> Any:
    """Best-effort scalar coercion for CLI string values."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        import json

        return json.loads(value)
    return value


def load_config(
    model_config_file: str | Path | None = None,
    overrides: list[str] | None = None,
    base_config_file: str | Path | None = None,
) -> Config:
    """Load and validate the full configuration.

    Parameters
    ----------
    model_config_file:
        Path to a model-arm YAML (e.g. ``configs/model/tcn.yaml``). Relative paths
        resolve against the repo root (the directory containing ``configs/``).
    overrides:
        ``--key=value`` CLI overrides applied on top of the YAML files.
    base_config_file:
        Override the default ``configs/base.yaml`` (mostly for tests).

    Returns
    -------
    Config:
        Validated, fully-merged configuration.
    """
    from movement.utils.env import load_env, repo_root

    load_env()
    repo_root = repo_root()
    if base_config_file is None:
        base_config_file = repo_root / "configs" / "base.yaml"
    if model_config_file is not None:
        path = Path(model_config_file)
        if not path.is_absolute():
            path = repo_root / path
    else:
        path = None

    cfg: dict[str, Any] = _load_yaml(Path(base_config_file))
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Model config file not found: {path}")
        cfg = merge_dict(cfg, _load_yaml(path))
    cfg = _interpolate_env(cfg)
    cfg = apply_overrides(cfg, overrides)
    return Config.model_validate(cfg)


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Serialize a config to a plain dict (for YAML snapshots)."""
    return cfg.model_dump(mode="json")


@model_validator(mode="after")
def _validate_windowing(self: WindowingConfig) -> WindowingConfig:
    if self.input_len < 1 or self.horizon < 1:
        raise ValueError("input_len and horizon must be >= 1")
    return self
