"""Model registry: name -> builder. Adding a model arm = one entry here."""

from __future__ import annotations

import logging
from collections.abc import Callable

from movement.config import (
    CovariateTransformerModelConfig,
    LSTMModelConfig,
    ModelConfig,
    TransformerModelConfig,
    TransformsConfig,
    WindowingConfig,
)
from movement.models.base import ForecastModel
from movement.models.cov_transformer import CovariateTransformer, FaunaFormer
from movement.models.lstm import LSTMForecaster
from movement.models.tcn import TCN
from movement.models.transformer import TransformerForecaster

logger = logging.getLogger(__name__)

ModelBuilder = Callable[..., ForecastModel]


def _n_features(transforms: TransformsConfig, windowing: WindowingConfig) -> int:
    from movement.data.transforms import WindowTransform

    return WindowTransform(
        input_len=windowing.input_len,
        horizon=windowing.horizon,
        delta_encoding=transforms.delta_encoding,
        step_length=transforms.step_length,
        turning_angle=transforms.turning_angle,
        speed=transforms.speed,
        delta_t=transforms.delta_t,
        cyclical_time=transforms.cyclical_time,
    ).input_features


def _build_tcn(
    windowing: WindowingConfig,
    transforms: TransformsConfig,
    model: ModelConfig,
    data_spec: dict | None = None,
) -> ForecastModel:
    channels = list(model.channels)
    # Scale the number of residual blocks so the dilated receptive field covers
    # the (possibly per-dataset-scaled) input_len. With kernel_size=k and
    # dilations 1,2,4,... over n blocks the field is 1 + (k-1) * (2^n - 1);
    # extend the channel list with the last width until the field covers it.
    kernel = model.kernel_size
    blocks = len(channels)

    def rf(n: int) -> int:
        return 1 + (kernel - 1) * (2**n - 1)

    while rf(blocks) < windowing.input_len:
        channels.append(channels[-1])
        blocks += 1
    if blocks != len(model.channels):
        logger.info(
            "TCN channels extended %d -> %d blocks to cover input_len=%d (RF=%d)",
            len(model.channels), blocks, windowing.input_len, rf(blocks),
        )
    return TCN(
        input_len=windowing.input_len,
        horizon=windowing.horizon,
        n_features=_n_features(transforms, windowing),
        channels=channels,
        kernel_size=kernel,
        dropout=model.dropout,
        activation=model.activation,
        norm=model.norm,
        head=model.head,
    )


def _build_lstm(
    windowing: WindowingConfig,
    transforms: TransformsConfig,
    model: LSTMModelConfig,
    data_spec: dict | None = None,
) -> ForecastModel:
    return LSTMForecaster(
        input_len=windowing.input_len,
        horizon=windowing.horizon,
        n_features=_n_features(transforms, windowing),
        hidden_size=model.hidden_size,
        num_layers=model.num_layers,
        dropout=model.dropout,
        bidirectional=model.bidirectional,
        decoder=model.decoder,
        teacher_forcing_ratio=model.teacher_forcing_ratio,
        scheduled_sampling=model.scheduled_sampling,
        scheduled_sampling_epochs=model.scheduled_sampling_epochs,
        attention=model.attention,
    )


def _build_transformer(
    windowing: WindowingConfig,
    transforms: TransformsConfig,
    model: TransformerModelConfig,
    data_spec: dict | None = None,
) -> ForecastModel:
    return TransformerForecaster(
        input_len=windowing.input_len,
        horizon=windowing.horizon,
        n_features=_n_features(transforms, windowing),
        d_model=model.d_model,
        nhead=model.nhead,
        num_layers=model.num_layers,
        dim_feedforward=model.dim_feedforward,
        dropout=model.dropout,
        mode=model.mode,
        pos_encoding=model.pos_encoding,
        pooling=model.pooling,
        causal_mask=model.causal_mask,
        norm_first=model.norm_first,
    )


def _build_cov_transformer(
    windowing: WindowingConfig,
    transforms: TransformsConfig,
    model: CovariateTransformerModelConfig,
    data_spec: dict | None = None,
) -> ForecastModel:
    """Needs data-dependent widths (number of covariate columns, time features),
    which only the datamodule knows — hence ``data_spec``."""
    spec = data_spec or {}
    n_time = int(spec.get("n_time_features", 0))
    n_cov = int(spec.get("n_covariates", 0))
    if n_time <= 0:
        raise ValueError(
            "cov_transformer needs per-fix time context: set transforms.time_context=true "
            "and pass data_spec=datamodule.model_data_spec() to build_model()."
        )
    if model.use_covariates and n_cov <= 0:
        raise ValueError(
            "cov_transformer with model.use_covariates=true needs covariates: set "
            "covariates.enabled=true (and covariates.path), or use_covariates=false for "
            "the no-covariate ablation."
        )
    cls = FaunaFormer if model.name == "faunaformer" else CovariateTransformer
    return cls(
        input_len=windowing.input_len,
        horizon=windowing.horizon,
        n_features=_n_features(transforms, windowing),
        n_covariates=n_cov if model.use_covariates else 0,
        n_time_features=n_time,
        d_model=model.d_model,
        nhead=model.nhead,
        num_encoder_layers=model.num_encoder_layers,
        num_decoder_layers=model.num_decoder_layers,
        dim_feedforward=model.dim_feedforward,
        dropout=model.dropout,
        var_embed_dim=model.var_embed_dim,
        covariate_dropout=model.covariate_dropout,
        context_dropout=model.context_dropout,
        min_context=model.min_context,
        covariate_columns=list(spec.get("covariate_columns", [])) if model.use_covariates else [],
        target_scale=float(spec.get("target_scale", 1.0)),
        probabilistic=model.probabilistic,
        noise_dim=model.noise_dim,
        n_samples_train=model.n_samples_train,
        n_samples_eval=model.n_samples_eval,
        covariate_fusion=model.covariate_fusion,
        covariate_features=model.covariate_features,
        fusion_gate_init=model.fusion_gate_init,
        covariate_change_scale=spec.get("covariate_change_scale") if model.use_covariates else None,
        n_memory_step=int(spec.get("n_memory_step", 0)),
        n_memory_global=int(spec.get("n_memory_global", 0)),
        hex_rings=model.hex_rings,
        hex_edge_m=model.hex_edge_m,
    )


REGISTRY: dict[str, ModelBuilder] = {
    "tcn": _build_tcn,
    "lstm": _build_lstm,
    "transformer": _build_transformer,
    "cov_transformer": _build_cov_transformer,
    "faunaformer": _build_cov_transformer,
}


def build_model(
    model: ModelConfig,
    windowing: WindowingConfig,
    transforms: TransformsConfig,
    *,
    data_spec: dict | None = None,
) -> ForecastModel:
    """Instantiate a model by config name; unknown names fail loudly.

    ``data_spec`` (from :meth:`DataModule.model_data_spec`) carries data-dependent
    input widths; arms that do not need it ignore it.
    """
    builder = REGISTRY.get(model.name)
    if builder is None:
        raise ValueError(
            f"Unknown model name {model.name!r}. Registered: {sorted(REGISTRY)}"
        )
    logger.info("Building model '%s'", model.name)
    return builder(windowing, transforms, model, data_spec)
