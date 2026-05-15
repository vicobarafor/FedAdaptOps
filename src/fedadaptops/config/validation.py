from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from fedadaptops.config.schema import ExperimentConfig


def validate_config(cfg: DictConfig) -> ExperimentConfig:
    schema = OmegaConf.structured(ExperimentConfig)
    merged = OmegaConf.merge(schema, cfg)
    resolved = OmegaConf.to_object(merged)

    if resolved.data.num_clients <= 0:
        raise ValueError("data.num_clients must be positive.")
    if resolved.data.dirichlet_alpha <= 0:
        raise ValueError("data.dirichlet_alpha must be positive.")
    if resolved.training.epochs <= 0:
        raise ValueError("training.epochs must be positive.")
    if resolved.training.lr <= 0:
        raise ValueError("training.lr must be positive.")
    if resolved.data.batch_size <= 0:
        raise ValueError("data.batch_size must be positive.")

    return resolved
