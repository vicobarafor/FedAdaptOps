from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from fedadaptops.config.personalization_schema import PersonalizationExperimentConfig
from fedadaptops.personalization.policies import SUPPORTED_POLICIES


def validate_personalization_config(cfg: DictConfig) -> PersonalizationExperimentConfig:
    schema = OmegaConf.structured(PersonalizationExperimentConfig)
    merged = OmegaConf.merge(schema, cfg)
    resolved = OmegaConf.to_object(merged)

    if resolved.data.dataset != "CIFAR10":
        raise ValueError("Only CIFAR10 is supported in Phase 3.")
    if resolved.model.name != "SimpleCNN":
        raise ValueError("Only SimpleCNN is supported in Phase 3.")
    if resolved.data.num_clients <= 0:
        raise ValueError("data.num_clients must be positive.")
    if not 0.0 < resolved.data.validation_fraction < 1.0:
        raise ValueError("data.validation_fraction must be in (0.0, 1.0).")
    if (
        resolved.personalization.clients_limit is not None
        and resolved.personalization.clients_limit <= 0
    ):
        raise ValueError("personalization.clients_limit must be positive or null.")
    if resolved.personalization.local_epochs <= 0:
        raise ValueError("personalization.local_epochs must be positive.")
    if resolved.personalization.lr <= 0:
        raise ValueError("personalization.lr must be positive.")

    unknown = [
        policy for policy in resolved.personalization.policies if policy not in SUPPORTED_POLICIES
    ]
    if unknown:
        raise ValueError(f"Unsupported personalization policies: {unknown}")

    return resolved
