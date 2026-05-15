from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from fedadaptops.config.federated_schema import FederatedExperimentConfig


def validate_federated_config(cfg: DictConfig) -> FederatedExperimentConfig:
    schema = OmegaConf.structured(FederatedExperimentConfig)
    merged = OmegaConf.merge(schema, cfg)
    resolved = OmegaConf.to_object(merged)

    if resolved.data.dataset != "CIFAR10":
        raise ValueError("Only CIFAR10 is supported in Phase 2.")
    if resolved.model.name != "SimpleCNN":
        raise ValueError("Only SimpleCNN is supported in Phase 2.")
    if resolved.data.num_clients <= 0:
        raise ValueError("data.num_clients must be positive.")
    if resolved.data.dirichlet_alpha <= 0:
        raise ValueError("data.dirichlet_alpha must be positive.")
    if resolved.federation.num_rounds <= 0:
        raise ValueError("federation.num_rounds must be positive.")
    if resolved.federation.clients_per_round <= 0:
        raise ValueError("federation.clients_per_round must be positive.")
    if resolved.federation.clients_per_round > resolved.data.num_clients:
        raise ValueError("federation.clients_per_round cannot exceed data.num_clients.")
    if not 0.0 <= resolved.federation.dropout_probability < 1.0:
        raise ValueError("federation.dropout_probability must be in [0.0, 1.0).")
    if resolved.client_training.local_epochs <= 0:
        raise ValueError("client_training.local_epochs must be positive.")
    if resolved.client_training.lr <= 0:
        raise ValueError("client_training.lr must be positive.")
    if resolved.evaluation.eval_every_round <= 0:
        raise ValueError("evaluation.eval_every_round must be positive.")
    if resolved.federation.aggregation != "sample_weighted":
        raise ValueError("Only sample_weighted aggregation is supported in Phase 2.")

    return resolved
