from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from fedadaptops.config.routing_schema import RoutingExperimentConfig
from fedadaptops.selectors.registry import SUPPORTED_SELECTORS


def validate_routing_config(cfg: DictConfig) -> RoutingExperimentConfig:
    schema = OmegaConf.structured(RoutingExperimentConfig)
    merged = OmegaConf.merge(schema, cfg)
    resolved = OmegaConf.to_object(merged)

    unknown = [
        selector for selector in resolved.routing.selectors if selector not in SUPPORTED_SELECTORS
    ]
    if unknown:
        raise ValueError(f"Unsupported selectors: {unknown}")

    if resolved.routing.personalization_results_path is None:
        raise ValueError(
            "routing.personalization_results_path must point to a Phase 3 personalization_results.csv file."
        )

    if resolved.routing.clients_limit is not None and resolved.routing.clients_limit <= 0:
        raise ValueError("routing.clients_limit must be positive or null.")

    if resolved.resources.compute_budget_min <= 0:
        raise ValueError("resources.compute_budget_min must be positive.")
    if resolved.resources.compute_budget_min > resolved.resources.compute_budget_max:
        raise ValueError("compute budget min cannot exceed max.")
    if resolved.resources.memory_budget_mb_min > resolved.resources.memory_budget_mb_max:
        raise ValueError("memory budget min cannot exceed max.")
    if resolved.resources.latency_budget_ms_min > resolved.resources.latency_budget_ms_max:
        raise ValueError("latency budget min cannot exceed max.")
    if resolved.resources.bandwidth_budget_mb_min > resolved.resources.bandwidth_budget_mb_max:
        raise ValueError("bandwidth budget min cannot exceed max.")
    if resolved.resources.energy_budget_j_min > resolved.resources.energy_budget_j_max:
        raise ValueError("energy budget min cannot exceed max.")

    return resolved
