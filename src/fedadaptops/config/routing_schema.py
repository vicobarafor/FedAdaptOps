from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProjectConfig:
    name: str = "FedAdaptOps"
    run_name: str = "cifar10_phase4_adaptive_routing"
    output_dir: str = "runs"


@dataclass
class RoutingConfig:
    personalization_results_path: str | None = None
    selectors: list[str] = field(default_factory=lambda: ["metadata", "resource_aware", "oracle"])
    target_accuracy_weight: float = 1.0
    cost_weight: float = 0.35
    latency_weight: float = 0.15
    memory_weight: float = 0.15
    bandwidth_weight: float = 0.05
    clients_limit: int | None = None


@dataclass
class ResourceSimulationConfig:
    compute_budget_min: float = 0.20
    compute_budget_max: float = 1.00
    memory_budget_mb_min: int = 64
    memory_budget_mb_max: int = 512
    latency_budget_ms_min: int = 30
    latency_budget_ms_max: int = 250
    bandwidth_budget_mb_min: int = 1
    bandwidth_budget_mb_max: int = 25
    energy_budget_j_min: int = 5
    energy_budget_j_max: int = 100


@dataclass
class ArtifactConfig:
    save_environment: bool = True


@dataclass
class RoutingExperimentConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    resources: ResourceSimulationConfig = field(default_factory=ResourceSimulationConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    seed: int = 42
