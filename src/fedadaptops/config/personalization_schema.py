from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProjectConfig:
    name: str = "FedAdaptOps"
    run_name: str = "cifar10_phase3_personalization"
    output_dir: str = "runs"


@dataclass
class DataConfig:
    dataset: str = "CIFAR10"
    data_dir: str = "data"
    num_clients: int = 20
    dirichlet_alpha: float = 0.3
    min_samples_per_client: int = 50
    batch_size: int = 64
    num_workers: int = 2
    validation_fraction: float = 0.2


@dataclass
class ModelConfig:
    name: str = "SimpleCNN"
    num_classes: int = 10


@dataclass
class PersonalizationConfig:
    policies: list[str] = field(
        default_factory=lambda: ["head_only", "partial_finetune", "full_finetune"]
    )
    clients_limit: int | None = 8
    local_epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_train_batches_per_client: int | None = 10
    max_eval_batches_per_client: int | None = 10
    checkpoint_path: str | None = None


@dataclass
class RuntimeTrainingConfig:
    device: str = "auto"


@dataclass
class ArtifactConfig:
    save_environment: bool = True


@dataclass
class PersonalizationExperimentConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    personalization: PersonalizationConfig = field(default_factory=PersonalizationConfig)
    training: RuntimeTrainingConfig = field(default_factory=RuntimeTrainingConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    seed: int = 42
