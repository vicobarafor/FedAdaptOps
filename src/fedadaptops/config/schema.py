from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProjectConfig:
    name: str
    run_name: str
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


@dataclass
class ModelConfig:
    name: str = "SimpleCNN"
    num_classes: int = 10


@dataclass
class TrainingConfig:
    device: str = "auto"
    epochs: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    log_every: int = 50
    max_train_batches: int | None = None
    max_eval_batches: int | None = None


@dataclass
class ArtifactConfig:
    save_checkpoint: bool = True
    save_environment: bool = True


@dataclass
class ExperimentConfig:
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    seed: int = 42
