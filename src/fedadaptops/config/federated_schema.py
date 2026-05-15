from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProjectConfig:
    name: str = "FedAdaptOps"
    run_name: str = "cifar10_phase2_fedavg"
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
class FederationConfig:
    num_rounds: int = 3
    clients_per_round: int = 5
    dropout_probability: float = 0.0
    aggregation: str = "sample_weighted"


@dataclass
class ClientTrainingConfig:
    local_epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_batches_per_client: int | None = 20


@dataclass
class EvaluationConfig:
    eval_every_round: int = 1
    max_eval_batches: int | None = 50


@dataclass
class RuntimeTrainingConfig:
    device: str = "auto"


@dataclass
class ArtifactConfig:
    save_checkpoint: bool = True
    save_environment: bool = True


@dataclass
class FederatedExperimentConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    federation: FederationConfig = field(default_factory=FederationConfig)
    client_training: ClientTrainingConfig = field(default_factory=ClientTrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    training: RuntimeTrainingConfig = field(default_factory=RuntimeTrainingConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    seed: int = 42
