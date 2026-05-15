from pathlib import Path

from fedadaptops.config.federated_validation import validate_federated_config
from fedadaptops.utils.config import load_config


def test_cifar10_fedavg_config_validates_successfully():
    raw_cfg = load_config(Path("configs/cifar10_fedavg.yaml"))
    cfg = validate_federated_config(raw_cfg)

    assert cfg.project.name == "FedAdaptOps"
    assert cfg.data.dataset == "CIFAR10"
    assert cfg.model.name == "SimpleCNN"
    assert cfg.federation.num_rounds > 0
    assert cfg.federation.clients_per_round > 0
    assert cfg.federation.clients_per_round <= cfg.data.num_clients
    assert cfg.client_training.local_epochs > 0
