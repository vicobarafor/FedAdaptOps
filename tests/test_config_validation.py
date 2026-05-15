from pathlib import Path

from fedadaptops.config.validation import validate_config
from fedadaptops.utils.config import load_config


def test_cifar10_config_validates_successfully():
    cfg_path = Path("configs/cifar10_dirichlet.yaml")
    raw_cfg = load_config(cfg_path)
    cfg = validate_config(raw_cfg)

    assert cfg.project.name == "FedAdaptOps"
    assert cfg.data.dataset == "CIFAR10"
    assert cfg.model.name == "SimpleCNN"
    assert cfg.training.epochs > 0
    assert cfg.data.num_clients > 0
    assert cfg.data.dirichlet_alpha > 0
