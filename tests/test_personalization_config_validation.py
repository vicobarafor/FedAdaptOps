from pathlib import Path

from fedadaptops.config.personalization_validation import validate_personalization_config
from fedadaptops.utils.config import load_config


def test_cifar10_personalization_config_validates_successfully():
    raw_cfg = load_config(Path("configs/cifar10_personalization.yaml"))
    cfg = validate_personalization_config(raw_cfg)
    assert cfg.project.name == "FedAdaptOps"
    assert cfg.data.dataset == "CIFAR10"
    assert cfg.model.name == "SimpleCNN"
    assert "head_only" in cfg.personalization.policies
    assert "partial_finetune" in cfg.personalization.policies
    assert "full_finetune" in cfg.personalization.policies
    assert cfg.personalization.local_epochs > 0
