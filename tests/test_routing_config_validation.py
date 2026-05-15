from pathlib import Path

from omegaconf import OmegaConf

from fedadaptops.config.routing_validation import validate_routing_config


def test_routing_config_validates_when_results_path_is_set():
    cfg = OmegaConf.load(Path("configs/cifar10_routing.yaml"))
    cfg.routing.personalization_results_path = "runs/example/personalization_results.csv"

    resolved = validate_routing_config(cfg)

    assert "metadata" in resolved.routing.selectors
    assert "resource_aware" in resolved.routing.selectors
    assert "oracle" in resolved.routing.selectors
