from __future__ import annotations

from torch import nn

from fedadaptops.models.simple_cnn import SimpleCNN


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "SimpleCNN":
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")
