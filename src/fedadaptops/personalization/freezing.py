from __future__ import annotations

from torch import nn


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def freeze_all(model: nn.Module) -> None:
    set_trainable(model, False)


def unfreeze_all(model: nn.Module) -> None:
    set_trainable(model, True)


def train_classifier_only(model: nn.Module) -> None:
    freeze_all(model)
    if not hasattr(model, "classifier"):
        raise AttributeError("Model does not expose a 'classifier' module.")
    set_trainable(model.classifier, True)


def train_last_feature_block_and_classifier(model: nn.Module) -> None:
    freeze_all(model)
    if not hasattr(model, "features"):
        raise AttributeError("Model does not expose a 'features' module.")
    if not hasattr(model, "classifier"):
        raise AttributeError("Model does not expose a 'classifier' module.")
    feature_children = list(model.features.children())
    if not feature_children:
        raise ValueError("Model feature extractor has no children.")
    for child in feature_children[-4:]:
        set_trainable(child, True)
    set_trainable(model.classifier, True)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
