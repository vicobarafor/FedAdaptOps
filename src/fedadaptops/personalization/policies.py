from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import nn

from fedadaptops.personalization.freezing import (
    count_total_parameters,
    count_trainable_parameters,
    train_classifier_only,
    train_last_feature_block_and_classifier,
    unfreeze_all,
)


@dataclass(frozen=True)
class PersonalizationPolicy:
    name: str
    description: str
    apply: Callable[[nn.Module], None]
    relative_compute_cost: float


def _head_only(model: nn.Module) -> None:
    train_classifier_only(model)


def _partial_finetune(model: nn.Module) -> None:
    train_last_feature_block_and_classifier(model)


def _full_finetune(model: nn.Module) -> None:
    unfreeze_all(model)


_POLICY_REGISTRY = {
    "head_only": PersonalizationPolicy(
        name="head_only",
        description="Freeze the feature extractor and adapt the classifier head only.",
        apply=_head_only,
        relative_compute_cost=0.25,
    ),
    "partial_finetune": PersonalizationPolicy(
        name="partial_finetune",
        description="Freeze early layers and adapt the final feature block plus classifier.",
        apply=_partial_finetune,
        relative_compute_cost=0.60,
    ),
    "full_finetune": PersonalizationPolicy(
        name="full_finetune",
        description="Adapt all model parameters for maximum flexibility.",
        apply=_full_finetune,
        relative_compute_cost=1.00,
    ),
}
SUPPORTED_POLICIES = tuple(_POLICY_REGISTRY.keys())


def build_policy(name: str) -> PersonalizationPolicy:
    try:
        return _POLICY_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown personalization policy: {name}") from exc


def policy_metadata(model: nn.Module, policy: PersonalizationPolicy) -> dict:
    policy.apply(model)
    trainable = count_trainable_parameters(model)
    total = count_total_parameters(model)
    return {
        "policy": policy.name,
        "description": policy.description,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_parameter_fraction": trainable / total if total else 0.0,
        "relative_compute_cost": policy.relative_compute_cost,
    }
