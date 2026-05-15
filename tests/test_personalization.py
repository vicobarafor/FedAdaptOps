import torch

from fedadaptops.models.simple_cnn import SimpleCNN
from fedadaptops.personalization.freezing import (
    count_total_parameters,
    count_trainable_parameters,
    train_classifier_only,
    train_last_feature_block_and_classifier,
)
from fedadaptops.personalization.policies import build_policy


def test_head_only_policy_makes_classifier_trainable_only():
    model = SimpleCNN(num_classes=10)
    policy = build_policy("head_only")
    policy.apply(model)
    total = count_total_parameters(model)
    trainable = count_trainable_parameters(model)
    assert trainable < total
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert not any(parameter.requires_grad for parameter in model.features[0].parameters())


def test_partial_finetune_policy_unfreezes_more_than_head_only():
    head_model = SimpleCNN(num_classes=10)
    partial_model = SimpleCNN(num_classes=10)
    train_classifier_only(head_model)
    train_last_feature_block_and_classifier(partial_model)
    assert count_trainable_parameters(partial_model) > count_trainable_parameters(head_model)


def test_full_finetune_policy_unfreezes_all_parameters():
    model = SimpleCNN(num_classes=10)
    train_classifier_only(model)
    policy = build_policy("full_finetune")
    policy.apply(model)
    assert count_trainable_parameters(model) == count_total_parameters(model)


def test_unknown_policy_raises_error():
    try:
        build_policy("does_not_exist")
    except ValueError as exc:
        assert "Unknown personalization policy" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown policy.")


def test_optimizer_can_be_built_for_head_only_policy():
    model = SimpleCNN(num_classes=10)
    build_policy("head_only").apply(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    assert optimizer is not None
