from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from fedadaptops.personalization.policies import PersonalizationPolicy
from fedadaptops.training.trainer import evaluate, train_one_epoch


@dataclass(frozen=True)
class PersonalizationResult:
    client_id: int
    policy: str
    num_train_samples: int
    num_eval_samples: int
    pre_accuracy: float
    post_accuracy: float
    accuracy_delta: float
    post_loss: float
    train_loss: float
    train_accuracy: float
    trainable_parameters: int
    total_parameters: int
    trainable_parameter_fraction: float
    relative_compute_cost: float


def personalize_client(
    *,
    client_id: int,
    model: nn.Module,
    initial_state_dict: dict[str, torch.Tensor],
    policy: PersonalizationPolicy,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    local_epochs: int,
    lr: float,
    weight_decay: float,
    max_train_batches: int | None,
    max_eval_batches: int | None,
) -> PersonalizationResult:
    model.load_state_dict(initial_state_dict)
    model.to(device)
    pre_metrics = evaluate(model, eval_loader, device, max_batches=max_eval_batches)
    policy.apply(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    epoch_metrics = []
    for _ in range(local_epochs):
        metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches,
        )
        epoch_metrics.append(metrics)
    post_metrics = evaluate(model, eval_loader, device, max_batches=max_eval_batches)
    train_loss = sum(m["loss"] for m in epoch_metrics) / len(epoch_metrics)
    train_acc = sum(m["accuracy"] for m in epoch_metrics) / len(epoch_metrics)
    return PersonalizationResult(
        client_id=client_id,
        policy=policy.name,
        num_train_samples=len(train_loader.dataset),
        num_eval_samples=len(eval_loader.dataset),
        pre_accuracy=float(pre_metrics["accuracy"]),
        post_accuracy=float(post_metrics["accuracy"]),
        accuracy_delta=float(post_metrics["accuracy"] - pre_metrics["accuracy"]),
        post_loss=float(post_metrics["loss"]),
        train_loss=float(train_loss),
        train_accuracy=float(train_acc),
        trainable_parameters=int(trainable_params),
        total_parameters=int(total_params),
        trainable_parameter_fraction=float(
            trainable_params / total_params if total_params else 0.0
        ),
        relative_compute_cost=float(policy.relative_compute_cost),
    )
