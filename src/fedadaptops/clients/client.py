from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from fedadaptops.training.trainer import train_one_epoch


@dataclass(frozen=True)
class ClientTrainResult:
    client_id: int
    num_samples: int
    train_loss: float
    train_accuracy: float
    local_epochs: int
    state_dict: dict[str, torch.Tensor]


class FederatedClient:
    def __init__(self, client_id: int, train_loader: DataLoader):
        self.client_id = client_id
        self.train_loader = train_loader
        self.num_samples = len(train_loader.dataset)

    def train(
        self,
        *,
        model: nn.Module,
        global_state_dict: dict[str, torch.Tensor],
        device: torch.device,
        local_epochs: int,
        lr: float,
        weight_decay: float,
        max_batches_per_epoch: int | None = None,
    ) -> ClientTrainResult:
        model.load_state_dict(global_state_dict)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        epoch_metrics: list[dict[str, Any]] = []
        for _ in range(local_epochs):
            metrics = train_one_epoch(
                model=model,
                loader=self.train_loader,
                optimizer=optimizer,
                device=device,
                max_batches=max_batches_per_epoch,
            )
            epoch_metrics.append(metrics)
        avg_loss = sum(m["loss"] for m in epoch_metrics) / len(epoch_metrics)
        avg_acc = sum(m["accuracy"] for m in epoch_metrics) / len(epoch_metrics)
        local_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        return ClientTrainResult(
            client_id=self.client_id,
            num_samples=self.num_samples,
            train_loss=avg_loss,
            train_accuracy=avg_acc,
            local_epochs=local_epochs,
            state_dict=local_state,
        )
