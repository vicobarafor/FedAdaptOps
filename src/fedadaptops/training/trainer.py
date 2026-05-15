from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fedadaptops.evaluation.metrics import accuracy_from_logits


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for batch_idx, (x, y) in enumerate(tqdm(loader, desc="train", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits.detach(), y)
        total_batches += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "accuracy": total_acc / max(total_batches, 1),
        "batches": total_batches,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for batch_idx, (x, y) in enumerate(tqdm(loader, desc="eval", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, y)
        total_batches += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "accuracy": total_acc / max(total_batches, 1),
        "batches": total_batches,
    }
