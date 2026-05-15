from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class CheckpointRecord:
    path: str
    epoch: int
    metric_name: str
    metric_value: float


class CheckpointManager:
    def __init__(
        self, checkpoint_dir: str | Path, monitor: str = "eval_accuracy", mode: str = "max"
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_record: CheckpointRecord | None = None

    def maybe_save_best(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> CheckpointRecord | None:
        value = float(metrics[self.monitor])
        improved = value > self.best_value if self.mode == "max" else value < self.best_value

        if not improved:
            return None

        self.best_value = value
        path = self.checkpoint_dir / "best.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "extra": extra or {},
            },
            path,
        )
        self.best_record = CheckpointRecord(
            path=str(path),
            epoch=epoch,
            metric_name=self.monitor,
            metric_value=value,
        )
        return self.best_record
