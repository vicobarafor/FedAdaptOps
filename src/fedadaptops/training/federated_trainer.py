from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from fedadaptops.clients.client import FederatedClient
from fedadaptops.training.aggregation import sample_weighted_average
from fedadaptops.training.fedavg import ClientSampler
from fedadaptops.training.trainer import evaluate


class FederatedTrainer:
    def __init__(
        self,
        *,
        global_model: nn.Module,
        client_model_factory,
        clients: list[FederatedClient],
        test_loader,
        device: torch.device,
        num_rounds: int,
        clients_per_round: int,
        dropout_probability: float,
        local_epochs: int,
        lr: float,
        weight_decay: float,
        max_batches_per_client: int | None,
        eval_every_round: int,
        max_eval_batches: int | None,
        seed: int,
        run_dir: str | Path,
        checkpoint_enabled: bool = True,
    ):
        self.global_model = global_model.to(device)
        self.client_model_factory = client_model_factory
        self.clients = {client.client_id: client for client in clients}
        self.test_loader = test_loader
        self.device = device
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_batches_per_client = max_batches_per_client
        self.eval_every_round = eval_every_round
        self.max_eval_batches = max_eval_batches
        self.run_dir = Path(run_dir)
        self.checkpoint_enabled = checkpoint_enabled
        self.best_eval_accuracy = float("-inf")
        self.round_metrics: list[dict] = []
        self.client_metrics: list[dict] = []
        self.selections: list[dict] = []
        self.sampler = ClientSampler(
            num_clients=len(clients),
            clients_per_round=clients_per_round,
            dropout_probability=dropout_probability,
            seed=seed,
        )

    def fit(self) -> dict:
        for round_id in range(1, self.num_rounds + 1):
            round_start = time.time()
            selection = self.sampler.sample(round_id)
            self.selections.append(asdict(selection))
            global_state = {
                key: value.detach().cpu().clone()
                for key, value in self.global_model.state_dict().items()
            }
            client_results = []
            for client_id in selection.active_client_ids:
                client = self.clients[client_id]
                local_model = self.client_model_factory()
                result = client.train(
                    model=local_model,
                    global_state_dict=global_state,
                    device=self.device,
                    local_epochs=self.local_epochs,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    max_batches_per_epoch=self.max_batches_per_client,
                )
                client_results.append(result)
                self.client_metrics.append(
                    {
                        "round_id": round_id,
                        "client_id": client_id,
                        "num_samples": result.num_samples,
                        "train_loss": result.train_loss,
                        "train_accuracy": result.train_accuracy,
                        "local_epochs": result.local_epochs,
                        "status": "completed",
                    }
                )
            for client_id in selection.dropped_client_ids:
                self.client_metrics.append(
                    {
                        "round_id": round_id,
                        "client_id": client_id,
                        "num_samples": self.clients[client_id].num_samples,
                        "train_loss": None,
                        "train_accuracy": None,
                        "local_epochs": 0,
                        "status": "dropped",
                    }
                )
            aggregated = sample_weighted_average(client_results)
            self.global_model.load_state_dict(aggregated)
            eval_metrics = {"loss": None, "accuracy": None, "batches": None}
            if round_id % self.eval_every_round == 0:
                eval_metrics = evaluate(
                    self.global_model, self.test_loader, self.device, self.max_eval_batches
                )
            round_row = {
                "round_id": round_id,
                "selected_clients": len(selection.selected_client_ids),
                "active_clients": len(selection.active_client_ids),
                "dropped_clients": len(selection.dropped_client_ids),
                "total_active_samples": sum(r.num_samples for r in client_results),
                "mean_client_train_loss": sum(r.train_loss for r in client_results)
                / len(client_results),
                "mean_client_train_accuracy": sum(r.train_accuracy for r in client_results)
                / len(client_results),
                "eval_loss": eval_metrics["loss"],
                "eval_accuracy": eval_metrics["accuracy"],
                "round_seconds": time.time() - round_start,
            }
            self.round_metrics.append(round_row)
            if (
                eval_metrics["accuracy"] is not None
                and eval_metrics["accuracy"] > self.best_eval_accuracy
            ):
                self.best_eval_accuracy = eval_metrics["accuracy"]
                if self.checkpoint_enabled:
                    checkpoint_path = self.run_dir / "checkpoints" / "global_round_best.pt"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "round_id": round_id,
                            "model_state_dict": self.global_model.state_dict(),
                            "eval_metrics": eval_metrics,
                        },
                        checkpoint_path,
                    )
            self._flush_metrics()
            print(round_row)
        return {
            "num_rounds": self.num_rounds,
            "best_eval_accuracy": (
                None if self.best_eval_accuracy == float("-inf") else self.best_eval_accuracy
            ),
            "round_metrics_path": str(self.run_dir / "federated_round_metrics.csv"),
            "client_metrics_path": str(self.run_dir / "client_round_metrics.csv"),
        }

    def _flush_metrics(self) -> None:
        pd.DataFrame(self.round_metrics).to_csv(
            self.run_dir / "federated_round_metrics.csv", index=False
        )
        pd.DataFrame(self.client_metrics).to_csv(
            self.run_dir / "client_round_metrics.csv", index=False
        )
