from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from fedadaptops.personalization.personalize import personalize_client
from fedadaptops.personalization.policies import build_policy


class PersonalizationEvaluator:
    def __init__(
        self,
        *,
        model_factory,
        initial_state_dict: dict[str, torch.Tensor],
        client_loaders: dict[int, tuple],
        policies: list[str],
        device: torch.device,
        local_epochs: int,
        lr: float,
        weight_decay: float,
        max_train_batches: int | None,
        max_eval_batches: int | None,
        run_dir: str | Path,
    ):
        self.model_factory = model_factory
        self.initial_state_dict = {
            key: value.detach().cpu().clone() for key, value in initial_state_dict.items()
        }
        self.client_loaders = client_loaders
        self.policies = policies
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_train_batches = max_train_batches
        self.max_eval_batches = max_eval_batches
        self.run_dir = Path(run_dir)
        self.results: list[dict] = []

    def run(self) -> dict:
        for client_id, (train_loader, eval_loader) in self.client_loaders.items():
            for policy_name in self.policies:
                policy = build_policy(policy_name)
                model = self.model_factory()
                result = personalize_client(
                    client_id=client_id,
                    model=model,
                    initial_state_dict=self.initial_state_dict,
                    policy=policy,
                    train_loader=train_loader,
                    eval_loader=eval_loader,
                    device=self.device,
                    local_epochs=self.local_epochs,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    max_train_batches=self.max_train_batches,
                    max_eval_batches=self.max_eval_batches,
                )
                row = asdict(result)
                self.results.append(row)
                self._flush()
                print(row)
        frame = pd.DataFrame(self.results)
        return self._summarize(frame)

    def _flush(self) -> None:
        frame = pd.DataFrame(self.results)
        frame.to_csv(self.run_dir / "personalization_results.csv", index=False)
        frame.to_csv(self.run_dir / "client_policy_metrics.csv", index=False)

    @staticmethod
    def _summarize(frame: pd.DataFrame) -> dict:
        if frame.empty:
            return {
                "num_clients": 0,
                "num_policies": 0,
                "best_mean_policy": None,
                "best_mean_post_accuracy": None,
            }
        by_policy = (
            frame.groupby("policy")
            .agg(
                mean_pre_accuracy=("pre_accuracy", "mean"),
                mean_post_accuracy=("post_accuracy", "mean"),
                mean_accuracy_delta=("accuracy_delta", "mean"),
                mean_relative_compute_cost=("relative_compute_cost", "mean"),
                mean_trainable_parameter_fraction=("trainable_parameter_fraction", "mean"),
            )
            .reset_index()
        )
        best_row = by_policy.sort_values("mean_post_accuracy", ascending=False).iloc[0]
        return {
            "num_clients": int(frame["client_id"].nunique()),
            "num_policies": int(frame["policy"].nunique()),
            "best_mean_policy": str(best_row["policy"]),
            "best_mean_post_accuracy": float(best_row["mean_post_accuracy"]),
            "policy_summary": by_policy.to_dict(orient="records"),
        }
