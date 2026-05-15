from __future__ import annotations

import pandas as pd

from fedadaptops.selectors.base import SelectorDecision
from fedadaptops.selectors.scoring import attach_resource_feasibility, utility_score


class ResourceAwareSelector:
    name = "resource_aware"

    def __init__(
        self,
        *,
        accuracy_weight: float = 1.0,
        cost_weight: float = 0.35,
        latency_weight: float = 0.15,
        memory_weight: float = 0.15,
        bandwidth_weight: float = 0.05,
    ):
        self.accuracy_weight = accuracy_weight
        self.cost_weight = cost_weight
        self.latency_weight = latency_weight
        self.memory_weight = memory_weight
        self.bandwidth_weight = bandwidth_weight

    def recommend(
        self,
        *,
        client_metrics: pd.DataFrame,
        resource_profiles: pd.DataFrame,
    ) -> list[SelectorDecision]:
        frame = attach_resource_feasibility(client_metrics, resource_profiles)
        frame["utility_score"] = frame.apply(
            utility_score,
            axis=1,
            accuracy_weight=self.accuracy_weight,
            cost_weight=self.cost_weight,
            latency_weight=self.latency_weight,
            memory_weight=self.memory_weight,
            bandwidth_weight=self.bandwidth_weight,
        )

        decisions = []
        for client_id, group in frame.groupby("client_id"):
            feasible = group[group["feasibility_score"] >= 1.0]
            candidates = feasible if not feasible.empty else group
            row = candidates.sort_values("utility_score", ascending=False).iloc[0]

            reason = (
                "best_feasible_utility"
                if not feasible.empty
                else "fallback_best_utility_no_fully_feasible_policy"
            )
            decisions.append(
                SelectorDecision(
                    client_id=int(client_id),
                    selector=self.name,
                    recommended_policy=str(row["policy"]),
                    expected_accuracy=float(row["post_accuracy"]),
                    expected_accuracy_delta=float(row["accuracy_delta"]),
                    expected_cost=float(row["compute_cost"]),
                    feasibility_score=float(row["feasibility_score"]),
                    utility_score=float(row["utility_score"]),
                    reason=reason,
                )
            )

        return decisions
