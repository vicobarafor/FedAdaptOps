from __future__ import annotations

import pandas as pd

from fedadaptops.selectors.base import SelectorDecision
from fedadaptops.selectors.scoring import attach_resource_feasibility


class MetadataSelector:
    name = "metadata"

    def recommend(
        self,
        *,
        client_metrics: pd.DataFrame,
        resource_profiles: pd.DataFrame,
    ) -> list[SelectorDecision]:
        frame = attach_resource_feasibility(client_metrics, resource_profiles)
        decisions = []

        for client_id, group in frame.groupby("client_id"):
            resource_tier = str(group["resource_tier"].iloc[0])

            if resource_tier == "high":
                preferred = "full_finetune"
            elif resource_tier == "medium":
                preferred = "partial_finetune"
            else:
                preferred = "head_only"

            candidates = group[group["policy"] == preferred]
            if candidates.empty:
                candidates = group.sort_values("post_accuracy", ascending=False).head(1)

            row = candidates.iloc[0]
            decisions.append(
                SelectorDecision(
                    client_id=int(client_id),
                    selector=self.name,
                    recommended_policy=str(row["policy"]),
                    expected_accuracy=float(row["post_accuracy"]),
                    expected_accuracy_delta=float(row["accuracy_delta"]),
                    expected_cost=float(row["compute_cost"]),
                    feasibility_score=float(row["feasibility_score"]),
                    utility_score=float(row["post_accuracy"] - row["compute_cost"]),
                    reason=f"resource_tier={resource_tier}",
                )
            )

        return decisions
