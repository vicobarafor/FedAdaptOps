from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from fedadaptops.config.routing_schema import RoutingExperimentConfig
from fedadaptops.resources.simulation import simulate_resource_profiles
from fedadaptops.selectors.registry import build_selector


class PolicyRecommendationEngine:
    def __init__(
        self,
        *,
        cfg: RoutingExperimentConfig,
        run_dir: str | Path,
    ):
        self.cfg = cfg
        self.run_dir = Path(run_dir)

    def run(self) -> dict:
        metrics = pd.read_csv(self.cfg.routing.personalization_results_path)
        if self.cfg.routing.clients_limit is not None:
            client_ids = sorted(metrics["client_id"].unique())[: self.cfg.routing.clients_limit]
            metrics = metrics[metrics["client_id"].isin(client_ids)]

        client_ids = sorted(metrics["client_id"].unique())
        resources = simulate_resource_profiles(
            client_ids=[int(client_id) for client_id in client_ids],
            config=self.cfg.resources,
            seed=self.cfg.seed,
        )

        resources.to_csv(self.run_dir / "client_resource_profiles.csv", index=False)

        all_decisions = []
        for selector_name in self.cfg.routing.selectors:
            kwargs = {}
            if selector_name == "resource_aware":
                kwargs = {
                    "accuracy_weight": self.cfg.routing.target_accuracy_weight,
                    "cost_weight": self.cfg.routing.cost_weight,
                    "latency_weight": self.cfg.routing.latency_weight,
                    "memory_weight": self.cfg.routing.memory_weight,
                    "bandwidth_weight": self.cfg.routing.bandwidth_weight,
                }

            selector = build_selector(selector_name, **kwargs)
            decisions = selector.recommend(client_metrics=metrics, resource_profiles=resources)
            all_decisions.extend(asdict(decision) for decision in decisions)

        decisions_df = pd.DataFrame(all_decisions)
        decisions_df.to_csv(self.run_dir / "selector_recommendations.csv", index=False)

        summary_df = (
            decisions_df.groupby("selector")
            .agg(
                mean_expected_accuracy=("expected_accuracy", "mean"),
                mean_expected_accuracy_delta=("expected_accuracy_delta", "mean"),
                mean_expected_cost=("expected_cost", "mean"),
                mean_feasibility_score=("feasibility_score", "mean"),
                mean_utility_score=("utility_score", "mean"),
                num_clients=("client_id", "nunique"),
            )
            .reset_index()
        )
        summary_df.to_csv(self.run_dir / "selector_summary.csv", index=False)

        oracle = decisions_df[decisions_df["selector"] == "oracle"][
            ["client_id", "expected_accuracy"]
        ]
        oracle = oracle.rename(columns={"expected_accuracy": "oracle_accuracy"})

        headroom_rows = []
        for selector_name in self.cfg.routing.selectors:
            if selector_name == "oracle":
                continue
            selector_rows = decisions_df[decisions_df["selector"] == selector_name]
            merged = selector_rows.merge(oracle, on="client_id", how="left")
            merged["oracle_headroom"] = merged["oracle_accuracy"] - merged["expected_accuracy"]
            headroom_rows.append(
                {
                    "selector": selector_name,
                    "mean_oracle_headroom": float(merged["oracle_headroom"].mean()),
                    "max_oracle_headroom": float(merged["oracle_headroom"].max()),
                }
            )

        headroom_df = pd.DataFrame(headroom_rows)
        headroom_df.to_csv(self.run_dir / "oracle_headroom.csv", index=False)

        return {
            "num_clients": len(client_ids),
            "selectors": self.cfg.routing.selectors,
            "recommendations_path": str(self.run_dir / "selector_recommendations.csv"),
            "selector_summary_path": str(self.run_dir / "selector_summary.csv"),
            "oracle_headroom_path": str(self.run_dir / "oracle_headroom.csv"),
            "resource_profiles_path": str(self.run_dir / "client_resource_profiles.csv"),
            "selector_summary": summary_df.to_dict(orient="records"),
            "oracle_headroom": headroom_df.to_dict(orient="records"),
        }
