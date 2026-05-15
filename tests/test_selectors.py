import pandas as pd

from fedadaptops.selectors.metadata_selector import MetadataSelector
from fedadaptops.selectors.oracle_selector import OracleSelector
from fedadaptops.selectors.resource_aware_selector import ResourceAwareSelector


def _metrics():
    return pd.DataFrame(
        [
            {"client_id": 0, "policy": "head_only", "post_accuracy": 0.70, "accuracy_delta": 0.01},
            {
                "client_id": 0,
                "policy": "partial_finetune",
                "post_accuracy": 0.75,
                "accuracy_delta": 0.03,
            },
            {
                "client_id": 0,
                "policy": "full_finetune",
                "post_accuracy": 0.80,
                "accuracy_delta": 0.05,
            },
            {"client_id": 1, "policy": "head_only", "post_accuracy": 0.60, "accuracy_delta": 0.01},
            {
                "client_id": 1,
                "policy": "partial_finetune",
                "post_accuracy": 0.65,
                "accuracy_delta": 0.02,
            },
            {
                "client_id": 1,
                "policy": "full_finetune",
                "post_accuracy": 0.62,
                "accuracy_delta": 0.00,
            },
        ]
    )


def _resources():
    return pd.DataFrame(
        [
            {
                "client_id": 0,
                "compute_budget": 1.0,
                "memory_budget_mb": 512,
                "latency_budget_ms": 250,
                "bandwidth_budget_mb": 25,
                "energy_budget_j": 100,
                "resource_tier": "high",
            },
            {
                "client_id": 1,
                "compute_budget": 0.3,
                "memory_budget_mb": 96,
                "latency_budget_ms": 50,
                "bandwidth_budget_mb": 2,
                "energy_budget_j": 10,
                "resource_tier": "low",
            },
        ]
    )


def test_oracle_selector_picks_best_accuracy():
    decisions = OracleSelector().recommend(
        client_metrics=_metrics(), resource_profiles=_resources()
    )

    by_client = {decision.client_id: decision.recommended_policy for decision in decisions}

    assert by_client[0] == "full_finetune"
    assert by_client[1] == "partial_finetune"


def test_metadata_selector_uses_resource_tier():
    decisions = MetadataSelector().recommend(
        client_metrics=_metrics(), resource_profiles=_resources()
    )

    by_client = {decision.client_id: decision.recommended_policy for decision in decisions}

    assert by_client[0] == "full_finetune"
    assert by_client[1] == "head_only"


def test_resource_aware_selector_returns_one_decision_per_client():
    decisions = ResourceAwareSelector().recommend(
        client_metrics=_metrics(), resource_profiles=_resources()
    )

    assert len(decisions) == 2
    assert {decision.client_id for decision in decisions} == {0, 1}
