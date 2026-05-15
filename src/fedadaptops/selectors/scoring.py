from __future__ import annotations

import pandas as pd

POLICY_COST_PRIORS = {
    "head_only": {
        "compute_cost": 0.25,
        "memory_cost_mb": 64.0,
        "latency_cost_ms": 35.0,
        "bandwidth_cost_mb": 1.0,
        "energy_cost_j": 8.0,
    },
    "partial_finetune": {
        "compute_cost": 0.60,
        "memory_cost_mb": 192.0,
        "latency_cost_ms": 90.0,
        "bandwidth_cost_mb": 3.0,
        "energy_cost_j": 30.0,
    },
    "full_finetune": {
        "compute_cost": 1.00,
        "memory_cost_mb": 384.0,
        "latency_cost_ms": 180.0,
        "bandwidth_cost_mb": 8.0,
        "energy_cost_j": 75.0,
    },
}


def attach_policy_costs(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.copy()
    for column in [
        "compute_cost",
        "memory_cost_mb",
        "latency_cost_ms",
        "bandwidth_cost_mb",
        "energy_cost_j",
    ]:
        frame[column] = frame["policy"].map(lambda policy: POLICY_COST_PRIORS[str(policy)][column])
    return frame


def attach_resource_feasibility(
    metrics: pd.DataFrame, resource_profiles: pd.DataFrame
) -> pd.DataFrame:
    frame = attach_policy_costs(metrics)
    frame = frame.merge(resource_profiles, on="client_id", how="left")

    frame["compute_feasible"] = frame["compute_cost"] <= frame["compute_budget"]
    frame["memory_feasible"] = frame["memory_cost_mb"] <= frame["memory_budget_mb"]
    frame["latency_feasible"] = frame["latency_cost_ms"] <= frame["latency_budget_ms"]
    frame["bandwidth_feasible"] = frame["bandwidth_cost_mb"] <= frame["bandwidth_budget_mb"]
    frame["energy_feasible"] = frame["energy_cost_j"] <= frame["energy_budget_j"]

    feasibility_cols = [
        "compute_feasible",
        "memory_feasible",
        "latency_feasible",
        "bandwidth_feasible",
        "energy_feasible",
    ]
    frame["feasibility_score"] = frame[feasibility_cols].mean(axis=1)

    return frame


def utility_score(
    row,
    *,
    accuracy_weight: float,
    cost_weight: float,
    latency_weight: float,
    memory_weight: float,
    bandwidth_weight: float,
) -> float:
    return (
        accuracy_weight * float(row["post_accuracy"])
        - cost_weight * float(row["compute_cost"])
        - latency_weight
        * (float(row["latency_cost_ms"]) / max(float(row["latency_budget_ms"]), 1.0))
        - memory_weight * (float(row["memory_cost_mb"]) / max(float(row["memory_budget_mb"]), 1.0))
        - bandwidth_weight
        * (float(row["bandwidth_cost_mb"]) / max(float(row["bandwidth_budget_mb"]), 1.0))
        + 0.25 * float(row["feasibility_score"])
    )
