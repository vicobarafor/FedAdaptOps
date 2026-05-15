from __future__ import annotations

import random

import pandas as pd

from fedadaptops.config.routing_schema import ResourceSimulationConfig
from fedadaptops.resources.profile import ClientResourceProfile


def simulate_resource_profiles(
    *,
    client_ids: list[int],
    config: ResourceSimulationConfig,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    profiles = []

    for client_id in client_ids:
        profile = ClientResourceProfile(
            client_id=client_id,
            compute_budget=rng.uniform(config.compute_budget_min, config.compute_budget_max),
            memory_budget_mb=rng.uniform(config.memory_budget_mb_min, config.memory_budget_mb_max),
            latency_budget_ms=rng.uniform(
                config.latency_budget_ms_min, config.latency_budget_ms_max
            ),
            bandwidth_budget_mb=rng.uniform(
                config.bandwidth_budget_mb_min, config.bandwidth_budget_mb_max
            ),
            energy_budget_j=rng.uniform(config.energy_budget_j_min, config.energy_budget_j_max),
        )
        profiles.append(
            {
                "client_id": profile.client_id,
                "compute_budget": profile.compute_budget,
                "memory_budget_mb": profile.memory_budget_mb,
                "latency_budget_ms": profile.latency_budget_ms,
                "bandwidth_budget_mb": profile.bandwidth_budget_mb,
                "energy_budget_j": profile.energy_budget_j,
                "resource_tier": profile.resource_tier,
            }
        )

    return pd.DataFrame(profiles).sort_values("client_id").reset_index(drop=True)
