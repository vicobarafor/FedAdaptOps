from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClientResourceProfile:
    client_id: int
    compute_budget: float
    memory_budget_mb: float
    latency_budget_ms: float
    bandwidth_budget_mb: float
    energy_budget_j: float

    @property
    def resource_tier(self) -> str:
        if self.compute_budget >= 0.75 and self.memory_budget_mb >= 384:
            return "high"
        if self.compute_budget >= 0.45 and self.memory_budget_mb >= 192:
            return "medium"
        return "low"
