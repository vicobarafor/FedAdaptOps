from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class SelectorDecision:
    client_id: int
    selector: str
    recommended_policy: str
    expected_accuracy: float
    expected_accuracy_delta: float
    expected_cost: float
    feasibility_score: float
    utility_score: float
    reason: str


class PolicySelector(Protocol):
    name: str

    def recommend(
        self,
        *,
        client_metrics: pd.DataFrame,
        resource_profiles: pd.DataFrame,
    ) -> list[SelectorDecision]: ...
