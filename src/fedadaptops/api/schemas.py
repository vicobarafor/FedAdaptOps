from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class RunSummary(BaseModel):
    run_id: str
    run_type: str
    path: str
    status: str | None = None
    summary: dict[str, Any] | None = None


class RunDetail(BaseModel):
    run_id: str
    run_type: str
    path: str
    summary: dict[str, Any] | None = None
    run_metadata: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None


class SelectorRecommendRequest(BaseModel):
    personalization_results_path: str = Field(
        ...,
        description="Path to a Phase 3 personalization_results.csv file.",
    )
    selectors: list[str] = Field(default_factory=lambda: ["metadata", "resource_aware", "oracle"])
    clients_limit: int | None = None
    seed: int = 42

    target_accuracy_weight: float = 1.0
    cost_weight: float = 0.35
    latency_weight: float = 0.15
    memory_weight: float = 0.15
    bandwidth_weight: float = 0.05


class SelectorRecommendResponse(BaseModel):
    recommendations: list[dict[str, Any]]
    selector_summary: list[dict[str, Any]]
    oracle_headroom: list[dict[str, Any]]
    resource_profiles: list[dict[str, Any]]
