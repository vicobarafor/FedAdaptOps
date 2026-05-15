from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from fedadaptops import __version__
from fedadaptops.api.run_registry import RunRegistry
from fedadaptops.api.schemas import (
    HealthResponse,
    RunDetail,
    RunSummary,
    SelectorRecommendRequest,
    SelectorRecommendResponse,
)
from fedadaptops.config.routing_schema import (
    ResourceSimulationConfig,
)
from fedadaptops.resources.simulation import simulate_resource_profiles
from fedadaptops.selectors.registry import build_selector

app = FastAPI(
    title="FedAdaptOps API",
    description="Production-style API for inspecting federated personalization experiments and routing recommendations.",
    version=__version__,
)


def registry() -> RunRegistry:
    return RunRegistry("runs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="fedadaptops-api", version=__version__)


@app.get("/runs", response_model=list[RunSummary])
def list_runs() -> list[RunSummary]:
    return [RunSummary(**row) for row in registry().list_runs()]


@app.get("/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str) -> RunDetail:
    try:
        return RunDetail(**registry().get_run(run_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/runs/{run_id}/metrics")
def get_metrics(
    run_id: str,
    filename: str | None = Query(default=None, description="Specific CSV metric file to read."),
):
    reg = registry()
    try:
        if filename is None:
            return {"run_id": run_id, "metric_files": reg.list_metric_files(run_id)}
        return {
            "run_id": run_id,
            "filename": filename,
            "records": reg.read_metric_file(run_id, filename),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/clients/{client_id}")
def get_client_metrics(client_id: int):
    records = []
    for run in registry().list_runs():
        run_id = run["run_id"]
        for filename in [
            "client_round_metrics.csv",
            "personalization_results.csv",
            "client_policy_metrics.csv",
            "selector_recommendations.csv",
            "client_resource_profiles.csv",
        ]:
            try:
                rows = registry().read_metric_file(run_id, filename)
            except FileNotFoundError:
                continue

            for row in rows:
                if int(row.get("client_id", -1)) == client_id:
                    row["_run_id"] = run_id
                    row["_source_file"] = filename
                    records.append(row)

    if not records:
        raise HTTPException(
            status_code=404, detail=f"No client metrics found for client_id={client_id}"
        )

    return {"client_id": client_id, "records": records}


@app.post("/selector/recommend", response_model=SelectorRecommendResponse)
def recommend_selector_policies(request: SelectorRecommendRequest) -> SelectorRecommendResponse:
    path = Path(request.personalization_results_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing personalization results: {path}")

    metrics = pd.read_csv(path)
    if request.clients_limit is not None:
        client_ids = sorted(metrics["client_id"].unique())[: request.clients_limit]
        metrics = metrics[metrics["client_id"].isin(client_ids)]

    client_ids = [int(client_id) for client_id in sorted(metrics["client_id"].unique())]

    resource_cfg = ResourceSimulationConfig()
    resources = simulate_resource_profiles(
        client_ids=client_ids, config=resource_cfg, seed=request.seed
    )

    all_decisions = []
    for selector_name in request.selectors:
        kwargs = {}
        if selector_name == "resource_aware":
            kwargs = {
                "accuracy_weight": request.target_accuracy_weight,
                "cost_weight": request.cost_weight,
                "latency_weight": request.latency_weight,
                "memory_weight": request.memory_weight,
                "bandwidth_weight": request.bandwidth_weight,
            }

        try:
            selector = build_selector(selector_name, **kwargs)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        decisions = selector.recommend(client_metrics=metrics, resource_profiles=resources)
        all_decisions.extend(decision.__dict__ for decision in decisions)

    decisions_df = pd.DataFrame(all_decisions)

    selector_summary = (
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
        .to_dict(orient="records")
    )

    oracle_headroom = []
    if "oracle" in decisions_df["selector"].unique():
        oracle = decisions_df[decisions_df["selector"] == "oracle"][
            ["client_id", "expected_accuracy"]
        ]
        oracle = oracle.rename(columns={"expected_accuracy": "oracle_accuracy"})
        for selector_name in request.selectors:
            if selector_name == "oracle":
                continue
            selector_rows = decisions_df[decisions_df["selector"] == selector_name]
            merged = selector_rows.merge(oracle, on="client_id", how="left")
            merged["oracle_headroom"] = merged["oracle_accuracy"] - merged["expected_accuracy"]
            oracle_headroom.append(
                {
                    "selector": selector_name,
                    "mean_oracle_headroom": float(merged["oracle_headroom"].mean()),
                    "max_oracle_headroom": float(merged["oracle_headroom"].max()),
                }
            )

    return SelectorRecommendResponse(
        recommendations=decisions_df.to_dict(orient="records"),
        selector_summary=selector_summary,
        oracle_headroom=oracle_headroom,
        resource_profiles=resources.to_dict(orient="records"),
    )
