from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from fedadaptops.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_selector_recommend_endpoint(tmp_path: Path):
    personalization_path = tmp_path / "personalization_results.csv"
    pd.DataFrame(
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
    ).to_csv(personalization_path, index=False)

    response = client.post(
        "/selector/recommend",
        json={
            "personalization_results_path": str(personalization_path),
            "selectors": ["metadata", "resource_aware", "oracle"],
            "seed": 123,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "recommendations" in payload
    assert "selector_summary" in payload
    assert "oracle_headroom" in payload
    assert "resource_profiles" in payload
    assert len(payload["recommendations"]) == 6
