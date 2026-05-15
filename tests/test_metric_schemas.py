from fedadaptops.tracking.metric_schemas import (
    CLIENT_ROUND_METRIC_COLUMNS,
    FEDERATED_ROUND_METRIC_COLUMNS,
    validate_columns,
)


def test_round_metric_schema_contains_dashboard_ready_columns():
    assert "round_id" in FEDERATED_ROUND_METRIC_COLUMNS
    assert "eval_accuracy" in FEDERATED_ROUND_METRIC_COLUMNS
    assert "active_clients" in FEDERATED_ROUND_METRIC_COLUMNS
    assert "dropped_clients" in FEDERATED_ROUND_METRIC_COLUMNS


def test_client_metric_schema_contains_client_observability_columns():
    assert "client_id" in CLIENT_ROUND_METRIC_COLUMNS
    assert "train_accuracy" in CLIENT_ROUND_METRIC_COLUMNS
    assert "status" in CLIENT_ROUND_METRIC_COLUMNS


def test_validate_columns_accepts_complete_schema():
    validate_columns(
        actual_columns=["round_id", "eval_accuracy", "extra"],
        expected_columns=["round_id", "eval_accuracy"],
    )


def test_validate_columns_rejects_missing_schema():
    try:
        validate_columns(
            actual_columns=["round_id"], expected_columns=["round_id", "eval_accuracy"]
        )
    except ValueError as exc:
        assert "eval_accuracy" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing metric column.")
