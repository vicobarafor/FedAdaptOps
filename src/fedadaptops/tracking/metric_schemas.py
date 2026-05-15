from __future__ import annotations

FEDERATED_ROUND_METRIC_COLUMNS = [
    "round_id",
    "selected_clients",
    "active_clients",
    "dropped_clients",
    "total_active_samples",
    "mean_client_train_loss",
    "mean_client_train_accuracy",
    "eval_loss",
    "eval_accuracy",
    "round_seconds",
]

CLIENT_ROUND_METRIC_COLUMNS = [
    "round_id",
    "client_id",
    "num_samples",
    "train_loss",
    "train_accuracy",
    "local_epochs",
    "status",
]


def validate_columns(actual_columns, expected_columns: list[str]) -> None:
    missing = [col for col in expected_columns if col not in actual_columns]
    if missing:
        raise ValueError(f"Missing expected metric columns: {missing}")
