from pathlib import Path

import pandas as pd

from fedadaptops.evaluation.reporting import generate_federated_run_report


def test_generate_federated_run_report(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    pd.DataFrame(
        [
            {
                "round_id": 1,
                "selected_clients": 2,
                "active_clients": 2,
                "dropped_clients": 0,
                "total_active_samples": 10,
                "mean_client_train_loss": 1.0,
                "mean_client_train_accuracy": 0.5,
                "eval_loss": 1.2,
                "eval_accuracy": 0.4,
                "round_seconds": 3.0,
            }
        ]
    ).to_csv(run_dir / "federated_round_metrics.csv", index=False)

    pd.DataFrame(
        [
            {
                "round_id": 1,
                "client_id": 0,
                "num_samples": 5,
                "train_loss": 1.0,
                "train_accuracy": 0.5,
                "local_epochs": 1,
                "status": "completed",
            }
        ]
    ).to_csv(run_dir / "client_round_metrics.csv", index=False)

    report_path = generate_federated_run_report(run_dir)

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "FedAdaptOps Federated Run Report" in content
    assert "Best eval accuracy" in content
