from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_federated_run_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    round_metrics_path = run_dir / "federated_round_metrics.csv"
    client_metrics_path = run_dir / "client_round_metrics.csv"

    if not round_metrics_path.exists():
        raise FileNotFoundError(f"Missing round metrics: {round_metrics_path}")
    if not client_metrics_path.exists():
        raise FileNotFoundError(f"Missing client metrics: {client_metrics_path}")

    round_df = pd.read_csv(round_metrics_path)
    client_df = pd.read_csv(client_metrics_path)

    best_eval = None
    if "eval_accuracy" in round_df.columns and round_df["eval_accuracy"].notna().any():
        best_eval = float(round_df["eval_accuracy"].max())

    total_dropped = 0
    if "status" in client_df.columns:
        total_dropped = int((client_df["status"] == "dropped").sum())

    final_round = int(round_df["round_id"].max()) if not round_df.empty else 0
    mean_round_seconds = (
        float(round_df["round_seconds"].mean()) if "round_seconds" in round_df else None
    )

    report = [
        "# FedAdaptOps Federated Run Report",
        "",
        "## Summary",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Completed rounds: `{final_round}`",
        f"- Best eval accuracy: `{best_eval}`",
        f"- Dropped client events: `{total_dropped}`",
        f"- Mean round seconds: `{mean_round_seconds}`",
        "",
        "## Artifact contract",
        "",
        "- `config.yaml`",
        "- `environment.json`",
        "- `run_metadata.json`",
        "- `client_partitions.json`",
        "- `partition_summary.csv`",
        "- `federated_round_metrics.csv`",
        "- `client_round_metrics.csv`",
        "- `selected_clients.json`",
        "- `summary.json`",
        "- `checkpoints/global_round_best.pt`",
        "",
        "## Notes",
        "",
        (
            "This report is generated from persisted run artifacts and is intended to support "
            "reproducible experiment review, dashboard ingestion, and API-backed run inspection."
        ),
        "",
    ]

    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "sample_run.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    return report_path
