from __future__ import annotations

from pathlib import Path

import pandas as pd


def list_run_dirs(runs_dir: str | Path = "runs") -> list[Path]:
    root = Path(runs_dir)
    if not root.exists():
        return []
    return sorted(
        [path for path in root.iterdir() if path.is_dir()], key=lambda p: p.name, reverse=True
    )


def detect_run_type(run_dir: str | Path) -> str:
    run_dir = Path(run_dir)
    if (run_dir / "selector_recommendations.csv").exists():
        return "adaptive_routing"
    if (run_dir / "personalization_results.csv").exists():
        return "personalization"
    if (run_dir / "federated_round_metrics.csv").exists():
        return "fedavg"
    if (run_dir / "metrics.csv").exists():
        return "centralized_baseline"
    return "unknown"


def read_csv_if_exists(run_dir: str | Path, filename: str) -> pd.DataFrame | None:
    path = Path(run_dir) / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


def read_json_if_exists(run_dir: str | Path, filename: str) -> dict | None:
    import json

    path = Path(run_dir) / filename
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_run_by_type(run_type: str, runs_dir: str | Path = "runs") -> Path | None:
    for run_dir in list_run_dirs(runs_dir):
        if detect_run_type(run_dir) == run_type:
            return run_dir
    return None
