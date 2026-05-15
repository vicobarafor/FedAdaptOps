from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fedadaptops.dashboard.data import detect_run_type, list_run_dirs, read_json_if_exists


class RunRegistry:
    def __init__(self, runs_dir: str | Path = "runs"):
        self.runs_dir = Path(runs_dir)

    def list_runs(self) -> list[dict[str, Any]]:
        rows = []
        for run_dir in list_run_dirs(self.runs_dir):
            summary = read_json_if_exists(run_dir, "summary.json") or {}
            rows.append(
                {
                    "run_id": run_dir.name,
                    "run_type": detect_run_type(run_dir),
                    "path": str(run_dir),
                    "status": summary.get("status"),
                    "summary": summary,
                }
            )
        return rows

    def get_run_dir(self, run_id: str) -> Path:
        run_dir = self.runs_dir / run_id
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"Run not found: {run_id}")
        return run_dir

    def get_run(self, run_id: str) -> dict[str, Any]:
        run_dir = self.get_run_dir(run_id)
        return {
            "run_id": run_id,
            "run_type": detect_run_type(run_dir),
            "path": str(run_dir),
            "summary": read_json_if_exists(run_dir, "summary.json"),
            "run_metadata": read_json_if_exists(run_dir, "run_metadata.json"),
            "environment": read_json_if_exists(run_dir, "environment.json"),
        }

    def read_metric_file(self, run_id: str, filename: str) -> list[dict[str, Any]]:
        run_dir = self.get_run_dir(run_id)
        path = run_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Metric file not found for run {run_id}: {filename}")
        return pd.read_csv(path).to_dict(orient="records")

    def list_metric_files(self, run_id: str) -> list[str]:
        run_dir = self.get_run_dir(run_id)
        return sorted(path.name for path in run_dir.glob("*.csv"))
