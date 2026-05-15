from __future__ import annotations

from pathlib import Path

import pandas as pd


class MetricsLogger:
    def __init__(self, run_dir: str | Path):
        self.path = Path(run_dir) / "metrics.csv"
        self.rows: list[dict] = []

    def log(self, row: dict) -> None:
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)
