from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def client_partition_frame(
    labels: list[int] | np.ndarray, partitions: dict[int, list[int]]
) -> pd.DataFrame:
    labels = np.asarray(labels)
    rows = []

    for client_id, indices in partitions.items():
        counts = Counter(labels[indices].tolist())
        row = {"client_id": client_id, "num_samples": len(indices)}
        for cls, count in sorted(counts.items()):
            row[f"class_{cls}"] = count
        rows.append(row)

    return pd.DataFrame(rows).fillna(0).sort_values("client_id")


def save_partition_reports(
    labels: list[int] | np.ndarray,
    partitions: dict[int, list[int]],
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = client_partition_frame(labels, partitions)
    csv_path = output_dir / "partition_summary.csv"
    json_path = output_dir / "partition_summary_records.json"

    frame.to_csv(csv_path, index=False)
    frame.to_json(json_path, orient="records", indent=2)

    return {
        "partition_summary_csv": str(csv_path),
        "partition_summary_json": str(json_path),
    }
