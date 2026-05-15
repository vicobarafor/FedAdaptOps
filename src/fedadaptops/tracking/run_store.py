from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def create_run_dir(output_dir: str | Path, run_name: str) -> tuple[str, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{run_name}_{uuid4().hex[:8]}"
    run_dir = Path(output_dir) / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def write_summary(run_dir: str | Path, summary: dict) -> None:
    path = Path(run_dir) / "summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
