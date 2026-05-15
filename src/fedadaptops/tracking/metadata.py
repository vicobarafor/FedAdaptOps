from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any


def build_run_metadata(
    *,
    run_id: str,
    config: Any,
    device: str,
    status: str = "running",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "status": status,
        "config": asdict(config) if is_dataclass(config) else config,
    }
