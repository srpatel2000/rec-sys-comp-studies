"""Wall-clock timing for GPT4Rec pipeline components (CSV under trained_models/time_tracking/gpt4rec/)."""

from __future__ import annotations

import csv
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class GPT4RecRuntimeTracker:
    """Collects per-segment durations for one pipeline run; writes a single CSV."""

    def __init__(self, data_type: str, run_id: Optional[str] = None):
        self.data_type = data_type
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.rows: List[Dict[str, Any]] = []
        self._t0_wall = time.perf_counter()

    def log(
        self,
        component: str,
        duration_sec: float,
        epoch: Optional[int] = None,
        detail: str = "",
    ) -> None:
        self.rows.append(
            {
                "run_id": self.run_id,
                "data_type": self.data_type,
                "component": component,
                "duration_sec": round(float(duration_sec), 6),
                "epoch": "" if epoch is None else str(int(epoch)),
                "detail": detail or "",
            }
        )

    def save(self, trained_models_dir: Path) -> Path:
        """Write CSV to trained_models/time_tracking/gpt4rec/."""

        out_dir = Path(trained_models_dir) / "time_tracking" / "gpt4rec"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"gpt4rec_runtime_{self.data_type}_{self.run_id}.csv"
        total_wall = time.perf_counter() - self._t0_wall
        self.rows.append(
            {
                "run_id": self.run_id,
                "data_type": self.data_type,
                "component": "pipeline_end_to_end_wall",
                "duration_sec": round(total_wall, 6),
                "epoch": "",
                "detail": "perf_counter from tracker init to save()",
            }
        )
        fieldnames = ["run_id", "data_type", "component", "duration_sec", "epoch", "detail"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in self.rows:
                w.writerow(row)
        return out_path
