"""Append-only persistence for recommendation inputs and outcomes."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .contracts import AUDIT_SCHEMA_VERSION, RecommendationOutcome, RecommendationRequest


@dataclass(frozen=True)
class RecommendationAuditRecord:
    schema_version: str
    recorded_at: str
    request: Mapping[str, Any]
    outcome: Mapping[str, Any]


class JsonlAuditStore:
    """Durably append one complete recommendation interaction per JSON line."""

    def __init__(self, path: str | Path = "data/omegaclaw/recommendations.jsonl") -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(
        self,
        request: RecommendationRequest,
        outcome: RecommendationOutcome,
    ) -> RecommendationAuditRecord:
        record = RecommendationAuditRecord(
            schema_version=AUDIT_SCHEMA_VERSION,
            recorded_at=datetime.now(timezone.utc).isoformat(),
            request=request.to_dict(),
            outcome=outcome.to_dict(),
        )
        line = json.dumps(asdict(record), sort_keys=True, allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        return record
