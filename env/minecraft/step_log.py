from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from village_sim.state import VillageState


def _json_safe(value: Any) -> Any:
    if isinstance(value, frozenset):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if is_dataclass(value):
        return _json_safe(asdict(value))
    return value


class StepLogger:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        root = Path(__file__).resolve().parents[2]
        self._path = root / "minecraft" / "logs" / "{}.jsonl".format(run_id)

    def log_step(
        self,
        run_id: str,
        snapshot_id: str,
        step: int,
        tick_before: int,
        tick_after: int,
        action: str,
        accepted: bool,
        refusal: Optional[str],
        action_ticks: int,
        finished: bool,
        state_before: VillageState,
        state_after: VillageState,
        task_payoff: float,
        events: List[str],
    ) -> None:
        before = _json_safe(asdict(state_before))
        after = _json_safe(asdict(state_after))
        record: Dict[str, Any] = {
            "run_id": run_id,
            "snapshot_id": snapshot_id,
            "step": step,
            "tick_before": tick_before,
            "tick_after": tick_after,
            "action": action,
            "accepted": accepted,
            "refusal": refusal,
            "action_ticks": action_ticks,
            "finished": finished,
            "state_before": before,
            "state_after": after,
            "task_payoff": task_payoff,
            "events": events,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
