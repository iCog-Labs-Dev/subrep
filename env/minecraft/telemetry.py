from __future__ import annotations

from numbers import Number
from typing import Any, Dict


class TelemetryError(ValueError):
    pass


def _position(value: Any) -> None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise TelemetryError("position must be a list or tuple of 3 integers")


def _number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Number):
        raise TelemetryError("{} must be a number".format(name))
    if value < 0 or value > 20:
        raise TelemetryError("{} must be between 0 and 20".format(name))


def _inventory(value: Any) -> None:
    if not isinstance(value, list):
        raise TelemetryError("inventory must be a list")
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise TelemetryError("inventory[{}] must be a dict".format(index))
        item_id = entry.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise TelemetryError(
                "inventory[{}].id must be a non-empty string".format(index)
            )
        count = entry.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise TelemetryError(
                "inventory[{}].count must be a non-negative integer".format(index)
            )


def _validate_payload(payload: Dict[str, Any]) -> None:
    for key in ("position", "health", "food", "inventory"):
        if key not in payload:
            raise TelemetryError("telemetry is missing {}".format(key))
    _position(payload["position"])
    _number(payload["health"], "health")
    _number(payload["food"], "food")
    _inventory(payload["inventory"])


def validate_observation(observation: Any) -> Dict[str, Any]:
    if not isinstance(observation, dict):
        raise TelemetryError("observation must be a dict")
    if observation.get("status") != "ok":
        raise TelemetryError("observation status must be 'ok'")
    _validate_payload(observation)
    return observation


def validate_action_result(result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise TelemetryError("action result must be a dict")
    status = result.get("status")
    if status not in ("ok", "error"):
        raise TelemetryError("action result status must be 'ok' or 'error'")
    if status == "error":
        reason = result.get("reason")
        if not isinstance(reason, str) or not reason:
            raise TelemetryError("action result error requires a non-empty reason")
    else:
        _validate_payload(result)
    return result
