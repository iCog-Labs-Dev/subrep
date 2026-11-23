
"""Loading and validating the frozen candidate-ranking configuration."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def validate_motive_weights(weights):
    """Return validated (Safety, Fuel) simplex weights."""
    values = np.asarray(weights, dtype=np.float64).reshape(-1)

    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError(
            "motive weights must contain finite Safety and Fuel values"
        )

    if np.any(values < 0.0) or not np.isclose(values.sum(), 1.0):
        raise ValueError(
            "motive weights must be non-negative and sum to 1"
        )

    return float(values[0]), float(values[1])

def load_motive_weights(path,expected_split_signature=None,):
    """Load frozen validation-tuned weights or use documented defaults.

    Tuning is optional for running the filter. If no configuration exists,
    equal Safety/Fuel weights are used. If a configuration exists, it must
    be valid and match the expected validation split.
    """
    config_path = Path(path)

    if not config_path.is_file():
        return 0.5, 0.5

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read ranking configuration {config_path}: {exc}"
        ) from exc

    weights = config.get("motive_weights")

    if not isinstance(weights, dict) or not {
        "safety", "fuel"
    } <= set(weights):
        raise ValueError(
            "Ranking configuration must contain safety and fuel weights"
        )

    if not config.get("split_signature"):
        raise ValueError(
            "Ranking configuration is missing its validation split signature"
        )

    if (
        expected_split_signature is not None
        and config["split_signature"] != expected_split_signature
    ):
        raise ValueError(
            "Ranking configuration was tuned on a different split; "
            "rerun motive-weight tuning"
        )

    return validate_motive_weights([
        weights["safety"],
        weights["fuel"],
    ])