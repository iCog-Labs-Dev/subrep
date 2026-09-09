"""
Generic environment contract and metadata validation for SubRep.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple, runtime_checkable
import numpy as np

REQUIRED_METADATA_KEYS = {
    "environment_id": str,
    "motive_names": list,
    "motive_schema_version": str,
    "payoff_schema_version": str,
    "observation_schema_version": str,
    "action_schema_version": str,
}


def validate_env_metadata(metadata: Dict[str, Any]) -> None:
    """Validate that environment metadata conforms to the SubRep contract.

    Args:
        metadata: Dictionary containing environment metadata.

    Raises:
        ValueError: If any required key is missing or has an invalid type/value.
    """
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata must be a dict, got {type(metadata).__name__}")

    for key, expected_type in REQUIRED_METADATA_KEYS.items():
        if key not in metadata:
            raise ValueError(f"Missing required metadata key: '{key}'")
        val = metadata[key]
        if not isinstance(val, expected_type):
            raise ValueError(
                f"Metadata key '{key}' must be of type {expected_type.__name__}, got {type(val).__name__}"
            )

    motive_names = metadata["motive_names"]
    if len(motive_names) == 0:
        raise ValueError("metadata['motive_names'] must be non-empty")
    for name in motive_names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("All entries in metadata['motive_names'] must be non-empty strings")


@runtime_checkable
class SubRepBaseEnv(Protocol):
    """Protocol defining the standard interface for all SubRep environments."""

    @property
    def metadata(self) -> Dict[str, Any]:
        """Environment metadata dictionary conforming to SubRep specification."""
        ...

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment to initial state.

        Returns:
            Tuple of (initial_observation, info_dict).
        """
        ...

    def step(self, action: Any) -> Tuple[Any, np.ndarray, bool, bool, Dict[str, Any]]:
        """Advance the environment by one action step.

        Returns:
            Tuple of (observation, motive_vector, terminated, truncated, info).
            The info dict MUST contain 'task_payoff' as a scalar float.
        """
        ...

    def close(self) -> None:
        """Close and clean up environment resources."""
        ...
