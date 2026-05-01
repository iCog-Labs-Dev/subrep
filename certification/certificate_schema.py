"""
Certificate schema for SubRep admission records.

This module defines the validated data contract used before serializing
certificates into MeTTA-shaped expressions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any

_VALID_GATE_TYPES = {"CDS", "PDS"}

@dataclass(frozen=True)
class Certificate:
    """
    Validated certificate record for an admitted skill.

    The schema groups:
    - identity fields (`skill_id`, `gate_type`)
    - certification metrics (`delta_r`, `delta_n`, `admission_margin`, `epsilon`)
    - auditability fields (`timestamp`, `seed`, `gamma`, `baseline_id`)
    - metadata (`environment`, `episode_length`, `version`)
    """

    skill_id: str
    gate_type: str
    delta_r: float
    delta_n: tuple[float, float]
    admission_margin: float
    epsilon: float
    timestamp: str
    seed: int
    gamma: float
    baseline_id: str
    environment: str
    episode_length: int
    version: str

    def __post_init__(self) -> None:
        # Normalize gate labels once so downstream query logic is consistent.
        gate_type = self.gate_type.strip().upper()
        object.__setattr__(self, "gate_type", gate_type)

        # Required identity/context strings.
        self._validate_non_empty_string("skill_id", self.skill_id)
        self._validate_non_empty_string("baseline_id", self.baseline_id)
        self._validate_non_empty_string("environment", self.environment)
        self._validate_non_empty_string("version", self.version)

        if gate_type not in _VALID_GATE_TYPES:
            raise ValueError(f"gate_type must be one of {_VALID_GATE_TYPES}, got {self.gate_type!r}")

        # In this phase, motive vector is fixed to 2D: [Safety, Fuel].
        dn = tuple(float(v) for v in self.delta_n)
        if len(dn) != 2:
            raise ValueError(f"delta_n must have length 2, got {len(dn)}")
        if not all(isfinite(v) for v in dn):
            raise ValueError(f"delta_n must contain only finite values, got {dn}")
        object.__setattr__(self, "delta_n", dn)

        # Numeric invariants used by gate logic and reproducibility.
        self._validate_finite("delta_r", self.delta_r)
        self._validate_finite("admission_margin", self.admission_margin)
        self._validate_finite("epsilon", self.epsilon)
        self._validate_finite("gamma", self.gamma)

        if float(self.admission_margin) < 0.0:
            raise ValueError(f"admission_margin must be >= 0, got {self.admission_margin}")
        if float(self.epsilon) < 0.0:
            raise ValueError(f"epsilon must be >= 0, got {self.epsilon}")
        if not (0.0 <= float(self.gamma) <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")

        # Keep timestamps parseable for audit and replay tooling.
        self._validate_iso_timestamp(self.timestamp)

        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {type(self.seed).__name__}")
        if not isinstance(self.episode_length, int) or isinstance(self.episode_length, bool):
            raise ValueError(
                f"episode_length must be an int, got {type(self.episode_length).__name__}"
            )
        if self.episode_length <= 0:
            raise ValueError(f"episode_length must be > 0, got {self.episode_length}")

        # CDS has no epsilon budget in this phase.
        if gate_type == "CDS" and float(self.epsilon) != 0.0:
            raise ValueError("CDS certificates must have epsilon == 0.0")

    def to_dict(self) -> dict[str, Any]:
        """Convert certificate to a JSON/MeTTA serialization-ready dictionary."""
        return {
            "skill_id": self.skill_id,
            "gate_type": self.gate_type,
            "delta_r": float(self.delta_r),
            "delta_n": [float(v) for v in self.delta_n],
            "admission_margin": float(self.admission_margin),
            "epsilon": float(self.epsilon),
            "timestamp": self.timestamp,
            "seed": int(self.seed),
            "gamma": float(self.gamma),
            "baseline_id": self.baseline_id,
            "environment": self.environment,
            "episode_length": int(self.episode_length),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Certificate":
        """Create a certificate from dictionary input via constructor validation."""
        return cls(**data)

    @staticmethod
    def _validate_non_empty_string(field: str, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")

    @staticmethod
    def _validate_finite(field: str, value: float) -> None:
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError(f"{field} must be finite, got {value}")

    @staticmethod
    def _validate_iso_timestamp(value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("timestamp must be a non-empty ISO string")
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"timestamp must be ISO-8601, got {value!r}") from exc
