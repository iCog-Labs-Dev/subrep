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
_VALID_WEIGHT_REGION_TYPES = {"FULL_SIMPLEX", "MDN_WX"}
_MDN_AUDIT_FIELDS = (
    "certification_context",
    "mdn_alpha",
    "wx_support_directions",
    "wx_support_values",
)

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
    weight_region_type: str = "FULL_SIMPLEX"
    certification_context: tuple[float, ...] | None = None
    mdn_alpha: tuple[float, ...] | None = None
    wx_support_directions: tuple[tuple[float, ...], ...] | None = None
    wx_support_values: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        # Normalize gate labels once so downstream query logic is consistent.
        gate_type = self.gate_type.strip().upper()
        object.__setattr__(self, "gate_type", gate_type)
        weight_region_type = self.weight_region_type.strip().upper()
        object.__setattr__(self, "weight_region_type", weight_region_type)

        # Required identity/context strings.
        self._validate_non_empty_string("skill_id", self.skill_id)
        self._validate_non_empty_string("baseline_id", self.baseline_id)
        self._validate_non_empty_string("environment", self.environment)
        self._validate_non_empty_string("version", self.version)

        if gate_type not in _VALID_GATE_TYPES:
            raise ValueError(f"gate_type must be one of {_VALID_GATE_TYPES}, got {self.gate_type!r}")
        if weight_region_type not in _VALID_WEIGHT_REGION_TYPES:
            raise ValueError(
                f"weight_region_type must be one of {_VALID_WEIGHT_REGION_TYPES}, "
                f"got {self.weight_region_type!r}"
            )

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

        if weight_region_type == "FULL_SIMPLEX":
            for field in _MDN_AUDIT_FIELDS:
                if getattr(self, field) is not None:
                    raise ValueError(f"FULL_SIMPLEX certificates must have {field} == None")
        else:
            validate_mdn_certificate(self)

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
            "weight_region_type": self.weight_region_type,
            "certification_context": _optional_vector_to_list(self.certification_context),
            "mdn_alpha": _optional_vector_to_list(self.mdn_alpha),
            "wx_support_directions": _optional_matrix_to_list(self.wx_support_directions),
            "wx_support_values": _optional_vector_to_list(self.wx_support_values),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Certificate":
        """Create a certificate from dictionary input via constructor validation."""
        data = dict(data)
        data.setdefault("weight_region_type", "FULL_SIMPLEX")
        data.setdefault("certification_context", None)
        data.setdefault("mdn_alpha", None)
        data.setdefault("wx_support_directions", None)
        data.setdefault("wx_support_values", None)
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


def is_mdn_certificate(cert: Certificate) -> bool:
    """Return True when a certificate was issued under contextual MDN W_x."""
    return cert.weight_region_type == "MDN_WX"


def validate_mdn_certificate(cert: Certificate) -> None:
    """Validate required MDN W_x audit metadata on a certificate."""
    if cert.weight_region_type != "MDN_WX":
        return

    context = _finite_vector(
        "certification_context",
        cert.certification_context,
        positive=False,
        non_negative=False,
    )
    alpha = _finite_vector(
        "mdn_alpha",
        cert.mdn_alpha,
        positive=True,
        non_negative=False,
    )
    directions = _finite_matrix("wx_support_directions", cert.wx_support_directions)
    values = _finite_vector(
        "wx_support_values",
        cert.wx_support_values,
        positive=False,
        non_negative=True,
    )

    if len(values) != len(directions):
        raise ValueError(
            "wx_support_values must have the same number of items as "
            "wx_support_directions rows"
        )

    object.__setattr__(cert, "certification_context", context)
    object.__setattr__(cert, "mdn_alpha", alpha)
    object.__setattr__(cert, "wx_support_directions", directions)
    object.__setattr__(cert, "wx_support_values", values)


def _finite_vector(
    field: str,
    value: Any,
    *,
    positive: bool,
    non_negative: bool,
) -> tuple[float, ...]:
    if value is None:
        raise ValueError(f"MDN_WX certificates must have non-None {field}")
    try:
        vector = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite non-empty vector") from exc
    if not vector:
        raise ValueError(f"{field} must be non-empty")
    if not all(isfinite(v) for v in vector):
        raise ValueError(f"{field} must contain only finite values")
    if positive and any(v <= 0.0 for v in vector):
        raise ValueError(f"{field} must contain only positive values")
    if non_negative and any(v < 0.0 for v in vector):
        raise ValueError(f"{field} must contain only non-negative values")
    return vector


def _finite_matrix(field: str, value: Any) -> tuple[tuple[float, ...], ...]:
    if value is None:
        raise ValueError(f"MDN_WX certificates must have non-None {field}")
    try:
        rows = tuple(tuple(float(item) for item in row) for row in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite two-dimensional matrix") from exc
    if not rows:
        raise ValueError(f"{field} must have at least one row")
    row_width = len(rows[0])
    if row_width == 0:
        raise ValueError(f"{field} rows must be non-empty")
    for row in rows:
        if len(row) != row_width:
            raise ValueError(f"{field} rows must all have the same length")
        if not all(isfinite(v) for v in row):
            raise ValueError(f"{field} must contain only finite values")
    return rows


def _optional_vector_to_list(value: tuple[float, ...] | None) -> list[float] | None:
    if value is None:
        return None
    return [float(v) for v in value]


def _optional_matrix_to_list(value: tuple[tuple[float, ...], ...] | None) -> list[list[float]] | None:
    if value is None:
        return None
    return [[float(v) for v in row] for row in value]
