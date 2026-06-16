from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from certification.certificate_schema import Certificate

FULL_SIMPLEX = "FULL_SIMPLEX"
MDN_WX = "MDN_WX"
_VALID_WEIGHT_REGIONS = {FULL_SIMPLEX, MDN_WX}


@dataclass
class SkillEntry:
    """ Runtime record for a single admitted skill in the Skill Library. """

    skill_id: str
    gate_type: str
    certificate: Certificate
    policy: Optional[Callable] = field(default=None, repr=False) # runtime only, can't be serialized
    executions: int = 0
    success_rate: float = 0.0
    avg_payoff: float = 0.0

    weight_region_type: str = FULL_SIMPLEX
    certification_context: Optional[tuple[float, ...]] = None
    mdn_alpha: Optional[tuple[float, ...]] = None
    wx_support_directions: Optional[tuple[tuple[float, ...], ...]] = None
    wx_support_values: Optional[tuple[float, ...]] = None

    def __post_init__(self) -> None:
        """post-init validation to ensure gate_type is valid and matches the certificate."""
        valid_gates = {"CDS", "PDS"}
        if self.gate_type not in valid_gates:
            raise ValueError(
                f"gate_type must be one of {valid_gates}, got '{self.gate_type}'"
            )
        if self.gate_type != self.certificate.gate_type:
            raise ValueError(
                f"gate_type '{self.gate_type}' does not match certificate gate_type '{self.certificate.gate_type}'"
            )
        if self.weight_region_type not in _VALID_WEIGHT_REGIONS:
            raise ValueError(
                f"weight_region_type must be one of {_VALID_WEIGHT_REGIONS}, got '{self.weight_region_type}'"
            )

        if self.weight_region_type == MDN_WX:
            _missing = []
            if self.certification_context is None:
                _missing.append("certification_context")
            if self.mdn_alpha is None:
                _missing.append("mdn_alpha")
            if self.wx_support_directions is None:
                _missing.append("wx_support_directions")
            if self.wx_support_values is None:
                _missing.append("wx_support_values")
            if _missing:
                raise ValueError(
                    f"MDN_WX entries require all audit fields for traceability. Missing: {', '.join(_missing)}"
                )

    @property
    def delta_r(self) -> float:
        """Scalar payoff improvement from the certificate."""
        return self.certificate.delta_r

    @property
    def delta_n(self) -> tuple[float, float]:
        """Motive improvement vector from the certificate."""
        return self.certificate.delta_n

    @property
    def admission_margin(self) -> float:
        """Admission margin from the certification gate."""
        return self.certificate.admission_margin

    @property
    def epsilon(self) -> float:
        """PDS epsilon budget (0.0 for CDS skills)."""
        return self.certificate.epsilon

    def to_dict(self) -> dict:
        """ 
        Convert to a JSON-safe dictionary.

        The `policy` field is intentionally excluded — callables cannot be
        serialized to JSON.  After loading, the caller must re-register
        policies via SkillLibrary.register_policy()
        """
        data = {
            "skill_id": self.skill_id,
            "gate_type": self.gate_type,
            "certificate": self.certificate.to_dict(),
            "executions": int(self.executions),
            "success_rate": float(self.success_rate),
            "avg_payoff": float(self.avg_payoff),
            "weight_region_type": self.weight_region_type,
        }

        # Only persist MDN fields when they carry data.
        if self.certification_context is not None:
            data["certification_context"] = list(self.certification_context)
        if self.mdn_alpha is not None:
            data["mdn_alpha"] = list(self.mdn_alpha)
        if self.wx_support_directions is not None:
            data["wx_support_directions"] = [list(d) for d in self.wx_support_directions]
        if self.wx_support_values is not None:
            data["wx_support_values"] = list(self.wx_support_values)

        return data

    @classmethod
    def from_dict(cls, data: dict) -> SkillEntry:
        """ Reconstruct a SkillEntry from a JSON-loaded dictionary. """
        certificate = Certificate.from_dict(data["certificate"])

        weight_region_type = data.get("weight_region_type", FULL_SIMPLEX)

        certification_context = data.get("certification_context")
        if certification_context is not None:
            certification_context = tuple(float(v) for v in certification_context)

        mdn_alpha = data.get("mdn_alpha")
        if mdn_alpha is not None:
            mdn_alpha = tuple(float(v) for v in mdn_alpha)

        wx_support_directions = data.get("wx_support_directions")
        if wx_support_directions is not None:
            wx_support_directions = tuple(
                tuple(float(v) for v in d) for d in wx_support_directions
            )

        wx_support_values = data.get("wx_support_values")
        if wx_support_values is not None:
            wx_support_values = tuple(float(v) for v in wx_support_values)

        return cls(
            skill_id=data["skill_id"],
            gate_type=data["gate_type"],
            certificate=certificate,
            policy=None, 
            executions=int(data.get("executions", 0)),
            success_rate=float(data.get("success_rate", 0.0)),
            avg_payoff=float(data.get("avg_payoff", 0.0)),
            weight_region_type=weight_region_type,
            certification_context=certification_context,
            mdn_alpha=mdn_alpha,
            wx_support_directions=wx_support_directions,
            wx_support_values=wx_support_values,
        )