"""
ObjectiveSchema — named, ordered motive-space descriptor for SubRep.

This module provides the central schema abstraction so that certificates,
candidate records, and decision records carry enough identity information
to prevent incompatible objective spaces from being mixed.

Design constraints:
  * Supports arbitrary N objectives (N >= 1).
  * Objective ordering is semantically meaningful.
  * Schema compatibility is based on identity, not vector shape alone —
    [Safety, Fuel] and [Safety, Reputation] are incompatible even though
    both have N=2.
  * Artifacts that carry no schema fields (domain_id, motive_schema_version,
    motive_names all None) are treated as legacy/unknown and are always
    accepted by compatibility checks.
  * MDN_WX contextual geometry is a separate concern handled elsewhere.

Usage example:

    from schemas.objective_schema import ObjectiveSchema, schemas_compatible

    lunar = ObjectiveSchema(
        domain_id="lunarlander",
        motive_schema_version="1.0",
        motive_names=("Safety", "Fuel"),
    )

    mc = ObjectiveSchema(
        domain_id="minecraft_village_defense_trade",
        motive_schema_version="1.0",
        motive_names=(
            "Safety", "Reputation", "DeadlineSlack",
            "InventoryValue", "Sustainability", "Infrastructure",
        ),
    )

    schemas_compatible(lunar, mc)    # False — different domain + names
    schemas_compatible(lunar, lunar) # True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ObjectiveSchema:
    """
    Descriptor for a named, ordered set of motive coordinates.

    Attributes:
        domain_id: Unique identifier for the domain/environment family.
            Example: "lunarlander" or "minecraft_village_defense_trade".
        motive_schema_version: Semantic version string for the motive
            definitions.  Increment when coordinate semantics change.
            Example: "1.0", "2.0".
        motive_names: Ordered tuple of non-empty, non-duplicate motive
            names.  Order is semantically meaningful — the i-th name
            corresponds to the i-th coordinate of every motive vector
            that claims this schema.

    Derived:
        n_objectives: Number of motive coordinates (len(motive_names)).
    """

    domain_id: str
    motive_schema_version: str
    motive_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.domain_id, str) or not self.domain_id.strip():
            raise ValueError("domain_id must be a non-empty string")
        if not isinstance(self.motive_schema_version, str) or not self.motive_schema_version.strip():
            raise ValueError("motive_schema_version must be a non-empty string")

        # Coerce to tuple of strings once so hashing is stable.
        names = tuple(self.motive_names)
        object.__setattr__(self, "motive_names", names)

        if len(names) == 0:
            raise ValueError("motive_names must contain at least one name")
        seen: set[str] = set()
        for name in names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"Every motive name must be a non-empty string, got {name!r}"
                )
            if name in seen:
                raise ValueError(f"Duplicate motive name: {name!r}")
            seen.add(name)

    @property
    def n_objectives(self) -> int:
        """Number of motive coordinates described by this schema."""
        return len(self.motive_names)

    def validate_vector(self, vector: Sequence[float], field_name: str = "motive_vector") -> None:
        """
        Raise ValueError when *vector* length does not match this schema.

        Args:
            vector: The motive (or delta_n) vector to validate.
            field_name: Name used in the error message.

        Raises:
            ValueError: If the vector length does not equal n_objectives.
        """
        n = len(vector)
        if n != self.n_objectives:
            raise ValueError(
                f"{field_name} has length {n} but schema "
                f"'{self.domain_id}' expects {self.n_objectives} "
                f"({', '.join(self.motive_names)})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "domain_id": self.domain_id,
            "motive_schema_version": self.motive_schema_version,
            "motive_names": list(self.motive_names),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectiveSchema":
        """Reconstruct an ObjectiveSchema from a dictionary."""
        return cls(
            domain_id=str(data["domain_id"]),
            motive_schema_version=str(data["motive_schema_version"]),
            motive_names=tuple(str(n) for n in data["motive_names"]),
        )


def schemas_compatible(a: ObjectiveSchema | None, b: ObjectiveSchema | None) -> bool:
    """
    Return True when two schemas are compatible (i.e., can be compared).

    Rules:
        - If either schema is None (legacy / unknown), accept.
        - Two schemas are compatible only when domain_id,
          motive_schema_version, AND motive_names all match exactly.
        - Same length but different names → incompatible.
        - Different lengths → incompatible (unless one is None).

    Args:
        a: First schema, or None for legacy artifacts.
        b: Second schema, or None for legacy artifacts.

    Returns:
        True if the schemas can be compared/combined without error.
    """
    if a is None or b is None:
        # One or both sides are legacy/unknown — accept with a warning
        # at the call site if desired.
        return True
    return (
        a.domain_id == b.domain_id
        and a.motive_schema_version == b.motive_schema_version
        and a.motive_names == b.motive_names
    )


def validate_vector_against_schema(
    vector: Sequence[float],
    schema: ObjectiveSchema | None,
    field_name: str = "motive_vector",
) -> None:
    """
    Validate that *vector* matches *schema* if schema is not None.

    A None schema means the objective space is unknown (legacy artifact),
    so no length enforcement is applied.

    Args:
        vector: The motive vector to validate.
        schema: The expected objective schema, or None to skip enforcement.
        field_name: Label used in error messages.

    Raises:
        ValueError: If schema is not None and the vector length is wrong.
    """
    if schema is None:
        return
    schema.validate_vector(vector, field_name)
