"""
SubRep objective schema definitions.

This package provides the ObjectiveSchema abstraction used to identify
and validate named, ordered motive vectors across all SubRep domains.
"""

from schemas.objective_schema import (
    ObjectiveSchema,
    schemas_compatible,
    schemas_loadable,
    validate_vector_against_schema,
)
from schemas.minecraft_objective_schema import (
    MINECRAFT_OBJECTIVE_SCHEMA,
    LUNARLANDER_OBJECTIVE_SCHEMA,
    SAFETY_GYMNASIUM_OBJECTIVE_SCHEMA,
)

__all__ = [
    "ObjectiveSchema",
    "schemas_compatible",
    "schemas_loadable",
    "validate_vector_against_schema",
    "MINECRAFT_OBJECTIVE_SCHEMA",
    "LUNARLANDER_OBJECTIVE_SCHEMA",
    "SAFETY_GYMNASIUM_OBJECTIVE_SCHEMA",
]
