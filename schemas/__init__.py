"""
SubRep objective schema definitions.

This package provides the ObjectiveSchema abstraction used to identify
and validate named, ordered motive vectors across all SubRep domains.
"""

from schemas.objective_schema import (
    ObjectiveSchema,
    schemas_compatible,
    validate_vector_against_schema,
)

__all__ = [
    "ObjectiveSchema",
    "schemas_compatible",
    "validate_vector_against_schema",
]
