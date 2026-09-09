"""
Canonical Minecraft objective schema for SubRep Village Defense & Trade.

This module provides the single authoritative ObjectiveSchema instance
for the Minecraft domain, matching the canonical motive order defined in
the SubRep Minecraft roadmap (Section 8):

    [Safety, Reputation, DeadlineSlack, InventoryValue,
     Sustainability, Infrastructure]

All coordinates follow the convention: larger value = better outcome.

Usage:

    from schemas.minecraft_objective_schema import MINECRAFT_OBJECTIVE_SCHEMA

    # Validate a 6D motive vector
    MINECRAFT_OBJECTIVE_SCHEMA.validate_vector(my_delta_n)

    # Check compatibility with a certificate's schema
    from schemas.objective_schema import schemas_compatible
    schemas_compatible(cert_schema, MINECRAFT_OBJECTIVE_SCHEMA)

Note: This module defines the schema only.  The actual Minecraft runtime
environment, Mineflayer sidecar, Paper server connection, and SMDP option
execution are NOT implemented here — those belong to Phase 5–7.
"""

from schemas.objective_schema import ObjectiveSchema

# Canonical Minecraft motive order — must match roadmap Section 8 exactly.
# Do NOT reorder. The i-th name corresponds to the i-th coordinate of
# every Minecraft motive vector (phi, delta_n, n_hat).
MINECRAFT_MOTIVE_NAMES: tuple[str, ...] = (
    "Safety",
    "Reputation",
    "DeadlineSlack",
    "InventoryValue",
    "Sustainability",
    "Infrastructure",
)

MINECRAFT_OBJECTIVE_SCHEMA = ObjectiveSchema(
    domain_id="minecraft_village_defense_trade",
    motive_schema_version="1.0",
    motive_names=MINECRAFT_MOTIVE_NAMES,
)

# Convenience alias for the canonical LunarLander 2D schema used in
# existing tests and legacy artifacts.
LUNARLANDER_OBJECTIVE_SCHEMA = ObjectiveSchema(
    domain_id="lunarlander",
    motive_schema_version="1.0",
    motive_names=("Safety", "Fuel"),
)
