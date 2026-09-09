# Named N-Dimensional Objectives — Schema Migration

## Purpose

This document describes SubRep's schema generalization for arbitrary N-dimensional
objective spaces. It covers the `ObjectiveSchema` abstraction, named motive vectors,
cross-domain rejection, and backward compatibility for existing artifacts.

The change generalizes core schemas and artifacts to support:
* Existing 2D LunarLander certificates and records
* New 6D Minecraft Village Defense & Trade certificates and records
* Any future objective space of any dimension N ≥ 1

This is a **schema/data-contract migration**, not a Minecraft runtime implementation.
No Mineflayer, Paper server, SMDP options, or RL training is introduced here.

---

## Named Objective Schemas

`schemas/objective_schema.py` introduces the `ObjectiveSchema` dataclass:

```python
from schemas.objective_schema import ObjectiveSchema

schema = ObjectiveSchema(
    domain_id="my_domain",
    motive_schema_version="1.0",
    motive_names=("Motive_A", "Motive_B"),
)
```

Fields:
| Field | Type | Meaning |
|---|---|---|
| `domain_id` | `str` | Unique domain/environment family identifier |
| `motive_schema_version` | `str` | Semantic version; increment when semantics change |
| `motive_names` | `tuple[str, ...]` | Ordered, non-duplicate, non-empty names |

Derived property `n_objectives = len(motive_names)`.

### Constraints enforced at construction time

* `motive_names` must be non-empty.
* Every name must be a non-empty, non-whitespace string.
* No duplicate names.
* `domain_id` and `motive_schema_version` must be non-empty strings.

---

## 2D LunarLander vs 6D Minecraft

Two pre-built constants are provided in
`schemas/minecraft_objective_schema.py`:

### LunarLander (existing, 2D)

```python
from schemas.minecraft_objective_schema import LUNARLANDER_OBJECTIVE_SCHEMA

# domain_id = "lunarlander"
# motive_schema_version = "1.0"
# motive_names = ("Safety", "Fuel")
# n_objectives = 2
```

### Minecraft Village Defense & Trade (6D)

```python
from schemas.minecraft_objective_schema import MINECRAFT_OBJECTIVE_SCHEMA

# domain_id = "minecraft_village_defense_trade"
# motive_schema_version = "1.0"
# motive_names = (
#     "Safety", "Reputation", "DeadlineSlack",
#     "InventoryValue", "Sustainability", "Infrastructure"
# )
# n_objectives = 6
```

The canonical Minecraft motive order matches Section 8 of the roadmap.
**Do not reorder** — the i-th name corresponds to the i-th coordinate of
every Minecraft motive vector (phi, delta_n, n_hat).

---

## Schema / Domain Compatibility

```python
from schemas.objective_schema import schemas_compatible

schemas_compatible(LUNARLANDER_OBJECTIVE_SCHEMA, MINECRAFT_OBJECTIVE_SCHEMA)
# False — different domain_id, different names, different dimension

schemas_compatible(LUNARLANDER_OBJECTIVE_SCHEMA, LUNARLANDER_OBJECTIVE_SCHEMA)
# True — same object

schemas_compatible(None, MINECRAFT_OBJECTIVE_SCHEMA)
# True — None means "unknown/legacy", always accepted
```

Compatibility requires **all three** to match:
* `domain_id`
* `motive_schema_version`
* `motive_names` (ordered)

Two schemas with the same dimension but different names are **incompatible**:
`[Safety, Fuel]` vs `[Safety, Reputation]` are rejected even though both
have N=2.

### Enforcement in the Skill Library

`SkillLibrary.query_by_schema(schema)` returns only skills whose
certificate is compatible with the requested schema:

```python
from library.skill_library import SkillLibrary
from schemas.minecraft_objective_schema import MINECRAFT_OBJECTIVE_SCHEMA

lib = SkillLibrary()
# ... add skills ...

# Returns only skills from the Minecraft domain.
mc_skills = lib.query_by_schema(MINECRAFT_OBJECTIVE_SCHEMA)
```

Skills whose certificate carries no schema fields (legacy) are always
included.

---

## Backward Compatibility

All schema identity fields (`domain_id`, `motive_schema_version`,
`motive_names`) **default to `None`** in `Certificate`, `CandidateSkillRecord`,
and `MDNDecisionRecord`.

An artifact that lacks these fields loads cleanly:

```python
cert = Certificate.from_dict(legacy_dict)  # no schema fields → None
assert cert.domain_id is None
assert len(cert.delta_n) == 2  # 2D still works
```

`schemas_compatible(None, any_schema)` always returns `True`, so legacy
artifacts appear in all `query_by_schema` results — they are never
silently excluded.

**Never reinterpret** a LunarLander artifact as a Minecraft artifact.
The `None` schema means "unknown", not "compatible with everything". Callers
that need strict enforcement should check `cert.domain_id is not None` before
comparing.

---

## N-Dimensional Simplex Support

The standard simplex generalizes to N dimensions naturally:

```
W = {w >= 0 : sum(w) = 1}
```

`validate_simplex_weights(w)` accepts any 1D non-negative, unit-sum vector.
`CertificateStore.query_by_weights` accepts N-dimensional weight vectors.

Example:

```python
import numpy as np
from utils.cone_utils import validate_simplex_weights

w6 = np.full(6, 1.0 / 6)        # uniform 6D weight
validate_simplex_weights(w6)     # True

store.query_by_weights([1/6]*6)  # works for 6D certificates
```

> **Note:** `MDN_WX` contextual geometry operates in a fixed 2D space
> and is a separate concern. The `_validate_wx_geometry` and
> `_compute_wx_worst_case` functions in `skill_library.py` retain their
> explicit M=2 requirement by design.

---

## What This Migration Deliberately Does NOT Implement

| Feature | Notes |
|---|---|
| Minecraft runtime environment | Separate workstream |
| Mineflayer / Paper server | Separate workstream |
| Snapshot restoration | Separate workstream |
| SMDP options | Separate workstream |
| Real paired rollouts | Separate workstream |
| RL options | Separate workstream |
| MDN training on Minecraft data | Separate workstream |
| Contextual W_x geometry (6D) | MDN_WX is intentionally 2D |
| CVaR with 6D weights | Depends on MDN_WX generalization |

---

## Tests

```bash
# Named objective schema tests:
python -m pytest tests/test_objective_schema.py -v

# Regression: village sim + certification + baseline
python -m pytest tests/test_village_sim.py tests/test_certificate_storage.py tests/test_baseline.py -v
```

### Coverage summary (14 categories)

| # | Category | Test class |
|---|---|---|
| 1 | Named 2D schema creation | `TestNamedTwoDimensionalSchema` |
| 2 | Named 6D Minecraft schema creation | `TestNamedSixDimensionalSchema` |
| 3 | Vector/schema length validation | `TestVectorSchemaLengthValidation` |
| 4 | Invalid/duplicate objective names | `TestInvalidObjectiveNames` |
| 5 | 6D certificates | `TestSixDimensionalCertificates` |
| 6 | 6D candidate records | `TestSixDimensionalCandidateRecords` |
| 7 | 6D decision records | `TestSixDimensionalDecisionRecords` |
| 8 | 6D simplex weights | `TestSixDimensionalSimplexWeights` |
| 9 | Cross-domain rejection | `TestCrossDomainRejection` |
| 10 | Same-dim/different-schema rejection | `TestSameDimDifferentSchemaRejection` |
| 11 | Legacy artifact loading | `TestLegacyArtifactBackwardCompatibility` |
| 12 | Serialization preserving schema metadata | `TestSerializationWithSchemaMetadata` |
| 13 | LunarLander behavior unchanged | `TestExistingLunarLanderBehavior` |
| 14 | Removed fixed-dimension assumptions | `TestRemovedTwoDimensionalAssumptions` |

---

## Files Changed

| File | Change |
|---|---|
| `schemas/__init__.py` | **New** — schemas package |
| `schemas/objective_schema.py` | **New** — `ObjectiveSchema`, `schemas_compatible`, `validate_vector_against_schema` |
| `schemas/minecraft_objective_schema.py` | **New** — canonical Minecraft 6D and LunarLander 2D constants |
| `certification/certificate_schema.py` | `delta_n` N-dim; add `domain_id`, `motive_schema_version`, `motive_names` fields |
| `certification/metta_bridge.py` | Add schema fields to `OPTIONAL_AUDIT_FIELDS`; `motive_names` tuple coercion |
| `certification/metta_storage.py` | `query_by_weights` accepts N-dim weights |
| `utils/mdn_contracts.py` | `CandidateSkillRecord.delta_n`, `MDNDecisionRecord.actual_motives` → N-dim |
| `utils/mdn_record_builder.py` | `PreparedCandidateOutcome.motives` → N-dim |
| `library/skill_metadata.py` | `delta_n` property type hint; `to_dict` emits schema identity fields |
| `library/skill_library.py` | Add `query_by_schema(schema)` method |
| `tests/test_objective_schema.py` | **New** — 14-category test suite |
| `docs/NAMED_OBJECTIVES.md` | **New** — this document |
