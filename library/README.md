# Skill Library: Runtime Storage and Selection

The Skill Library is the runtime interface for certified SubRep skills. After a
skill passes a CDS or PDS gate, it can be registered in the library and made
available for safe selection.

```text
Execute -> Certify -> Store Certificate -> Register Skill -> Query -> Select
```

## Certificate Store vs Skill Library

| Aspect | Certificate Store | Skill Library |
|---|---|---|
| Purpose | Formal certificate storage and auditability | Runtime skill lookup and selection |
| Main question | Is this skill certified? | Which certified skill can I use now? |
| Implementation | Hyperon/MeTTa atoms via `certification/metta_storage.py` | Python `dict[str, SkillEntry]` plus JSON persistence |
| Stored data | Certificate math and audit metadata | Certificate plus runtime policy callable |
| Persistence | `data/certificates.metta` | `data/library.json` |

They work together. The certificate store is the source of formal admission
evidence, and the library is the fast runtime access layer. When a
`CertificateStore` is attached, `SkillLibrary.add_skill()` requires the
certificate to already exist in that store.

## Files

```text
library/
├── __init__.py           # Public API exports
├── skill_metadata.py     # SkillEntry schema and region constants
├── skill_library.py      # Add/get/remove/query/save/load behavior
└── skill_selector.py     # Random and MDN-weighted selection strategies
```

## Admission Behavior

`SkillLibrary.add_skill()` checks:

- caller `skill_id` matches `certificate.skill_id`,
- certificate exists in the attached `CertificateStore`, when provided,
- gate type is supported (`CDS` or `PDS`),
- CDS/PDS math still passes before runtime admission,
- `MDN_WX` skills include support directions and support values.

This re-check protects the runtime library from accepting malformed or stale
certificate data.

## Query Modes

- `get_skill(skill_id)`: return one stored skill.
- `get_admitted_skills()`: return all registered skills.
- `query_by_gate_type(gate_type)`: filter by `CDS` or `PDS`.
- `query_by_weights(weights)`: legacy full-simplex query for valid simplex weights.
- `query_admissible(current_weight, support_directions, support_values)`: unified runtime query for both `FULL_SIMPLEX` and `MDN_WX` skills.

`FULL_SIMPLEX` skills are globally eligible under valid simplex weights.
`MDN_WX` skills require feasible current support geometry.

## Selection Strategies

- `select_random(obs)`: seeded random selection from all admitted skills.
- `select_by_mdn(obs)`: uses MDN alpha/support output, calls
  `query_admissible()`, and chooses the highest scoring admissible skill.

`select_by_payoff()` is intentionally not the main runtime path. Context-aware
selection should use the MDN path.

## Persistence

`SkillLibrary.save()` writes serializable skill metadata to JSON. Policy
callables cannot be serialized, so loaded libraries need policies re-registered
with `register_policy(skill_id, policy)` before execution.

## Validation

```bash
python -m pytest tests/test_skill_library.py tests/test_mdn_skill_selection.py -v
```

Tests cover:

- skill entry validation,
- add/get/remove behavior,
- certificate-store synchronization,
- CDS/PDS re-checks,
- full-simplex and `MDN_WX` admissibility,
- random selection,
- MDN-guided selection,
- JSON save/load behavior.
