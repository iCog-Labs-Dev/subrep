# Skill Library — Runtime Storage & Selection

The Skill Library is the **deployment interface** for SubRep. After a skill passes
certification (CDS or PDS gate), it enters the library and becomes available for
selection and execution.

```
Execute → Certify → Store → Select → Execute
```

### Certificate Storage vs. Skill Library

| Aspect | Certificate Storage (MeTTA/PLN) | Skill Library (Python) |
|--------|-------------------------------|----------------------|
| **Purpose** | Formal safety proofs | Runtime skill access |
| **Question it answers** | "Is this skill safe?" | "Give me a safe skill." |
| **Optimized for** | PLN reasoning & auditability | Fast O(1) Python lookup |
| **Format** | AtomSpace atoms | Dict[str, SkillEntry] + JSON |
| **Lives in** | `metta/` | `library/` |

They complement each other: the formal store provides the safety guarantee and the library provides the fast runtime interface.


```
library/
├── __init__.py           # Public API exports
├── skill_metadata.py     # Certificate + SkillEntry dataclasses
├── skill_library.py      # SkillLibrary — storage with query/save/load
└── skill_selector.py     # SkillSelector — pluggable selection strategies
```

## Usage

- **Adding Skills** — Add a skill with its certificate and policy function. The library validates against the certificate store if provided.
- **Querying Skills** — Retrieve skills by ID, gate type (CDS or PDS), or by weight vector to find admissible skills under specific stakeholder preferences.
- **Removing Skills** — Remove a skill from the library by ID.
- **Save / Load** — Persist the library to JSON and restore it later. Policies cannot be serialized, so re-register them after loading.


## Selection Strategies

- **Random Baseline (Stage 3-4) — Implemented** — Uniform random selection from all admitted skills. Reproducible with seed control, returns `None` for empty library.
- **Greedy Payoff (Stage 5) — Not Implemented** — Selects the skill with highest predicted payoff using SkillGenerator predictions.
- **MDN-Weighted (Stage 6) — Not Implemented** — Context-aware selection using MDN for adaptive weights and SkillGenerator for predictions.


## Validation

```bash
python -m pytest tests/test_skill_library.py -v
```

**Tests cover**: SkillEntry creation/validation, Certificate serialization, add/get/remove, gate-type queries, weight-vector queries, JSON roundtrips, random selection with reproducibility, and a full integration test wiring certification gates → library → selector.
