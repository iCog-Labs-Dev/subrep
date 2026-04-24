
# Certification Logic 

**Purpose:** Implements mathematical admission gates (CDS/PDS) to certify skills before storage.  


## Goal
Ensure only skills that improve backed-up value (without unacceptable harm) are admitted to the library.


## Key Files
| File | Purpose |
|------|---------|
| `cds_test.py` | Implements Cone-Dominant Subtask math (universal benefit) |
| `pds_test.py` | Implements Pareto-Dominant Subtask math (acceptable trade-off) |
| `gate.py` | Unified admission controller with logging |


## Validation
Run `python tests/test_certification.py` to verify:
- Universally beneficial skills pass CDS
- Harmful skills fail CDS
- Trade-off skills pass PDS only within epsilon bounds
- Gate returns clear admission reasons for debugging




