# Certification

`certification/` contains SubRep's mathematical admission gates and certificate
storage layer.

## Gates

| File | Purpose |
|---|---|
| `cds_test.py` | Cone-Dominant Subtask gate: `delta_r + min(delta_n) >= 0` |
| `pds_test.py` | Pareto-Dominant Subtask gate: `delta_r + min(delta_n) >= -epsilon` |
| `cvar_test.py` | Optional distribution-aware CVaR gate using MDN alpha |
| `gate.py` | Shared input validation base class |

The gates support full-simplex certification, optional `WeightSet`-restricted certification (legacy
M = 2 path), and `support_values`-restricted certification for contextual `W_x` flows at any M.
The `W_x` path evaluates the region's support function via an exact O(M log M) greedy solver rather
than enumerating vertices.

## Certificates

| File | Purpose |
|---|---|
| `certificate_schema.py` | Validated immutable certificate dataclass |
| `metta_bridge.py` | Certificate <-> Hyperon MeTTa atom conversion |
| `metta_storage.py` | Hyperon space-backed certificate store |

Certificates record gate metrics, baseline metadata, environment metadata, and
optional MDN/W_x audit evidence for contextual certificates.

## Tests

```bash
python -m pytest tests/test_certification_gates.py -v
python -m pytest tests/test_certificate_storage.py tests/test_certificate_mdn_audit_storage.py -v
```
