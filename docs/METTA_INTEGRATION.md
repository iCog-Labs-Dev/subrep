# MeTTa Integration

**Purpose:** Stores and retrieves certified skills as native Atoms in AtomSpace using the `hyperon` Python package.

## Goal

Enable Hyperon compatibility by serializing certificates as MeTTa Atoms for logical querying and zero-shot reuse.

- `certification/certificate_schema.py`
- `certification/metta_bridge.py`
- `certification/metta_storage.py`

Certificates are converted to Hyperon MeTTa atoms, stored in a Hyperon space,
and can be saved to `data/certificates.metta` for audit/replay.

## Validation

Verify:

- Atoms are successfully added to AtomSpace
- Query returns correct skills for given weights
- Certificate retrieval preserves numerical precision

```bash
python -m pytest tests/test_certificate_storage.py tests/test_certificate_mdn_audit_storage.py -v
```

These tests verify atom conversion, parse/serialize roundtrips, duplicate
rejection, query behavior, and MDN/W_x audit metadata preservation.
