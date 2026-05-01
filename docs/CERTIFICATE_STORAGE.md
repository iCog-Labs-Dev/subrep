# Certificate Storage

This document describes the SubRep certificate schema, Hyperon MeTTA atom
format, and storage/query behavior used in this phase.

## Overview

The certificate system stores admitted skills as MeTTA atoms in a Hyperon
space for auditability and reproducibility.

Implemented modules:

- `certification/certificate_schema.py`
- `certification/metta_bridge.py`
- `certification/metta_storage.py`

Tests:

- `tests/test_certificate_storage.py`

## Certificate Schema

`Certificate` is an immutable dataclass validated at construction time.

Fields:

- `skill_id: str`
- `gate_type: str` (`CDS` or `PDS`, normalized to uppercase)
- `delta_r: float`
- `delta_n: vector<float>` (exactly 2 values)
- `admission_margin: float` (`>= 0`)
- `epsilon: float` (`>= 0`, and `epsilon == 0` for `CDS`)
- `timestamp: str` (ISO format)
- `seed: int`
- `gamma: float` (`0 <= gamma <= 1`)
- `baseline_id: str`
- `environment: str`
- `episode_length: int` (`> 0`)
- `version: str`

Methods:

- `to_dict() -> dict`
- `from_dict(data: dict) -> Certificate`

## MeTTA Atom Format

Canonical expression shape:

```metta
(Certificate
  (skill_id "landing_skill_01")
  (gate_type "CDS")
  (delta_r 0.5)
  (delta_n (vec 0.2 -0.1))
  (admission_margin 0.4)
  (epsilon 0.0)
  (timestamp "2025-01-15T10:30:00")
  (seed 42)
  (gamma 0.99)
  (baseline_id "idle_policy")
  (environment "MO-LunarLander-v0")
  (episode_length 120)
  (version "subrep-q1-v0.1"))
```

Note: `delta_n` is fixed to 2D in this phase for MO-LunarLander compatibility.
The paper's broader formulation uses an `m`-dimensional motive vector.

`metta_bridge.py` (Hyperon-backed) provides:

- `cert_to_atom(cert)`
- `atom_to_cert(atom)`
- deterministic serializer/parser for this certificate format

The parser is intentionally narrow and supports only this emitted schema.
Persistence note:
- `save_to_file()` serializes via Hyperon atom text representation (`str(atom)`).
- `load_from_file()` parses those lines back with `MeTTa.parse_single(...)`.
- This roundtrip is validated in tests, but it depends on Hyperon's emitted atom
  text remaining parse-compatible with `parse_single`.

## Storage API

`CertificateStore` (Hyperon space-backed) supports:

- `add(certificate) -> bool`
- `contains(skill_id) -> bool`
- `get_certificate(skill_id) -> Optional[Certificate]`
- `query_by_gate_type(gate_type) -> List[Certificate]`
- `query_by_weights(weights) -> List[Certificate]`
- `remove_skill(skill_id) -> bool`
- `load_all() -> List[Certificate]`
- `count() -> int`
- `save_to_file(path)`
- `load_from_file(path)`

Behavior policies:

- Duplicate `skill_id` on `add` is rejected (`False`).
- `query_by_gate_type` normalizes gate type to uppercase.
- Weight vectors must be valid simplex vectors:
  - length 2
  - finite
  - non-negative
  - sum approximately 1
- Invalid weights raise `ValueError`.
- `load_from_file(path)` replaces current store content.

Testing note:
- `tests/test_certificate_storage.py` requires `hyperon`.
- In environments without `hyperon`, that test module is skipped.

Environment note:
- If `hyperon` is unavailable in your Windows Python environment, run the
  certificate-storage tests from WSL/Linux with a dedicated virtualenv.
- Example:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `python -m pip install -U pip`
  - `python -m pip install hyperon pytest`
  - `python -m pytest tests/test_certificate_storage.py -v`

## Running MeTTA/Hyperon Tests

Use one of the following command paths depending on where you run tests.

From Windows terminal (PowerShell) using WSL Python:

- `wsl bash -lc "cd /mnt/c/Users/lisan/Desktop/subrep && /home/lisan/subrep-hyperon-venv/bin/python -m pytest tests/test_certificate_storage.py -v"`

From an interactive WSL shell:

- `cd /mnt/c/Users/lisan/Desktop/subrep`
- `/home/lisan/subrep-hyperon-venv/bin/python -m pytest tests/test_certificate_storage.py -v`

Optional full suite in WSL Hyperon environment:

- `/home/lisan/subrep-hyperon-venv/bin/python -m pytest tests/ -v`

## Query Semantics

For valid simplex weights:

- `CDS`: certificates are globally admissible (returned).
- `PDS`: certificate returned if:
  - `delta_r + w^T delta_n >= -epsilon`

## Usage Example

```python
from certification.certificate_schema import Certificate
from certification.metta_storage import CertificateStore

cert = Certificate(
    skill_id="landing_skill_01",
    gate_type="CDS",
    delta_r=0.5,
    delta_n=(0.2, -0.1),
    admission_margin=0.4,
    epsilon=0.0,
    timestamp="2025-01-15T10:30:00",
    seed=42,
    gamma=0.99,
    baseline_id="idle_policy",
    environment="MO-LunarLander-v0",
    episode_length=120,
    version="subrep-q1-v0.1",
)

store = CertificateStore()
store.add(cert)
store.save_to_file("data/certificates.metta")
```
