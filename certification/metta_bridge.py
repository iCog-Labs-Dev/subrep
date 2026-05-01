"""
Certificate <-> MeTTA atom conversion utilities backed by Hyperon.
"""

from __future__ import annotations

from typing import Any

from hyperon import E, S, ValueAtom, MeTTa, ExpressionAtom, SymbolAtom, GroundedAtom

from certification.certificate_schema import Certificate


CERTIFICATE_FIELDS = [
    "skill_id",
    "gate_type",
    "delta_r",
    "delta_n",
    "admission_margin",
    "epsilon",
    "timestamp",
    "seed",
    "gamma",
    "baseline_id",
    "environment",
    "episode_length",
    "version",
]

# A shared parser instance is sufficient for line-by-line certificate parsing.
_PARSER = MeTTa()


def cert_to_atom(cert: Certificate) -> ExpressionAtom:
    """Convert a Certificate into canonical MeTTA expression atom."""
    d = cert.to_dict()
    # Canonical shape:
    # (Certificate (field value) ... (delta_n (vec x y)) ...)
    return E(
        S("Certificate"),
        E(S("skill_id"), ValueAtom(d["skill_id"])),
        E(S("gate_type"), ValueAtom(d["gate_type"])),
        E(S("delta_r"), ValueAtom(float(d["delta_r"]))),
        E(S("delta_n"), E(S("vec"), ValueAtom(float(d["delta_n"][0])), ValueAtom(float(d["delta_n"][1])))),
        E(S("admission_margin"), ValueAtom(float(d["admission_margin"]))),
        E(S("epsilon"), ValueAtom(float(d["epsilon"]))),
        E(S("timestamp"), ValueAtom(d["timestamp"])),
        E(S("seed"), ValueAtom(int(d["seed"]))),
        E(S("gamma"), ValueAtom(float(d["gamma"]))),
        E(S("baseline_id"), ValueAtom(d["baseline_id"])),
        E(S("environment"), ValueAtom(d["environment"])),
        E(S("episode_length"), ValueAtom(int(d["episode_length"]))),
        E(S("version"), ValueAtom(d["version"])),
    )

def atom_to_cert(atom: Any) -> Certificate:
    """Convert canonical MeTTA certificate atom back to Certificate."""
    if not isinstance(atom, ExpressionAtom):
        raise ValueError(f"Certificate atom must be ExpressionAtom, got {type(atom).__name__}")

    # Root must be the `Certificate` constructor symbol.
    children = atom.get_children()
    if not children or not isinstance(children[0], SymbolAtom) or children[0].get_name() != "Certificate":
        raise ValueError("Certificate atom must start with symbol 'Certificate'")

    fields: dict[str, Any] = {}
    # Each child is expected as a key/value pair expression.
    for pair in children[1:]:
        if not isinstance(pair, ExpressionAtom):
            raise ValueError(f"Certificate field must be expression, got {pair}")
        pair_children = pair.get_children()
        if len(pair_children) != 2:
            raise ValueError(f"Certificate field pair must have 2 elements, got {pair}")
        key_atom, value_atom = pair_children
        if not isinstance(key_atom, SymbolAtom):
            raise ValueError(f"Certificate field key must be symbol, got {key_atom}")
        key = key_atom.get_name()

        if key == "delta_n":
            # `delta_n` is nested to preserve vector structure explicitly.
            fields[key] = _parse_delta_n(value_atom)
        else:
            fields[key] = _atom_value(value_atom)

    # Enforce strict schema completeness to avoid silent drift.
    expected = set(CERTIFICATE_FIELDS)
    actual = set(fields.keys())
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Certificate fields mismatch. missing={missing}, extra={extra}")

    return Certificate.from_dict(fields)


def serialize_atom(atom: Any) -> str:
    """Serialize Hyperon atom to textual MeTTA expression."""
    return str(atom)

def parse_atom(text: str) -> ExpressionAtom:
    """Parse a single textual certificate expression into Hyperon atom."""
    try:
        atom = _PARSER.parse_single(text)
    except Exception as exc:  # Hyperon raises runtime exceptions on parse errors.
        raise ValueError(f"Failed to parse certificate expression: {exc}") from exc

    if not isinstance(atom, ExpressionAtom):
        raise ValueError("Parsed certificate expression is not an ExpressionAtom")
    # Validate shape eagerly so callers only handle valid certificate atoms.
    atom_to_cert(atom)
    return atom

def _parse_delta_n(value_atom: Any) -> list[float]:
    """Parse `(vec x y)` motive representation into Python float list."""
    if not isinstance(value_atom, ExpressionAtom):
        raise ValueError(f"delta_n value must be expression, got {value_atom}")
    vec_children = value_atom.get_children()
    if (
        len(vec_children) != 3
        or not isinstance(vec_children[0], SymbolAtom)
        or vec_children[0].get_name() != "vec"
    ):
        raise ValueError(f"delta_n must be in format (vec x y), got {value_atom}")
    return [float(_atom_value(vec_children[1])), float(_atom_value(vec_children[2]))]

def _atom_value(atom: Any) -> Any:
    """Extract Python value from supported Hyperon atom wrappers."""
    if isinstance(atom, SymbolAtom):
        return atom.get_name()
    if isinstance(atom, GroundedAtom):
        obj = atom.get_object()
        if hasattr(obj, "value"):
            return obj.value
        return obj
    raise ValueError(f"Unsupported atom value type: {type(atom).__name__}")
