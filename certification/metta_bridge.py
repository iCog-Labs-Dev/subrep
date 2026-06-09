"""
Certificate <-> MeTTA atom conversion utilities backed by Hyperon.
"""

from __future__ import annotations

from typing import Any

from hyperon import E, S, ValueAtom, MeTTa, ExpressionAtom, SymbolAtom, GroundedAtom

from certification.certificate_schema import Certificate


BASE_CERTIFICATE_FIELDS = [
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
OPTIONAL_AUDIT_FIELDS = [
    "weight_region_type",
    "certification_context",
    "mdn_alpha",
    "wx_support_directions",
    "wx_support_values",
]
CERTIFICATE_FIELDS = BASE_CERTIFICATE_FIELDS + OPTIONAL_AUDIT_FIELDS

# A shared parser instance is sufficient for line-by-line certificate parsing.
_PARSER = MeTTa()


def cert_to_atom(cert: Certificate) -> ExpressionAtom:
    """Convert a Certificate into canonical MeTTA expression atom."""
    d = cert.to_dict()
    # Canonical shape:
    # (Certificate (field value) ... (delta_n (vec x y)) ...)
    return E(
        S("Certificate"),
        *[E(S(field), python_to_metta_value(d[field])) for field in CERTIFICATE_FIELDS],
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

        fields[key] = metta_to_python_value(value_atom)

    # Enforce required fields while allowing old certificates to omit optional
    # MDN audit metadata. Extra fields still fail to avoid silent schema drift.
    required = set(BASE_CERTIFICATE_FIELDS)
    optional = set(OPTIONAL_AUDIT_FIELDS)
    expected = required | optional
    actual = set(fields.keys())
    missing_required = sorted(required - actual)
    if missing_required or not actual <= expected:
        extra = sorted(actual - expected)
        raise ValueError(
            f"Certificate fields mismatch. missing={missing_required}, extra={extra}"
        )
    for field in OPTIONAL_AUDIT_FIELDS:
        fields.setdefault(field, None)
    fields["weight_region_type"] = fields["weight_region_type"] or "FULL_SIMPLEX"

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

def python_to_metta_value(value: Any) -> Any:
    """Convert supported Python values into MeTTA atoms."""
    if value is None:
        return S("Nil")
    if _is_numeric_vector(value):
        return E(S("vec"), *[ValueAtom(float(item)) for item in value])
    if _is_list_of_numeric_vectors(value):
        return E(
            S("list"),
            *[
                E(S("vec"), *[ValueAtom(float(item)) for item in row])
                for row in value
            ],
        )
    if isinstance(value, bool):
        return ValueAtom(bool(value))
    if isinstance(value, int):
        return ValueAtom(int(value))
    if isinstance(value, float):
        return ValueAtom(float(value))
    if isinstance(value, str):
        return ValueAtom(value)
    raise ValueError(f"Unsupported Python value for MeTTA serialization: {value!r}")


def metta_to_python_value(metta_value: Any) -> Any:
    """Convert supported MeTTA atoms into Python values."""
    if isinstance(metta_value, SymbolAtom):
        name = metta_value.get_name()
        if name == "Nil":
            return None
        return name
    if isinstance(metta_value, ExpressionAtom):
        children = metta_value.get_children()
        if not children or not isinstance(children[0], SymbolAtom):
            raise ValueError(f"Unsupported expression value: {metta_value}")
        head = children[0].get_name()
        if head == "vec":
            return [float(metta_to_python_value(child)) for child in children[1:]]
        if head == "list":
            return [metta_to_python_value(child) for child in children[1:]]
        raise ValueError(f"Unsupported expression value: {metta_value}")
    return _atom_value(metta_value)

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


def _is_numeric_vector(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    if not isinstance(value, (list, tuple)):
        return False
    if not value:
        return True
    return all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)


def _is_list_of_numeric_vectors(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    return all(_is_numeric_vector(item) for item in value)
