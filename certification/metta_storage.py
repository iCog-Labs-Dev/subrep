"""
Certificate storage backed by a real Hyperon MeTTA space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from hyperon import MeTTa, ExpressionAtom, SymbolAtom

from certification.certificate_schema import Certificate
from certification.metta_bridge import atom_to_cert, cert_to_atom, parse_atom, serialize_atom
from utils.cone_utils import validate_simplex_weights


class CertificateStore:
    """Manage admitted skill certificates in a Hyperon space."""

    def __init__(self, space: Optional[Any] = None):
        # Keep a runner for parsing/space lifecycle.
        self._metta = MeTTa()
        # Default to this runner's program space unless caller injects one.
        self._space = space if space is not None else self._metta.space()
        # Keep a Python-side index for stable runtime access. Hyperon remains
        # the backing audit space, but some Hyperon builds can be fragile when
        # repeatedly scanning atoms through get_atoms().
        self._certificates: dict[str, Certificate] = {}
        self._atoms_by_skill: dict[str, Any] = {}

    def add(self, certificate: Certificate) -> bool:
        """Add a certificate if its skill_id is not already present."""
        # Skill IDs are treated as unique store keys.
        if self.contains(certificate.skill_id):
            return False
        atom = cert_to_atom(certificate)
        self._space.add_atom(atom)
        self._certificates[certificate.skill_id] = certificate
        self._atoms_by_skill[certificate.skill_id] = atom
        return True

    def contains(self, skill_id: str) -> bool:
        return skill_id in self._certificates

    def get_certificate(self, skill_id: str) -> Optional[Certificate]:
        """Return certificate by skill_id, or None when missing."""
        return self._certificates.get(skill_id)

    def query_by_gate_type(self, gate_type: str) -> list[Certificate]:
        """Return certificates matching gate type ('CDS' or 'PDS')."""
        # Normalize input to match schema normalization behavior.
        normalized = gate_type.strip().upper()
        if normalized not in {"CDS", "PDS"}:
            raise ValueError(f"gate_type must be 'CDS' or 'PDS', got {gate_type!r}")

        return [c for c in self.load_all() if c.gate_type == normalized]

    def query_by_weights(self, weights: list[float]) -> list[Certificate]:
        """
        Return certificates admissible under given simplex weights.

        CDS certificates are globally admitted under valid simplex constraints.
        PDS certificates are checked with: delta_r + w^T delta_n >= -epsilon.
        """
        w = self._validated_weights_array(weights)
        results: list[Certificate] = []

        for cert in self.load_all():
            if cert.gate_type == "CDS":
                # CDS is globally admissible under the simplex assumption.
                results.append(cert)
                continue
            # PDS check from project task: delta_r + w^T delta_n >= -epsilon.
            score = float(cert.delta_r) + float(np.dot(w, np.asarray(cert.delta_n, dtype=float)))
            if score >= -float(cert.epsilon):
                results.append(cert)
        return results

    def remove_skill(self, skill_id: str) -> bool:
        """Remove a certificate by skill_id. Returns False when missing."""
        atom = self._atoms_by_skill.pop(skill_id, None)
        if atom is None:
            return False
        self._certificates.pop(skill_id, None)
        self._space.remove_atom(atom)
        return True

    def load_all(self) -> list[Certificate]:
        """Load all certificate atoms and convert them to certificate objects."""
        return list(self._certificates.values())

    def count(self) -> int:
        return len(self._certificates)

    def save_to_file(self, path: str | Path) -> None:
        """Persist all certificates as one deterministic expression per line."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Deterministic representation is useful for auditing and diffing.
        lines = [serialize_atom(cert_to_atom(cert)) for cert in self.load_all()]
        content = "\n".join(lines)
        if content:
            content += "\n"
        file_path.write_text(content, encoding="utf-8")

    def load_from_file(self, path: str | Path) -> None:
        """
        Replace current store content with certificates loaded from file.

        Duplicate skill_id entries inside the file are rejected.
        """
        file_path = Path(path)
        raw_text = file_path.read_text(encoding="utf-8")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        # Parse and validate first.
        parsed_certs: list[Certificate] = []
        seen: set[str] = set()
        for line in lines:
            atom = parse_atom(line)
            cert = atom_to_cert(atom)
            if cert.skill_id in seen:
                raise ValueError(f"Duplicate skill_id in file: {cert.skill_id}")
            seen.add(cert.skill_id)
            parsed_certs.append(cert)

        # Replace semantics: remove all current certificate atoms, then add parsed.
        for atom in list(self._atoms_by_skill.values()):
            self._space.remove_atom(atom)
        self._certificates.clear()
        self._atoms_by_skill.clear()
        for cert in parsed_certs:
            atom = cert_to_atom(cert)
            self._space.add_atom(atom)
            self._certificates[cert.skill_id] = cert
            self._atoms_by_skill[cert.skill_id] = atom

    def _certificate_atoms(self) -> list[Any]:
        return [atom for atom in self._space.get_atoms() if self._is_certificate_atom(atom)]

    @staticmethod
    def _is_certificate_atom(atom: Any) -> bool:
        if not isinstance(atom, ExpressionAtom):
            return False
        children = atom.get_children()
        return bool(children) and isinstance(children[0], SymbolAtom) and children[0].get_name() == "Certificate"

    def _find_certificate_atom(self, skill_id: str) -> Optional[Any]:
        # Resolve by decoded certificate content, not by raw atom string shape.
        for atom in self._certificate_atoms():
            try:
                cert = atom_to_cert(atom)
            except ValueError:
                continue
            if cert.skill_id == skill_id:
                return atom
        return None

    @staticmethod
    def _validated_weights_array(weights: list[float]) -> np.ndarray:
        """Validate and return a 2D simplex weight array for this phase."""
        if weights is None:
            raise ValueError("weights must not be None")
        arr = np.asarray(weights, dtype=float)
        if arr.shape != (2,):
            raise ValueError(f"weights must be a length-2 vector, got shape {arr.shape}")
        if not validate_simplex_weights(arr):
            raise ValueError(
                "weights must be a valid simplex vector (finite, non-negative, sum to 1)"
            )
        return arr
