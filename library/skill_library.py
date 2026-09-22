"""
Skill Library — runtime storage for certified SubRep skills.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional
import warnings
import logging

import numpy as np

from .skill_metadata import SkillEntry, FULL_SIMPLEX, MDN_WX
from utils.cone_utils import validate_simplex_weights
from utils.support_geometry import FEASIBILITY_TOLERANCE, greedy_support_function
from certification.certificate_schema import Certificate
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate

logger = logging.getLogger(__name__)

# Tolerance for the sum(s) >= 1 feasibility test, shared with the greedy solver
# so the library and utils.support_geometry cannot drift apart on what counts as
# feasible.
_FEAS_TOL = FEASIBILITY_TOLERANCE

def _validate_wx_geometry(support_directions: np.ndarray, support_values: np.ndarray,) -> tuple[np.ndarray, np.ndarray]:
    """Validate W_x support geometry for standard-basis directions, any M >= 2.

    W_x = { w in simplex : w_i <= s_i } is non-empty if and only if every
    s_i lies in [0, 1] and sum(s) >= 1. Both conditions hold for any M without
    modification, so this validator carries no per-M special case.
    """
    sd = np.asarray(support_directions, dtype=np.float64)
    sv = np.asarray(support_values, dtype=np.float64)

    if sv.ndim != 1 or sv.size < 2:
        raise ValueError(
            f"support_values must be a vector of length >= 2, got shape {sv.shape}"
        )

    num_obj = sv.size
    expected_shape = (num_obj, num_obj)
    if sd.shape != expected_shape:
        raise ValueError(
            f"support_directions must have shape {expected_shape},got {sd.shape}"
        )

    basis = np.eye(num_obj, dtype=np.float64)
    if not np.allclose(sd, basis, atol=1e-6):
        raise ValueError(
            f"W_x evaluation requires standard basis support directions (identity matrix), got {sd.tolist()}"
        )

    if not np.all(np.isfinite(sv)):
        raise ValueError(
            f"support_values must be finite, got {sv.tolist()}"
        )

    if not np.all((sv >= 0.0) & (sv <= 1.0)):
        raise ValueError(
            f"support_values must satisfy 0 ≤ s_i ≤ 1, got {sv.tolist()}"
        )

    if float(np.sum(sv)) < 1.0 - _FEAS_TOL:
        raise ValueError(
            f"support values must satisfy sum(s) ≥ 1, (otherwise W_x is empty), got sum={float(np.sum(sv)):.6f} from {sv.tolist()}"
        )

    return sd, sv

def _support_values_feasible(support_values: np.ndarray) -> bool:
    """Check runtime feasibility of MDN-predicted support values.

    Kept permanently as defense in depth. SASP makes infeasible output
    mathematically impossible at the head, so a False here means a regression
    somewhere else (a refactor, a new head, a checkpoint mismatch) -- which is
    exactly why this check must not be deleted as unreachable.
    """
    sv = np.asarray(support_values, dtype=np.float64)
    if sv.ndim != 1 or sv.size < 2:
        return False
    if not np.all(np.isfinite(sv)):
        return False
    if not np.all((sv >= 0.0) & (sv <= 1.0)):
        return False
    if float(np.sum(sv)) < 1.0 - _FEAS_TOL:
        return False
    return True

def support_values_feasible(support_values: np.ndarray) -> bool:
    """Public entry point for the W_x feasibility test.

    Exists so consumers outside this package (the demo pipeline, the Streamlit
    app) share one definition of feasibility instead of re-deriving the
    expression inline or importing a private name.
    """
    return _support_values_feasible(support_values)

def _compute_wx_worst_case(delta_n: np.ndarray, support_directions: np.ndarray, support_values: np.ndarray,) -> float:
    """Compute h_{W_x}(-Δn) = max_{w ∈ W_x} w · (-Δn) exactly, for any M.

    Delegates to the shared exact greedy solver (O(M log M)) instead of
    enumerating vertices, whose count grows combinatorially with M.

    At M = 2 this returns exactly the same value as the legacy two-vertex
    enumeration: with s_i in [0, 1], W_x is the segment w_0 in [1 - s_1, s_0],
    whose endpoints are precisely the vertices [s_0, 1-s_0] and [1-s_1, s_1]
    that the old code built by hand.
    """
    _, sv = _validate_wx_geometry(support_directions, support_values)
    neg_delta_n = -np.asarray(delta_n, dtype=np.float64).reshape(-1)

    if neg_delta_n.size != sv.size:
        raise ValueError(
            f"delta_n length {neg_delta_n.size} != support length {sv.size}"
        )

    return greedy_support_function(neg_delta_n, sv)

# _build_wx_weight_set was removed here. It reconstructed the two vertices of a
# two-objective W_x, and existed only to hand a WeightSet to the admission gates.
# Certificate re-verification now evaluates the support function directly via
# _compute_wx_worst_case, which is exact at every M, so no vertex enumeration
# remains on any certification path.

class SkillLibrary:
    """ In-memory store of certified skills """

    def __init__(self, cert_store=None, save_path: str = "data/library.json") -> None:
        self.cert_store = cert_store
        self.save_path = save_path
        self._skills: Dict[str, SkillEntry] = {}
        # Counted rather than merely logged: a silent exclusion of every
        # MDN_WX skill was the original bug's worst property. Expected to stay
        # at zero under SASP; any nonzero reading is a regression signal.
        self.infeasible_support_events: int = 0

    def add_skill(
        self,
        skill_id: str,
        certificate: Certificate,
        policy: Callable,
        *,
        weight_region_type: str = FULL_SIMPLEX,
        certification_context: Optional[tuple[float, ...]] = None,
        mdn_alpha: Optional[tuple[float, ...]] = None,
        wx_support_directions: Optional[tuple[tuple[float, ...], ...]] = None,
        wx_support_values: Optional[tuple[float, ...]] = None,
    ) -> bool:
        """Add a certified skill to the library"""
        # 1. Identity & Store Check
        if skill_id != certificate.skill_id:
            return False
            
        if self.cert_store is not None:
            if not self.cert_store.contains(certificate.skill_id):
                return False

        # 2. Mathematical Check (The "Chain of Safety")
        # We re-verify the certificate's math at the library entry point.
        if certificate.gate_type == "CDS":
            gate = CDSGate()
        elif certificate.gate_type == "PDS":
            gate = PDSGate(epsilon=certificate.epsilon)
        else:
            return False

        delta_n_vec = np.asarray(certificate.delta_n, dtype=np.float64)

        # Re-verify the certificate's math at the library entry point.
        #
        # For MDN_WX we evaluate the region's support function directly rather
        # than materializing vertices, so this works at any M. The gates test
        # Δr + min_w(w·Δn) >= -ε, and min_w(w·Δn) = -h_{W_x}(-Δn), so the
        # condition below is identical -- at M = 2 it is exactly the value the
        # previous two-vertex WeightSet produced.
        if weight_region_type == MDN_WX:
            if wx_support_directions is None or wx_support_values is None:
                raise ValueError(
                    f"MDN_WX skill '{skill_id}' requires wx_support_directions and wx_support_values for certificate verification."
                )
            h_wx = _compute_wx_worst_case(
                delta_n_vec,
                np.asarray(wx_support_directions, dtype=np.float64),
                np.asarray(wx_support_values, dtype=np.float64),
            )
            epsilon = float(certificate.epsilon) if certificate.gate_type == "PDS" else 0.0
            if not float(certificate.delta_r) >= h_wx - epsilon:
                return False
        else:
            if not gate.admit(certificate.delta_r, delta_n_vec, None):
                return False

        entry = SkillEntry(
            skill_id=skill_id,
            gate_type=certificate.gate_type,
            certificate=certificate,
            policy=policy,
            weight_region_type=weight_region_type,
            certification_context=certification_context,
            mdn_alpha=mdn_alpha,
            wx_support_directions=wx_support_directions,
            wx_support_values=wx_support_values,
        )

        self._skills[skill_id] = entry
        return True

    def remove_skill(self, skill_id: str) -> bool:
        """ Remove a skill from the library. """
        return self._skills.pop(skill_id, None) is not None

    def get_skill(self, skill_id: str) -> Optional[SkillEntry]:
        """ Retrieve a single skill by its unique ID. """
        return self._skills.get(skill_id)

    def get_admitted_skills(self) -> List[SkillEntry]:
        """ Return all skills currently in the library. """
        return list(self._skills.values())

    def query_by_gate_type(self, gate_type: str) -> List[SkillEntry]:
        """ Filter skills by the gate that admitted them. """
        return [s for s in self._skills.values() if s.gate_type == gate_type]

    def query_by_weights(self, weights: List[float]) -> List[SkillEntry]:
        """ 
        Return skills that are admissible under a specific weight vector. 

        For a given weight vector w, a skill is admissible if:
            Δr + w^T Δn  ≥  -ε

        where ε = 0 for CDS skills (they pass for ALL w by definition)
        and ε = certificate.epsilon for PDS skills.
        
        """

        warnings.warn(
            "query_by_weights() is deprecated for mixed libraries.Use query_admissible() which handles both FULL_SIMPLEX and MDN_WX skills.",
            DeprecationWarning,
            stacklevel=2,
        )
        
        w = np.asarray(weights, dtype=np.float64)

        if not validate_simplex_weights(w):
            raise ValueError(
                f"weights must be a valid simplex vector (non-negative, sum to 1), "
                f"got {weights}"
            )

        admissible = []
        for entry in self._skills.values():
            if entry.weight_region_type == MDN_WX:
                continue
            if len(entry.delta_n) != w.size:
                continue

            if entry.gate_type == "CDS":
                # CDS skills pass for all weight vectors
                admissible.append(entry)
            else:
                # PDS: check  Δr + w^T Δn ≥ -ε  for this specific w
                delta_n = np.asarray(entry.delta_n, dtype=np.float64)
                score = entry.delta_r + float(np.dot(w, delta_n))
                if score >= -entry.epsilon:
                    admissible.append(entry)

        return admissible

    def query_admissible(
        self,
        current_weight: np.ndarray,
        support_directions: Optional[np.ndarray] = None,
        support_values: Optional[np.ndarray] = None,
    ) -> List[SkillEntry]:
        """Return skills admissible under the current MDN weight and W_x region."""
        w = np.asarray(current_weight, dtype=np.float64).reshape(-1)
        if not validate_simplex_weights(w):
            raise ValueError(
                f"current_weight must be a valid simplex vector, got {current_weight}"
            )

        wx_feasible = True
        if support_values is not None:
            if not _support_values_feasible(support_values):
                self.infeasible_support_events += 1
                logger.warning(
                    "Infeasible support values (event #%d, outside [0,1] or sum < 1): %s. "
                    "Excluding MDN_WX skills for this step. "
                    "This should be impossible under SASP - investigate immediately.",
                    self.infeasible_support_events,
                    np.asarray(support_values).tolist(),
                )
                wx_feasible = False

        admissible: list[SkillEntry] = []
        for entry in self._skills.values():
            if entry.weight_region_type == FULL_SIMPLEX:
                # Globally certified
                admissible.append(entry)

            elif entry.weight_region_type == MDN_WX:
                if not wx_feasible:
                    continue

                if support_directions is None or support_values is None:
                    raise ValueError(
                        f"MDN_WX skill '{entry.skill_id}' requires current support_directions and support_values for runtime admissibility, but they were not provided."
                    )
                sd = np.asarray(support_directions, dtype=np.float64)
                sv = np.asarray(support_values, dtype=np.float64)
                delta_n = np.asarray(entry.delta_n, dtype=np.float64)
                if delta_n.size != w.size:
                    continue

                h_wx = _compute_wx_worst_case(delta_n, sd, sv)

                if entry.gate_type == "CDS":
                    if entry.delta_r >= h_wx:
                        admissible.append(entry)
                elif entry.gate_type == "PDS":
                    if entry.delta_r >= h_wx - entry.epsilon:
                        admissible.append(entry)

        return admissible

    def count(self) -> int:
        """Return the number of skills in the library."""
        return len(self._skills)

    def register_policy(self, skill_id: str, policy: Callable) -> bool:
        """ Attach a policy to a skill that was loaded from disk. """
        entry = self._skills.get(skill_id)
        if entry is None:
            return False
        entry.policy = policy
        return True

    def save(self, path: Optional[str] = None) -> None:
        """ Save the library to a JSON file. """
        path = path or self.save_path
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "skill_count": self.count(),
            "skills": {
                sid: entry.to_dict()
                for sid, entry in self._skills.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[str] = None) -> None:
        """ Load a library from a JSON file """
        path = path or self.save_path
        filepath = Path(path)

        with open(filepath, "r") as f:
            data = json.load(f)

        self._skills = {
            sid: SkillEntry.from_dict(entry_data)
            for sid, entry_data in data["skills"].items()
        }
