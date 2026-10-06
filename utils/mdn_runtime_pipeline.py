"""Runtime certification pipeline with W_x tracking and certificate permanence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from baseline.improvement_calculator import ImprovementCalculator
from certification.cds_test import CDSGate
from certification.cvar_test import CVaRGate
from certification.pds_test import PDSGate
from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_support_trainer import MDNSupportTrainer
from utils.mdn_contracts import CandidateSkillRecord
from utils.mdn_selection import alpha_to_mean_weights, select_best_candidate
from utils.support_geometry import make_basis_query_directions, simplex_support_values
from utils.weight_set_store import WeightSet, WeightSetStore


@dataclass(frozen=True)
class CertificationResult:
    """Result of a single skill certification attempt."""
    skill_id: str
    is_certified: bool
    gate_type: str
    was_already_certified: bool
    admission_margin: float
    delta_r: float
    delta_n: tuple[float, ...]
    epsilon: float = 0.0
    weight_region_type: str = "FULL_SIMPLEX"
    certification_context: tuple[float, ...] | None = None
    mdn_alpha: tuple[float, ...] | None = None
    wx_support_directions: tuple[tuple[float, ...], ...] | None = None
    wx_support_values: tuple[float, ...] | None = None
    # CVaR tail level the result was issued under. Together with `epsilon` it
    # records the risk budgets in force, so a cached result can be rechecked
    # when either budget changes.
    cvar_confidence: float | None = None


def certification_result_to_certificate_kwargs(
    result: CertificationResult,
    *,
    timestamp: str,
    seed: int,
    gamma: float,
    baseline_id: str,
    environment: str,
    episode_length: int,
    version: str,
    epsilon: float | None = None,
) -> dict[str, object]:
    """Build Certificate constructor kwargs without dropping runtime audit fields."""
    if not result.is_certified:
        raise ValueError("Only certified runtime results can be converted to certificates")

    gate_type = result.gate_type.strip().upper()
    if gate_type == "CDS":
        certificate_epsilon = 0.0 if epsilon is None else float(epsilon)
        if certificate_epsilon != 0.0:
            raise ValueError("CDS certificates must use epsilon == 0.0")
    elif gate_type == "PDS":
        if epsilon is None:
            if result.epsilon is None:
                raise ValueError("PDS certificate conversion requires epsilon")
            certificate_epsilon = float(result.epsilon)
        else:
            certificate_epsilon = float(epsilon)
    else:
        raise ValueError(f"Unsupported certificate gate_type: {result.gate_type!r}")

    return {
        "skill_id": result.skill_id,
        "gate_type": gate_type,
        "delta_r": float(result.delta_r),
        "delta_n": tuple(float(v) for v in result.delta_n),
        "admission_margin": float(result.admission_margin),
        "epsilon": certificate_epsilon,
        "timestamp": timestamp,
        "seed": seed,
        "gamma": gamma,
        "baseline_id": baseline_id,
        "environment": environment,
        "episode_length": episode_length,
        "version": version,
        "weight_region_type": result.weight_region_type,
        "certification_context": result.certification_context,
        "mdn_alpha": result.mdn_alpha,
        "wx_support_directions": result.wx_support_directions,
        "wx_support_values": result.wx_support_values,
    }


@dataclass
class RuntimePipelineConfig:
    gate_type: str = "CDS"
    cvar_confidence: float = 0.1
    cvar_samples: int = 1000
    pds_epsilon: float = 0.1
    use_cvar: bool = False
    require_cds_or_cvar: bool = True
    train_support_after_certify: bool = True
    store_path: Optional[str] = None


class RuntimeCertificationPipeline:
    """Full runtime certification pipeline with W_x tracking and certificate permanence.

    This pipeline:
    1. Checks if a skill is already certified (permanence). The stored
       certificate is kept as the historical record; if the risk budgets (PDS
       epsilon, CVaR tail level) differ from the ones it was issued under,
       eligibility is rechecked under the current budgets.
    2. Runs gate tests against actual W_x (not simplex fallback)
    3. Records certified weights into W_x store
    4. Trains the support head after certification
    5. Persists W_x store to disk
    """

    def __init__(
        self,
        model: MotiveDecompositionNetwork,
        weight_store: WeightSetStore,
        support_trainer: Optional[MDNSupportTrainer] = None,
        config: Optional[RuntimePipelineConfig] = None,
    ) -> None:
        self.model = model
        self.weight_store = weight_store
        self.support_trainer = support_trainer
        self.config = config or RuntimePipelineConfig()
        self._certified_skills: dict[tuple[tuple[float, ...], str], CertificationResult] = {}
        # The W_x region each stored result was gated against, snapshotted at
        # certification time. A budget recheck must judge the same claim over
        # the same region: W_x grows from the weights of certified skills, so
        # rechecking against the CURRENT W_x would judge a skill partly by the
        # weights recorded when it was certified.
        self._certified_regions: dict[
            tuple[tuple[float, ...], str], Optional[WeightSet]
        ] = {}

        if self.config.store_path is not None:
            self._load_store()

    def certify_skill(
        self,
        *,
        context: np.ndarray,
        skill_id: str,
        skill_payoff: float,
        skill_motives: np.ndarray,
        baseline_stats: dict[str, Any],
        weights_used: Optional[np.ndarray] = None,
        epsilon: Optional[float] = None,
        cvar_confidence: Optional[float] = None,
    ) -> CertificationResult:
        """Run the full certification pipeline for a single skill.

        Returns a CertificationResult with is_certified=True if the skill passes
        gate tests (or was already certified).

        `epsilon` and `cvar_confidence` allow a caller (e.g. a motivational
        governor) to supply per-step risk budgets. Both default to None, which
        falls back to the values on `self.config`.

        A result already stored for this context and skill is returned as-is
        while the budgets are unchanged. If they differ from the ones it was
        issued under, the stored delta is rechecked under the current budgets
        and a copy carrying the current verdict is returned; the stored result
        itself is never modified.
        """
        context_key = self.weight_store._context_key(context)
        permanence_key = (context_key, skill_id)

        if permanence_key in self._certified_skills:
            stored = self._certified_skills[permanence_key]
            current_epsilon = self._effective_epsilon(epsilon)
            current_confidence = self._effective_cvar_confidence(cvar_confidence)
            if not self._budgets_changed(
                stored, epsilon=current_epsilon, cvar_confidence=current_confidence
            ):
                return stored
            is_eligible, margin = self._recheck_eligibility(
                stored,
                context=context,
                weight_set=self._certified_regions.get(permanence_key),
                epsilon=current_epsilon,
                cvar_confidence=current_confidence,
            )
            return replace(
                stored,
                is_certified=is_eligible,
                admission_margin=margin,
                epsilon=current_epsilon,
                cvar_confidence=current_confidence,
            )

        calculator = ImprovementCalculator(baseline_stats)
        delta_r, delta_n = calculator.compute_improvements(
            skill_payoff=skill_payoff,
            skill_motives=skill_motives,
        )

        weight_set = self.weight_store.get_weight_set(context)
        effective_epsilon = self._effective_epsilon(epsilon)
        is_certified = self._run_gate_tests(
            delta_r,
            delta_n,
            context,
            weight_set,
            epsilon=effective_epsilon,
            cvar_confidence=cvar_confidence,
        )
        # Before observe_certified_weight below can grow this context's W_x.
        region = _snapshot_region(weight_set)
        audit_fields = self._build_audit_fields(context, weight_set)

        if is_certified and weights_used is not None:
            self.weight_store.observe_certified_weight(context, weights_used)
            if self.config.train_support_after_certify and self.support_trainer is not None:
                self.support_trainer.training_step()

        result = CertificationResult(
            skill_id=skill_id,
            is_certified=is_certified,
            gate_type=self.config.gate_type,
            was_already_certified=False,
            admission_margin=self._compute_admission_margin(
                delta_r,
                delta_n,
                context,
                weight_set,
                epsilon=effective_epsilon,
                cvar_confidence=cvar_confidence,
            ),
            delta_r=float(delta_r),
            delta_n=tuple(float(v) for v in delta_n),
            epsilon=effective_epsilon,
            cvar_confidence=self._effective_cvar_confidence(cvar_confidence),
            **audit_fields,
        )

        self._certified_skills[permanence_key] = result
        self._certified_regions[permanence_key] = region
        return result

    def certify_candidate_skills(
        self,
        *,
        context: np.ndarray,
        candidate_skills: list[CandidateSkillRecord],
        baseline_stats: dict[str, Any],
        weights_used: Optional[np.ndarray] = None,
        cvar_confidence: Optional[float] = None,
    ) -> list[CandidateSkillRecord]:
        """Certify a batch of candidate skills and return updated records.

        For each candidate, checks permanence, runs gate tests against W_x,
        and records certified weights.

        A certificate already stored for this context and skill stays as the
        historical record. While the candidate's PDS epsilon and this call's
        CVaR tail level match the ones it was issued under, its verdict is
        reused. If either differs, the stored delta is rechecked under the
        current budgets, and the returned record carries the CURRENT verdict,
        margin and epsilon -- so a skill certified under a looser budget is
        not admitted once the budget tightens past its margin.
        """
        context_key = self.weight_store._context_key(context)
        weight_set = self.weight_store.get_weight_set(context)
        current_confidence = self._effective_cvar_confidence(cvar_confidence)
        updated_records = []

        for candidate in candidate_skills:
            permanence_key = (context_key, candidate.skill_id)

            if permanence_key in self._certified_skills:
                stored = self._certified_skills[permanence_key]
                current_epsilon = self._candidate_epsilon(candidate)
                if self._budgets_changed(
                    stored, epsilon=current_epsilon, cvar_confidence=current_confidence
                ):
                    is_eligible, margin = self._recheck_eligibility(
                        stored,
                        context=context,
                        weight_set=self._certified_regions.get(permanence_key),
                        epsilon=current_epsilon,
                        cvar_confidence=current_confidence,
                    )
                    epsilon_now = current_epsilon
                else:
                    is_eligible = stored.is_certified
                    margin = stored.admission_margin
                    epsilon_now = stored.epsilon
                if is_eligible and weights_used is not None:
                    self._observe_certified_weight(context, weights_used)
                updated_records.append(
                    CandidateSkillRecord(
                        skill_id=candidate.skill_id,
                        delta_r=stored.delta_r,
                        delta_n=stored.delta_n,
                        is_certified=is_eligible,
                        gate_type=stored.gate_type,
                        metadata=dict(candidate.metadata),
                        admission_margin=margin,
                        epsilon=epsilon_now,
                        baseline_id=candidate.baseline_id,
                    )
                )
                continue

            effective_epsilon = self._candidate_epsilon(candidate)

            is_certified = self._run_gate_tests(
                candidate.delta_r,
                np.array(candidate.delta_n),
                context,
                weight_set,
                epsilon=effective_epsilon,
                cvar_confidence=cvar_confidence,
            )
            # Before _observe_certified_weight below can grow this context's W_x.
            region = _snapshot_region(weight_set)
            audit_fields = self._build_audit_fields(
                context,
                weight_set,
            )

            if is_certified:
                if weights_used is not None:
                    self._observe_certified_weight(context, weights_used)
                result = CertificationResult(
                    skill_id=candidate.skill_id,
                    is_certified=True,
                    gate_type=self.config.gate_type,
                    was_already_certified=False,
                    admission_margin=self._compute_admission_margin(
                        candidate.delta_r,
                        np.array(candidate.delta_n),
                        context,
                        weight_set,
                        epsilon=effective_epsilon,
                        cvar_confidence=cvar_confidence,
                    ),
                    delta_r=candidate.delta_r,
                    delta_n=candidate.delta_n,
                    epsilon=effective_epsilon,
                    cvar_confidence=current_confidence,
                    **audit_fields,
                )
                self._certified_skills[permanence_key] = result
                self._certified_regions[permanence_key] = region
                updated_records.append(
                    CandidateSkillRecord(
                        skill_id=candidate.skill_id,
                        delta_r=candidate.delta_r,
                        delta_n=candidate.delta_n,
                        is_certified=True,
                        gate_type=result.gate_type,
                        metadata=dict(candidate.metadata),
                        admission_margin=result.admission_margin,
                        epsilon=result.epsilon,
                        baseline_id=candidate.baseline_id,
                    )
                )
                continue

            updated_records.append(candidate)

        return updated_records

    def get_certification_result(
        self,
        *,
        context: np.ndarray,
        skill_id: str,
    ) -> CertificationResult | None:
        """Return the stored certification result for a context/skill pair, if any."""
        context_key = self.weight_store._context_key(context)
        return self._certified_skills.get((context_key, skill_id))

    def _candidate_epsilon(self, candidate: CandidateSkillRecord) -> float:
        if self.config.gate_type.upper() == "PDS":
            return self._effective_epsilon(candidate.epsilon)
        return 0.0

    def _effective_epsilon(self, epsilon: float | None) -> float:
        if self.config.gate_type.upper() == "PDS":
            return self.config.pds_epsilon if epsilon is None else float(epsilon)
        return 0.0

    def _effective_cvar_confidence(self, cvar_confidence: float | None) -> float:
        """Resolve the CVaR tail level for this call.

        Mirrors `_effective_epsilon`: a per-call value wins, otherwise the
        configured default applies. This is the scalar tail-mass argument to
        `CVaRGate(confidence=...)`, NOT the MDN's Dirichlet concentration
        vector `mdn_alpha`, which is a separate positive array of length m.
        """
        if cvar_confidence is None:
            return self.config.cvar_confidence
        return float(cvar_confidence)

    def _uses_cvar(self) -> bool:
        """Whether the CVaR tail level can affect this pipeline's verdicts."""
        return self.config.gate_type.upper() == "CVAR" or bool(self.config.use_cvar)

    def _budgets_changed(
        self,
        stored: CertificationResult,
        *,
        epsilon: float,
        cvar_confidence: float,
    ) -> bool:
        """Whether the current risk budgets differ from a stored result's.

        `epsilon` and `cvar_confidence` are effective (already resolved)
        values. The CVaR tail level only counts when a CVaR gate is in use.
        """
        if stored.epsilon != epsilon:
            return True
        return (
            self._uses_cvar()
            and stored.cvar_confidence is not None
            and stored.cvar_confidence != cvar_confidence
        )

    def _recheck_eligibility(
        self,
        stored: CertificationResult,
        *,
        context: np.ndarray,
        weight_set: Optional[WeightSet],
        epsilon: float,
        cvar_confidence: float,
    ) -> tuple[bool, float]:
        """Re-run the gates on a stored result's delta under current budgets.

        `weight_set` is the region the result was originally gated against
        (see `_certified_regions`), so only the risk budgets differ from the
        original check. Returns (is_eligible, admission_margin). Read-only:
        the stored result stays as the historical certificate.
        """
        delta_n = np.array(stored.delta_n)
        is_eligible = self._run_gate_tests(
            stored.delta_r,
            delta_n,
            context,
            weight_set,
            epsilon=epsilon,
            cvar_confidence=cvar_confidence,
        )
        margin = self._compute_admission_margin(
            stored.delta_r,
            delta_n,
            context,
            weight_set,
            epsilon=epsilon,
            cvar_confidence=cvar_confidence,
        )
        return bool(is_eligible), float(margin)

    def _observe_certified_weight(self, context: np.ndarray, weights_used: np.ndarray) -> None:
        self.weight_store.observe_certified_weight(context, weights_used)
        if self.config.train_support_after_certify and self.support_trainer is not None:
            self.support_trainer.training_step()

    def select_and_certify(
        self,
        *,
        context: np.ndarray,
        candidate_skills: list[CandidateSkillRecord],
        baseline_stats: dict[str, Any],
    ) -> tuple[str, np.ndarray, list[CandidateSkillRecord]]:
        """Select best candidate using MDN alpha, then certify all candidates.

        Returns (selected_skill_id, weights_used, updated_candidate_records).
        """
        context_tensor = __import__("torch").tensor(context, dtype=__import__("torch").float32, device=self.model.device if hasattr(self.model, "device") else "cpu")
        with __import__("torch").no_grad():
            alpha, support_values = self.model.forward_inference(context_tensor)
        alpha_np = alpha.detach().cpu().numpy()
        weights_used = alpha_to_mean_weights(alpha_np)

        updated_records = self.certify_candidate_skills(
            context=context,
            candidate_skills=candidate_skills,
            baseline_stats=baseline_stats,
            weights_used=weights_used,
        )

        selected_skill_id, _ = select_best_candidate(updated_records, weights_used)
        return selected_skill_id, weights_used, updated_records

    def get_support_values(self, context: np.ndarray) -> np.ndarray:
        """Get W_x support function values for a context."""
        return self.weight_store.get_support_values(context)

    def observe_certified_weight(self, context: np.ndarray, weights_used: np.ndarray) -> None:
        """Record a selected weight vector for a certified or stored skill reuse."""
        self._observe_certified_weight(context, weights_used)

    def save_store(self, path: Optional[str] = None) -> str:
        """Persist W_x store to disk."""
        save_path = path or self.config.store_path or "data/weight_store.json"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.weight_store.save(save_path)
        return save_path

    def _load_store(self) -> None:
        """Load W_x store from disk if it exists and is non-empty."""
        if self.config.store_path and Path(self.config.store_path).exists():
            content = Path(self.config.store_path).read_text(encoding="utf-8")
            if content.strip():
                loaded = WeightSetStore.load(self.config.store_path)
                self.weight_store._store = loaded._store

    def _run_gate_tests(
        self,
        delta_r: float,
        delta_n: np.ndarray,
        context: np.ndarray,
        weight_set: Optional[WeightSet],
        *,
        epsilon: float | None = None,
        cvar_confidence: float | None = None,
    ) -> bool:
        """Run configured gate tests against W_x."""
        gate_type = self.config.gate_type.upper()
        effective_confidence = self._effective_cvar_confidence(cvar_confidence)

        if gate_type == "CDS":
            gate = CDSGate()
            result = gate.admit(delta_r, delta_n, weight_set=weight_set)
        elif gate_type == "PDS":
            gate = PDSGate(epsilon=self.config.pds_epsilon if epsilon is None else float(epsilon))
            result = gate.admit(delta_r, delta_n, weight_set=weight_set)
        elif gate_type == "CVAR":
            with __import__("torch").no_grad():
                context_tensor = __import__("torch").tensor(context, dtype=__import__("torch").float32, device=self.model.device if hasattr(self.model, "device") else "cpu")
                alpha, _ = self.model.forward_inference(context_tensor)
            alpha_np = alpha.detach().cpu().numpy()
            gate = CVaRGate(confidence=effective_confidence, n_samples=self.config.cvar_samples)
            result = gate.admit(delta_r, delta_n, mdn_alpha=alpha_np)
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")

        if self.config.use_cvar and gate_type != "CVAR":
            with __import__("torch").no_grad():
                context_tensor = __import__("torch").tensor(context, dtype=__import__("torch").float32, device=self.model.device if hasattr(self.model, "device") else "cpu")
                alpha, _ = self.model.forward_inference(context_tensor)
            alpha_np = alpha.detach().cpu().numpy()
            cvar_gate = CVaRGate(confidence=effective_confidence, n_samples=self.config.cvar_samples)
            cvar_result = cvar_gate.admit(delta_r, delta_n, mdn_alpha=alpha_np)

            if self.config.require_cds_or_cvar:
                return result or cvar_result
            return cvar_result

        return result

    def _build_audit_fields(
        self,
        context: np.ndarray,
        weight_set: Optional[WeightSet],
    ) -> dict[str, object]:
        """Return certificate audit fields for the active certification region."""
        gate_type = self.config.gate_type.upper()
        uses_mdn_distribution = gate_type == "CVAR" or bool(self.config.use_cvar)
        uses_contextual_weight_set = weight_set is not None and not weight_set.is_empty()

        if not uses_mdn_distribution and not uses_contextual_weight_set:
            return {
                "weight_region_type": "FULL_SIMPLEX",
                "certification_context": None,
                "mdn_alpha": None,
                "wx_support_directions": None,
                "wx_support_values": None,
            }

        torch = __import__("torch")
        context_array = np.asarray(context, dtype=np.float32).reshape(-1)
        context_tensor = torch.tensor(
            context_array,
            dtype=torch.float32,
            device=self.model.device if hasattr(self.model, "device") else "cpu",
        )
        with torch.no_grad():
            alpha, _ = self.model.forward_inference(context_tensor)

        support_directions, support_values = _wx_support_evidence(
            self.weight_store.num_objectives,
            weight_set,
        )

        return {
            "weight_region_type": "MDN_WX",
            "certification_context": tuple(float(v) for v in context_array),
            "mdn_alpha": tuple(float(v) for v in alpha.detach().cpu().numpy().reshape(-1)),
            "wx_support_directions": tuple(
                tuple(float(v) for v in row)
                for row in support_directions
            ),
            "wx_support_values": tuple(float(v) for v in support_values),
        }

    def _compute_admission_margin(
        self,
        delta_r: float,
        delta_n: np.ndarray,
        context: np.ndarray,
        weight_set: Optional[WeightSet],
        *,
        epsilon: float | None = None,
        cvar_confidence: float | None = None,
    ) -> float:
        gate_type = self.config.gate_type.upper()

        if gate_type == "CDS":
            return CDSGate().get_admission_margin(delta_r, delta_n, weight_set=weight_set)
        if gate_type == "PDS":
            return PDSGate(
                epsilon=self.config.pds_epsilon if epsilon is None else float(epsilon)
            ).get_admission_margin(
                delta_r,
                delta_n,
                weight_set=weight_set,
            )
        if gate_type == "CVAR":
            with __import__("torch").no_grad():
                context_tensor = __import__("torch").tensor(
                    context,
                    dtype=__import__("torch").float32,
                    device=self.model.device if hasattr(self.model, "device") else "cpu",
                )
                alpha, _ = self.model.forward_inference(context_tensor)
            alpha_np = alpha.detach().cpu().numpy()
            return CVaRGate(
                confidence=self._effective_cvar_confidence(cvar_confidence),
                n_samples=self.config.cvar_samples,
            ).get_cvar(delta_r, delta_n, mdn_alpha=alpha_np)

        raise ValueError(f"Unknown gate_type: {gate_type}")


def _snapshot_region(weight_set: Optional[WeightSet]) -> Optional[WeightSet]:
    """Copy a W_x region so later observations cannot change it."""
    if weight_set is None or weight_set.is_empty():
        return None
    return WeightSet(vertices=[np.array(v, copy=True) for v in weight_set.vertices])


def _wx_support_evidence(
    num_objectives: int,
    weight_set: Optional[WeightSet],
) -> tuple[np.ndarray, np.ndarray]:
    """Return standard-basis W_x support geometry for SkillLibrary replay."""
    support_directions = make_basis_query_directions(num_objectives)
    if weight_set is None or weight_set.is_empty():
        support_values = simplex_support_values(support_directions)
    else:
        support_values = weight_set.get_support_values(support_directions)
    return support_directions, support_values.astype(np.float32)
