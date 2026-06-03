"""Runtime certification pipeline with W_x tracking and certificate permanence."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    1. Checks if a skill is already certified (permanence)
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
    ) -> CertificationResult:
        """Run the full certification pipeline for a single skill.

        Returns a CertificationResult with is_certified=True if the skill passes
        gate tests (or was already certified).
        """
        context_key = self.weight_store._context_key(context)
        permanence_key = (context_key, skill_id)

        if permanence_key in self._certified_skills:
            return self._certified_skills[permanence_key]

        calculator = ImprovementCalculator(baseline_stats)
        delta_r, delta_n = calculator.compute_improvements(
            skill_payoff=skill_payoff,
            skill_motives=skill_motives,
        )

        weight_set = self.weight_store.get_weight_set(context)
        is_certified = self._run_gate_tests(delta_r, delta_n, context, weight_set)

        if is_certified and weights_used is not None:
            self.weight_store.observe_certified_weight(context, weights_used)
            if self.config.train_support_after_certify and self.support_trainer is not None:
                self.support_trainer.training_step()

        result = CertificationResult(
            skill_id=skill_id,
            is_certified=is_certified,
            gate_type=self.config.gate_type,
            was_already_certified=False,
            admission_margin=self._compute_admission_margin(delta_r, delta_n, context, weight_set),
            delta_r=float(delta_r),
            delta_n=tuple(float(v) for v in delta_n),
        )

        self._certified_skills[permanence_key] = result
        return result

    def certify_candidate_skills(
        self,
        *,
        context: np.ndarray,
        candidate_skills: list[CandidateSkillRecord],
        baseline_stats: dict[str, Any],
        weights_used: Optional[np.ndarray] = None,
    ) -> list[CandidateSkillRecord]:
        """Certify a batch of candidate skills and return updated records.

        For each candidate, checks permanence, runs gate tests against W_x,
        and records certified weights.
        """
        context_key = self.weight_store._context_key(context)
        weight_set = self.weight_store.get_weight_set(context)
        updated_records = []

        for candidate in candidate_skills:
            permanence_key = (context_key, candidate.skill_id)

            if permanence_key in self._certified_skills:
                updated_records.append(candidate)
                continue

            is_certified = self._run_gate_tests(
                candidate.delta_r,
                np.array(candidate.delta_n),
                context,
                weight_set,
            )

            if is_certified and weights_used is not None:
                self.weight_store.observe_certified_weight(context, weights_used)
                if self.config.train_support_after_certify and self.support_trainer is not None:
                    self.support_trainer.training_step()

                result = CertificationResult(
                    skill_id=candidate.skill_id,
                    is_certified=True,
                    gate_type=self.config.gate_type,
                    was_already_certified=False,
                    admission_margin=candidate.admission_margin,
                    delta_r=candidate.delta_r,
                    delta_n=candidate.delta_n,
                )
                self._certified_skills[permanence_key] = result

            updated_records.append(candidate)

        return updated_records

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
    ) -> bool:
        """Run configured gate tests against W_x."""
        gate_type = self.config.gate_type.upper()

        if gate_type == "CDS":
            gate = CDSGate()
            result = gate.admit(delta_r, delta_n, weight_set=weight_set)
        elif gate_type == "PDS":
            gate = PDSGate(epsilon=self.config.pds_epsilon)
            result = gate.admit(delta_r, delta_n, weight_set=weight_set)
        elif gate_type == "CVAR":
            with __import__("torch").no_grad():
                context_tensor = __import__("torch").tensor(context, dtype=__import__("torch").float32, device=self.model.device if hasattr(self.model, "device") else "cpu")
                alpha, _ = self.model.forward_inference(context_tensor)
            alpha_np = alpha.detach().cpu().numpy()
            gate = CVaRGate(confidence=self.config.cvar_confidence, n_samples=self.config.cvar_samples)
            result = gate.admit(delta_r, delta_n, mdn_alpha=alpha_np)
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")

        if self.config.use_cvar and gate_type != "CVAR":
            with __import__("torch").no_grad():
                context_tensor = __import__("torch").tensor(context, dtype=__import__("torch").float32, device=self.model.device if hasattr(self.model, "device") else "cpu")
                alpha, _ = self.model.forward_inference(context_tensor)
            alpha_np = alpha.detach().cpu().numpy()
            cvar_gate = CVaRGate(confidence=self.config.cvar_confidence, n_samples=self.config.cvar_samples)
            cvar_result = cvar_gate.admit(delta_r, delta_n, mdn_alpha=alpha_np)

            if self.config.require_cds_or_cvar:
                return result or cvar_result
            return cvar_result

        return result

    def _compute_admission_margin(
        self,
        delta_r: float,
        delta_n: np.ndarray,
        context: np.ndarray,
        weight_set: Optional[WeightSet],
    ) -> float:
        gate_type = self.config.gate_type.upper()

        if gate_type == "CDS":
            return CDSGate().get_admission_margin(delta_r, delta_n, weight_set=weight_set)
        if gate_type == "PDS":
            return PDSGate(epsilon=self.config.pds_epsilon).get_admission_margin(
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
                confidence=self.config.cvar_confidence,
                n_samples=self.config.cvar_samples,
            ).get_cvar(delta_r, delta_n, mdn_alpha=alpha_np)

        raise ValueError(f"Unknown gate_type: {gate_type}")
