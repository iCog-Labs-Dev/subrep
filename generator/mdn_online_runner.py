"""Runtime MDN orchestration for certification, selection, logging, and updates."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_auxiliary_trainer import AuxiliaryTrainingRecord, MDNAuxiliaryTrainer
from generator.mdn_runtime_selector import MDNRuntimeSelector
from generator.mdn_trainer import MDNTrainer
from utils.mdn_contracts import CandidateSkillRecord, MDNDecisionRecord
from utils.mdn_record_builder import build_candidate_skill_records
from utils.mdn_runtime_pipeline import RuntimeCertificationPipeline
from utils.weight_set_store import WeightSetStore


@dataclass(frozen=True)
class StepResult:
    """Result of one online MDN step."""

    selected_skill_id: str | None
    behavior_probability: float | None
    weights_used: np.ndarray | None
    decision_record: MDNDecisionRecord | None
    policy_metrics: dict[str, float] | None
    certified_skill_ids: tuple[str, ...]
    auxiliary_metrics: dict[str, float] | None = None


class MDNOnlineRunner:
    """Wire the MDN runtime pieces into one decision/update step."""

    def __init__(
        self,
        *,
        model: MotiveDecompositionNetwork,
        certification_pipeline: RuntimeCertificationPipeline,
        policy_trainer: MDNTrainer,
        baseline_stats: dict[str, Any],
        checkpoint_path: str = "models/mdn_policy_best.pth",
        store_path: Optional[str] = None,
        save_every_n_steps: int = 10,
        auxiliary_trainer: Optional[MDNAuxiliaryTrainer] = None,
        device: Optional[str] = None,
        certificate_store: Optional[Any] = None,
        certificate_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if save_every_n_steps <= 0:
            raise ValueError("save_every_n_steps must be positive")

        self.model = model
        self.certification_pipeline = certification_pipeline
        self.policy_trainer = policy_trainer
        self.auxiliary_trainer = auxiliary_trainer
        self.baseline_stats = dict(baseline_stats)
        self.checkpoint_path = checkpoint_path
        self.store_path = store_path or certification_pipeline.config.store_path
        self.save_every_n_steps = int(save_every_n_steps)
        self.selector = MDNRuntimeSelector(model=model, device=device or str(policy_trainer.device))
        self.certificate_store = certificate_store
        self.certificate_metadata: dict[str, Any] = dict(certificate_metadata) if certificate_metadata else {}
        self._step_count = 0

    def step(
        self,
        *,
        observation: np.ndarray,
        candidate_skill_payloads: list[dict[str, Any]],
        execute_skill: Callable[[str], dict[str, Any]],
        payoff_weight: Optional[float] = None,
    ) -> StepResult:
        """Run one full runtime step from observation to policy update."""
        if not candidate_skill_payloads:
            raise ValueError("candidate_skill_payloads must not be empty")

        context = np.asarray(observation, dtype=np.float32).reshape(-1)
        candidate_skills = list(
            build_candidate_skill_records(
                skill_outcomes=candidate_skill_payloads,
                baseline_stats=self.baseline_stats,
                weight_store=self.certification_pipeline.weight_store,
            )
        )
        certified_candidates = self.certification_pipeline.certify_candidate_skills(
            context=context,
            candidate_skills=candidate_skills,
            baseline_stats=self.baseline_stats,
        )
        certified_skill_ids = tuple(
            candidate.skill_id for candidate in certified_candidates if candidate.is_certified
        )
        if self.certificate_store is not None:
            self._write_certified_to_store(certified_candidates)
        if not certified_skill_ids:
            self._step_count += 1
            self._maybe_save()
            return StepResult(
                selected_skill_id=None,
                behavior_probability=None,
                weights_used=None,
                decision_record=None,
                policy_metrics=None,
                certified_skill_ids=certified_skill_ids,
            )

        selection = self.selector.select(context, certified_candidates)
        outcome = dict(execute_skill(selection.selected_skill_id))
        if "actual_payoff" not in outcome or "actual_motives" not in outcome:
            raise ValueError("execute_skill must return 'actual_payoff' and 'actual_motives'")

        record = selection.build_decision_record(
            actual_payoff=float(outcome["actual_payoff"]),
            actual_motives=outcome["actual_motives"],
            utility=outcome.get("utility"),
            payoff_weight=self.policy_trainer.config.payoff_weight if payoff_weight is None else float(payoff_weight),
        )
        policy_metrics = self.policy_trainer.training_step(record)

        auxiliary_metrics: dict[str, float] | None = None
        if self.auxiliary_trainer is not None:
            aux_record = self._build_aux_record(
                context=context,
                certified_candidates=certified_candidates,
                selected_skill_id=selection.selected_skill_id,
                behavior_probability=selection.behavior_probability,
                actual_motives=tuple(float(v) for v in outcome["actual_motives"]),
            )
            auxiliary_metrics = self.auxiliary_trainer.online_step(aux_record)

        self.certification_pipeline.certify_candidate_skills(
            context=context,
            candidate_skills=certified_candidates,
            baseline_stats=self.baseline_stats,
            weights_used=selection.weights_used,
        )

        self._step_count += 1
        self._maybe_save()
        return StepResult(
            selected_skill_id=selection.selected_skill_id,
            behavior_probability=selection.behavior_probability,
            weights_used=selection.weights_used,
            decision_record=record,
            policy_metrics=policy_metrics,
            certified_skill_ids=certified_skill_ids,
            auxiliary_metrics=auxiliary_metrics,
        )

    def _write_certified_to_store(self, candidates: list[CandidateSkillRecord]) -> None:
        """Write newly certified candidates to the MeTTa CertificateStore.

        The store deduplicates by skill_id so calling this on every step is safe.
        Candidates with missing or negative admission_margin are skipped — this
        guards against malformed records reaching the certificate schema validator.
        Certificate and hyperon imports are deferred so the runner can be used
        without hyperon installed when no certificate_store is provided.
        """
        from certification.certificate_schema import Certificate  # deferred — requires no hyperon

        timestamp = datetime.now().isoformat()
        meta = self.certificate_metadata
        for candidate in candidates:
            if not candidate.is_certified:
                continue
            if candidate.admission_margin is None or candidate.admission_margin < 0.0:
                continue
            try:
                cert = Certificate(
                    skill_id=candidate.skill_id,
                    gate_type=candidate.gate_type,
                    delta_r=candidate.delta_r,
                    delta_n=candidate.delta_n,
                    admission_margin=candidate.admission_margin,
                    epsilon=candidate.epsilon if candidate.epsilon is not None else 0.0,
                    timestamp=timestamp,
                    seed=int(meta.get("seed", 0)),
                    gamma=float(meta.get("gamma", 1.0)),
                    baseline_id=candidate.baseline_id or "default",
                    environment=str(meta.get("environment", "mo-lunar-lander-v3")),
                    episode_length=int(meta.get("episode_length", 1)),
                    version=str(meta.get("version", "1.0")),
                )
                self.certificate_store.add(cert)
            except (ValueError, TypeError):
                pass

    def _build_aux_record(
        self,
        *,
        context: np.ndarray,
        certified_candidates: list[CandidateSkillRecord],
        selected_skill_id: str,
        behavior_probability: float,
        actual_motives: tuple[float, ...],
    ) -> AuxiliaryTrainingRecord:
        """Build an AuxiliaryTrainingRecord from one online step for gate/motive head training.
        """
        certified_only = [c for c in certified_candidates if c.is_certified]
        selected_idx = next(
            (i for i, c in enumerate(certified_only) if c.skill_id == selected_skill_id), 0
        )
        skill_id_int = zlib.crc32(selected_skill_id.encode()) % self.model.num_skills
        return AuxiliaryTrainingRecord(
            context=tuple(float(v) for v in context),
            skill_id=skill_id_int,
            accept_label=1.0,
            q_target=actual_motives,
            behavior_probability=behavior_probability,
            candidate_delta_r=tuple(c.delta_r for c in certified_only),
            candidate_delta_n=tuple(tuple(c.delta_n) for c in certified_only),
            selected_candidate_index=selected_idx,
        )

    def save(self) -> None:
        """Persist policy checkpoint and weight store."""
        self.policy_trainer.save_checkpoint(self.checkpoint_path)
        if self.store_path is not None:
            self.certification_pipeline.save_store(self.store_path)

    def _maybe_save(self) -> None:
        if self._step_count % self.save_every_n_steps == 0:
            self.save()

    @classmethod
    def load(
        cls,
        *,
        model: MotiveDecompositionNetwork,
        certification_pipeline: RuntimeCertificationPipeline,
        policy_trainer: MDNTrainer,
        baseline_stats: dict[str, Any],
        checkpoint_path: str = "models/mdn_policy_best.pth",
        store_path: Optional[str] = None,
        save_every_n_steps: int = 10,
        auxiliary_trainer: Optional[MDNAuxiliaryTrainer] = None,
        device: Optional[str] = None,
        certificate_store: Optional[Any] = None,
        certificate_metadata: Optional[dict[str, Any]] = None,
    ) -> "MDNOnlineRunner":
        """Load persisted runtime state into a new online runner."""
        runner = cls(
            model=model,
            certification_pipeline=certification_pipeline,
            policy_trainer=policy_trainer,
            baseline_stats=baseline_stats,
            checkpoint_path=checkpoint_path,
            store_path=store_path,
            save_every_n_steps=save_every_n_steps,
            auxiliary_trainer=auxiliary_trainer,
            device=device,
            certificate_store=certificate_store,
            certificate_metadata=certificate_metadata,
        )
        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.exists():
            restored = MDNTrainer.from_checkpoint(
                checkpoint_file,
                model=model,
                device=device or str(policy_trainer.device),
            )
            runner.policy_trainer = restored
            runner.selector = MDNRuntimeSelector(model=restored.model, device=device or str(restored.device))
            runner.model = restored.model

        store_file = Path(runner.store_path) if runner.store_path is not None else None
        if store_file is not None and store_file.exists():
            loaded_store = WeightSetStore.load(store_file)
            runner.certification_pipeline.weight_store._store = loaded_store._store

        return runner
