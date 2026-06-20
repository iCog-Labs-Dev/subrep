"""Runtime MDN orchestration for certification, selection, logging, and updates."""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from certification.certificate_schema import Certificate
from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_auxiliary_replay import (
    AuxiliaryReplayBuffer,
    AuxiliaryReplayEntry,
    replay_entry_to_auxiliary_records,
    replay_entry_to_selected_auxiliary_record,
)
from generator.mdn_auxiliary_trainer import AuxiliaryTrainingRecord, MDNAuxiliaryTrainer
from generator.mdn_runtime_selector import MDNRuntimeSelector
from generator.mdn_trainer import MDNTrainer
from library.skill_library import SkillLibrary
from utils.mdn_contracts import CandidateSkillRecord, MDNDecisionRecord
from utils.mdn_record_builder import build_candidate_skill_records
from utils.mdn_runtime_pipeline import RuntimeCertificationPipeline, certification_result_to_certificate_kwargs
from utils.weight_set_store import WeightSetStore


logger = logging.getLogger(__name__)


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
        auxiliary_replay_buffer: Optional[AuxiliaryReplayBuffer] = None,
        auxiliary_replay_train_every_n_steps: Optional[int] = None,
        device: Optional[str] = None,
        certificate_store: Optional[Any] = None,
        certificate_metadata: Optional[dict[str, Any]] = None,
        skill_library: Optional[SkillLibrary] = None,
    ) -> None:
        if save_every_n_steps <= 0:
            raise ValueError("save_every_n_steps must be positive")
        if auxiliary_replay_train_every_n_steps is not None and auxiliary_replay_train_every_n_steps <= 0:
            raise ValueError("auxiliary_replay_train_every_n_steps must be positive when provided")

        self.model = model
        self.certification_pipeline = certification_pipeline
        self.policy_trainer = policy_trainer
        self.auxiliary_trainer = auxiliary_trainer
        self.auxiliary_replay_buffer = auxiliary_replay_buffer
        self.auxiliary_replay_train_every_n_steps = auxiliary_replay_train_every_n_steps
        self.baseline_stats = dict(baseline_stats)
        self.checkpoint_path = checkpoint_path
        self.store_path = store_path or certification_pipeline.config.store_path
        self.save_every_n_steps = int(save_every_n_steps)
        self.selector = MDNRuntimeSelector(model=model, device=device or str(policy_trainer.device))
        self.certificate_store = certificate_store
        self.certificate_metadata: dict[str, Any] = dict(certificate_metadata) if certificate_metadata else {}
        self.skill_library = skill_library
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
            self._write_certified_to_store(context, certified_candidates)
        if self.skill_library is not None:
            self._write_certified_to_skill_library(
                context=context,
                candidates=certified_candidates,
                execute_skill=execute_skill,
            )
        if self.skill_library is None and not certified_skill_ids:
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

        if self.skill_library is not None:
            try:
                selection = self.selector.select_from_library(context, self.skill_library)
            except ValueError as exc:
                if "admissible stored skill" not in str(exc):
                    raise
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
        else:
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

        if self.auxiliary_replay_buffer is not None:
            self.auxiliary_replay_buffer.append(
                self._build_aux_replay_entry(
                    context=context,
                    candidates=list(selection.candidate_skills),
                    selected_skill_id=selection.selected_skill_id,
                    behavior_probability=selection.behavior_probability,
                    actual_payoff=float(outcome["actual_payoff"]),
                    actual_motives=tuple(float(v) for v in outcome["actual_motives"]),
                )
            )

        auxiliary_metrics: dict[str, float] | None = None
        if self.auxiliary_trainer is not None and self.auxiliary_replay_train_every_n_steps is None:
            aux_record = self._build_aux_record(
                context=context,
                candidate_skills=selection.candidate_skills,
                selected_skill_id=selection.selected_skill_id,
                behavior_probability=selection.behavior_probability,
                actual_motives=tuple(float(v) for v in outcome["actual_motives"]),
            )
            auxiliary_metrics = self.auxiliary_trainer.online_step(aux_record)

        self.certification_pipeline.observe_certified_weight(context, selection.weights_used)

        self._step_count += 1
        replay_metrics = self._maybe_train_auxiliary_from_replay()
        if replay_metrics is not None:
            auxiliary_metrics = replay_metrics
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

    def _write_certified_to_store(
        self,
        context: np.ndarray,
        candidates: list[CandidateSkillRecord],
    ) -> None:
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
                result = self.certification_pipeline.get_certification_result(
                    context=context,
                    skill_id=candidate.skill_id,
                )
                if result is None:
                    continue
                kwargs = certification_result_to_certificate_kwargs(
                    result,
                    timestamp=timestamp,
                    seed=int(meta.get("seed", 0)),
                    gamma=float(meta.get("gamma", 1.0)),
                    baseline_id=candidate.baseline_id or str(meta.get("baseline_id", "default")),
                    environment=str(meta.get("environment", "mo-lunar-lander-v3")),
                    episode_length=int(meta.get("episode_length", 1)),
                    version=str(meta.get("version", "1.0")),
                    epsilon=result.epsilon,
                )
                cert = Certificate(**kwargs)
                added = self.certificate_store.add(cert)
                if not added:
                    logger.warning(
                        "Failed to write runtime certificate for skill '%s': store rejected it",
                        candidate.skill_id,
                    )
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Failed to write runtime certificate for skill '%s': %s",
                    candidate.skill_id,
                    exc,
                )

    def _write_certified_to_skill_library(
        self,
        *,
        context: np.ndarray,
        candidates: list[CandidateSkillRecord],
        execute_skill: Callable[[str], dict[str, Any]],
    ) -> None:
        """Promote runtime-certified candidates into static SkillLibrary storage."""
        if self.skill_library is None:
            return

        timestamp = datetime.now().isoformat()
        meta = self.certificate_metadata
        for candidate in candidates:
            if not candidate.is_certified:
                continue
            result = self.certification_pipeline.get_certification_result(
                context=context,
                skill_id=candidate.skill_id,
            )
            if result is None:
                continue

            try:
                kwargs = certification_result_to_certificate_kwargs(
                    result,
                    timestamp=timestamp,
                    seed=int(meta.get("seed", 0)),
                    gamma=float(meta.get("gamma", 1.0)),
                    baseline_id=candidate.baseline_id or str(meta.get("baseline_id", "default")),
                    environment=str(meta.get("environment", "mo-lunar-lander-v3")),
                    episode_length=int(meta.get("episode_length", 1)),
                    version=str(meta.get("version", "1.0")),
                    epsilon=result.epsilon,
                )
                certificate = Certificate(**kwargs)
                policy = candidate.metadata.get("policy")
                if policy is None:
                    policy = lambda _obs=None, _skill_id=candidate.skill_id, _execute_skill=execute_skill: _execute_skill(_skill_id)
                promoted = self.skill_library.add_skill(
                    candidate.skill_id,
                    certificate,
                    policy,
                    weight_region_type=certificate.weight_region_type,
                    certification_context=certificate.certification_context,
                    mdn_alpha=certificate.mdn_alpha,
                    wx_support_directions=certificate.wx_support_directions,
                    wx_support_values=certificate.wx_support_values,
                )
                if not promoted:
                    logger.warning(
                        "Failed to promote certified skill '%s' into SkillLibrary",
                        candidate.skill_id,
                    )
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Failed to promote certified skill '%s' into SkillLibrary: %s",
                    candidate.skill_id,
                    exc,
                )
                continue

    def _build_aux_record(
        self,
        *,
        context: np.ndarray,
        candidate_skills: tuple[CandidateSkillRecord, ...],
        selected_skill_id: str,
        behavior_probability: float,
        actual_motives: tuple[float, ...],
    ) -> AuxiliaryTrainingRecord:
        """Build an AuxiliaryTrainingRecord from one online step for gate/motive head training.
        """
        candidates = tuple(candidate_skills)
        selected_matches = [i for i, c in enumerate(candidates) if c.skill_id == selected_skill_id]
        if not selected_matches:
            raise ValueError("selected_skill_id must be present in auxiliary candidate_skills")
        selected_idx = selected_matches[0]
        skill_id_int = zlib.crc32(selected_skill_id.encode()) % self.model.num_skills
        return AuxiliaryTrainingRecord(
            context=tuple(float(v) for v in context),
            skill_id=skill_id_int,
            accept_label=1.0,
            q_target=actual_motives,
            behavior_probability=behavior_probability,
            candidate_delta_r=tuple(c.delta_r for c in candidates),
            candidate_delta_n=tuple(tuple(c.delta_n) for c in candidates),
            selected_candidate_index=selected_idx,
        )

    def _build_aux_replay_entry(
        self,
        *,
        context: np.ndarray,
        candidates: list[CandidateSkillRecord],
        selected_skill_id: str,
        behavior_probability: float,
        actual_payoff: float,
        actual_motives: tuple[float, ...],
    ) -> AuxiliaryReplayEntry:
        selected_idx = next(
            (i for i, candidate in enumerate(candidates) if candidate.skill_id == selected_skill_id),
            None,
        )
        if selected_idx is None:
            raise ValueError(f"selected_skill_id {selected_skill_id!r} not found in candidates")
        certified_candidate_indices = tuple(
            index for index, candidate in enumerate(candidates) if candidate.is_certified
        )
        if selected_idx not in certified_candidate_indices:
            raise ValueError(f"selected_skill_id {selected_skill_id!r} is not certified")
        return AuxiliaryReplayEntry(
            context=tuple(float(v) for v in context),
            selected_skill_id=selected_skill_id,
            selected_candidate_index=selected_idx,
            behavior_probability=float(behavior_probability),
            actual_payoff=float(actual_payoff),
            actual_motives=tuple(float(v) for v in actual_motives),
            candidate_skill_ids=tuple(candidate.skill_id for candidate in candidates),
            candidate_accept_labels=tuple(float(candidate.is_certified) for candidate in candidates),
            candidate_delta_r=tuple(float(candidate.delta_r) for candidate in candidates),
            candidate_delta_n=tuple(tuple(float(v) for v in candidate.delta_n) for candidate in candidates),
            certified_candidate_indices=certified_candidate_indices,
        )

    def _maybe_train_auxiliary_from_replay(self) -> dict[str, float] | None:
        if self.auxiliary_trainer is None:
            return None
        if self.auxiliary_replay_buffer is None:
            return None
        if self.auxiliary_replay_train_every_n_steps is None:
            return None
        if self._step_count % self.auxiliary_replay_train_every_n_steps != 0:
            return None

        entries = self.auxiliary_replay_buffer.sample_all()
        if not entries:
            return None
        if len(entries) < 2:
            return None

        records: list[AuxiliaryTrainingRecord] = []
        for entry in entries:
            records.extend(replay_entry_to_auxiliary_records(entry, num_skills=self.model.num_skills))
        if self.auxiliary_trainer.config.use_ips or self.auxiliary_trainer.config.use_doubly_robust:
            return self.auxiliary_trainer.train_probability_aware_records(records)
        return self.auxiliary_trainer.train_records(records)

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
        auxiliary_replay_buffer: Optional[AuxiliaryReplayBuffer] = None,
        auxiliary_replay_train_every_n_steps: Optional[int] = None,
        device: Optional[str] = None,
        certificate_store: Optional[Any] = None,
        certificate_metadata: Optional[dict[str, Any]] = None,
        skill_library: Optional[SkillLibrary] = None,
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
            auxiliary_replay_buffer=auxiliary_replay_buffer,
            auxiliary_replay_train_every_n_steps=auxiliary_replay_train_every_n_steps,
            device=device,
            certificate_store=certificate_store,
            certificate_metadata=certificate_metadata,
            skill_library=skill_library,
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
            runner.certification_pipeline.model = restored.model
            if runner.auxiliary_trainer is not None:
                runner.auxiliary_trainer.model = restored.model

        store_file = Path(runner.store_path) if runner.store_path is not None else None
        if store_file is not None and store_file.exists():
            loaded_store = WeightSetStore.load(store_file)
            runner.certification_pipeline.weight_store._store = loaded_store._store

        return runner
