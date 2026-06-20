from __future__ import annotations

import numpy as np
import pytest

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_auxiliary_replay import AuxiliaryReplayBuffer
from generator.mdn_auxiliary_trainer import MDNAuxiliaryTrainer, MDNAuxiliaryTrainerConfig
from generator.mdn_online_runner import MDNOnlineRunner, StepResult
from generator.mdn_trainer import MDNTrainer
from library.skill_library import SkillLibrary
from utils.mdn_contracts import CandidateSkillRecord
from utils.mdn_runtime_pipeline import RuntimeCertificationPipeline, RuntimePipelineConfig
from utils.weight_set_store import WeightSetStore


def _baseline_stats() -> dict[str, object]:
    return {
        "baseline_payoff": 1.0,
        "baseline_motives": np.array([0.5, 0.2], dtype=np.float32),
    }


def _make_runner(
    tmp_path,
    save_every_n_steps: int = 10,
    skill_library: SkillLibrary | None = None,
) -> MDNOnlineRunner:
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    store = WeightSetStore(num_objectives=2)
    pipeline = RuntimeCertificationPipeline(
        model=model,
        weight_store=store,
        config=RuntimePipelineConfig(
            gate_type="CDS",
            train_support_after_certify=False,
            store_path=str(tmp_path / "weight_store.json"),
        ),
    )
    trainer = MDNTrainer(model=model, device="cpu")
    return MDNOnlineRunner(
        model=model,
        certification_pipeline=pipeline,
        policy_trainer=trainer,
        baseline_stats=_baseline_stats(),
        checkpoint_path=str(tmp_path / "mdn_policy_best.pth"),
        store_path=str(tmp_path / "weight_store.json"),
        save_every_n_steps=save_every_n_steps,
        device="cpu",
        skill_library=skill_library,
    )


class _CollectingCertificateStore:
    def __init__(self) -> None:
        self.certificates = []

    def add(self, certificate):
        self.certificates.append(certificate)
        return True


class _RejectingCertificateStore:
    def add(self, certificate):
        return False


class _RejectingSkillLibrary(SkillLibrary):
    def add_skill(self, *args, **kwargs) -> bool:
        return False


def _candidate_payload(skill_id: str, payoff: float, motives: tuple[float, float]) -> dict[str, object]:
    return {
        "context": (0.1,) * 8,
        "skill_id": skill_id,
        "payoff": payoff,
        "motives": motives,
    }


def _execute_skill(skill_id: str) -> dict[str, object]:
    outcomes = {
        "skill_a": {"actual_payoff": 1.7, "actual_motives": (0.8, 0.4)},
        "skill_b": {"actual_payoff": 1.1, "actual_motives": (0.3, 0.7)},
    }
    return outcomes[skill_id]


def test_step_returns_valid_result(tmp_path):
    runner = _make_runner(tmp_path)
    result = runner.step(
        observation=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
        execute_skill=_execute_skill,
    )

    assert isinstance(result, StepResult)
    assert result.selected_skill_id == "skill_a"
    assert result.behavior_probability is not None
    assert 0.0 < result.behavior_probability <= 1.0
    assert result.decision_record is not None
    assert result.policy_metrics is not None


def test_step_skips_uncertified_candidates(tmp_path):
    runner = _make_runner(tmp_path)
    result = runner.step(
        observation=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_payloads=[_candidate_payload("skill_b", 0.1, (-0.5, -0.4))],
        execute_skill=_execute_skill,
    )

    assert result.selected_skill_id is None
    assert result.decision_record is None
    assert result.policy_metrics is None


def test_decision_record_has_behavior_probability(tmp_path):
    runner = _make_runner(tmp_path)
    result = runner.step(
        observation=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
        execute_skill=_execute_skill,
    )

    assert result.decision_record is not None
    assert result.decision_record.behavior_probability is not None


def test_wx_expands_on_certification(tmp_path):
    runner = _make_runner(tmp_path)
    context = np.array([0.1] * 8, dtype=np.float32)
    runner.step(
        observation=context,
        candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
        execute_skill=_execute_skill,
    )

    assert runner.certification_pipeline.weight_store.context_count() == 1


def test_observation_dimension_is_enforced(tmp_path):
    runner = _make_runner(tmp_path)
    with pytest.raises(ValueError, match="8"):
        runner.step(
            observation=np.zeros(14, dtype=np.float32),
            candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
            execute_skill=_execute_skill,
        )


def test_library_runner_enforces_observation_dimension(tmp_path):
    runner = _make_runner(tmp_path, skill_library=SkillLibrary())
    with pytest.raises(ValueError, match="8"):
        runner.step(
            observation=np.zeros(14, dtype=np.float32),
            candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
            execute_skill=_execute_skill,
        )


def test_save_and_load_restores_state(tmp_path):
    runner = _make_runner(tmp_path, save_every_n_steps=1)
    context = np.array([0.1] * 8, dtype=np.float32)
    runner.step(
        observation=context,
        candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
        execute_skill=_execute_skill,
    )

    restored_model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    restored_store = WeightSetStore(num_objectives=2)
    restored_pipeline = RuntimeCertificationPipeline(
        model=restored_model,
        weight_store=restored_store,
        config=RuntimePipelineConfig(
            gate_type="CDS",
            train_support_after_certify=False,
            store_path=str(tmp_path / "weight_store.json"),
        ),
    )
    restored_trainer = MDNTrainer(model=restored_model, device="cpu")
    restored_runner = MDNOnlineRunner.load(
        model=restored_model,
        certification_pipeline=restored_pipeline,
        policy_trainer=restored_trainer,
        baseline_stats=_baseline_stats(),
        checkpoint_path=str(tmp_path / "mdn_policy_best.pth"),
        store_path=str(tmp_path / "weight_store.json"),
        device="cpu",
    )

    assert restored_runner.certification_pipeline.weight_store.context_count() == 1


def test_full_loop_runs_for_five_steps(tmp_path):
    runner = _make_runner(tmp_path)
    for _ in range(5):
        result = runner.step(
            observation=np.array([0.1] * 8, dtype=np.float32),
            candidate_skill_payloads=[
                _candidate_payload("skill_a", 1.7, (0.8, 0.4)),
                _candidate_payload("skill_b", 0.1, (-0.5, -0.4)),
            ],
            execute_skill=_execute_skill,
        )
        assert isinstance(result, StepResult)


def _make_runner_with_auxiliary(
    tmp_path,
    skill_library: SkillLibrary | None = None,
) -> MDNOnlineRunner:
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    store = WeightSetStore(num_objectives=2)
    pipeline = RuntimeCertificationPipeline(
        model=model,
        weight_store=store,
        config=RuntimePipelineConfig(
            gate_type="CDS",
            train_support_after_certify=False,
            store_path=str(tmp_path / "weight_store.json"),
        ),
    )
    trainer = MDNTrainer(model=model, device="cpu")
    aux_trainer = MDNAuxiliaryTrainer(
        model=model,
        config=MDNAuxiliaryTrainerConfig(use_ips=True),
        device="cpu",
    )
    return MDNOnlineRunner(
        model=model,
        certification_pipeline=pipeline,
        policy_trainer=trainer,
        auxiliary_trainer=aux_trainer,
        baseline_stats=_baseline_stats(),
        checkpoint_path=str(tmp_path / "mdn_policy_best.pth"),
        store_path=str(tmp_path / "weight_store.json"),
        save_every_n_steps=10,
        device="cpu",
        skill_library=skill_library,
    )


def test_auxiliary_metrics_returned_when_trainer_wired(tmp_path):
    runner = _make_runner_with_auxiliary(tmp_path)
    result = runner.step(
        observation=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
        execute_skill=_execute_skill,
    )
    assert result.auxiliary_metrics is not None
    assert "loss" in result.auxiliary_metrics
    assert "gate_loss" in result.auxiliary_metrics
    assert "gate_accuracy" in result.auxiliary_metrics
    assert np.isfinite(result.auxiliary_metrics["loss"])


def test_auxiliary_metrics_none_without_trainer(tmp_path):
    runner = _make_runner(tmp_path)
    result = runner.step(
        observation=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
        execute_skill=_execute_skill,
    )
    assert result.auxiliary_metrics is None


def test_replay_entry_added_when_buffer_is_wired(tmp_path):
    replay = AuxiliaryReplayBuffer(capacity=10)
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    store = WeightSetStore(num_objectives=2)
    pipeline = RuntimeCertificationPipeline(
        model=model,
        weight_store=store,
        config=RuntimePipelineConfig(
            gate_type="CDS",
            train_support_after_certify=False,
            store_path=str(tmp_path / "weight_store.json"),
        ),
    )
    trainer = MDNTrainer(model=model, device="cpu")
    runner = MDNOnlineRunner(
        model=model,
        certification_pipeline=pipeline,
        policy_trainer=trainer,
        auxiliary_replay_buffer=replay,
        baseline_stats=_baseline_stats(),
        checkpoint_path=str(tmp_path / "mdn_policy_best.pth"),
        store_path=str(tmp_path / "weight_store.json"),
        save_every_n_steps=10,
        device="cpu",
    )

    runner.step(
        observation=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
        execute_skill=_execute_skill,
    )

    assert len(replay) == 1
    entry = replay.last()
    assert entry is not None
    assert entry.selected_skill_id == "skill_a"
    assert entry.candidate_skill_ids == ("skill_a",)
    assert entry.certified_candidate_indices == (0,)


def test_step_promotes_certified_skill_to_library(tmp_path):
    skill_library = SkillLibrary()
    runner = _make_runner(tmp_path, skill_library=skill_library)

    result = runner.step(
        observation=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
        execute_skill=_execute_skill,
    )

    assert result.selected_skill_id == "skill_a"
    assert skill_library.get_skill("skill_a") is not None
    assert result.decision_record is not None
    assert result.decision_record.candidate_skills[0].skill_id == "skill_a"


def test_replay_entry_not_added_when_no_certified_candidate_selected(tmp_path):
    replay = AuxiliaryReplayBuffer(capacity=10)
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    store = WeightSetStore(num_objectives=2)
    pipeline = RuntimeCertificationPipeline(
        model=model,
        weight_store=store,
        config=RuntimePipelineConfig(
            gate_type="CDS",
            train_support_after_certify=False,
            store_path=str(tmp_path / "weight_store.json"),
        ),
    )
    trainer = MDNTrainer(model=model, device="cpu")
    runner = MDNOnlineRunner(
        model=model,
        certification_pipeline=pipeline,
        policy_trainer=trainer,
        auxiliary_replay_buffer=replay,
        baseline_stats=_baseline_stats(),
        checkpoint_path=str(tmp_path / "mdn_policy_best.pth"),
        store_path=str(tmp_path / "weight_store.json"),
        save_every_n_steps=10,
        device="cpu",
    )

    runner.step(
        observation=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_payloads=[_candidate_payload("skill_b", 0.1, (-0.5, -0.4))],
        execute_skill=_execute_skill,
    )

    assert len(replay) == 0


def test_step_selects_existing_library_skill_without_new_certification(tmp_path):
    skill_library = SkillLibrary()
    runner = _make_runner(tmp_path, skill_library=skill_library)
    context = np.array([0.1] * 8, dtype=np.float32)

    runner.step(
        observation=context,
        candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
        execute_skill=_execute_skill,
    )
    result = runner.step(
        observation=context,
        candidate_skill_payloads=[_candidate_payload("skill_b", 0.1, (-0.5, -0.4))],
        execute_skill=_execute_skill,
    )

    assert result.certified_skill_ids == ()
    assert result.selected_skill_id == "skill_a"
    assert result.decision_record is not None
    assert runner.certification_pipeline.weight_store.total_vertex_count() == 2
    vertices = runner.certification_pipeline.weight_store.get_weight_set(context).get_vertices_array()
    assert vertices is not None
    np.testing.assert_allclose(vertices[-1], result.weights_used)


def test_library_selection_adds_replay_entry_for_stored_skill(tmp_path):
    replay = AuxiliaryReplayBuffer(capacity=10)
    skill_library = SkillLibrary()
    runner = _make_runner(tmp_path, skill_library=skill_library)
    runner.auxiliary_replay_buffer = replay
    context = np.array([0.1] * 8, dtype=np.float32)

    runner.step(
        observation=context,
        candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
        execute_skill=_execute_skill,
    )
    result = runner.step(
        observation=context,
        candidate_skill_payloads=[_candidate_payload("skill_b", 0.1, (-0.5, -0.4))],
        execute_skill=_execute_skill,
    )

    assert result.selected_skill_id == "skill_a"
    assert len(replay) == 2
    entry = replay.last()
    assert entry is not None
    assert entry.selected_skill_id == "skill_a"
    assert entry.candidate_skill_ids == ("skill_a",)
    assert entry.certified_candidate_indices == (0,)


def test_certificate_store_write_preserves_mdn_audit_fields(tmp_path):
    store = _CollectingCertificateStore()
    runner = _make_runner(tmp_path)
    runner.certificate_store = store
    runner.certificate_metadata = {"baseline_id": "runtime_baseline"}
    context = np.array([0.1] * 8, dtype=np.float32)
    runner.certification_pipeline.weight_store.observe_certified_weight(
        context,
        np.array([0.7, 0.3], dtype=np.float32),
    )

    runner.step(
        observation=context,
        candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
        execute_skill=_execute_skill,
    )

    assert len(store.certificates) == 1
    certificate = store.certificates[0]
    assert certificate.weight_region_type == "MDN_WX"
    assert certificate.baseline_id == "runtime_baseline"
    assert certificate.certification_context == tuple(float(v) for v in context)
    assert certificate.mdn_alpha is not None
    assert certificate.wx_support_directions == ((1.0, 0.0), (0.0, 1.0))
    assert certificate.wx_support_values == pytest.approx((0.7, 0.3))


def test_skill_library_promotion_failure_is_logged(tmp_path, caplog):
    runner = _make_runner(tmp_path, skill_library=_RejectingSkillLibrary())

    with caplog.at_level("WARNING", logger="generator.mdn_online_runner"):
        result = runner.step(
            observation=np.array([0.1] * 8, dtype=np.float32),
            candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
            execute_skill=_execute_skill,
        )

    assert result.selected_skill_id is None
    assert "Failed to promote certified skill 'skill_a' into SkillLibrary" in caplog.text


def test_certificate_store_rejection_is_logged(tmp_path, caplog):
    runner = _make_runner(tmp_path)
    runner.certificate_store = _RejectingCertificateStore()

    with caplog.at_level("WARNING", logger="generator.mdn_online_runner"):
        runner.step(
            observation=np.array([0.1] * 8, dtype=np.float32),
            candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
            execute_skill=_execute_skill,
        )

    assert "Failed to write runtime certificate for skill 'skill_a': store rejected it" in caplog.text


def test_auxiliary_record_uses_library_selection_candidates(tmp_path):
    skill_library = SkillLibrary()
    runner = _make_runner_with_auxiliary(tmp_path, skill_library=skill_library)
    context = np.array([0.1] * 8, dtype=np.float32)

    runner.step(
        observation=context,
        candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
        execute_skill=_execute_skill,
    )
    result = runner.step(
        observation=context,
        candidate_skill_payloads=[_candidate_payload("skill_b", 0.1, (-0.5, -0.4))],
        execute_skill=_execute_skill,
    )

    assert result.certified_skill_ids == ()
    assert result.selected_skill_id == "skill_a"
    assert result.auxiliary_metrics is not None


def test_auxiliary_record_excludes_uncertified_candidates_from_ips_set(tmp_path):
    runner = _make_runner_with_auxiliary(tmp_path)

    result = runner.step(
        observation=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_payloads=[
            _candidate_payload("skill_a", 1.7, (0.8, 0.4)),
            _candidate_payload("skill_b", 0.1, (-0.5, -0.4)),
        ],
        execute_skill=_execute_skill,
    )

    assert result.decision_record is not None
    assert result.auxiliary_metrics is not None
    assert tuple(candidate.skill_id for candidate in result.decision_record.candidate_skills) == ("skill_a",)


def test_aux_record_raises_when_selected_skill_missing(tmp_path):
    runner = _make_runner(tmp_path)
    candidate = CandidateSkillRecord(
        skill_id="skill_a",
        delta_r=0.7,
        delta_n=(0.3, 0.2),
        is_certified=True,
        gate_type="CDS",
        admission_margin=0.5,
        epsilon=0.0,
    )

    with pytest.raises(ValueError, match="present"):
        runner._build_aux_record(
            context=np.array([0.1] * 8, dtype=np.float32),
            candidate_skills=(candidate,),
            selected_skill_id="missing",
            behavior_probability=1.0,
            actual_motives=(0.8, 0.4),
        )


def test_aux_replay_entry_raises_when_selected_skill_missing(tmp_path):
    runner = _make_runner(tmp_path)
    candidate = CandidateSkillRecord(
        skill_id="skill_a",
        delta_r=0.7,
        delta_n=(0.3, 0.2),
        is_certified=True,
        gate_type="CDS",
        admission_margin=0.5,
        epsilon=0.0,
    )

    with pytest.raises(ValueError, match="not found"):
        runner._build_aux_replay_entry(
            context=np.array([0.1] * 8, dtype=np.float32),
            candidates=[candidate],
            selected_skill_id="missing",
            behavior_probability=1.0,
            actual_payoff=1.7,
            actual_motives=(0.8, 0.4),
        )


def test_aux_replay_entry_raises_when_selected_skill_uncertified(tmp_path):
    runner = _make_runner(tmp_path)
    candidate = CandidateSkillRecord(
        skill_id="skill_a",
        delta_r=-0.2,
        delta_n=(-0.3, -0.2),
        is_certified=False,
        gate_type="CDS",
        admission_margin=None,
        epsilon=0.0,
    )

    with pytest.raises(ValueError, match="not certified"):
        runner._build_aux_replay_entry(
            context=np.array([0.1] * 8, dtype=np.float32),
            candidates=[candidate],
            selected_skill_id="skill_a",
            behavior_probability=1.0,
            actual_payoff=1.7,
            actual_motives=(0.8, 0.4),
        )


def test_replay_training_triggers_auxiliary_batch_update(tmp_path):
    replay = AuxiliaryReplayBuffer(capacity=10)
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    store = WeightSetStore(num_objectives=2)
    pipeline = RuntimeCertificationPipeline(
        model=model,
        weight_store=store,
        config=RuntimePipelineConfig(
            gate_type="CDS",
            train_support_after_certify=False,
            store_path=str(tmp_path / "weight_store.json"),
        ),
    )
    trainer = MDNTrainer(model=model, device="cpu")
    aux_trainer = MDNAuxiliaryTrainer(
        model=model,
        config=MDNAuxiliaryTrainerConfig(use_ips=True, max_epochs=1, batch_size=1),
        device="cpu",
    )
    runner = MDNOnlineRunner(
        model=model,
        certification_pipeline=pipeline,
        policy_trainer=trainer,
        auxiliary_trainer=aux_trainer,
        auxiliary_replay_buffer=replay,
        auxiliary_replay_train_every_n_steps=1,
        baseline_stats=_baseline_stats(),
        checkpoint_path=str(tmp_path / "mdn_policy_best.pth"),
        store_path=str(tmp_path / "weight_store.json"),
        save_every_n_steps=10,
        device="cpu",
    )

    first_result = runner.step(
        observation=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
        execute_skill=_execute_skill,
    )

    assert len(replay) == 1
    assert first_result.auxiliary_metrics is None

    result = runner.step(
        observation=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
        execute_skill=_execute_skill,
    )

    assert len(replay) == 2
    assert result.auxiliary_metrics is not None
    assert "best_val_loss" in result.auxiliary_metrics


def test_replay_training_works_with_selected_and_gate_only_expansion(tmp_path):
    replay = AuxiliaryReplayBuffer(capacity=10)
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    store = WeightSetStore(num_objectives=2)
    pipeline = RuntimeCertificationPipeline(
        model=model,
        weight_store=store,
        config=RuntimePipelineConfig(
            gate_type="CDS",
            train_support_after_certify=False,
            store_path=str(tmp_path / "weight_store.json"),
        ),
    )
    trainer = MDNTrainer(model=model, device="cpu")
    aux_trainer = MDNAuxiliaryTrainer(
        model=model,
        config=MDNAuxiliaryTrainerConfig(use_ips=True, max_epochs=1, batch_size=1),
        device="cpu",
    )
    runner = MDNOnlineRunner(
        model=model,
        certification_pipeline=pipeline,
        policy_trainer=trainer,
        auxiliary_trainer=aux_trainer,
        auxiliary_replay_buffer=replay,
        auxiliary_replay_train_every_n_steps=1,
        baseline_stats=_baseline_stats(),
        checkpoint_path=str(tmp_path / "mdn_policy_best.pth"),
        store_path=str(tmp_path / "weight_store.json"),
        save_every_n_steps=10,
        device="cpu",
    )

    runner.step(
        observation=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_payloads=[
            _candidate_payload("skill_a", 1.7, (0.8, 0.4)),
            _candidate_payload("skill_b", 0.1, (-0.5, -0.4)),
        ],
        execute_skill=_execute_skill,
    )

    result = runner.step(
        observation=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_payloads=[
            _candidate_payload("skill_a", 1.7, (0.8, 0.4)),
            _candidate_payload("skill_b", 0.1, (-0.5, -0.4)),
        ],
        execute_skill=_execute_skill,
    )

    assert result.auxiliary_metrics is not None
    assert "best_val_loss" in result.auxiliary_metrics


def test_replay_training_supports_dr_estimator(tmp_path):
    replay = AuxiliaryReplayBuffer(capacity=10)
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    store = WeightSetStore(num_objectives=2)
    pipeline = RuntimeCertificationPipeline(
        model=model,
        weight_store=store,
        config=RuntimePipelineConfig(
            gate_type="CDS",
            train_support_after_certify=False,
            store_path=str(tmp_path / "weight_store.json"),
        ),
    )
    trainer = MDNTrainer(model=model, device="cpu")
    aux_trainer = MDNAuxiliaryTrainer(
        model=model,
        config=MDNAuxiliaryTrainerConfig(use_doubly_robust=True, max_epochs=1, batch_size=1),
        device="cpu",
    )
    runner = MDNOnlineRunner(
        model=model,
        certification_pipeline=pipeline,
        policy_trainer=trainer,
        auxiliary_trainer=aux_trainer,
        auxiliary_replay_buffer=replay,
        auxiliary_replay_train_every_n_steps=1,
        baseline_stats=_baseline_stats(),
        checkpoint_path=str(tmp_path / "mdn_policy_best.pth"),
        store_path=str(tmp_path / "weight_store.json"),
        save_every_n_steps=10,
        device="cpu",
    )

    runner.step(
        observation=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
        execute_skill=_execute_skill,
    )
    result = runner.step(
        observation=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
        execute_skill=_execute_skill,
    )

    assert result.auxiliary_metrics is not None
    assert "best_val_loss" in result.auxiliary_metrics


def test_wx_records_selected_weight_once_per_step_with_multiple_certified_candidates(tmp_path):
    runner = _make_runner(tmp_path)

    runner.step(
        observation=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_payloads=[
            _candidate_payload("skill_a", 1.7, (0.8, 0.4)),
            _candidate_payload("skill_c", 1.8, (0.9, 0.5)),
        ],
        execute_skill=lambda skill_id: {
            "actual_payoff": 1.8,
            "actual_motives": (0.9, 0.5),
        },
    )

    assert runner.certification_pipeline.weight_store.total_vertex_count() == 1


def test_load_aligns_certification_pipeline_model(tmp_path):
    runner = _make_runner(tmp_path, save_every_n_steps=1)
    runner.step(
        observation=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
        execute_skill=_execute_skill,
    )

    restored_model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    pipeline_model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    restored_store = WeightSetStore(num_objectives=2)
    restored_pipeline = RuntimeCertificationPipeline(
        model=pipeline_model,
        weight_store=restored_store,
        config=RuntimePipelineConfig(
            gate_type="CDS",
            train_support_after_certify=False,
            store_path=str(tmp_path / "weight_store.json"),
        ),
    )
    restored_trainer = MDNTrainer(model=restored_model, device="cpu")

    restored_runner = MDNOnlineRunner.load(
        model=restored_model,
        certification_pipeline=restored_pipeline,
        policy_trainer=restored_trainer,
        baseline_stats=_baseline_stats(),
        checkpoint_path=str(tmp_path / "mdn_policy_best.pth"),
        store_path=str(tmp_path / "weight_store.json"),
        device="cpu",
    )

    assert restored_runner.certification_pipeline.model is restored_runner.model


def test_replay_training_can_be_disabled_while_buffer_collects(tmp_path):
    replay = AuxiliaryReplayBuffer(capacity=10)
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    store = WeightSetStore(num_objectives=2)
    pipeline = RuntimeCertificationPipeline(
        model=model,
        weight_store=store,
        config=RuntimePipelineConfig(
            gate_type="CDS",
            train_support_after_certify=False,
            store_path=str(tmp_path / "weight_store.json"),
        ),
    )
    trainer = MDNTrainer(model=model, device="cpu")
    aux_trainer = MDNAuxiliaryTrainer(
        model=model,
        config=MDNAuxiliaryTrainerConfig(use_ips=True, max_epochs=1, batch_size=1),
        device="cpu",
    )
    runner = MDNOnlineRunner(
        model=model,
        certification_pipeline=pipeline,
        policy_trainer=trainer,
        auxiliary_trainer=aux_trainer,
        auxiliary_replay_buffer=replay,
        baseline_stats=_baseline_stats(),
        checkpoint_path=str(tmp_path / "mdn_policy_best.pth"),
        store_path=str(tmp_path / "weight_store.json"),
        save_every_n_steps=10,
        device="cpu",
    )

    result = runner.step(
        observation=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_payloads=[_candidate_payload("skill_a", 1.7, (0.8, 0.4))],
        execute_skill=_execute_skill,
    )

    assert len(replay) == 1
    assert result.auxiliary_metrics is not None
    assert "loss" in result.auxiliary_metrics
