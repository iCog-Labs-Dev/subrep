from __future__ import annotations

import numpy as np
import pytest

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_auxiliary_trainer import MDNAuxiliaryTrainer, MDNAuxiliaryTrainerConfig
from generator.mdn_online_runner import MDNOnlineRunner, StepResult
from generator.mdn_trainer import MDNTrainer
from utils.mdn_runtime_pipeline import RuntimeCertificationPipeline, RuntimePipelineConfig
from utils.weight_set_store import WeightSetStore


def _baseline_stats() -> dict[str, object]:
    return {
        "baseline_payoff": 1.0,
        "baseline_motives": np.array([0.5, 0.2], dtype=np.float32),
    }


def _make_runner(tmp_path, save_every_n_steps: int = 10) -> MDNOnlineRunner:
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
    )


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


def _make_runner_with_auxiliary(tmp_path) -> MDNOnlineRunner:
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
