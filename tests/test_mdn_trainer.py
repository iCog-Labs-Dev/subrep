from __future__ import annotations

from pathlib import Path

import torch

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_trainer import MDNTrainer, MDNTrainerConfig
from utils.mdn_contracts import CandidateSkillRecord, MDNDecisionRecord


def _decision_record() -> MDNDecisionRecord:
    candidates = (
        CandidateSkillRecord(
            skill_id="skill_a",
            delta_r=0.5,
            delta_n=(0.2, -0.1),
            is_certified=True,
            gate_type="CDS",
        ),
        CandidateSkillRecord(
            skill_id="skill_b",
            delta_r=0.2,
            delta_n=(0.1, 0.3),
            is_certified=True,
            gate_type="PDS",
            epsilon=0.1,
        ),
    )
    return MDNDecisionRecord(
        context=(0.1,) * 8,
        alpha=(2.0, 3.0),
        support_values=(0.7, 0.3),
        weights_used=(0.4, 0.6),
        candidate_skills=candidates,
        selected_skill_id="skill_b",
        selected_score=0.42,
        actual_payoff=1.2,
        actual_motives=(0.2, 0.6),
        utility=0.44,
    )


def test_trainer_one_step_update_runs_without_nan():
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    trainer = MDNTrainer(model, config=MDNTrainerConfig(), device="cpu")

    metrics = trainer.training_step(_decision_record())

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert torch.isfinite(torch.tensor(metrics["utility"]))
    assert torch.isfinite(torch.tensor(metrics["advantage"]))


def test_trainer_train_records_aggregates_metrics():
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    trainer = MDNTrainer(model, config=MDNTrainerConfig(), device="cpu")

    metrics = trainer.train_records([_decision_record()])

    assert "loss" in metrics
    assert "utility" in metrics
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_trainer_requires_actual_motives_for_offline_training():
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    trainer = MDNTrainer(model, config=MDNTrainerConfig(), device="cpu")
    record = MDNDecisionRecord(
        context=(0.1,) * 8,
        alpha=(2.0, 3.0),
        support_values=(0.7, 0.3),
        weights_used=(0.4, 0.6),
        candidate_skills=(
            CandidateSkillRecord(
                skill_id="skill_a",
                delta_r=0.5,
                delta_n=(0.2, -0.1),
                is_certified=True,
                gate_type="CDS",
            ),
        ),
        selected_skill_id="skill_a",
    )

    try:
        trainer.training_step(record)
    except ValueError as exc:
        assert "actual_motives" in str(exc)
    else:
        raise AssertionError("Expected ValueError when actual_motives are missing")


def test_trainer_uses_recorded_weights_without_resampling_selector_alignment():
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    trainer = MDNTrainer(model, config=MDNTrainerConfig(strict_validation=False), device="cpu")
    record = MDNDecisionRecord(
        context=(0.1,) * 8,
        alpha=(2.0, 3.0),
        support_values=(0.7, 0.3),
        weights_used=(0.95, 0.05),
        candidate_skills=(
            CandidateSkillRecord(
                skill_id="safe_skill",
                delta_r=0.2,
                delta_n=(0.8, 0.1),
                is_certified=True,
                gate_type="CDS",
            ),
            CandidateSkillRecord(
                skill_id="fuel_skill",
                delta_r=0.2,
                delta_n=(0.1, 0.8),
                is_certified=True,
                gate_type="CDS",
            ),
        ),
        selected_skill_id="fuel_skill",
        actual_payoff=1.0,
        actual_motives=(0.9, 0.1),
    )

    metrics = trainer.training_step(record)

    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_trainer_respects_recorded_utility_when_present():
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    trainer = MDNTrainer(model, config=MDNTrainerConfig(), device="cpu")
    record = _decision_record()
    record = MDNDecisionRecord(
        context=record.context,
        alpha=record.alpha,
        support_values=record.support_values,
        weights_used=record.weights_used,
        candidate_skills=record.candidate_skills,
        selected_skill_id=record.selected_skill_id,
        selected_score=record.selected_score,
        actual_payoff=record.actual_payoff,
        actual_motives=record.actual_motives,
        utility=5.0,
    )

    metrics = trainer.training_step(record)

    assert metrics["utility"] == 5.0


def test_trainer_strict_validation_catches_selected_skill_mismatch():
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    trainer = MDNTrainer(model, config=MDNTrainerConfig(strict_validation=True), device="cpu")
    record = MDNDecisionRecord(
        context=(0.1,) * 8,
        alpha=(2.0, 3.0),
        support_values=(0.7, 0.3),
        weights_used=(0.95, 0.05),
        candidate_skills=(
            CandidateSkillRecord(
                skill_id="safe_skill",
                delta_r=0.2,
                delta_n=(0.8, 0.1),
                is_certified=True,
                gate_type="CDS",
            ),
            CandidateSkillRecord(
                skill_id="fuel_skill",
                delta_r=0.2,
                delta_n=(0.1, 0.8),
                is_certified=True,
                gate_type="CDS",
            ),
        ),
        selected_skill_id="fuel_skill",
        actual_payoff=1.0,
        actual_motives=(0.9, 0.1),
    )

    try:
        trainer.training_step(record)
    except ValueError as exc:
        assert "selected_skill_id" in str(exc)
    else:
        raise AssertionError("Expected ValueError for strict selector mismatch")


def test_trainer_checkpoint_round_trip(tmp_path: Path):
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    trainer = MDNTrainer(model, config=MDNTrainerConfig(checkpoint_path=str(tmp_path / "mdn_policy_best.pth")), device="cpu")
    trainer.training_step(_decision_record())

    checkpoint_path = trainer.save_checkpoint()

    restored_model = MotiveDecompositionNetwork()
    restored_trainer = MDNTrainer.from_checkpoint(checkpoint_path, model=restored_model, device="cpu")

    assert restored_trainer.running_baseline is not None


def test_trainer_callback_receives_metrics_for_step_and_batch():
    class Callback:
        def __init__(self) -> None:
            self.calls: list[tuple[int, dict[str, float]]] = []

        def on_step(self, step: int, metrics: dict[str, float]) -> None:
            self.calls.append((step, metrics))

    torch.manual_seed(0)
    callback = Callback()
    model = MotiveDecompositionNetwork()
    trainer = MDNTrainer(model, config=MDNTrainerConfig(), device="cpu", callback=callback)

    trainer.training_step(_decision_record())
    trainer.train_records([_decision_record()])

    assert len(callback.calls) == 2
    assert callback.calls[0][0] == 1
    assert callback.calls[1][0] == 2
    assert "loss" in callback.calls[0][1]
