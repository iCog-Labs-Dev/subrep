from __future__ import annotations

import numpy as np
import torch

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_trainer import MDNTrainer, MDNTrainerConfig
from utils.mdn_contracts import CandidateSkillRecord, MDNDecisionRecord
from utils.mdn_selection import alpha_to_mean_weights


def _make_record(
    *,
    context_value: float,
    weights_used: tuple[float, float],
    selected_skill_id: str,
    actual_motives: tuple[float, float],
) -> MDNDecisionRecord:
    candidates = (
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
    )
    return MDNDecisionRecord(
        context=(context_value,) * 14,
        alpha=(1.0, 1.0),
        support_values=(0.5, 0.5),
        weights_used=weights_used,
        candidate_skills=candidates,
        selected_skill_id=selected_skill_id,
        selected_score=0.0,
        actual_payoff=1.0,
        actual_motives=actual_motives,
        utility=None,
    )


def _mean_weights_for_context(model: MotiveDecompositionNetwork, context_value: float) -> np.ndarray:
    with torch.no_grad():
        alpha, _ = model(torch.tensor((context_value,) * 14, dtype=torch.float32))
    return alpha_to_mean_weights(alpha.detach().cpu().numpy())


def test_behavior_safety_dominant_records_increase_safety_weight():
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    trainer = MDNTrainer(model, config=MDNTrainerConfig(learning_rate=5e-3), device="cpu")

    before = _mean_weights_for_context(model, 0.1)
    record = _make_record(
        context_value=0.1,
        weights_used=(0.8, 0.2),
        selected_skill_id="safe_skill",
        actual_motives=(0.9, 0.1),
    )
    for _ in range(30):
        trainer.training_step(record)
    after = _mean_weights_for_context(model, 0.1)

    assert after[0] > before[0]


def test_behavior_fuel_dominant_records_increase_fuel_weight():
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    trainer = MDNTrainer(model, config=MDNTrainerConfig(learning_rate=5e-3), device="cpu")

    before = _mean_weights_for_context(model, 0.2)
    record = _make_record(
        context_value=0.2,
        weights_used=(0.2, 0.8),
        selected_skill_id="fuel_skill",
        actual_motives=(0.1, 0.9),
    )
    for _ in range(30):
        trainer.training_step(record)
    after = _mean_weights_for_context(model, 0.2)

    assert after[1] > before[1]

