from __future__ import annotations

import numpy as np
import pytest
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
        context=(context_value,) * 8,
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
        alpha, _ = model(torch.tensor((context_value,) * 8, dtype=torch.float32))
    return alpha_to_mean_weights(alpha.detach().cpu().numpy())


# Both directional-learning tests below are xfail(strict=False): they assert a
# property MDNTrainer does not reliably exhibit, and they should be treated as
# open work on the trainer rather than as a passing guarantee.
#
# WHY, with measurements. Each test fixes torch.manual_seed(0) and asserts that
# 30 training steps on a motive-dominant record move the mean weight for that
# objective upward. Sweeping the seed instead of fixing it shows the assertion
# is a coin flip, not a property:
#
#   original design, 30 steps, lr=5e-3
#     safety: 5/12 seeds pass   (deltas -0.016 .. +0.045)
#     fuel:   4/12 seeds pass   (deltas -0.016 .. +0.048)
#
# A stronger paired design -- train two models from the SAME seed, one on
# safety-dominant and one on fuel-dominant records, then compare, so identical
# initialization cancels out and only the record differs -- does not rescue it:
#
#   paired, 20 seeds:  11/20 (30 steps, lr=5e-3)
#                      12/20 (60 steps, lr=5e-3)
#                      12/20 (30 steps, lr=2e-2)
#   mean effect +0.001 .. +0.010 on a value that sits at ~0.5
#
# So the effect is real but far smaller than the run-to-run variance. The cause
# is the training setup, not the assertion: MDNTrainer applies a REINFORCE-style
# policy loss whose gradient direction depends on weights sampled from
# Dirichlet(alpha) each step, and a single record replayed 30 times is dominated
# by that sampling noise.
#
# These tests were previously green only because seed 0 happened to land on the
# favourable side. The SASP change (support head widened from M to 2M outputs)
# consumes a different amount of RNG when the layer is constructed, which shifts
# the global stream feeding rsample() during training and reshuffled that luck.
# Note the pre-training value is unchanged (0.4986076 before and after the
# change) because distribution_head is initialized before support_head -- only
# the training trajectory moved. SASP itself is unaffected: support-value
# feasibility is proven algebraically and covered in tests/test_mdn.py.
#
# To fix properly, address the trainer: a variance-reduced advantage baseline,
# many distinct records instead of one replayed record, or enough steps for the
# signal to dominate. Do NOT simply pick a luckier seed -- that restores the
# illusion and the next change to parameter shapes will silently flip it again.

@pytest.mark.xfail(
    strict=False,
    reason=(
        "MDNTrainer directional learning is not reliable: 5/12 seeds pass as "
        "written, 11/20 with a paired design. Pre-existing trainer variance, "
        "not a support-geometry regression. See module comment above."
    ),
)
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


@pytest.mark.xfail(
    strict=False,
    reason=(
        "MDNTrainer directional learning is not reliable: 4/12 seeds pass as "
        "written, 11/20 with a paired design. Pre-existing trainer variance, "
        "not a support-geometry regression. See module comment above."
    ),
)
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

