from __future__ import annotations

from pathlib import Path

import torch

from generator.train_mdn import train_mdn_from_records
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
        context=(0.1,) * 14,
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


def test_train_mdn_from_records_runs_and_saves_checkpoint(tmp_path: Path):
    checkpoint_path = tmp_path / "mdn_policy_best.pth"

    metrics = train_mdn_from_records(
        [_decision_record()],
        checkpoint_path=str(checkpoint_path),
        seed=0,
        device="cpu",
    )

    assert checkpoint_path.exists()
    assert "loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_train_mdn_from_records_rejects_empty_records(tmp_path: Path):
    try:
        train_mdn_from_records([], checkpoint_path=str(tmp_path / "mdn_policy_best.pth"), seed=0, device="cpu")
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty decision records")
