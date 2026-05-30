from __future__ import annotations

from pathlib import Path

import torch

from generator.train_mdn import (
    build_auxiliary_records_from_prepared_candidate_outcomes,
    build_records_from_prepared_candidate_outcomes,
    train_mdn_from_prepared_outcomes,
)
from utils.mdn_record_builder import PreparedCandidateOutcome


def _baseline_stats() -> dict[str, object]:
    return {
        "baseline_payoff": 1.0,
        "baseline_motives": (0.5, 0.2),
    }


def _prepared_outcomes() -> tuple[PreparedCandidateOutcome, ...]:
    return (
        PreparedCandidateOutcome(context=(0.1,) * 14, skill_id="skill_a", payoff=1.7, motives=(0.8, 0.4)),
        PreparedCandidateOutcome(context=(0.1,) * 14, skill_id="skill_b", payoff=1.1, motives=(0.3, 0.7)),
        PreparedCandidateOutcome(context=(0.2,) * 14, skill_id="skill_c", payoff=1.5, motives=(0.7, 0.3)),
        PreparedCandidateOutcome(context=(0.2,) * 14, skill_id="skill_d", payoff=1.2, motives=(0.6, 0.5)),
    )


def test_build_records_from_prepared_candidate_outcomes_groups_by_context():
    records = build_records_from_prepared_candidate_outcomes(
        prepared_outcomes=_prepared_outcomes(),
        baseline_stats=_baseline_stats(),
        seed=0,
        device="cpu",
    )

    assert len(records) == 2
    assert all(record.selected_skill_id for record in records)


def test_train_mdn_from_prepared_outcomes_runs_and_saves_checkpoint(tmp_path: Path):
    checkpoint_path = tmp_path / "mdn_policy_best.pth"
    metrics = train_mdn_from_prepared_outcomes(
        prepared_outcomes=_prepared_outcomes(),
        baseline_stats=_baseline_stats(),
        checkpoint_path=str(checkpoint_path),
        seed=0,
        device="cpu",
    )

    assert checkpoint_path.exists()
    assert "loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_build_auxiliary_records_from_prepared_candidate_outcomes_includes_all_candidates():
    records = build_auxiliary_records_from_prepared_candidate_outcomes(
        prepared_outcomes=_prepared_outcomes(),
        baseline_stats=_baseline_stats(),
    )

    assert len(records) == len(_prepared_outcomes())
