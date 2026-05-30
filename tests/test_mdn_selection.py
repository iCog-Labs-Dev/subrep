from __future__ import annotations

import numpy as np
import pytest
import torch

from utils.mdn_contracts import CandidateSkillRecord
from utils.mdn_selection import (
    alpha_to_mean_weights,
    sample_dirichlet_weights,
    score_candidate,
    select_best_candidate,
)


def test_alpha_to_mean_weights_sums_to_one():
    alpha = np.array([2.0, 3.0], dtype=np.float32)
    weights = alpha_to_mean_weights(alpha)

    assert weights.shape == (2,)
    assert np.allclose(weights.sum(), 1.0)


def test_sample_dirichlet_weights_returns_simplex_sample_and_finite_log_prob():
    torch.manual_seed(0)
    alpha = torch.tensor([2.0, 3.0], dtype=torch.float32)

    weights, log_prob = sample_dirichlet_weights(alpha)

    assert weights.shape == (2,)
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(), torch.tensor(1.0))
    assert torch.isfinite(log_prob)


def test_sample_dirichlet_weights_supports_batched_alpha():
    torch.manual_seed(0)
    alpha = torch.tensor([[2.0, 3.0], [1.5, 4.5]], dtype=torch.float32)

    weights, log_prob = sample_dirichlet_weights(alpha)

    assert weights.shape == (2, 2)
    assert log_prob.shape == (2,)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))


def test_score_candidate_matches_manual_calculation():
    candidate = CandidateSkillRecord(
        skill_id="skill_a",
        delta_r=0.4,
        delta_n=(0.2, -0.1),
        is_certified=True,
        gate_type="CDS",
    )
    weights = np.array([0.25, 0.75], dtype=np.float32)

    score = score_candidate(candidate, weights)
    expected = 0.4 + 0.25 * 0.2 + 0.75 * (-0.1)

    assert np.isclose(score, expected)


def test_score_candidate_rejects_uncertified_candidate():
    candidate = CandidateSkillRecord(
        skill_id="skill_a",
        delta_r=0.4,
        delta_n=(0.2, -0.1),
        is_certified=False,
        gate_type="CDS",
    )

    with pytest.raises(ValueError, match="not certified"):
        score_candidate(candidate, np.array([0.5, 0.5], dtype=np.float32))


def test_select_best_candidate_ignores_uncertified_candidates():
    candidates = (
        CandidateSkillRecord(
            skill_id="bad_but_high",
            delta_r=10.0,
            delta_n=(10.0, 10.0),
            is_certified=False,
            gate_type="CDS",
        ),
        CandidateSkillRecord(
            skill_id="good_a",
            delta_r=0.4,
            delta_n=(0.2, 0.0),
            is_certified=True,
            gate_type="CDS",
        ),
        CandidateSkillRecord(
            skill_id="good_b",
            delta_r=0.3,
            delta_n=(0.5, 0.1),
            is_certified=True,
            gate_type="PDS",
        ),
    )
    weights = np.array([0.5, 0.5], dtype=np.float32)

    selected_skill_id, selected_score = select_best_candidate(candidates, weights)

    assert selected_skill_id in {"good_a", "good_b"}
    assert np.isfinite(selected_score)


def test_select_best_candidate_raises_when_no_certified_candidates_exist():
    candidates = (
        CandidateSkillRecord(
            skill_id="skill_a",
            delta_r=0.4,
            delta_n=(0.2, -0.1),
            is_certified=False,
            gate_type="CDS",
        ),
    )

    with pytest.raises(ValueError, match="No certified candidates"):
        select_best_candidate(candidates, np.array([0.5, 0.5], dtype=np.float32))
