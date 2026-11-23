"""Candidate filtering contracts, independent of certification internals."""
import numpy as np
import pytest
import torch
from generator.candidate_filter import select_top_k, execute_selected,  make_predicted_score_fn
from generator.policy_registry import POLICY_NAMES


class ControlledModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, obs, ids):
        return ids.float().unsqueeze(-1) + self.anchor, torch.zeros(len(ids), 2)


def test_ranking_budget_and_tie_breaking():
    model = ControlledModel()
    names = list(POLICY_NAMES)
    assert select_top_k(model, np.zeros(8), names, 2) == [names[-1], names[-2]]
    assert select_top_k(model, np.zeros(8), names, 2, score_fn=lambda p, m: 0) == sorted(names)[:2]
    with pytest.raises(ValueError):
        select_top_k(model, np.zeros(8), names, 0)
    with pytest.raises(ValueError):
        select_top_k(model, np.zeros(8), [names[0], names[0]], 1)


def test_only_selected_are_executed_and_certification_gets_measured_values():
    calls = []
    selected = [POLICY_NAMES[0], POLICY_NAMES[2]]
    def execute(policy):
        calls.append(policy)
        return -7., np.array([-2., -3.]), False
    def certify(payoff, motives):
        assert payoff == -7.
        np.testing.assert_array_equal(motives, [-2., -3.])
        return {'cds_admit': False, 'pds_admit': False}
    results = execute_selected(selected, {n: n for n in POLICY_NAMES}, execute, certify)
    assert calls == selected
    assert len(results) == 2
    assert all(not r['certification']['pds_admit'] for r in results)
class MotiveControlledModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, obs, ids):
        ids = ids.float()
        payoffs = torch.zeros(
            len(ids), 1, device=ids.device
        ) + self.anchor

        motives = torch.stack(
            [ids, len(POLICY_NAMES) - 1 - ids],
            dim=1,
        )
        return payoffs, motives


def test_subrep_motive_weights_can_change_candidate_ranking():
    model = MotiveControlledModel()

    baseline = {
        "baseline_payoff": 0.0,
        "baseline_motives": [0.0, 0.0],
    }

    names = list(POLICY_NAMES)

    safety_score = make_predicted_score_fn(
        baseline,
        [1.0, 0.0],
    )
    fuel_score = make_predicted_score_fn(
        baseline,
        [0.0, 1.0],
    )

    safety_selected = select_top_k(
        model,
        np.zeros(8),
        names,
        1,
        score_fn=safety_score,
    )

    fuel_selected = select_top_k(
        model,
        np.zeros(8),
        names,
        1,
        score_fn=fuel_score,
    )

    assert safety_selected == [names[-1]]
    assert fuel_selected == [names[0]]
    assert safety_selected != fuel_selected


def test_subrep_score_validates_motive_weights():
    baseline = {
        "baseline_payoff": 0.0,
        "baseline_motives": [0.0, 0.0],
    }

    with pytest.raises(ValueError, match="sum to 1"):
        make_predicted_score_fn(baseline, [0.7, 0.7])

    with pytest.raises(ValueError, match="non-negative"):
        make_predicted_score_fn(baseline, [1.1, -0.1])