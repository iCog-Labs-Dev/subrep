"""Offline metrics must reflect full measured candidate outcomes."""
import numpy as np
import pytest
from generator.evaluate_candidate_filter import assess
from generator.policy_registry import POLICY_NAMES


def test_savings_missed_candidates_and_measured_admissions():
    rows = [dict(policy=name, payoff=10. if i < 4 else -10.,
                 motives=np.array([2., -1.]) if i < 4 else np.array([-2., -10.]))
            for i, name in enumerate(POLICY_NAMES)]
    groups = {'seed:1': rows}
    result = assess(groups, {'seed:1': list(POLICY_NAMES[:2])},
                    {'baseline_payoff': 0., 'baseline_motives': [0., -5.]})
    assert result['executed'] == 2
    assert result['avoided'] == 7
    assert result['execution_savings_fraction'] == pytest.approx(7/9)
    assert result['genuinely_certifiable'] == 4
    assert result['certifiable_found'] == 2
    assert result['certifiable_missed'] == 2
    assert result['missed_certifiable_fraction'] == pytest.approx(.5)
    assert result['cds_admission_rate'] == 1.
    assert result['pds_admission_rate'] == 1.
    assert result['measured_payoff_mean'] == 10.
    assert result['task_success_rate'] is None


def test_invalid_selections_are_rejected():
    rows = [dict(policy=name, payoff=1., motives=np.array([1., -1.])) for name in POLICY_NAMES]
    baseline = {'baseline_payoff': 0., 'baseline_motives': [0., -2.]}
    with pytest.raises(ValueError, match='Invalid candidate selection'):
        assess({'seed:1': rows}, {'seed:1': [POLICY_NAMES[0], POLICY_NAMES[0]]}, baseline)
