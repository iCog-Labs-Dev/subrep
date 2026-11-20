import numpy as np
import pytest
from demo.diagnose_safety_policies import audit_arrays


def test_baseline_relative_margin_and_pds_tradeoff():
    rows, details = audit_arrays(['zero_action','move'], [[[0,0,0],[0,.1,-.25]]], [1,1,1], .2)
    assert rows[1]['cds_passes']==0
    assert rows[1]['pds_only_passes']==1
    assert details[1]['cds_margin']==pytest.approx(-.15)
    assert details[1]['worst_objective']=='efficiency'


def test_scaling_and_invalid_measurements():
    rows, details = audit_arrays(['zero_action','move'], [[[0,2,0],[0,4,-4]]], [1,2,4], .2)
    assert details[1]['delta_r']==1
    assert details[1]['delta_n']==[0,1,-1]
    assert rows[1]['mean_effort']==4
    with pytest.raises(ValueError):
        audit_arrays(['zero_action'], [[[0,np.nan,0]]],[1,1,1],.2)


def test_payoff_check_accepts_rounding_but_rejects_real_mismatch():
    from demo.diagnose_safety_policies import validate_task_payoff
    assert validate_task_payoff([.000048479], [[0,.000048506,0]]) < 2e-6
    with pytest.raises(ValueError, match='payoff differs'):
        validate_task_payoff([.01], [[0,.02,0]], source='bad.npz')
    with pytest.raises(ValueError, match='invalid'):
        validate_task_payoff([float('nan')], [[0,0,0]])
