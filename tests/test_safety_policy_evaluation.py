import numpy as np
import pytest
from demo.evaluate_safety_policies import summarize


def test_raw_metrics_and_paired_baseline():
    records=[dict(candidate_skill_ids=['zero_action','ppo'],candidate_motives=np.array([[0,0,0],[-2,3,-4]])),
             dict(candidate_skill_ids=['zero_action','ppo'],candidate_motives=np.array([[0,1,0],[0,-1,-2]]))]
    rows=summarize(records)
    assert rows[1]['mean_task_return']==1
    assert rows[1]['mean_safety_cost']==1
    assert rows[1]['mean_control_effort']==3
    assert rows[1]['task_better_than_idle_rate']==.5
    assert rows[1]['mean_improvement_safety_task_efficiency']==[-1,.5,-3]
    records[1]['candidate_skill_ids']=['ppo','zero_action']
    with pytest.raises(ValueError): summarize(records)
