import numpy as np
from utils.mdn_data_adapter import candidate_set_directory_to_prepared_candidate_outcomes
from utils.mdn_record_builder import build_candidate_skill_records
from generator.train_mdn import build_records_from_prepared_candidate_outcomes


def test_three_objectives_and_same_context_baseline(tmp_path):
    np.savez(tmp_path/'context.npz', context=np.zeros(5),
             candidate_skill_ids=['zero_action','move'], candidate_payoffs=[2.,3.],
             candidate_motives=[[1.,2.,3.],[2.,3.,4.]])
    outcomes = candidate_set_directory_to_prepared_candidate_outcomes(tmp_path, baseline_candidate_id='zero_action')
    fallback = {'baseline_payoff': 99., 'baseline_motives': [99.,99.,99.]}
    candidates = build_candidate_skill_records(skill_outcomes=outcomes, baseline_stats=fallback)
    assert candidates[0].delta_r == 0
    assert candidates[1].delta_r == 1
    assert candidates[1].delta_n == (1.,1.,1.)
    records = build_records_from_prepared_candidate_outcomes(prepared_outcomes=outcomes, baseline_stats=fallback, device='cpu')
    assert len(records[0].alpha) == 3
    assert len(records[0].context) == 5
