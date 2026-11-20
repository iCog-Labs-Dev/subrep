from types import SimpleNamespace
import numpy as np
import pytest
from generator.mdn_auxiliary_trainer import context_split_indices
from utils.mdn_data_adapter import candidate_set_file_to_prepared_candidate_outcomes


def test_contexts_do_not_cross_split():
    records = [SimpleNamespace(context=(i,)) for i in range(10) for _ in range(3)]
    train, val = context_split_indices(records, .2, 42)
    assert len(train) == 24 and len(val) == 6
    assert not {records[i].context for i in train} & {records[i].context for i in val}
    with pytest.raises(ValueError, match='two distinct'):
        context_split_indices(records[:3], .2, 42)


def test_scales_apply_to_payoff_motives_and_baseline(tmp_path):
    path = tmp_path / 'data.npz'
    np.savez(path, context=[1, 2], candidate_skill_ids=['zero_action', 'move'],
             candidate_payoffs=[4., 8.], candidate_motives=[[2., 4., -8.], [4., 8., -16.]],
             objective_scales=[2., 4., 8.], values_are_raw=True)
    rows = candidate_set_file_to_prepared_candidate_outcomes(path, baseline_candidate_id='zero_action')
    assert rows[1].payoff == 2.
    assert rows[1].motives == (2., 2., -2.)
    assert rows[1].metadata['same_context_baseline']['baseline_motives'] == [1., 1., -1.]
    with np.load(path) as raw:
        assert raw['candidate_payoffs'][1] == 8.
