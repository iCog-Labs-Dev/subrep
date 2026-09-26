from __future__ import annotations

import numpy as np
import pytest

from utils.mdn_support_data import runtime_logs_to_support_data, split_weight_set_store
from utils.probability_aware_logs import save_probability_aware_log
from utils.weight_set_store import WeightSetStore


def _store(context_count: int = 10) -> WeightSetStore:
    store = WeightSetStore(num_objectives=2)
    for index in range(context_count):
        context = np.full(8, index / 100.0, dtype=np.float32)
        first = 0.2 + 0.6 * index / max(context_count - 1, 1)
        store.observe_certified_weight(
            context,
            np.array([first, 1.0 - first], dtype=np.float32),
        )
    return store


def test_support_split_is_deterministic_complete_and_context_disjoint():
    first = split_weight_set_store(_store(), seed=17)
    second = split_weight_set_store(_store(), seed=17)

    assert first.manifest == second.manifest
    assert first.train.context_count() + first.validation.context_count() + first.test.context_count() == 10
    assert first.train.total_vertex_count() + first.validation.total_vertex_count() + first.test.total_vertex_count() == 10

    split_keys = [
        {tuple(context.tolist()) for context, _ in partition.get_all_context_vertices()}
        for partition in (first.train, first.validation, first.test)
    ]
    assert split_keys[0].isdisjoint(split_keys[1])
    assert split_keys[0].isdisjoint(split_keys[2])
    assert split_keys[1].isdisjoint(split_keys[2])
    assert all(split_keys)


def test_support_split_keeps_all_vertices_for_context_together():
    store = _store(context_count=3)
    context = np.zeros(8, dtype=np.float32)
    store.observe_certified_weight(context, np.array([0.9, 0.1], dtype=np.float32))

    split = split_weight_set_store(store, seed=3)

    vertex_counts = []
    for partition in (split.train, split.validation, split.test):
        weight_set = partition.get_weight_set(context)
        vertex_counts.append(0 if weight_set is None else len(weight_set.vertices))
    assert sorted(vertex_counts) == [0, 0, 2]


def test_support_split_rejects_too_few_contexts_and_invalid_fractions():
    with pytest.raises(ValueError, match="At least three"):
        split_weight_set_store(_store(context_count=2))
    with pytest.raises(ValueError, match="must sum to 1"):
        split_weight_set_store(
            _store(context_count=3),
            train_fraction=0.8,
            validation_fraction=0.15,
            test_fraction=0.15,
        )


def test_runtime_logs_build_support_targets_and_candidate_sets(tmp_path):
    log_path = tmp_path / "runtime_00001.npz"
    save_probability_aware_log(
        log_path,
        context=np.full(8, 0.2, dtype=np.float32),
        alpha=np.array([0.7, 0.3], dtype=np.float32),
        support_values=np.ones(2, dtype=np.float32),
        weights_used=np.array([0.7, 0.3], dtype=np.float32),
        candidate_skill_ids=np.array(["safe", "unsafe"]),
        candidate_delta_r=np.array([0.2, -0.2], dtype=np.float32),
        candidate_delta_n=np.array([[0.1, 0.0], [-0.2, -0.1]], dtype=np.float32),
        candidate_accept_labels=np.array([True, False]),
        candidate_gate_types=np.array(["CDS", "CDS"]),
        candidate_admission_margins=np.array([0.2, -0.3], dtype=np.float32),
        selected_candidate_index=np.asarray(0, dtype=np.int32),
        selected_skill_id=np.asarray("safe"),
        selected_score=np.asarray(0.27, dtype=np.float32),
        behavior_probability=np.asarray(1.0, dtype=np.float32),
        actual_payoff=np.asarray(0.2, dtype=np.float32),
        actual_motives=np.array([0.1, 0.0], dtype=np.float32),
    )

    store, candidate_sets = runtime_logs_to_support_data(str(tmp_path))

    assert store.context_count() == 1
    assert store.total_vertex_count() == 1
    assert np.allclose(
        store.get_support_values(np.full(8, 0.2, dtype=np.float32)),
        np.array([0.7, 0.3], dtype=np.float32),
    )
    assert len(candidate_sets) == 1
    assert len(candidate_sets[0].candidates) == 2
    assert candidate_sets[0].candidates[0].skill_id == "safe"


def test_runtime_logs_preserve_pds_gate_and_epsilon(tmp_path):
    save_probability_aware_log(
        tmp_path / "pds_00001.npz",
        context=np.full(8, 0.3, dtype=np.float32),
        alpha=np.array([0.6, 0.4], dtype=np.float32),
        support_values=np.ones(2, dtype=np.float32),
        weights_used=np.array([0.6, 0.4], dtype=np.float32),
        candidate_skill_ids=np.array(["pds_skill"]),
        candidate_delta_r=np.array([0.0], dtype=np.float32),
        candidate_delta_n=np.array([[-0.1, 0.0]], dtype=np.float32),
        candidate_accept_labels=np.array([True]),
        candidate_gate_types=np.array(["PDS"]),
        candidate_epsilons=np.array([0.2], dtype=np.float32),
        candidate_admission_margins=np.array([0.1], dtype=np.float32),
        selected_candidate_index=np.asarray(0, dtype=np.int32),
        selected_skill_id=np.asarray("pds_skill"),
        selected_score=np.asarray(-0.06, dtype=np.float32),
        behavior_probability=np.asarray(1.0, dtype=np.float32),
        actual_payoff=np.asarray(0.0, dtype=np.float32),
        actual_motives=np.array([-0.1, 0.0], dtype=np.float32),
    )

    _, candidate_sets = runtime_logs_to_support_data(str(tmp_path))

    assert candidate_sets[0].candidates[0].gate_type == "PDS"
    assert candidate_sets[0].candidates[0].epsilon == pytest.approx(0.2)
