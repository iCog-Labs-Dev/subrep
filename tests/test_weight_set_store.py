from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.weight_set_store import WeightSet, WeightSetStore


def test_weight_set_empty_uses_simplex_support_fallback():
    weight_set = WeightSet()
    query_directions = np.eye(2, dtype=np.float32)

    support_values = weight_set.get_support_values(query_directions)

    assert np.allclose(support_values, np.array([1.0, 1.0], dtype=np.float32))


def test_weight_set_add_vertex_and_return_vertices_array():
    weight_set = WeightSet()
    weight_set.add_vertex(np.array([0.8, 0.2], dtype=np.float32))
    weight_set.add_vertex(np.array([0.4, 0.6], dtype=np.float32))

    vertices = weight_set.get_vertices_array()

    assert vertices is not None
    assert vertices.shape == (2, 2)


def test_weight_set_store_groups_contexts_and_counts_vertices():
    store = WeightSetStore(num_objectives=2)
    store.observe_certified_weight(np.array([0.1] * 8, dtype=np.float32), np.array([0.8, 0.2], dtype=np.float32))
    store.observe_certified_weight(np.array([0.1] * 8, dtype=np.float32), np.array([0.4, 0.6], dtype=np.float32))
    store.observe_certified_weight(np.array([0.2] * 8, dtype=np.float32), np.array([0.1, 0.9], dtype=np.float32))

    assert store.context_count() == 2
    assert store.total_vertex_count() == 3


def test_weight_set_store_support_values_change_after_observing_weights():
    store = WeightSetStore(num_objectives=2)
    context = np.array([0.1] * 8, dtype=np.float32)

    before = store.get_support_values(context)
    store.observe_certified_weight(context, np.array([0.8, 0.2], dtype=np.float32))
    after = store.get_support_values(context)

    assert np.all(before >= 0.0)
    assert np.all(after >= 0.0)
    assert np.allclose(after, np.array([0.8, 0.2], dtype=np.float32))


def test_weight_set_store_get_all_support_targets_returns_context_value_pairs():
    store = WeightSetStore(num_objectives=2)
    store.observe_certified_weight(np.array([0.1] * 8, dtype=np.float32), np.array([0.8, 0.2], dtype=np.float32))

    targets = store.get_all_support_targets()

    assert len(targets) == 1
    context, support_values = targets[0]
    assert context.shape == (8,)
    assert support_values.shape == (2,)


def test_weight_set_store_save_load_round_trip(tmp_path: Path):
    store = WeightSetStore(num_objectives=2)
    store.observe_certified_weight(np.array([0.1] * 8, dtype=np.float32), np.array([0.8, 0.2], dtype=np.float32))
    store.observe_certified_weight(np.array([0.2] * 8, dtype=np.float32), np.array([0.3, 0.7], dtype=np.float32))

    save_path = tmp_path / "weight_set_store.json"
    store.save(save_path)
    loaded = WeightSetStore.load(save_path)

    assert loaded.context_count() == store.context_count()
    assert loaded.total_vertex_count() == store.total_vertex_count()
    original_support = store.get_support_values(np.array([0.1] * 8, dtype=np.float32))
    loaded_support = loaded.get_support_values(np.array([0.1] * 8, dtype=np.float32))
    assert np.allclose(original_support, loaded_support)
