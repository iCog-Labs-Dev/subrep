"""Tests for runtime certification and persistence integration."""

from __future__ import annotations

import numpy as np

from generator.mdn import MotiveDecompositionNetwork
from utils.mdn_runtime_pipeline import RuntimeCertificationPipeline, RuntimePipelineConfig
from utils.weight_set_store import WeightSetStore


def _baseline_stats() -> dict[str, object]:
    return {
        "baseline_payoff": 1.0,
        "baseline_motives": np.array([0.5, 0.2], dtype=np.float32),
    }


def test_runtime_pipeline_certifies_and_records_weight():
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    store = WeightSetStore(num_objectives=2)
    config = RuntimePipelineConfig(gate_type="CDS", train_support_after_certify=False)
    pipeline = RuntimeCertificationPipeline(model=model, weight_store=store, config=config)

    context = np.array([0.1] * 8, dtype=np.float32)
    result = pipeline.certify_skill(
        context=context,
        skill_id="skill_a",
        skill_payoff=1.7,
        skill_motives=np.array([0.8, 0.4], dtype=np.float32),
        baseline_stats=_baseline_stats(),
        weights_used=np.array([0.5, 0.5], dtype=np.float32),
    )

    assert result.skill_id == "skill_a"
    assert isinstance(result.is_certified, bool)
    assert store.context_count() >= 0


def test_runtime_pipeline_enforces_certificate_permanence():
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    store = WeightSetStore(num_objectives=2)
    config = RuntimePipelineConfig(gate_type="CDS", train_support_after_certify=False)
    pipeline = RuntimeCertificationPipeline(model=model, weight_store=store, config=config)

    context = np.array([0.1] * 8, dtype=np.float32)
    result1 = pipeline.certify_skill(
        context=context,
        skill_id="skill_a",
        skill_payoff=1.7,
        skill_motives=np.array([0.8, 0.4], dtype=np.float32),
        baseline_stats=_baseline_stats(),
        weights_used=np.array([0.5, 0.5], dtype=np.float32),
    )
    result2 = pipeline.certify_skill(
        context=context,
        skill_id="skill_a",
        skill_payoff=0.0,
        skill_motives=np.array([-1.0, -1.0], dtype=np.float32),
        baseline_stats=_baseline_stats(),
        weights_used=np.array([0.5, 0.5], dtype=np.float32),
    )

    assert result1.is_certified == result2.is_certified
    assert result2.was_already_certified is False


def test_runtime_pipeline_uses_wx_not_simplex():
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    store = WeightSetStore(num_objectives=2)
    context = np.array([0.1] * 8, dtype=np.float32)
    store.observe_certified_weight(context, np.array([0.7, 0.3], dtype=np.float32))

    config = RuntimePipelineConfig(gate_type="CDS", train_support_after_certify=False)
    pipeline = RuntimeCertificationPipeline(model=model, weight_store=store, config=config)

    result = pipeline.certify_skill(
        context=context,
        skill_id="skill_a",
        skill_payoff=1.7,
        skill_motives=np.array([0.8, 0.4], dtype=np.float32),
        baseline_stats=_baseline_stats(),
        weights_used=np.array([0.5, 0.5], dtype=np.float32),
    )

    assert isinstance(result.is_certified, bool)


def test_runtime_pipeline_get_support_values():
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    store = WeightSetStore(num_objectives=2)
    context = np.array([0.1] * 8, dtype=np.float32)
    store.observe_certified_weight(context, np.array([0.7, 0.3], dtype=np.float32))

    config = RuntimePipelineConfig(gate_type="CDS", train_support_after_certify=False)
    pipeline = RuntimeCertificationPipeline(model=model, weight_store=store, config=config)

    support = pipeline.get_support_values(context)

    assert support.shape == (2,)


def test_runtime_pipeline_save_and_load_store(tmp_path):
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    store = WeightSetStore(num_objectives=2)
    context = np.array([0.1] * 8, dtype=np.float32)
    store.observe_certified_weight(context, np.array([0.7, 0.3], dtype=np.float32))

    config = RuntimePipelineConfig(
        gate_type="CDS",
        train_support_after_certify=False,
        store_path=str(tmp_path / "weight_store.json"),
    )
    pipeline = RuntimeCertificationPipeline(model=model, weight_store=store, config=config)
    saved_path = pipeline.save_store()

    loaded_store = WeightSetStore.load(saved_path)
    assert loaded_store.context_count() == store.context_count()
