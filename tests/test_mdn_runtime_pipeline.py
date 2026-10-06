"""Tests for runtime certification and persistence integration."""

from __future__ import annotations

import numpy as np
import pytest

from certification.certificate_schema import Certificate
from generator.mdn import MotiveDecompositionNetwork
from library.skill_library import SkillLibrary
from utils.mdn_contracts import CandidateSkillRecord
from utils.mdn_runtime_pipeline import (
    CertificationResult,
    RuntimeCertificationPipeline,
    RuntimePipelineConfig,
    certification_result_to_certificate_kwargs,
)
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
    assert result.weight_region_type == "FULL_SIMPLEX"
    assert result.certification_context is None
    assert result.mdn_alpha is None
    assert result.wx_support_directions is None
    assert result.wx_support_values is None
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
    certification_time_vertices = np.array([[0.7, 0.3]], dtype=np.float32)
    store.observe_certified_weight(context, certification_time_vertices[0])

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
    assert result.weight_region_type == "MDN_WX"
    assert result.certification_context == tuple(float(v) for v in context)
    assert result.mdn_alpha is not None
    assert all(value > 0.0 for value in result.mdn_alpha)
    assert result.wx_support_directions is not None
    assert result.wx_support_values is not None
    assert len(result.wx_support_values) == len(result.wx_support_directions)

    np.testing.assert_allclose(
        np.array(result.wx_support_directions, dtype=np.float32),
        np.eye(2, dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.array(result.wx_support_values, dtype=np.float32),
        np.array([0.7, 0.3], dtype=np.float32),
    )


def test_runtime_result_to_certificate_kwargs_preserves_audit_fields():
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

    kwargs = certification_result_to_certificate_kwargs(
        result,
        timestamp="2026-06-09T12:00:00+00:00",
        seed=7,
        gamma=0.99,
        baseline_id="idle_policy",
        environment="MO-LunarLander-v3",
        episode_length=100,
        version="test",
    )
    cert = Certificate(**kwargs)

    assert cert.weight_region_type == "MDN_WX"
    assert cert.certification_context == result.certification_context
    assert cert.mdn_alpha == result.mdn_alpha
    assert cert.wx_support_directions == result.wx_support_directions
    assert cert.wx_support_values == result.wx_support_values


def test_runtime_mdn_wx_certificate_promotes_to_skill_library():
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
    kwargs = certification_result_to_certificate_kwargs(
        result,
        timestamp="2026-06-09T12:00:00+00:00",
        seed=7,
        gamma=0.99,
        baseline_id="idle_policy",
        environment="MO-LunarLander-v3",
        episode_length=100,
        version="test",
    )
    cert = Certificate(**kwargs)
    library = SkillLibrary()

    promoted = library.add_skill(
        "skill_a",
        cert,
        lambda _obs=None: None,
        weight_region_type=cert.weight_region_type,
        certification_context=cert.certification_context,
        mdn_alpha=cert.mdn_alpha,
        wx_support_directions=cert.wx_support_directions,
        wx_support_values=cert.wx_support_values,
    )

    assert promoted is True
    assert library.get_skill("skill_a") is not None


def test_pds_candidate_epsilon_is_preserved_in_runtime_certificate():
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    store = WeightSetStore(num_objectives=2)
    config = RuntimePipelineConfig(
        gate_type="PDS",
        pds_epsilon=0.1,
        train_support_after_certify=False,
    )
    pipeline = RuntimeCertificationPipeline(model=model, weight_store=store, config=config)
    context = np.array([0.1] * 8, dtype=np.float32)
    candidate = CandidateSkillRecord(
        skill_id="skill_pds",
        delta_r=0.0,
        delta_n=(-0.25, 0.0),
        is_certified=False,
        gate_type="PDS",
        admission_margin=None,
        epsilon=0.3,
    )

    updated = pipeline.certify_candidate_skills(
        context=context,
        candidate_skills=[candidate],
        baseline_stats=_baseline_stats(),
        weights_used=np.array([0.5, 0.5], dtype=np.float32),
    )
    result = pipeline.get_certification_result(context=context, skill_id="skill_pds")

    assert updated[0].is_certified is True
    assert updated[0].epsilon == pytest.approx(0.3)
    assert result is not None
    assert result.epsilon == pytest.approx(0.3)

    kwargs = certification_result_to_certificate_kwargs(
        result,
        timestamp="2026-06-09T12:00:00+00:00",
        seed=7,
        gamma=0.99,
        baseline_id="idle_policy",
        environment="MO-LunarLander-v3",
        episode_length=100,
        version="test",
    )
    cert = Certificate(**kwargs)

    assert cert.epsilon == pytest.approx(0.3)


def test_runtime_result_to_certificate_kwargs_rejects_unsupported_gate_type():
    result = CertificationResult(
        skill_id="skill_cvar",
        is_certified=True,
        gate_type="CVAR",
        was_already_certified=False,
        admission_margin=0.1,
        delta_r=1.0,
        delta_n=(0.2, 0.3),
        weight_region_type="MDN_WX",
        certification_context=(0.1, 0.2),
        mdn_alpha=(1.0, 2.0),
        wx_support_directions=((0.0, 0.1),),
        wx_support_values=(0.03,),
    )

    with pytest.raises(ValueError, match="Unsupported certificate gate_type"):
        certification_result_to_certificate_kwargs(
            result,
            timestamp="2026-06-09T12:00:00+00:00",
            seed=7,
            gamma=0.99,
            baseline_id="idle_policy",
            environment="MO-LunarLander-v3",
            episode_length=100,
            version="test",
        )


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


def test_certify_candidate_skills_returns_newly_certified_record():
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    store = WeightSetStore(num_objectives=2)
    config = RuntimePipelineConfig(gate_type="CDS", train_support_after_certify=False)
    pipeline = RuntimeCertificationPipeline(model=model, weight_store=store, config=config)

    context = np.array([0.1] * 8, dtype=np.float32)
    candidates = [
        CandidateSkillRecord(
            skill_id="skill_a",
            delta_r=0.7,
            delta_n=(0.3, 0.2),
            is_certified=False,
            gate_type="CDS",
            admission_margin=0.9,
            epsilon=0.0,
        )
    ]

    updated = pipeline.certify_candidate_skills(
        context=context,
        candidate_skills=candidates,
        baseline_stats=_baseline_stats(),
        weights_used=np.array([0.5, 0.5], dtype=np.float32),
    )

    assert len(updated) == 1
    assert updated[0].is_certified is True


def test_certify_candidate_skills_returns_certified_record_from_permanence_cache():
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    store = WeightSetStore(num_objectives=2)
    config = RuntimePipelineConfig(gate_type="CDS", train_support_after_certify=False)
    pipeline = RuntimeCertificationPipeline(model=model, weight_store=store, config=config)

    context = np.array([0.1] * 8, dtype=np.float32)
    pipeline.certify_skill(
        context=context,
        skill_id="skill_a",
        skill_payoff=1.7,
        skill_motives=np.array([0.8, 0.4], dtype=np.float32),
        baseline_stats=_baseline_stats(),
        weights_used=np.array([0.5, 0.5], dtype=np.float32),
    )

    stale_candidate = CandidateSkillRecord(
        skill_id="skill_a",
        delta_r=-1.0,
        delta_n=(-1.0, -1.0),
        is_certified=False,
        gate_type="CDS",
        admission_margin=-2.0,
        epsilon=0.0,
    )

    updated = pipeline.certify_candidate_skills(
        context=context,
        candidate_skills=[stale_candidate],
        baseline_stats=_baseline_stats(),
        weights_used=np.array([0.5, 0.5], dtype=np.float32),
    )

    assert len(updated) == 1
    assert updated[0].is_certified is True
    assert updated[0].delta_r != stale_candidate.delta_r


# ---------------------------------------------------------------------------
# A cached certificate stays as history, but eligibility follows the CURRENT
# risk budgets. Without the recheck, a skill certified at epsilon 0.10 kept
# being admitted (and selected) after epsilon fell to 0.00, although a fresh
# check rejects it.
# ---------------------------------------------------------------------------

BORDERLINE_DELTA_R = 0.0
BORDERLINE_DELTA_N = (0.2, -0.05)  # PDS margin over the full simplex: -0.05
UNIFORM = np.array([0.5, 0.5], dtype=np.float32)


def _pds_pipeline(*, use_cvar: bool = False) -> RuntimeCertificationPipeline:
    return RuntimeCertificationPipeline(
        model=MotiveDecompositionNetwork(input_dim=8, num_objectives=2),
        weight_store=WeightSetStore(num_objectives=2),
        config=RuntimePipelineConfig(
            gate_type="PDS",
            use_cvar=use_cvar,
            train_support_after_certify=False,
        ),
    )


def _borderline(epsilon: float) -> CandidateSkillRecord:
    return CandidateSkillRecord(
        skill_id="borderline",
        delta_r=BORDERLINE_DELTA_R,
        delta_n=BORDERLINE_DELTA_N,
        is_certified=False,
        gate_type="PDS",
        epsilon=epsilon,
    )


def _certify_batch(pipeline, context, epsilon, **kwargs):
    (record,) = pipeline.certify_candidate_skills(
        context=context,
        candidate_skills=[_borderline(epsilon)],
        baseline_stats=_baseline_stats(),
        weights_used=UNIFORM,
        **kwargs,
    )
    return record


def test_cached_certificate_is_rechecked_when_epsilon_tightens():
    """Same pipeline, same context, same skill: epsilon 0.10 -> 0.00.

    Margin -0.05 passes at epsilon 0.10 and fails at 0.00. The stored
    certificate must survive unchanged; the record returned for this call must
    carry the current verdict and budget.

    Certifying also records the uniform weight into this context's W_x. Over
    {uniform} alone the margin would be +0.075, so the recheck must use the
    region the certificate was issued under (the full simplex), not the W_x
    that certification itself grew -- as a fresh check would.
    """
    pipeline = _pds_pipeline()
    context = np.array([0.1] * 8, dtype=np.float32)

    loose = _certify_batch(pipeline, context, 0.10)
    assert loose.is_certified is True
    assert loose.epsilon == pytest.approx(0.10)

    tight = _certify_batch(pipeline, context, 0.0)
    assert tight.is_certified is False
    assert tight.epsilon == pytest.approx(0.0)
    assert tight.admission_margin < 0.0

    historical = pipeline.get_certification_result(context=context, skill_id="borderline")
    assert historical.is_certified is True
    assert historical.epsilon == pytest.approx(0.10)


def test_recheck_matches_a_fresh_pipeline():
    """The cached path and a fresh check must agree under the new budget."""
    context = np.array([0.1] * 8, dtype=np.float32)

    reused = _pds_pipeline()
    _certify_batch(reused, context, 0.10)
    cached_verdict = _certify_batch(reused, context, 0.0).is_certified

    fresh_verdict = _certify_batch(_pds_pipeline(), context, 0.0).is_certified

    assert cached_verdict == fresh_verdict is False


def test_skill_is_admitted_again_when_the_budget_loosens():
    pipeline = _pds_pipeline()
    context = np.array([0.1] * 8, dtype=np.float32)

    assert _certify_batch(pipeline, context, 0.10).is_certified is True
    assert _certify_batch(pipeline, context, 0.0).is_certified is False
    assert _certify_batch(pipeline, context, 0.10).is_certified is True


def test_rejected_recheck_does_not_record_the_weight():
    """W_x records weights only for skills eligible under the current budget."""
    pipeline = _pds_pipeline()
    context = np.array([0.1] * 8, dtype=np.float32)

    _certify_batch(pipeline, context, 0.10)
    vertices_before = len(pipeline.weight_store.get_weight_set(context).vertices)

    _certify_batch(pipeline, context, 0.0)

    assert len(pipeline.weight_store.get_weight_set(context).vertices) == vertices_before


def test_single_skill_path_rechecks_a_cached_rejection():
    """certify_skill caches rejections too; a looser budget re-admits, and the
    stored rejection stays as it was."""
    pipeline = _pds_pipeline()
    context = np.array([0.1] * 8, dtype=np.float32)
    kwargs = dict(
        context=context,
        skill_id="borderline",
        # Baseline is payoff 1.0, motives (0.5, 0.2): delta (0.0, (0.2, -0.05)).
        skill_payoff=1.0,
        skill_motives=np.array([0.7, 0.15], dtype=np.float32),
        baseline_stats=_baseline_stats(),
    )

    rejected = pipeline.certify_skill(**kwargs, epsilon=0.0)
    assert rejected.is_certified is False

    admitted = pipeline.certify_skill(**kwargs, epsilon=0.10)
    assert admitted.is_certified is True
    assert admitted.epsilon == pytest.approx(0.10)

    stored = pipeline.get_certification_result(context=context, skill_id="borderline")
    assert stored.is_certified is False
    assert stored.epsilon == pytest.approx(0.0)


def test_unchanged_budgets_reuse_the_stored_result():
    """No budget change, no recheck: today's permanence behaviour holds."""
    pipeline = _pds_pipeline()
    context = np.array([0.1] * 8, dtype=np.float32)
    kwargs = dict(
        context=context,
        skill_id="borderline",
        skill_payoff=1.0,
        skill_motives=np.array([0.7, 0.15], dtype=np.float32),
        baseline_stats=_baseline_stats(),
        epsilon=0.10,
    )

    first = pipeline.certify_skill(**kwargs)
    second = pipeline.certify_skill(**kwargs)

    assert second is first


def test_cvar_tail_level_change_triggers_a_recheck(monkeypatch):
    """The CVaR tail level is a risk budget too, when a CVaR gate is in use."""
    pipeline = _pds_pipeline(use_cvar=True)
    context = np.array([0.1] * 8, dtype=np.float32)

    calls = []
    original = pipeline._recheck_eligibility

    def spy(*args, **kwargs):
        calls.append(kwargs["cvar_confidence"])
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_recheck_eligibility", spy)

    # PDS admits at epsilon 0.10 (margin -0.05), so under OR semantics the
    # first call certifies and caches regardless of the untrained MDN.
    first = _certify_batch(pipeline, context, 0.10, cvar_confidence=0.10)
    assert first.is_certified is True

    _certify_batch(pipeline, context, 0.10, cvar_confidence=0.10)
    assert calls == [], "same budgets must reuse the cached verdict"

    _certify_batch(pipeline, context, 0.10, cvar_confidence=0.05)
    assert calls == [pytest.approx(0.05)]


def test_cvar_tail_level_is_ignored_without_a_cvar_gate(monkeypatch):
    pipeline = _pds_pipeline(use_cvar=False)
    context = np.array([0.1] * 8, dtype=np.float32)

    calls = []
    monkeypatch.setattr(
        pipeline,
        "_recheck_eligibility",
        lambda *a, **k: calls.append(k) or (True, 0.0),
    )

    _certify_batch(pipeline, context, 0.10, cvar_confidence=0.10)
    _certify_batch(pipeline, context, 0.10, cvar_confidence=0.05)

    assert calls == []
