from __future__ import annotations

import numpy as np
import pytest

from generator.evaluate_mdn_support import (
    evaluate_support_models,
    evaluate_support_predictions,
)
from utils.mdn_record_builder import PreparedCandidateOutcome
from utils.mdn_stub import StubMDN
from utils.weight_set_store import WeightSetStore


def test_support_prediction_metrics_include_error_feasibility_and_context_variation():
    targets = np.array([[0.8, 0.6], [0.4, 0.7]], dtype=np.float32)
    predictions = np.array([[0.8, 0.2], [0.4, 0.8]], dtype=np.float32)

    metrics = evaluate_support_predictions(predictions, targets)

    assert metrics["support_region_feasibility_violation_rate"] == 0.0
    assert metrics["support_value_mse"] == pytest.approx(0.0425)
    assert metrics["support_value_mae"] == pytest.approx(0.125)
    assert metrics["support_coordinate_underestimate_rate"] == pytest.approx(0.25)
    assert metrics["support_coordinate_overestimate_rate"] == pytest.approx(0.25)
    assert metrics["mean_prediction_context_variance"] > 0.0


def test_support_prediction_metrics_detect_infeasible_rows():
    targets = np.array([[0.8, 0.6], [0.4, 0.7]], dtype=np.float32)
    predictions = np.array([[0.4, 0.4], [1.1, 0.2]], dtype=np.float32)

    metrics = evaluate_support_predictions(predictions, targets)

    assert metrics["support_region_feasibility_violation_rate"] == 1.0


def test_support_evaluation_reports_admission_and_reuse_failures_by_arm():
    context = np.full(8, 0.1, dtype=np.float32)
    store = WeightSetStore(num_objectives=2)
    store.observe_certified_weight(context, np.array([0.8, 0.2], dtype=np.float32))
    store.observe_certified_weight(context, np.array([0.4, 0.6], dtype=np.float32))
    outcomes = (
        PreparedCandidateOutcome(
            context=tuple(context),
            skill_id="safe_specialist",
            payoff=0.0,
            motives=(0.2, -0.1),
            gate_type="CDS",
        ),
        PreparedCandidateOutcome(
            context=tuple(context),
            skill_id="unsafe_specialist",
            payoff=0.0,
            motives=(0.1, -0.2),
            gate_type="PDS",
            epsilon=0.0,
        ),
    )
    learned_model = StubMDN(
        input_dim=8,
        num_objectives=2,
        fixed_support_values=[0.8, 0.2],
    )

    report = evaluate_support_models(
        learned_model,
        store,
        candidate_outcomes=outcomes,
        baseline_stats={"baseline_payoff": 0.0, "baseline_motives": (0.0, 0.0)},
        device="cpu",
    )

    learned = report["arms"]["learned"]
    full_simplex = report["arms"]["full_simplex"]
    stub = report["arms"]["stub_mdn"]
    assert learned["support_value_mse"] == pytest.approx(0.08)
    assert full_simplex["support_value_mse"] == pytest.approx(0.10)
    assert learned["cds_agreement_rate"] == 1.0
    assert learned["pds_agreement_rate"] == 0.0
    assert learned["false_admission_rate"] == 1.0
    assert learned["false_rejection_rate"] == 0.0
    assert full_simplex["false_admission_rate"] == 0.0
    assert full_simplex["false_rejection_rate"] == 1.0
    assert learned["reuse_shift_opportunities"] == 2.0
    assert learned["reuse_motive_shift_context_coverage"] == 1.0
    assert learned["reuse_success_rate"] == 1.0
    assert learned["reuse_oracle_top1_agreement_rate"] == 1.0
    assert full_simplex["reuse_success_rate"] == 0.0
    assert report["stub_full_simplex_support_equivalent"] is True
    assert stub["support_value_mse"] == full_simplex["support_value_mse"]
    assert stub["false_rejection_rate"] == full_simplex["false_rejection_rate"]


def test_downstream_metrics_are_unavailable_without_matching_candidate_data():
    context = np.full(8, 0.1, dtype=np.float32)
    store = WeightSetStore(num_objectives=2)
    store.observe_certified_weight(context, np.array([0.8, 0.2], dtype=np.float32))
    model = StubMDN(input_dim=8, num_objectives=2)

    report = evaluate_support_models(model, store)

    assert np.isnan(report["arms"]["learned"]["cds_agreement_rate"])
    assert np.isnan(report["arms"]["learned"]["reuse_success_rate"])
    assert report["arms"]["learned"]["candidate_context_coverage"] == 0.0


def test_training_mean_baseline_is_reported_without_test_target_leakage():
    context = np.full(8, 0.1, dtype=np.float32)
    store = WeightSetStore(num_objectives=2)
    store.observe_certified_weight(context, np.array([0.8, 0.2], dtype=np.float32))
    model = StubMDN(
        input_dim=8,
        num_objectives=2,
        fixed_support_values=[0.8, 0.2],
    )

    report = evaluate_support_models(
        model,
        store,
        constant_baseline_support=np.array([0.6, 0.4], dtype=np.float32),
    )

    assert report["arms"]["train_mean_constant"]["support_value_mse"] == pytest.approx(0.04)
    assert report["comparisons"]["learned_support_mse_improvement_vs_train_mean_constant"] == pytest.approx(0.04)
