"""Tests for the runtime MDN selector."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from certification.certificate_schema import Certificate
from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_runtime_selector import MDNRuntimeSelector, SelectionResult
from library.skill_library import SkillLibrary
from utils.mdn_contracts import CandidateSkillRecord
from utils.mdn_logging import serialize_decision_record


def _make_model(input_dim: int = 8, num_objectives: int = 2) -> MotiveDecompositionNetwork:
    return MotiveDecompositionNetwork(input_dim=input_dim, num_objectives=num_objectives)


def _make_candidate(skill_id: str, delta_r: float, delta_n: tuple, certified: bool) -> CandidateSkillRecord:
    return CandidateSkillRecord(
        skill_id=skill_id,
        delta_r=delta_r,
        delta_n=delta_n,
        is_certified=certified,
        gate_type="CDS",
        admission_margin=delta_r + min(delta_n) if certified else None,
        epsilon=0.0,
        baseline_id=None,
    )


def _obs(dim: int = 8) -> np.ndarray:
    return np.zeros(dim, dtype=np.float32)


def _make_certificate(skill_id: str, delta_r: float, delta_n: tuple[float, float]) -> Certificate:
    return Certificate(
        skill_id=skill_id,
        gate_type="CDS",
        delta_r=delta_r,
        delta_n=delta_n,
        admission_margin=delta_r + min(delta_n),
        epsilon=0.0,
        timestamp=datetime.now().isoformat(),
        seed=0,
        gamma=1.0,
        baseline_id="default",
        environment="mo-lunar-lander-v3",
        episode_length=1,
        version="1.0",
    )


class TestMDNRuntimeSelectorSelect:
    def test_returns_selection_result(self):
        model = _make_model()
        selector = MDNRuntimeSelector(model)
        candidates = [_make_candidate("skill_a", 0.5, (0.3, 0.2), True)]
        result = selector.select(_obs(), candidates)
        assert isinstance(result, SelectionResult)

    def test_selected_skill_is_certified(self):
        model = _make_model()
        selector = MDNRuntimeSelector(model)
        candidates = [
            _make_candidate("skill_a", 0.5, (0.3, 0.2), True),
            _make_candidate("skill_b", -0.5, (-0.3, -0.2), False),
        ]
        result = selector.select(_obs(), candidates)
        assert result.selected_skill_id == "skill_a"

    def test_selects_higher_scoring_certified_skill(self):
        model = _make_model()
        selector = MDNRuntimeSelector(model)
        candidates = [
            _make_candidate("skill_low", 0.1, (0.1, 0.1), True),
            _make_candidate("skill_high", 0.9, (0.5, 0.5), True),
        ]
        result = selector.select(_obs(), candidates)
        assert result.selected_skill_id in {"skill_low", "skill_high"}

    def test_alpha_and_support_are_finite(self):
        model = _make_model()
        selector = MDNRuntimeSelector(model)
        candidates = [_make_candidate("skill_a", 0.5, (0.3, 0.2), True)]
        result = selector.select(_obs(), candidates)
        assert np.all(np.isfinite(result.alpha))
        assert np.all(np.isfinite(result.support_values))

    def test_weights_sum_to_one(self):
        model = _make_model()
        selector = MDNRuntimeSelector(model)
        candidates = [_make_candidate("skill_a", 0.5, (0.3, 0.2), True)]
        result = selector.select(_obs(), candidates)
        assert abs(np.sum(result.weights_used) - 1.0) < 1e-5

    def test_behavior_probability_is_valid(self):
        model = _make_model()
        selector = MDNRuntimeSelector(model)
        candidates = [
            _make_candidate("skill_a", 0.5, (0.3, 0.2), True),
            _make_candidate("skill_b", 0.3, (0.2, 0.1), True),
        ]
        result = selector.select(_obs(), candidates)
        assert result.behavior_probability == pytest.approx(1.0)

    def test_behavior_probability_is_one_with_single_certified_candidate(self):
        model = _make_model()
        selector = MDNRuntimeSelector(model)
        candidates = [_make_candidate("skill_a", 0.5, (0.3, 0.2), True)]
        result = selector.select(_obs(), candidates)
        assert abs(result.behavior_probability - 1.0) < 1e-5

    def test_select_forces_model_eval_mode(self):
        model = _make_model()
        selector = MDNRuntimeSelector(model)
        selector.model.train()
        candidates = [_make_candidate("skill_a", 0.5, (0.3, 0.2), True)]

        selector.select(_obs(), candidates)

        assert selector.model.training is False

    def test_context_stored_correctly(self):
        model = _make_model()
        selector = MDNRuntimeSelector(model)
        obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        candidates = [_make_candidate("skill_a", 0.5, (0.3, 0.2), True)]
        result = selector.select(obs, candidates)
        assert len(result.context) == 8
        assert abs(result.context[0] - 0.1) < 1e-6

    def test_only_selection_eligible_candidates_stored_in_result(self):
        model = _make_model()
        selector = MDNRuntimeSelector(model)
        candidates = [
            _make_candidate("skill_a", 0.5, (0.3, 0.2), True),
            _make_candidate("skill_b", -0.1, (-0.1, -0.1), False),
        ]
        result = selector.select(_obs(), candidates)
        assert len(result.candidate_skills) == 1
        assert result.candidate_skills[0].skill_id == "skill_a"

    def test_raises_if_no_certified_candidates(self):
        model = _make_model()
        selector = MDNRuntimeSelector(model)
        candidates = [_make_candidate("skill_b", -0.5, (-0.3, -0.2), False)]
        with pytest.raises(ValueError, match="certified"):
            selector.select(_obs(), candidates)

    def test_raises_if_observation_wrong_dim(self):
        model = _make_model(input_dim=8)
        selector = MDNRuntimeSelector(model)
        candidates = [_make_candidate("skill_a", 0.5, (0.3, 0.2), True)]
        with pytest.raises(ValueError, match="8"):
            selector.select(np.zeros(14, dtype=np.float32), candidates)

    def test_raises_if_observation_non_finite(self):
        model = _make_model()
        selector = MDNRuntimeSelector(model)
        candidates = [_make_candidate("skill_a", 0.5, (0.3, 0.2), True)]
        with pytest.raises(ValueError, match="finite"):
            selector.select(np.full(8, float("nan"), dtype=np.float32), candidates)

    def test_select_from_library_returns_loggable_result(self):
        model = _make_model()
        selector = MDNRuntimeSelector(model)
        library = SkillLibrary()
        certificate = _make_certificate("skill_a", 0.5, (0.3, 0.2))
        assert library.add_skill("skill_a", certificate, lambda obs: None)

        result = selector.select_from_library(_obs(), library)

        assert result.selected_skill_id == "skill_a"
        assert result.behavior_probability == pytest.approx(1.0)
        assert result.candidate_skills[0].metadata["weight_region_type"] == "FULL_SIMPLEX"

    def test_select_from_library_logs_deterministic_behavior_probability(self):
        model = _make_model()
        selector = MDNRuntimeSelector(model)
        library = SkillLibrary()
        assert library.add_skill("skill_a", _make_certificate("skill_a", 0.5, (0.3, 0.2)), lambda obs: None)
        assert library.add_skill("skill_b", _make_certificate("skill_b", 0.4, (0.2, 0.1)), lambda obs: None)

        result = selector.select_from_library(_obs(), library)

        assert result.behavior_probability == pytest.approx(1.0)


class TestSelectionResultBuildDecisionRecord:
    def _get_result(self) -> SelectionResult:
        model = _make_model()
        selector = MDNRuntimeSelector(model)
        candidates = [_make_candidate("skill_a", 0.5, (0.3, 0.2), True)]
        return selector.select(_obs(), candidates)

    def test_builds_decision_record_with_outcome(self):
        from utils.mdn_contracts import MDNDecisionRecord
        result = self._get_result()
        record = result.build_decision_record(actual_payoff=1.2, actual_motives=(0.8, 0.4))
        assert isinstance(record, MDNDecisionRecord)
        assert record.selected_skill_id == "skill_a"

    def test_utility_computed_automatically(self):
        result = self._get_result()
        record = result.build_decision_record(actual_payoff=1.0, actual_motives=(0.5, 0.5))
        assert record.utility is not None
        assert np.isfinite(record.utility)

    def test_behavior_probability_is_preserved_in_decision_record(self):
        result = self._get_result()
        record = result.build_decision_record(actual_payoff=1.0, actual_motives=(0.5, 0.5))
        assert record.behavior_probability is not None
        assert np.isclose(record.behavior_probability, result.behavior_probability)

    def test_behavior_probability_is_serialized(self):
        result = self._get_result()
        record = result.build_decision_record(actual_payoff=1.0, actual_motives=(0.5, 0.5))
        payload = serialize_decision_record(record)
        assert payload["behavior_probability"] is not None
        assert np.isclose(float(payload["behavior_probability"]), result.behavior_probability)

    def test_builds_without_outcome(self):
        from utils.mdn_contracts import MDNDecisionRecord
        result = self._get_result()
        record = result.build_decision_record()
        assert isinstance(record, MDNDecisionRecord)
        assert record.actual_payoff is None
        assert record.actual_motives is None
