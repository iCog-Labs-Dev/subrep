"""Runtime MDN selector — wires MDN inference into live skill selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from generator.mdn import MotiveDecompositionNetwork
from library.skill_library import SkillLibrary
from library.skill_metadata import SkillEntry
from library.skill_selector import select_best_skill_entry
from utils.mdn_contracts import CandidateSkillRecord, MDNDecisionRecord
from utils.mdn_logging import build_decision_record
from utils.mdn_reward import compute_mdn_utility
from utils.mdn_selection import (
    alpha_to_mean_weights,
    select_best_candidate,
)
from utils.support_geometry import make_basis_query_directions


@dataclass(frozen=True)
class SelectionResult:
    """Output of one MDN selection step, before the episode outcome is known."""

    selected_skill_id: str
    selected_score: float
    weights_used: np.ndarray
    alpha: np.ndarray
    support_values: np.ndarray
    behavior_probability: float
    candidate_skills: tuple[CandidateSkillRecord, ...]
    context: tuple[float, ...]

    def build_decision_record(
        self,
        *,
        actual_payoff: Optional[float] = None,
        actual_motives=None,
        utility: Optional[float] = None,
        payoff_weight: float = 0.0,
    ) -> MDNDecisionRecord:
        """Build a loggable MDNDecisionRecord once the episode outcome is known. 

        If actual_motives and actual_payoff are provided and utility is not,
        utility is computed automatically from weights_used.
        """
        if utility is None and actual_motives is not None:
            utility = compute_mdn_utility(
                actual_motives=actual_motives,
                weights_used=self.weights_used,
                actual_payoff=actual_payoff,
                payoff_weight=payoff_weight,
            )

        return build_decision_record(
            context=self.context,
            alpha=self.alpha,
            support_values=self.support_values,
            weights_used=self.weights_used,
            candidate_skills=self.candidate_skills,
            selected_skill_id=self.selected_skill_id,
            selected_score=self.selected_score,
            behavior_probability=self.behavior_probability,
            actual_payoff=actual_payoff,
            actual_motives=actual_motives,
            utility=utility,
        )


class MDNRuntimeSelector:
    """Wraps a trained MDN model for live skill selection.

    Typical usage per decision step:
        result = selector.select(obs, candidate_skills)
        # execute result.selected_skill_id in environment
        record = result.build_decision_record(actual_payoff=p, actual_motives=m)
    """

    def __init__(
        self,
        model: MotiveDecompositionNetwork,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

    def select(
        self,
        observation: np.ndarray,
        candidate_skills: tuple[CandidateSkillRecord, ...] | list[CandidateSkillRecord],
    ) -> SelectionResult:
        """Run MDN inference and select the best certified skill.

        Args:
            observation: Current environment observation — must match MDN input_dim.
            candidate_skills: All candidate skills (certified and uncertified).
                Only certified candidates are eligible for selection.

        Returns:
            SelectionResult with selected skill, weights, alpha, support_values,
            and deterministic behavior_probability for IPS logging.

        Raises:
            ValueError: If no certified candidates are available.
            ValueError: If observation shape does not match the MDN input_dim.
        """
        obs = self._validate_observation(observation)

        certified = [c for c in candidate_skills if c.is_certified]
        if not certified:
            raise ValueError("select() requires at least one certified candidate skill")

        alpha_np, support_np, weights = self._infer_mdn(obs)

        selected_skill_id, selected_score = select_best_candidate(certified, weights)

        return SelectionResult(
            selected_skill_id=selected_skill_id,
            selected_score=float(selected_score),
            weights_used=weights,
            alpha=alpha_np,
            support_values=support_np,
            behavior_probability=1.0,
            candidate_skills=tuple(certified),
            context=tuple(float(v) for v in obs),
        )

    def select_from_library(
        self,
        observation: np.ndarray,
        skill_library: SkillLibrary,
    ) -> SelectionResult:
        """Run MDN inference and select from stored, admissible SkillLibrary entries."""
        obs = self._validate_observation(observation)
        alpha_np, support_np, weights = self._infer_mdn(obs)
        support_directions = make_basis_query_directions(len(support_np))
        admissible_entries = skill_library.query_admissible(
            current_weight=weights,
            support_directions=support_directions,
            support_values=support_np,
        )
        if not admissible_entries:
            raise ValueError("select_from_library() requires at least one admissible stored skill")

        selected_skill_id, selected_score = select_best_skill_entry(
            admissible_entries,
            weights,
        )
        candidate_records = tuple(
            _candidate_record_from_entry(entry) for entry in admissible_entries
        )

        return SelectionResult(
            selected_skill_id=selected_skill_id,
            selected_score=float(selected_score),
            weights_used=weights,
            alpha=alpha_np,
            support_values=support_np,
            behavior_probability=1.0,
            candidate_skills=candidate_records,
            context=tuple(float(v) for v in obs),
        )

    def _validate_observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.model.input_dim:
            raise ValueError(
                f"observation has {obs.shape[0]} dimensions but MDN expects {self.model.input_dim}"
            )
        if not np.all(np.isfinite(obs)):
            raise ValueError("observation must contain only finite values")
        return obs

    def _infer_mdn(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        context_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            alpha_tensor, support_tensor = self.model.forward_inference(context_tensor)

        alpha_np = alpha_tensor.detach().cpu().numpy().reshape(-1)
        support_np = support_tensor.detach().cpu().numpy().reshape(-1)
        weights = alpha_to_mean_weights(alpha_np)
        return alpha_np, support_np, weights

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        input_dim: int,
        num_objectives: int,
        num_skills: int = 128,
        device: Optional[str] = None,
    ) -> "MDNRuntimeSelector":
        """Load a trained MDN from a checkpoint file."""
        model = MotiveDecompositionNetwork(
            input_dim=input_dim,
            num_objectives=num_objectives,
            num_skills=num_skills,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)
        return cls(model=model, device=device)


def _candidate_record_from_entry(entry: SkillEntry) -> CandidateSkillRecord:
    metadata = {
        "weight_region_type": entry.weight_region_type,
    }
    if entry.certification_context is not None:
        metadata["certification_context"] = entry.certification_context
    if entry.mdn_alpha is not None:
        metadata["mdn_alpha"] = entry.mdn_alpha
    if entry.wx_support_directions is not None:
        metadata["wx_support_directions"] = entry.wx_support_directions
    if entry.wx_support_values is not None:
        metadata["wx_support_values"] = entry.wx_support_values

    return CandidateSkillRecord(
        skill_id=entry.skill_id,
        delta_r=entry.delta_r,
        delta_n=entry.delta_n,
        is_certified=True,
        gate_type=entry.gate_type,
        metadata=metadata,
        admission_margin=entry.admission_margin,
        epsilon=entry.epsilon,
        baseline_id=entry.certificate.baseline_id,
    )
