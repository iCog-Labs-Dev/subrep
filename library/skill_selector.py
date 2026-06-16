"""
Skill Selector — strategy layer for choosing which skill to execute.

Sits on top of SkillLibrary and provides different selection strategies:

    Stage 3-4:   select_random()  — uniform random baseline
    Stage 5: select_by_payoff()   — greedy via SkillGenerator
    Stage 6: select_by_mdn()      — MDN-weighted contextual with W_x admissibility
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import torch
from .skill_library import SkillLibrary
from utils.mdn_selection import alpha_to_mean_weights
from utils.support_geometry import make_basis_query_directions


class SkillSelector:
    """
    Choose a skill from the library using a pluggable strategy.

    Attributes:
        library:   The SkillLibrary to select from.
        generator: Optional SkillGenerator for payoff-based selection (Stage 5).
        mdn:       Optional MotiveDecompositionNetwork for MDN selection (Stage 6).
        _rng:      Isolated random state — uses numpy's Generator API so
                   selections are reproducible with the same seed and never
                   interfere with other random generators in the pipeline.
    """

    def __init__(
        self,
        library: SkillLibrary,
        generator=None,
        mdn=None,
        seed: int = 42,
    ) -> None:
        """
        Initialize the selector.
        """
        self.library = library
        self.generator = generator
        self.mdn = mdn
        self._rng = np.random.default_rng(seed)


    def select_random(self, obs: np.ndarray) -> Optional[str]:
        """
        Uniformly random selection from all admitted skills.

        Args:
            obs: Current environment observation.
                 present for interface consistency with other selectors.
        """
        skills = self.library.get_admitted_skills()

        if not skills:
            return None

        # Build list of skill IDs then pick one uniformly at random.
        skill_ids = [s.skill_id for s in skills]
        idx = self._rng.integers(0, len(skill_ids))
        return skill_ids[idx]

    def select_by_payoff(self, obs: np.ndarray) -> Optional[str]:
        """
        Select the skill with the highest predicted payoff.

        Uses the SkillGenerator to predict (payoff, motives) for each skill,
        then picks the one with the highest scalar payoff.

        Args:
            obs: Current environment observation (8D for LunarLander).

        """
        raise NotImplementedError(
            "select_by_payoff() requires SkillGenerator integration(Stage 5). "
            "Use select_random() for the current baseline."
        )

    def select_by_mdn(self, obs: np.ndarray) -> Optional[str]:
        """Select the best admissible skill using MDN alpha and W_x support"""

        if self.mdn is None:
            raise ValueError(
                "select_by_mdn() requires a trained MotiveDecompositionNetwork"
            )

        # skip MDN inference when library is empty
        if self.library.count() == 0:
            return None

        obs_tensor = torch.tensor(
            np.asarray(obs, dtype=np.float32), dtype=torch.float32
        )

        with torch.no_grad():
            alpha, support_pred = self.mdn.forward_inference(obs_tensor)

        alpha_np = alpha.cpu().numpy()
        weight = alpha_to_mean_weights(alpha_np)
        support_values = support_pred.cpu().numpy()

        num_objectives = len(support_values)
        support_directions = make_basis_query_directions(num_objectives)

        admissible = self.library.query_admissible(
            current_weight=weight,
            support_directions=support_directions,
            support_values=support_values,
        )

        if not admissible:
            return None

        w = np.asarray(weight, dtype=np.float64).reshape(-1)

        best_id: Optional[str] = None
        best_score = -float("inf")

        for entry in admissible:
            delta_n = np.asarray(entry.delta_n, dtype=np.float64)
            score = float(entry.delta_r + np.dot(w, delta_n))

            if score > best_score or (
                score == best_score
                and (best_id is None or entry.skill_id < best_id)
            ):
                best_id = entry.skill_id
                best_score = score

        return best_id
