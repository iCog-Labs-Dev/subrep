"""
Skill Selector — strategy layer for choosing which skill to execute.

Sits on top of SkillLibrary and provides different selection strategies:

    Stage 3-4:   select_random()  — uniform random baseline
    Stage 5: select_by_payoff()   — greedy via SkillGenerator
    Stage 6: select_by_mdn()      — MDN-weighted contextual
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from .skill_library import SkillLibrary


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
        """
        Select the skill with the highest MDN-weighted score.

        Uses the MotiveDecompositionNetwork to get context-aware weights w,
        then scores each skill as:

            score = r̂ + w^T n̂

        where r̂ is the predicted payoff and n̂ is the predicted motive vector.
        This combines the Generator's predictions with the MDN's contextual
        weighting to balance payoff against motive trade-offs.

        Args:
            obs: Current environment observation (8D for LunarLander).

        """
        raise NotImplementedError(
            "select_by_mdn() requires SkillGenerator + MDN integration (Stage 6). "
            "Use select_random() for the current baseline."
        )
