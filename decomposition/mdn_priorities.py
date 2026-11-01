"""Priorities from a trained Motive Decomposition Network.

At each decision the MDN reads the observation and gives Dirichlet parameters
alpha and support values s. The weights are the mean of the Dirichlet, and the
region of plausible weights is W_x = {w in the simplex : w_i <= s_i}, which the
gates certify over.

An MDN can give weights outside its own region. Those priorities contradict
themselves, so they are refused: the source returns None, and the runtime then
flies nothing under them, as it does when no plan is certified.

The training follows the village MDN: the network is fitted to a schedule of
weights by situation, a stand-in for MetaMo until it sends priorities itself.

    alpha*(x) = CONCENTRATION * w_bar(x)
    s*(x)     = min(1, w_bar(x) + MOTIVE_SHIFT)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from decomposition.verifier import MdnRegion, Priorities
from generator.mdn import MotiveDecompositionNetwork

CONCENTRATION = 50.0
MOTIVE_SHIFT = 0.05
CHECKPOINT_VERSION = "1.0"
_NUM_SKILLS = 1  # the MDN's auxiliary skill heads are not trained here

Schedule = Callable[[np.ndarray], Sequence[float]]


def priorities_from_weights(
    weights: Sequence[float], cvar_tail_level: Optional[float] = None, pds_epsilon: float = 0.0
) -> Priorities:
    """Fixed priorities with the region an MDN trained on these weights would give."""
    values = np.asarray(weights, dtype=np.float64)
    region = MdnRegion(
        context=(0.0,),
        alpha=tuple(float(value) for value in CONCENTRATION * values),
        support_directions=tuple(tuple(map(float, row)) for row in np.eye(len(values))),
        support_values=tuple(float(value) for value in np.minimum(1.0, values + MOTIVE_SHIFT)),
    )
    return Priorities(
        weights=tuple(float(value) for value in values),
        pds_epsilon=pds_epsilon,
        cvar_tail_level=cvar_tail_level,
        region=region,
    )


class MdnPriorities:
    """Priorities for an observation, from a trained MDN."""

    def __init__(
        self,
        model: MotiveDecompositionNetwork,
        objectives: Sequence[str],
        cvar_tail_level: Optional[float] = None,
        pds_epsilon: float = 0.0,
    ) -> None:
        if model.num_objectives != len(objectives):
            raise ValueError(f"the MDN has {model.num_objectives} objectives, expected {len(objectives)}")
        self.model = model.eval()
        self.objectives = tuple(objectives)
        self.cvar_tail_level = cvar_tail_level
        self.pds_epsilon = pds_epsilon

    def __call__(self, observation: np.ndarray) -> Optional[Priorities]:
        """The priorities for ``observation``, or None if the MDN's weights lie outside its own region."""
        context = np.asarray(observation, dtype=np.float32)
        with torch.no_grad():
            alpha, support = self.model.forward_inference(torch.as_tensor(context))
        alpha_values = tuple(float(value) for value in alpha.numpy())
        support_values = tuple(float(value) for value in support.numpy())
        total = sum(alpha_values)
        weights = tuple(value / total for value in alpha_values)
        region = MdnRegion(
            context=tuple(float(value) for value in context),
            alpha=alpha_values,
            support_directions=tuple(tuple(map(float, row)) for row in np.eye(len(alpha_values))),
            support_values=support_values,
        )
        try:
            return Priorities(
                weights=weights, pds_epsilon=self.pds_epsilon, cvar_tail_level=self.cvar_tail_level, region=region
            )
        except ValueError:
            return None

    def save(self, path: Union[str, Path], metrics: Optional[Dict[str, float]] = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint_version": CHECKPOINT_VERSION,
                "objectives": list(self.objectives),
                "input_dim": self.model.input_dim,
                "hidden_dim": self.model.hidden_dim,
                "concentration": CONCENTRATION,
                "motive_shift": MOTIVE_SHIFT,
                "metrics": dict(metrics or {}),
                "state_dict": self.model.state_dict(),
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        objectives: Sequence[str],
        cvar_tail_level: Optional[float] = None,
        pds_epsilon: float = 0.0,
    ) -> "MdnPriorities":
        """Load a checkpoint, refusing one trained for other objectives."""
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported MDN checkpoint version {checkpoint.get('checkpoint_version')!r}")
        if checkpoint.get("objectives") != list(objectives):
            raise ValueError(f"MDN checkpoint was trained for objectives {checkpoint.get('objectives')}")
        model = MotiveDecompositionNetwork(
            input_dim=checkpoint["input_dim"],
            num_objectives=len(objectives),
            hidden_dim=checkpoint["hidden_dim"],
            num_skills=_NUM_SKILLS,
        )
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model, objectives, cvar_tail_level, pds_epsilon)


def train_mdn(
    observations: np.ndarray,
    schedule: Schedule,
    objectives: Sequence[str],
    epochs: int = 2000,
    learning_rate: float = 3e-3,
    hidden_dim: int = 64,
    seed: int = 0,
) -> Tuple[MdnPriorities, Dict[str, float]]:
    """Fit an MDN's weight and support heads to ``schedule`` over ``observations``."""
    torch.manual_seed(seed)
    contexts = torch.as_tensor(np.asarray(observations, dtype=np.float32))
    targets = np.asarray([schedule(observation) for observation in observations], dtype=np.float32)
    alpha_target = torch.as_tensor(CONCENTRATION * targets)
    support_target = torch.as_tensor(np.minimum(1.0, targets + MOTIVE_SHIFT))
    model = MotiveDecompositionNetwork(
        input_dim=contexts.shape[1], num_objectives=len(objectives), hidden_dim=hidden_dim, num_skills=_NUM_SKILLS
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        optimiser.zero_grad()
        alpha, support = model.forward_inference(contexts)
        alpha_loss = torch.mean((torch.log(alpha) - torch.log(alpha_target)) ** 2)
        support_loss = torch.mean((support - support_target) ** 2)
        (alpha_loss + support_loss).backward()
        optimiser.step()
    return MdnPriorities(model, objectives), {
        "alpha_log_mse": float(alpha_loss.item()),
        "support_mse": float(support_loss.item()),
    }


def evaluate_mdn(source: MdnPriorities, observations: np.ndarray, schedule: Schedule) -> Dict[str, float]:
    """Agreement with the schedule on held-out observations, and how often the weights lie in the region."""
    errors = []
    refused = 0
    for observation in observations:
        priorities = source(observation)
        if priorities is None:
            refused += 1
            continue
        errors.append(np.abs(np.asarray(priorities.weights) - np.asarray(schedule(observation))).max())
    return {
        "contexts": float(len(observations)),
        "refused_rate": refused / len(observations),
        "max_weight_error": float(np.max(errors)) if errors else float("nan"),
        "mean_weight_error": float(np.mean(errors)) if errors else float("nan"),
    }
