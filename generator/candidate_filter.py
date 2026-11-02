"""Prediction-only ranking. This module never certifies skills."""
from __future__ import annotations
import numpy as np
import torch
from generator.policy_registry import policy_index


def select_top_k(model, observation, policy_names, k: int, *, score_fn=None):
    """Return names of selected policies; never execute or certify them.

    Default ranking uses predicted payoff. An explicitly supplied score_fn
    may use predicted motives; tune it on validation only.
    """
    names = list(policy_names)
    if not 1 <= k <= len(names) or len(set(names)) != len(names):
        raise ValueError('k must be in 1..number of distinct candidates')
    obs = torch.as_tensor(np.asarray(observation), dtype=torch.float32).reshape(1, -1)
    batch = obs.expand(len(names), -1)
    ids = torch.tensor([policy_index(n) for n in names], dtype=torch.long)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        payoffs, motives = model(batch.to(device), ids.to(device))
    p = payoffs.cpu().numpy().reshape(-1)
    m = motives.cpu().numpy()
    scores = p if score_fn is None else np.asarray([score_fn(float(v), m[i]) for i, v in enumerate(p)])
    if scores.shape != (len(names),) or not np.isfinite(scores).all():
        raise ValueError('Ranking scores must be finite scalars')
    order = sorted(range(len(names)), key=lambda i: (-float(scores[i]), names[i]))
    return [names[i] for i in order[:k]]


def execute_selected(selected_names, candidates_by_name, execute_fn, certify_fn):
    """Runtime integration seam: certify_fn receives ONLY measured rollout outcomes.

    execute_fn(policy) -> (measured_payoff, measured_motives, terminated).
    certify_fn(measured_payoff, measured_motives) -> existing CDS/PDS decision.
    """
    results = []
    for name in selected_names:
        payoff, motives, terminated = execute_fn(candidates_by_name[name])
        result = certify_fn(float(payoff), np.asarray(motives, dtype=np.float32))
        results.append(dict(policy=name, measured_payoff=float(payoff),
                            measured_motives=np.asarray(motives).tolist(),
                            terminated=bool(terminated), certification=result))
    return results
