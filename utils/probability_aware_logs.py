"""Probability-aware runtime log helpers for IPS/DR MDN training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from utils.mdn_contracts import CandidateSkillRecord
from utils.mdn_selection import score_candidate


SCHEMA_VERSION = "probability_aware_runtime_log.v1"


def epsilon_softmax_candidate_probabilities(
    candidate_skills: tuple[CandidateSkillRecord, ...] | list[CandidateSkillRecord],
    weights: np.ndarray,
    *,
    epsilon: float = 0.2,
    temperature: float = 1.0,
) -> dict[int, float]:
    """Return exact behavior probabilities over certified candidate indices.
    """
    if not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError(f"epsilon must lie in [0, 1], got {epsilon}")
    if not np.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {temperature}")

    certified: list[tuple[int, CandidateSkillRecord]] = [
        (index, candidate)
        for index, candidate in enumerate(candidate_skills)
        if candidate.is_certified
    ]
    if not certified:
        raise ValueError("behavior policy requires at least one certified candidate")

    scores = np.asarray(
        [score_candidate(candidate, weights) for _, candidate in certified],
        dtype=np.float64,
    )
    scaled_scores = scores / float(temperature)
    scaled_scores -= np.max(scaled_scores)
    exp_scores = np.exp(scaled_scores)
    softmax_probs = exp_scores / np.sum(exp_scores)
    uniform_prob = 1.0 / float(len(certified))
    probs = (1.0 - float(epsilon)) * softmax_probs + float(epsilon) * uniform_prob
    probs = probs / np.sum(probs)

    return {
        int(original_index): float(probability)
        for (original_index, _), probability in zip(certified, probs)
    }


def sample_candidate_index(
    probabilities: dict[int, float],
    *,
    rng: np.random.Generator,
) -> tuple[int, float]:
    """Sample one candidate index from a validated probability map."""
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    indices = np.asarray(sorted(probabilities), dtype=np.int64)
    probs = np.asarray([probabilities[int(index)] for index in indices], dtype=np.float64)
    _validate_probability_vector(probs, field_name="probabilities")
    selected_position = int(rng.choice(len(indices), p=probs))
    selected_index = int(indices[selected_position])
    return selected_index, float(probabilities[selected_index])


def serialize_candidate_records(
    candidate_skills: tuple[CandidateSkillRecord, ...] | list[CandidateSkillRecord],
) -> dict[str, np.ndarray]:
    """Convert candidate records to arrays for a `.npz` runtime log."""
    if not candidate_skills:
        raise ValueError("candidate_skills must not be empty")
    return {
        "candidate_skill_ids": np.asarray([candidate.skill_id for candidate in candidate_skills]),
        "candidate_delta_r": np.asarray([candidate.delta_r for candidate in candidate_skills], dtype=np.float32),
        "candidate_delta_n": np.asarray([candidate.delta_n for candidate in candidate_skills], dtype=np.float32),
        "candidate_accept_labels": np.asarray([candidate.is_certified for candidate in candidate_skills], dtype=np.bool_),
        "candidate_gate_types": np.asarray([candidate.gate_type for candidate in candidate_skills]),
        "candidate_admission_margins": np.asarray(
            [np.nan if candidate.admission_margin is None else candidate.admission_margin for candidate in candidate_skills],
            dtype=np.float32,
        ),
    }


def save_probability_aware_log(path: str | Path, **record: Any) -> str:
    """Validate and save one probability-aware runtime decision as `.npz`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.setdefault("schema_version", SCHEMA_VERSION)
    validate_probability_aware_log(record)

    metadata = dict(record.get("metadata", {}))
    arrays = {
        key: value
        for key, value in record.items()
        if key not in {"metadata", "schema_version"}
    }
    arrays["schema_version"] = np.asarray(record["schema_version"])
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez(path, **arrays)
    return str(path)


def load_probability_aware_log(path: str | Path) -> dict[str, Any]:
    """Load and validate one probability-aware runtime decision log."""
    data = np.load(path, allow_pickle=True)
    record: dict[str, Any] = {key: data[key] for key in data.files if key != "metadata_json"}
    metadata_json = data["metadata_json"] if "metadata_json" in data.files else np.asarray("{}")
    record["metadata"] = json.loads(str(np.asarray(metadata_json).reshape(()).item()))
    validate_probability_aware_log(record)
    return record


def probability_aware_log_files(directory: str | Path, *, pattern: str = "*.npz") -> list[Path]:
    """Return sorted probability-aware log files from a directory."""
    directory_path = Path(directory)
    files = sorted(directory_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No probability-aware logs matching {pattern!r} found in {directory_path}")
    return files


def validate_probability_aware_log(record: dict[str, Any]) -> None:
    """Validate fields required for real IPS/DR auxiliary training."""
    required = {
        "schema_version",
        "context",
        "alpha",
        "support_values",
        "weights_used",
        "candidate_skill_ids",
        "candidate_delta_r",
        "candidate_delta_n",
        "candidate_accept_labels",
        "selected_candidate_index",
        "selected_skill_id",
        "selected_score",
        "behavior_probability",
        "actual_payoff",
        "actual_motives",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"probability-aware log is missing required fields: {missing}")

    schema_version = str(np.asarray(record["schema_version"]).reshape(()).item())
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported probability-aware log schema_version {schema_version!r}")

    context = _finite_vector(record["context"], field_name="context")
    alpha = _finite_vector(record["alpha"], field_name="alpha")
    support_values = _finite_vector(record["support_values"], field_name="support_values")
    weights_used = _finite_vector(record["weights_used"], field_name="weights_used")
    if len(context) == 0:
        raise ValueError("context must be non-empty")
    if len(alpha) == 0:
        raise ValueError("alpha must be non-empty")
    if len(support_values) != len(alpha) or len(weights_used) != len(alpha):
        raise ValueError("support_values and weights_used must match alpha length")
    if np.any(alpha <= 0.0):
        raise ValueError("alpha must be strictly positive")
    if np.any(support_values < 0.0):
        raise ValueError("support_values must be non-negative")
    if np.any(weights_used < 0.0) or not np.isclose(np.sum(weights_used), 1.0, atol=1e-5):
        raise ValueError("weights_used must be a valid simplex vector")

    skill_ids = np.asarray(record["candidate_skill_ids"]).reshape(-1)
    delta_r = _finite_vector(record["candidate_delta_r"], field_name="candidate_delta_r")
    delta_n = np.asarray(record["candidate_delta_n"], dtype=np.float64)
    labels = np.asarray(record["candidate_accept_labels"]).reshape(-1).astype(bool)
    if len(skill_ids) == 0:
        raise ValueError("candidate_skill_ids must be non-empty")
    if delta_r.shape != (len(skill_ids),):
        raise ValueError("candidate_delta_r length must match candidate_skill_ids")
    if delta_n.shape != (len(skill_ids), len(alpha)):
        raise ValueError("candidate_delta_n shape must be (num_candidates, num_objectives)")
    if not np.all(np.isfinite(delta_n)):
        raise ValueError("candidate_delta_n must contain only finite values")
    if labels.shape != (len(skill_ids),):
        raise ValueError("candidate_accept_labels length must match candidate_skill_ids")
    if not np.any(labels):
        raise ValueError("at least one candidate must be certified")

    selected_index = int(np.asarray(record["selected_candidate_index"]).reshape(()).item())
    if selected_index < 0 or selected_index >= len(skill_ids):
        raise ValueError("selected_candidate_index is out of range")
    if not bool(labels[selected_index]):
        raise ValueError("selected_candidate_index must point to a certified candidate")
    selected_skill_id = str(np.asarray(record["selected_skill_id"]).reshape(()).item())
    if selected_skill_id != str(skill_ids[selected_index]):
        raise ValueError("selected_skill_id must match selected_candidate_index")

    behavior_probability = float(np.asarray(record["behavior_probability"]).reshape(()).item())
    if not np.isfinite(behavior_probability) or behavior_probability <= 0.0 or behavior_probability > 1.0:
        raise ValueError("behavior_probability must be finite and lie in (0, 1]")
    selected_score = float(np.asarray(record["selected_score"]).reshape(()).item())
    if not np.isfinite(selected_score):
        raise ValueError("selected_score must be finite")
    actual_payoff = float(np.asarray(record["actual_payoff"]).reshape(()).item())
    if not np.isfinite(actual_payoff):
        raise ValueError("actual_payoff must be finite")
    actual_motives = _finite_vector(record["actual_motives"], field_name="actual_motives")
    if actual_motives.shape != (len(alpha),):
        raise ValueError("actual_motives length must match alpha length")


def _finite_vector(values: Any, *, field_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    return array


def _validate_probability_vector(probabilities: np.ndarray, *, field_name: str) -> None:
    if probabilities.ndim != 1 or len(probabilities) == 0:
        raise ValueError(f"{field_name} must be a non-empty 1D vector")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{field_name} must contain only finite values")
    if np.any(probabilities <= 0.0):
        raise ValueError(f"{field_name} must be strictly positive")
    if not np.isclose(np.sum(probabilities), 1.0, atol=1e-6):
        raise ValueError(f"{field_name} must sum to 1")
