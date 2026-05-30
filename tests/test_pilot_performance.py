"""Performance validation for the PPO-trained RL pilot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from baseline.idle_policy import IdlePolicy
from baseline.improvement_calculator import ImprovementCalculator
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from env.lunar_lander_wrapper import SubRepEnv
from env.skill_executor import SkillExecutor
from pilot.rl_pilot import RLPilot


CHECKPOINT_PATH = Path("models/pilot_ppo.pt")
# Keep these constants aligned with the task description: 100 episodes for
# landing competence and 10 episodes for certification-gate admission.
EVAL_EPISODES = 100
GATE_EPISODES = 10
MAX_STEPS = 1000
GAMMA = 0.99


@pytest.fixture(scope="module")
def trained_pilot() -> RLPilot:
    assert CHECKPOINT_PATH.exists(), "Expected trained pilot checkpoint at models/pilot_ppo.pt"
    return RLPilot.load(CHECKPOINT_PATH, map_location="cpu")


@pytest.fixture(scope="module")
def pilot_eval_metrics(trained_pilot: RLPilot) -> dict[str, object]:
    return _run_pilot_episodes(
        trained_pilot,
        episodes=EVAL_EPISODES,
        seed=2_000,
    )


def test_pilot_checkpoint_metadata_documents_reproducibility():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    required_keys = {
        "regeneration_command",
        "ppo_config",
        "reward_shaping_config",
        "seed",
        "dependencies",
        "expected_evaluation_metrics",
    }

    assert required_keys.issubset(metadata.keys())
    assert metadata["regeneration_command"] == (
        "python -m pilot.train_pilot --seed 7 --output models/pilot_ppo.pt"
    )
    assert metadata["seed"] == 7
    assert metadata.get("metadata_only") is False
    assert isinstance(metadata["ppo_config"], dict)
    assert isinstance(metadata["reward_shaping_config"], dict)
    assert isinstance(metadata["dependencies"], dict)

    expected_metrics = metadata["expected_evaluation_metrics"]
    assert expected_metrics["success_rate"] > 0.50
    assert expected_metrics["mean_delta_r_vs_idle"] > 0.0
    assert expected_metrics["gate_passes"] >= 1


def _run_pilot_episodes(
    pilot: RLPilot,
    *,
    episodes: int,
    seed: int,
    max_steps: int = MAX_STEPS,
) -> dict[str, object]:
    env = SubRepEnv(seed=seed)
    payoffs: list[float] = []
    motives: list[np.ndarray] = []
    successes = 0
    episode_lengths: list[int] = []

    try:
        for episode in range(episodes):
            # Use a different deterministic seed per episode. This avoids the
            # false confidence of evaluating the same initial lander state 100
            # times while keeping the test reproducible.
            obs, _ = env.reset(seed=seed + episode)
            total_payoff = 0.0
            motive_deltas = np.zeros(2, dtype=np.float32)
            discount = 1.0
            episode_success = False

            for step_index in range(max_steps):
                action = pilot.predict(obs, deterministic=True, return_probability=False)
                obs, reward_vec, terminated, truncated, info = env.step(action)
                reward_vec = np.asarray(reward_vec, dtype=np.float32)
                total_payoff += discount * float(np.sum(reward_vec))
                motive_deltas += discount * reward_vec

                if terminated or truncated:
                    # Strict success: real terminal landing only. Positive
                    # dense shaping or positive SubRep safety motive is not
                    # enough, because those can occur before a final crash.
                    episode_success = bool(
                        terminated
                        and not truncated
                        and info.get("landing_success", False)
                        and float(info.get("original_reward", 0.0)) == 100.0
                    )
                    break
                discount *= GAMMA

            payoffs.append(float(total_payoff))
            motives.append(motive_deltas)
            successes += int(episode_success)
            episode_lengths.append(step_index + 1)
    finally:
        env.close()

    return {
        "success_rate": float(successes / episodes),
        "mean_payoff": float(np.mean(payoffs)),
        "mean_motives": np.mean(np.asarray(motives, dtype=np.float32), axis=0).astype(np.float32),
        "episode_payoffs": np.asarray(payoffs, dtype=np.float32),
        "episode_motives": np.asarray(motives, dtype=np.float32),
        "mean_episode_length": float(np.mean(episode_lengths)),
    }


def test_pilot_success_rate_exceeds_random_baseline(pilot_eval_metrics: dict[str, object]):
    # Main competence requirement: the trained pilot must successfully land in
    # more than half of 100 varied episodes.
    assert pilot_eval_metrics["success_rate"] > 0.50


def test_pilot_mean_improvement_beats_idle_baseline(pilot_eval_metrics: dict[str, object]):
    baseline_env = SubRepEnv(seed=2_000)
    try:
        baseline_stats = IdlePolicy(baseline_env, gamma=GAMMA).run_baseline_episodes(
            num_episodes=EVAL_EPISODES,
            seed=2_000,
        )
    finally:
        baseline_env.close()

    calculator = ImprovementCalculator(baseline_stats)
    # SubRep admission is relative to an idle baseline, so this checks that the
    # policy is not merely landing but actually improves the scalarized rollout.
    delta_r, delta_n = calculator.compute_improvements(
        skill_payoff=pilot_eval_metrics["mean_payoff"],
        skill_motives=pilot_eval_metrics["mean_motives"],
    )

    calculator.validate_improvements(delta_r, delta_n)
    assert delta_r > 0.0


def test_pilot_generates_certifiable_skill_episode():
    env = SubRepEnv(seed=4_000)
    baseline_env = SubRepEnv(seed=4_000)
    try:
        executor = SkillExecutor.from_pilot_checkpoint(
            env=env,
            checkpoint_path=str(CHECKPOINT_PATH),
            gamma=GAMMA,
            max_steps=MAX_STEPS,
            deterministic=True,
        )
        baseline_stats = IdlePolicy(baseline_env, gamma=GAMMA).run_baseline_episodes(
            num_episodes=GATE_EPISODES,
            seed=4_000,
        )
        calculator = ImprovementCalculator(baseline_stats)
        cds_gate = CDSGate()
        pds_gate = PDSGate(epsilon=0.1)
        gate_passes = 0

        for episode in range(GATE_EPISODES):
            # Each episode is treated as a candidate skill rollout. At least one
            # must produce improvements that pass CDS or PDS certification.
            env.reset(seed=4_000 + episode)
            payoff, motives, _ = executor.run_episode()
            delta_r, delta_n = calculator.compute_improvements(payoff, motives)
            if cds_gate.admit(delta_r, delta_n) or pds_gate.admit(delta_r, delta_n):
                gate_passes += 1

        assert gate_passes >= 1
    finally:
        env.close()
        baseline_env.close()
