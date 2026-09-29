"""Tests for the MetaMo demo (demo/run_metamo_pipeline.py)."""

from __future__ import annotations

import pytest

from bridge._loader import is_available
from bridge.controller import MetaMoController
from demo.run_metamo_pipeline import main, start_new_episode
from env.minecraft_stub import MinecraftStubEnv


def test_new_episode_gets_a_fresh_pipeline_and_keeps_the_governor():
    """Certificates are cached per context, and every episode starts from the
    same observation -- so the pipeline must be replaced at each episode start,
    while MetaMo's state (the governor) carries over."""
    env = MinecraftStubEnv(seed=3, noise_scale=0.0)
    env.reset(seed=3)
    for _ in range(env.episode_length):
        env.step(0)
    assert env.episode_over

    governor = object()
    old_pipeline = object()
    new_pipeline = object()
    controller = MetaMoController(governor, old_pipeline, seed=3)

    obs = start_new_episode(
        env, controller, seed=3, make_pipeline=lambda: new_pipeline
    )

    assert controller.pipeline is new_pipeline
    assert controller.governor is governor
    assert not env.episode_over
    assert (obs == env.observation()).all()


@pytest.mark.skipif(
    not is_available(),
    reason="MetaMo checkout not found (see bridge/_loader.py for search paths)",
)
def test_demo_runs_across_an_episode_reset(capsys):
    """Smoke test: per-state evaluation every decision, and past the reset.

    At the default horizon an episode is 8 decisions, so 10 decisions crosses
    into a second episode. Nothing else in the suite runs the demo.
    """
    assert main(["--steps", "10", "--cvar-samples", "50"]) == 0
    out = capsys.readouterr().out
    assert "eps   range:" in out
