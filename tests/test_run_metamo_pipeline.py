"""Tests for the MetaMo demo's episode handling (demo/run_metamo_pipeline.py)."""

from __future__ import annotations

from bridge.controller import MetaMoController
from demo.run_metamo_pipeline import start_new_episode
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
