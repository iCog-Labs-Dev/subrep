"""Tests for the MO-LunarLander MetaMo demo (demo/run_lunar_lander_metamo.py).

The core tests use the `ci` candidates, which need no PPO checkpoint, and run a
few steps so the file stays fast. Every `ci` candidate crashes like the noop
baseline, so the PDS gate admits none of them and every step abstains; the
tests rely on that to check the two no-selection modes.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

import demo.run_lunar_lander_metamo as lunar_demo
from bridge._loader import is_available
from bridge.budget import BudgetCoefficients, alpha_bounds

pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="MetaMo checkout not found (see bridge/_loader.py for search paths)",
)

STEPS = 3
SEED = 5
CVAR_SAMPLES = 1000
ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+\d\.\d{3}\s")


def run_demo(argv: List[str]) -> Tuple[int, str, int]:
    """Run the demo, returning (exit code, printed output, episodes flown)."""
    flown = {"count": 0}
    real_execute = lunar_demo.execute_candidate

    def counting_execute(*args: Any, **kwargs: Any):
        flown["count"] += 1
        return real_execute(*args, **kwargs)

    lunar_demo.execute_candidate = counting_execute
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            code = lunar_demo.main(argv)
    finally:
        lunar_demo.execute_candidate = real_execute
    return code, buffer.getvalue(), flown["count"]


def table_rows(output: str) -> List[Tuple[int, int]]:
    """(step, seed) for every row of the per-step table."""
    return [
        (int(m.group(1)), int(m.group(2)))
        for m in (ROW.match(line) for line in output.splitlines())
        if m
    ]


@pytest.fixture(scope="module")
def noop_run(tmp_path_factory) -> Tuple[str, Dict[str, Any], int]:
    """One `ci` run in the default noop mode, shared by the tests that only read it."""
    path = tmp_path_factory.mktemp("demo") / "run.json"
    code, output, flown = run_demo(
        ["--steps", str(STEPS), "--seed", str(SEED), "--candidates", "ci",
         "--json-out", str(path)]
    )
    assert code == 0
    return output, json.loads(path.read_text(encoding="utf-8")), flown


def test_the_demo_runs(noop_run):
    output, _, _ = noop_run
    assert "step seed" in output
    assert "eps   range:" in output
    assert "eps sensitivity:" in output


def test_the_same_options_give_the_same_output(tmp_path):
    argv = ["--steps", "2", "--seed", str(SEED), "--candidates", "ci"]
    first = run_demo(argv)
    second = run_demo(argv)
    assert first[0] == second[0] == 0
    assert first[1] == second[1]


def test_one_row_per_step_with_consecutive_seeds(noop_run):
    output, data, _ = noop_run
    expected = [(k, SEED + k) for k in range(STEPS)]
    assert table_rows(output) == expected
    assert [(s["step"], s["seed"]) for s in data["steps"]] == expected
    assert data["config"]["seeds"] == [SEED + k for k in range(STEPS)]


def test_recorded_signals_are_valid(noop_run):
    _, data, _ = noop_run
    coefficients = BudgetCoefficients()
    alpha_low, alpha_high = alpha_bounds(coefficients, CVAR_SAMPLES)
    for step in data["steps"]:
        signal = step["signal"]
        weights = np.asarray(signal["weights"])
        assert weights.shape == (2,)
        assert np.all(weights >= 0.0)
        assert float(weights.sum()) == pytest.approx(1.0)
        assert 0.0 <= signal["epsilon"] <= coefficients.epsilon_max
        assert alpha_low - 1e-12 <= signal["alpha"] <= alpha_high + 1e-12


def test_noop_mode_flies_the_baseline_when_nothing_is_admitted(noop_run):
    output, data, flown = noop_run
    abstained = [s for s in data["steps"] if s["admitted_count"] == 0]
    assert abstained, "the ci candidates are expected to be rejected"
    for step in abstained:
        assert step["selected"] is None
        assert step["flown_skill_id"] == "noop"
        assert step["result"] in ("landed", "crashed", "timeout", "ended")
        assert step["flown_steps"] > 0
    assert flown == STEPS
    assert "STOP" not in output


def test_stop_mode_flies_nothing_when_nothing_is_admitted(tmp_path):
    path = tmp_path / "stop.json"
    code, output, flown = run_demo(
        ["--steps", "2", "--seed", str(SEED), "--candidates", "ci",
         "--no-selection", "stop", "--json-out", str(path)]
    )
    assert code == 0
    data = json.loads(path.read_text(encoding="utf-8"))
    abstained = [s for s in data["steps"] if s["admitted_count"] == 0]
    assert abstained, "the ci candidates are expected to be rejected"
    for step in abstained:
        assert step["flown_skill_id"] is None
        assert step["result"] == "stopped"
        assert step["flown_steps"] == 0
    # Only admitted steps fly an episode.
    assert flown == len(data["steps"]) - len(abstained)
    assert "STOP" in output


def test_json_output_holds_the_configuration_and_every_step(noop_run):
    _, data, _ = noop_run
    assert set(data) == {"config", "steps", "summary"}
    config = data["config"]
    assert config["candidates"] == "ci"
    assert config["no_selection"] == "noop"
    assert config["gates"] == "PDS"
    assert config["payoff_scale"] > 0 and config["motive_scale"] > 0
    assert len(data["steps"]) == STEPS
    for step in data["steps"]:
        assert step["candidate_count"] == len(step["candidates"]) == len(config["candidate_ids"])
        assert step["admitted_count"] == sum(c["admitted"] for c in step["candidates"])
        for candidate in step["candidates"]:
            assert len(candidate["delta_n"]) == 2
            assert candidate["pds_margin"] == pytest.approx(
                candidate["delta_r"] + min(candidate["delta_n"])
            )


def test_a_missing_checkpoint_is_an_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lunar_demo, "PILOT_CHECKPOINT", tmp_path / "missing.pt")
    code = lunar_demo.main(["--candidates", "ppo", "--steps", "1"])
    assert code != 0
    assert "PPO checkpoint" in capsys.readouterr().err


def test_the_summary_diagnostics_are_measured(noop_run):
    """The saturation step and eps sensitivity follow from the per-step records."""
    output, data, _ = noop_run
    steps = data["steps"]
    summary = data["summary"]

    saturated_at = next(
        (s["step"] for s in steps
         if s["modulators"]["securing"] >= lunar_demo.SATURATION_LEVEL),
        None,
    )
    assert summary["securing_saturated_at_step"] == saturated_at
    if saturated_at is None:
        assert "not reached" in output
    else:
        assert f">= {lunar_demo.SATURATION_LEVEL}: {saturated_at}" in output

    eps_max = BudgetCoefficients().epsilon_max
    sensitive = sum(
        1 for s in steps
        if lunar_demo.pds_admitted_count(s["candidates"], 0.0)
        != lunar_demo.pds_admitted_count(s["candidates"], eps_max)
    )
    assert summary["eps_sensitive_steps"] == sensitive
    if sensitive == 0:
        assert "did not change any PDS admission" in output
    else:
        assert f"on {sensitive}/{STEPS} steps" in output


@pytest.mark.skipif(
    not lunar_demo.PILOT_CHECKPOINT.is_file(),
    reason="PPO checkpoint models/pilot_ppo.pt not found",
)
def test_the_ppo_candidates_run_and_every_selection_was_admitted(tmp_path):
    path = tmp_path / "ppo.json"
    code, _, _ = run_demo(
        ["--steps", "2", "--seed", str(SEED), "--candidates", "ppo",
         "--json-out", str(path)]
    )
    assert code == 0
    data = json.loads(path.read_text(encoding="utf-8"))
    for step in data["steps"]:
        admitted = {c["skill_id"] for c in step["candidates"] if c["admitted"]}
        if step["selected"] is not None:
            assert step["selected"] in admitted
            assert step["flown_skill_id"] == step["selected"]
