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


@pytest.mark.parametrize("mode", ["noop", "stop"])
def test_abstention_feedback_is_recorded_in_both_modes(tmp_path, mode):
    """With the task risk, every abstention reaches MetaMo as not admitted.

    The ci candidates are all rejected. In noop mode the baseline flies and
    crashes, so the risk is the crash level (the rejection penalty then clips
    at 1.0). In stop mode nothing flies, so the risk is the penalty alone.
    """
    path = tmp_path / f"{mode}.json"
    code, output, _ = run_demo(
        ["--steps", "2", "--seed", str(SEED), "--candidates", "ci",
         "--no-selection", mode, "--risk", "task", "--json-out", str(path)]
    )
    assert code == 0
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["config"]["risk"] == "task"
    steps = data["steps"]
    for step in steps:
        assert step["admitted_count"] == 0
        assert step["feedback"]["admitted"] is False
        assert step["feedback"]["risk"] >= 0.25
        if mode == "stop":
            assert step["feedback"]["risk"] == pytest.approx(0.25)
        elif step["result"] == "crashed":
            assert step["feedback"]["risk"] == pytest.approx(1.0)
    assert steps[1]["modulators"] != steps[0]["modulators"]
    assert " risk" in output


def test_without_the_task_risk_the_json_has_no_feedback_fields(noop_run):
    _, data, _ = noop_run
    assert "risk" not in data["config"]
    assert all("feedback" not in step for step in data["steps"])


def test_an_all_fail_step_after_an_approved_step_reports_no_old_approval(tmp_path, monkeypatch):
    """Each step reports its own verdicts, even when the pipeline is skipped.

    Fake records, no flight: on step 0 `left_engine` lands and is admitted;
    on step 1 every candidate crashes, so every one fails the guard and the
    pipeline is never called. Step 1 must not show step 0's approval.
    """
    from utils.mdn_contracts import CandidateSkillRecord

    ids = ("left_engine", "main_engine", "right_engine", "random")

    def record(skill_id, *, lands, seed):
        return CandidateSkillRecord(
            skill_id=skill_id,
            delta_r=200.0 if lands else 0.0,
            delta_n=(300.0, -50.0) if lands else (-300.0, -60.0),
            is_certified=False,
            gate_type="PDS",
            baseline_id="noop",
            metadata={"seed": seed, "steps": 200 if lands else 70, "landing_success": lands,
                      "crashed": not lands, "truncated": False},
        )

    def fake_evaluate(env, candidates, *, seed, **_):
        baseline = {
            "baseline_payoff": -100.0, "baseline_motives": np.zeros(2),
            "baseline_id": "noop", "seed": seed,
            "info": {"start_obs": np.full(8, 0.01 * seed), "crashed": True},
        }
        first_step = seed == SEED
        return baseline, [record(i, lands=first_step and i == "left_engine", seed=seed) for i in ids]

    def fake_execute(env, policy, *, seed, **_):
        lands = policy.skill_id == "left_engine" and seed == SEED
        info = {"steps": 200 if lands else 70, "landing_success": lands, "crashed": not lands,
                "truncated": False, "stop_reason": "terminated", "max_steps": None}
        return (200.0 if lands else 0.0), np.array([300.0, -50.0] if lands else [0.0, 0.0]), info

    monkeypatch.setattr(lunar_demo, "evaluate_candidates", fake_evaluate)
    monkeypatch.setattr(lunar_demo, "execute_candidate", fake_execute)

    path = tmp_path / "stale.json"
    code, _, _ = run_demo(
        ["--steps", "2", "--seed", str(SEED), "--candidates", "ci",
         "--epsilon-scale", "on", "--json-out", str(path)]
    )
    assert code == 0
    first, second = json.loads(path.read_text(encoding="utf-8"))["steps"]

    admitted_first = {c["skill_id"]: c["admitted"] for c in first["candidates"]}
    assert admitted_first["left_engine"] is True
    assert first["selected"] == "left_engine"

    assert second["admitted_count"] == 0 and second["selected"] is None
    assert second["task_failures"] == len(ids)
    for candidate in second["candidates"]:
        assert candidate["admitted"] is False, candidate["skill_id"]
        assert candidate["task_failure"] is True
    for step in (first, second):
        assert step["admitted_count"] == sum(c["admitted"] for c in step["candidates"])


def test_without_the_epsilon_scale_the_json_has_no_scale_fields(noop_run):
    _, data, _ = noop_run
    assert "epsilon_scale" not in data["config"]
    for step in data["steps"]:
        assert "pds_tolerance" not in step and "task_failures" not in step
        assert all("task_failure" not in c for c in step["candidates"])
    assert "task_failures_total" not in data["summary"]


def test_the_epsilon_scale_records_the_tolerance_and_the_task_failures(tmp_path):
    from env.lunar_lander_metamo import lunar_lander_epsilon_scale

    scale = lunar_lander_epsilon_scale()
    path = tmp_path / "scaled.json"
    code, output, _ = run_demo(
        ["--steps", "2", "--seed", str(SEED), "--candidates", "ci",
         "--epsilon-scale", "on", "--json-out", str(path)]
    )
    assert code == 0
    assert " tol " in output and " fail " in output
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["config"]["epsilon_scale"] == {
        "unit": scale.unit, "payoff_floor": scale.payoff_floor
    }
    for step in data["steps"]:
        assert step["pds_tolerance"] == pytest.approx(step["signal"]["epsilon"] * scale.unit)
        assert step["task_failures"] == sum(c["task_failure"] for c in step["candidates"])
        for candidate in step["candidates"]:
            if candidate["crashed"]:
                assert candidate["task_failure"] and not candidate["admitted"]
        assert step["pds_only_admitted"] == lunar_demo.pds_admitted_count(
            step["candidates"], step["signal"]["epsilon"], scale
        )
    summary = data["summary"]
    assert summary["task_failures_total"] == sum(s["task_failures"] for s in data["steps"])
    sensitive = sum(1 for s in data["steps"] if lunar_demo.eps_sensitive(s["candidates"], scale))
    assert summary["eps_sensitive_steps"] == sensitive


@pytest.mark.skipif(
    not lunar_demo.PILOT_CHECKPOINT.is_file(),
    reason="PPO checkpoint models/pilot_ppo.pt not found",
)
def test_the_tradeoff_candidates_are_listed(tmp_path):
    path = tmp_path / "tradeoff.json"
    code, _, _ = run_demo(
        ["--steps", "1", "--seed", str(SEED), "--candidates", "tradeoff",
         "--epsilon-scale", "on", "--json-out", str(path)]
    )
    assert code == 0
    ids = json.loads(path.read_text(encoding="utf-8"))["config"]["candidate_ids"]
    for skill_id in ("ppo_deterministic", "ppo_then_side_tradeoff",
                     "ppo_waste_main_12", "ppo_waste_main_15", "ppo_waste_main_18",
                     "ppo_waste_main_20", "ppo_waste_side_15", "ppo_waste_side_20"):
        assert skill_id in ids


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


# ---------------------------------------------------------------------------
# The fixed-priority mode
# ---------------------------------------------------------------------------


def test_fixed_priorities_mode_runs(tmp_path, noop_run):
    """By default the fixed governor holds MetaMo's starting signal on every step."""
    path = tmp_path / "fixed.json"
    code, output, _ = run_demo(
        ["--steps", "2", "--seed", str(SEED), "--candidates", "ci",
         "--governor", "fixed", "--json-out", str(path)]
    )
    assert code == 0
    data = json.loads(path.read_text(encoding="utf-8"))
    constant = data["config"]["governor"]
    assert constant["kind"] == "fixed"

    _, metamo, _ = noop_run
    start = metamo["steps"][0]["signal"]
    assert constant["weights"] == start["weights"]
    assert constant["epsilon"] == start["epsilon"]
    assert constant["alpha"] == start["alpha"]
    for step in data["steps"]:
        assert step["signal"] == {
            "weights": constant["weights"],
            "epsilon": constant["epsilon"],
            "alpha": constant["alpha"],
        }
        assert step["modulators"] is None
    assert data["summary"]["securing_range"] is None
    assert data["summary"]["securing_saturated_at_step"] is None
    assert "governor: fixed" in output


def test_fixed_priorities_mode_runs_with_the_task_risk(tmp_path):
    """The path that reads the governor's stimulus and modulators, which the fixed governor lacks."""
    from env.lunar_lander_metamo import LUNAR_LANDER_EPSILON_UNIT

    path = tmp_path / "fixed_task.json"
    code, _, flown = run_demo(
        ["--steps", "2", "--seed", str(SEED), "--candidates", "ci",
         "--governor", "fixed", "--weights", "0.3,0.7", "--epsilon", "0.2", "--alpha", "0.25",
         "--risk", "task", "--epsilon-scale", "on", "--json-out", str(path)]
    )
    assert code == 0
    assert flown == 2  # the outcomes are still flown and reach the governor
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["config"]["risk"] == "task"
    for step in data["steps"]:
        assert step["signal"] == {"weights": [0.3, 0.7], "epsilon": 0.2, "alpha": 0.25}
        assert step["pds_tolerance"] == pytest.approx(0.2 * LUNAR_LANDER_EPSILON_UNIT)
        assert step["modulators"] is None
        assert step["feedback"] is None
    assert data["summary"]["final_modulators"] is None


@pytest.mark.parametrize("weights", ["0.7,0.4", "1.0", "-0.1,1.1"])
def test_fixed_priorities_reject_weights_off_the_simplex(weights):
    with pytest.raises(SystemExit) as raised:
        lunar_demo.main(["--governor", "fixed", "--weights", weights,
                         "--candidates", "ci", "--steps", "1"])
    assert raised.value.code == 2


@pytest.mark.parametrize("option", [["--weights", "0.7,0.3"], ["--epsilon", "0.2"], ["--alpha", "0.2"]])
def test_fixed_options_need_the_fixed_governor(option):
    with pytest.raises(SystemExit) as raised:
        lunar_demo.main([*option, "--candidates", "ci", "--steps", "1"])
    assert raised.value.code == 2
