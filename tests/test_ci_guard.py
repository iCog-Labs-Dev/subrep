"""Tests for the MetaMo CI prerequisite guard."""

from pathlib import Path

import pytest

from tests import metamo_guard
from tests.metamo_guard import _get_metamo_guard_errors


def test_guard_accepts_available_metamo_and_checkpoints(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "pilot_ppo.pt").touch()
    (tmp_path / "models" / "lunar_mdn.pth").touch()

    errors = _get_metamo_guard_errors(tmp_path, metamo_available=True)

    assert errors == []


def test_guard_reports_unavailable_metamo(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "pilot_ppo.pt").touch()
    (tmp_path / "models" / "lunar_mdn.pth").touch()

    errors = _get_metamo_guard_errors(tmp_path, metamo_available=False)

    assert "MetaMo checkout not found." in errors[0]
    assert "git submodule update --init --recursive" in errors[0]

    searched = [Path("/fake/metamo"), Path("/workspace/MetaMo-Python")]
    errors_with_paths = _get_metamo_guard_errors(
        tmp_path, metamo_available=False, searched_paths=searched
    )
    assert all(str(path) in errors_with_paths[0] for path in searched)


def test_guard_reports_missing_checkpoints(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "pilot_ppo.pt").touch()

    errors = _get_metamo_guard_errors(tmp_path, metamo_available=True)

    assert len(errors) == 1
    assert "models/lunar_mdn.pth" in errors[0]


def test_guard_reports_all_missing_prerequisites(tmp_path):
    errors = _get_metamo_guard_errors(tmp_path, metamo_available=False)

    assert len(errors) == 2
    assert "MetaMo checkout not found." in errors[0]
    assert "git submodule update --init --recursive" in errors[0]
    assert "models/pilot_ppo.pt" in errors[1]
    assert "models/lunar_mdn.pth" in errors[1]


def test_check_prerequisites_uses_loader_search_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("SUBREP_REQUIRE_METAMO", "1")
    monkeypatch.setattr(metamo_guard, "REPO_ROOT", tmp_path)

    searched_paths = [
        Path("/fake/subrep/external/metamo"),
        Path("/fake/MetaMo-Python"),
    ]

    monkeypatch.setattr(
        "bridge._loader.is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "bridge._loader.candidate_roots",
        lambda: searched_paths,
    )

    with pytest.raises(pytest.UsageError) as exc_info:
        metamo_guard.check_prerequisites()

    message = str(exc_info.value)

    assert "MetaMo checkout not found." in message
    assert "git submodule update --init --recursive" in message

    for path in searched_paths:
        assert str(path) in message


def test_guard_does_nothing_when_environment_variable_is_unset(monkeypatch):
    monkeypatch.delenv("SUBREP_REQUIRE_METAMO", raising=False)

    metamo_guard.check_prerequisites()


def test_guard_fails_with_actionable_message_when_metamo_is_missing(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SUBREP_REQUIRE_METAMO", "1")
    monkeypatch.setattr(metamo_guard, "REPO_ROOT", tmp_path)
    monkeypatch.setattr("bridge._loader.is_available", lambda: False)
    monkeypatch.setattr("bridge._loader.candidate_roots", lambda: [])

    with pytest.raises(pytest.UsageError, match="git submodule update"):
        metamo_guard.check_prerequisites()


def test_unexpected_skip_fails_the_test_session(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("SUBREP_REQUIRE_METAMO", "1")
    monkeypatch.setattr(metamo_guard, "_skipped", [])
    monkeypatch.setattr(metamo_guard, "ALLOWED_SKIPS", {})

    report = SimpleNamespace(
        skipped=True,
        longrepr=("test_file.py", 10, "Skipped: unexpected reason"),
        nodeid="tests/test_example.py::test_example",
    )

    metamo_guard.record_skip(report)

    session = SimpleNamespace(exitstatus=0)
    metamo_guard.fail_on_skips(session)

    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


def test_allowed_skip_does_not_fail_the_test_session(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("SUBREP_REQUIRE_METAMO", "1")
    monkeypatch.setattr(metamo_guard, "_skipped", [])
    monkeypatch.setattr(
        metamo_guard,
        "ALLOWED_SKIPS",
        {"tests/test_example.py::test_example": "approved reason"},
    )

    report = SimpleNamespace(
        skipped=True,
        longrepr=("test_file.py", 10, "Skipped: approved reason"),
        nodeid="tests/test_example.py::test_example",
    )

    metamo_guard.record_skip(report)

    session = SimpleNamespace(exitstatus=0)
    metamo_guard.fail_on_skips(session)

    assert session.exitstatus == 0


def test_empty_allowlist_reason_does_not_allow_skip(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("SUBREP_REQUIRE_METAMO", "1")
    monkeypatch.setattr(metamo_guard, "_skipped", [
        ("tests/test_example.py::test_example", "unexpected reason")
    ])
    monkeypatch.setattr(
        metamo_guard,
        "ALLOWED_SKIPS",
        {"tests/test_example.py::test_example": ""},
    )

    session = SimpleNamespace(exitstatus=pytest.ExitCode.OK)
    metamo_guard.fail_on_skips(session)

    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


def test_expected_xfail_is_not_recorded_as_skip(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(metamo_guard, "_skipped", [])

    report = SimpleNamespace(
        skipped=True,
        wasxfail="known expected failure",
        longrepr=("test_file.py", 10, "XFAIL"),
        nodeid="tests/test_example.py::test_example",
    )

    metamo_guard.record_skip(report)

    assert metamo_guard._skipped == []



def test_reset_skips_clears_recorded_skips(monkeypatch):
    monkeypatch.setattr(
        metamo_guard,
        "_skipped",
        [("tests/test_example.py::test_example", "previous skip")],
    )

    metamo_guard.reset_skips()

    assert metamo_guard._skipped == []


def test_root_conftest_collection_skip_hook_fails_session(
    pytestconfig, monkeypatch
):
    from types import SimpleNamespace

    monkeypatch.setenv("SUBREP_REQUIRE_METAMO", "1")
    monkeypatch.setattr(metamo_guard, "_skipped", [])

    root_conftest_path = Path(__file__).resolve().parents[1] / "conftest.py"
    root_conftest = next(
        (
            plugin
            for plugin in pytestconfig.pluginmanager.get_plugins()
            if getattr(plugin, "__file__", None)
            and Path(plugin.__file__).resolve() == root_conftest_path
        ),
        None,
    )
    assert root_conftest is not None

    report = SimpleNamespace(
        skipped=True,
        longrepr=("test_module.py", 1, "Skipped: module-level skip"),
        nodeid="tests/test_module.py",
    )

    root_conftest.pytest_collectreport(report)

    session = SimpleNamespace(exitstatus=pytest.ExitCode.OK)
    root_conftest.pytest_sessionfinish(session, session.exitstatus)

    assert metamo_guard._skipped == [
        ("tests/test_module.py", "Skipped: module-level skip")
    ]
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


def test_root_conftest_runtest_skip_hook_fails_session(
    pytestconfig, monkeypatch
):
    from types import SimpleNamespace

    monkeypatch.setenv("SUBREP_REQUIRE_METAMO", "1")
    monkeypatch.setattr(metamo_guard, "_skipped", [])

    root_conftest_path = Path(__file__).resolve().parents[1] / "conftest.py"
    root_conftest = next(
        (
            plugin
            for plugin in pytestconfig.pluginmanager.get_plugins()
            if getattr(plugin, "__file__", None)
            and Path(plugin.__file__).resolve() == root_conftest_path
        ),
        None,
    )
    assert root_conftest is not None

    report = SimpleNamespace(
        skipped=True,
        longrepr=("test_example.py", 10, "Skipped: test-level skip"),
        nodeid="tests/test_example.py::test_example",
    )

    root_conftest.pytest_runtest_logreport(report)

    session = SimpleNamespace(exitstatus=pytest.ExitCode.OK)
    root_conftest.pytest_sessionfinish(session, session.exitstatus)

    assert metamo_guard._skipped == [
        (
            "tests/test_example.py::test_example",
            "Skipped: test-level skip",
        )
    ]
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
