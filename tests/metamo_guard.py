"""Repository-wide prerequisite and test-skip requirements."""

import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_CHECKPOINTS = (
    Path("models/pilot_ppo.pt"),
    Path("models/lunar_mdn.pth"),
)


def _get_metamo_guard_errors(repo_root, metamo_available, searched_paths=None):
    """Return actionable errors for missing MetaMo prerequisites."""
    errors = []

    if not metamo_available:
        searched = "\n".join(
            f"        {path}" for path in (searched_paths or [])
        )
        message = "MetaMo checkout not found."

        if searched:
            message += f" Searched:\n{searched}"

        message += "\n      Fix: git submodule update --init --recursive"
        errors.append(message)

    missing_checkpoints = [
        str(checkpoint)
        for checkpoint in REQUIRED_CHECKPOINTS
        if not (repo_root / checkpoint).is_file()
    ]

    if missing_checkpoints:
        errors.append(
            "Required checkpoint(s) missing: "
            + ", ".join(missing_checkpoints)
            + "\n      These files are tracked in git; check the checkout."
        )

    return errors


def check_prerequisites():
    """Fail test collection when required MetaMo prerequisites are absent."""
    if os.environ.get("SUBREP_REQUIRE_METAMO") != "1":
        return

    from bridge._loader import candidate_roots, is_available

    errors = _get_metamo_guard_errors(
        REPO_ROOT,
        is_available(),
        candidate_roots(),
    )

    if errors:
        message = (
            "SUBREP_REQUIRE_METAMO=1, but required MetaMo prerequisites "
            "are not available:\n  - "
            + "\n  - ".join(errors)
        )
        raise pytest.UsageError(message)


ALLOWED_SKIPS: dict[str, str] = {}

_skipped: list[tuple[str, str]] = []


def guard_enabled() -> bool:
    """Return whether strict MetaMo CI requirements are enabled."""
    return os.environ.get("SUBREP_REQUIRE_METAMO") == "1"


def record_skip(report) -> None:
    """Record skipped tests, excluding expected xfail results."""
    if report.skipped and not hasattr(report, "wasxfail"):
        reason = (
            report.longrepr[2]
            if isinstance(report.longrepr, tuple)
            else str(report.longrepr)
        )
        _skipped.append((report.nodeid, reason))


def fail_on_skips(session) -> None:
    """Fail the test session when unexpected skips occurred."""
    if not guard_enabled():
        return

    unexpected = [
        (nodeid, reason)
        for nodeid, reason in _skipped
        if not ALLOWED_SKIPS.get(nodeid, "").strip()
    ]

    if unexpected:
        print("\nSUBREP_REQUIRE_METAMO=1, but these tests were skipped:")
        for nodeid, reason in unexpected:
            print(f"  - {nodeid}: {reason}")

        if session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def reset_skips() -> None:
    """Clear recorded skips at the beginning of a test session."""
    _skipped.clear()
