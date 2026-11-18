"""Pytest hooks for repository-wide test requirements."""

from tests.metamo_guard import (
    check_prerequisites,
    fail_on_skips,
    record_skip,
    reset_skips,
)


def pytest_sessionstart(session):
    reset_skips()


def pytest_collection_modifyitems(session, config, items):
    check_prerequisites()


def pytest_runtest_logreport(report):
    record_skip(report)


def pytest_collectreport(report):
    record_skip(report)


def pytest_sessionfinish(session, exitstatus):
    fail_on_skips(session)
