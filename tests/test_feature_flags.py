"""
tests/test_feature_flags.py
Tests for utils/feature_flags.py (ADR-007).

Verifies:
- All 8 flags default OFF when env vars are absent.
- Setting SUBREP_FLAG_<NAME>=1 enables the flag.
- Setting to '0', 'false', '' leaves flag OFF.
- is_enabled() raises ValueError for unknown names.
- override_flag() context manager restores the original state.
- all_flags() returns all 8 flags.
- list_flags() returns all 8 names.
- flags singleton is accessible module-level.
"""
from __future__ import annotations

import os
import importlib

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPECTED_FLAGS = [
    "minecraft_env_enabled",
    "real_server_enabled",
    "paired_rollout_enabled",
    "minecraft_certification_enabled",
    "skill_generator_enabled",
    "mdn_ranking_enabled",
    "rl_options_enabled",
    "contextual_geometry_enabled",
]


def _clear_flags():
    """Remove all SUBREP_FLAG_* env vars to ensure a clean test state."""
    for key in list(os.environ):
        if key.startswith("SUBREP_FLAG_"):
            del os.environ[key]


@pytest.fixture(autouse=True)
def clean_env():
    """Ensure no SUBREP_FLAG_* vars leak between tests."""
    _clear_flags()
    yield
    _clear_flags()


# ---------------------------------------------------------------------------
# Import and singleton
# ---------------------------------------------------------------------------

def test_module_imports():
    """utils.feature_flags imports without error."""
    from utils import feature_flags  # noqa: F401


def test_flags_singleton_accessible():
    """The module-level `flags` singleton is a FeatureFlags instance."""
    from utils.feature_flags import flags, FeatureFlags
    assert isinstance(flags, FeatureFlags)


def test_list_flags_returns_all_eight():
    """list_flags() returns exactly the 8 documented flag names."""
    from utils.feature_flags import list_flags
    result = list_flags()
    assert sorted(result) == sorted(_EXPECTED_FLAGS)


# ---------------------------------------------------------------------------
# Default OFF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag_name", _EXPECTED_FLAGS)
def test_all_flags_default_off(flag_name):
    """Every flag is OFF when its env var is not set."""
    from utils.feature_flags import flags
    assert flags.is_enabled(flag_name) is False


def test_all_flags_dict_default_off():
    """all_flags() returns all False when no env vars are set."""
    from utils.feature_flags import flags
    result = flags.all_flags()
    assert set(result.keys()) == set(_EXPECTED_FLAGS)
    assert all(v is False for v in result.values())


# ---------------------------------------------------------------------------
# Enabling flags via env vars
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag_name", _EXPECTED_FLAGS)
def test_flag_enabled_by_env_var_1(flag_name):
    """Setting SUBREP_FLAG_<NAME>=1 enables the flag."""
    from utils.feature_flags import flags
    env_key = f"SUBREP_FLAG_{flag_name.upper()}"
    os.environ[env_key] = "1"
    assert flags.is_enabled(flag_name) is True


@pytest.mark.parametrize("truthy_val", ["1", "true", "True", "TRUE", "yes", "on"])
def test_truthy_values_enable_flag(truthy_val):
    """Various truthy string values enable the flag."""
    from utils.feature_flags import flags
    os.environ["SUBREP_FLAG_MINECRAFT_ENV_ENABLED"] = truthy_val
    assert flags.minecraft_env_enabled is True


@pytest.mark.parametrize("falsy_val", ["0", "false", "False", "no", "off", ""])
def test_falsy_values_disable_flag(falsy_val):
    """Various falsy string values keep the flag OFF."""
    from utils.feature_flags import flags
    os.environ["SUBREP_FLAG_MINECRAFT_ENV_ENABLED"] = falsy_val
    assert flags.minecraft_env_enabled is False


# ---------------------------------------------------------------------------
# Property accessors match is_enabled()
# ---------------------------------------------------------------------------

def test_property_matches_is_enabled_when_on():
    """Direct property and is_enabled() agree when flag is ON."""
    from utils.feature_flags import flags
    os.environ["SUBREP_FLAG_MINECRAFT_ENV_ENABLED"] = "1"
    assert flags.minecraft_env_enabled is flags.is_enabled("minecraft_env_enabled")


def test_property_matches_is_enabled_when_off():
    """Direct property and is_enabled() agree when flag is OFF."""
    from utils.feature_flags import flags
    assert flags.minecraft_env_enabled is flags.is_enabled("minecraft_env_enabled")


# ---------------------------------------------------------------------------
# Unknown flag raises ValueError
# ---------------------------------------------------------------------------

def test_is_enabled_raises_for_unknown_flag():
    """is_enabled() raises ValueError for an unrecognised flag name."""
    from utils.feature_flags import flags
    with pytest.raises(ValueError, match="Unknown feature flag"):
        flags.is_enabled("not_a_real_flag")


def test_override_flag_raises_for_unknown_flag():
    """override_flag() raises ValueError for an unrecognised flag name."""
    from utils.feature_flags import override_flag
    with pytest.raises(ValueError, match="Unknown feature flag"):
        with override_flag("not_a_real_flag", True):
            pass


# ---------------------------------------------------------------------------
# override_flag context manager
# ---------------------------------------------------------------------------

def test_override_flag_enables_then_restores():
    """override_flag sets flag ON inside context and restores OFF after."""
    from utils.feature_flags import flags, override_flag
    assert flags.minecraft_env_enabled is False
    with override_flag("minecraft_env_enabled", True):
        assert flags.minecraft_env_enabled is True
    assert flags.minecraft_env_enabled is False


def test_override_flag_disables_then_restores():
    """override_flag can set flag to False and restore the previous True."""
    from utils.feature_flags import flags, override_flag
    os.environ["SUBREP_FLAG_MINECRAFT_ENV_ENABLED"] = "1"
    assert flags.minecraft_env_enabled is True
    with override_flag("minecraft_env_enabled", False):
        assert flags.minecraft_env_enabled is False
    assert flags.minecraft_env_enabled is True


def test_override_flag_restores_on_exception():
    """override_flag restores original state even if an exception occurs."""
    from utils.feature_flags import flags, override_flag
    assert flags.real_server_enabled is False
    try:
        with override_flag("real_server_enabled", True):
            assert flags.real_server_enabled is True
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass
    assert flags.real_server_enabled is False


def test_override_flag_restores_original_env_var():
    """override_flag restores the exact env var value that was there before."""
    from utils.feature_flags import flags, override_flag
    # Set a pre-existing value
    os.environ["SUBREP_FLAG_MINECRAFT_ENV_ENABLED"] = "1"
    with override_flag("minecraft_env_enabled", False):
        assert flags.minecraft_env_enabled is False
    # Original '1' should be restored
    assert os.environ.get("SUBREP_FLAG_MINECRAFT_ENV_ENABLED") == "1"


def test_override_flag_removes_env_var_if_not_originally_set():
    """override_flag removes the env var entirely if it wasn't set before."""
    from utils.feature_flags import flags, override_flag
    assert "SUBREP_FLAG_MINECRAFT_ENV_ENABLED" not in os.environ
    with override_flag("minecraft_env_enabled", True):
        assert flags.minecraft_env_enabled is True
    assert "SUBREP_FLAG_MINECRAFT_ENV_ENABLED" not in os.environ


# ---------------------------------------------------------------------------
# Independence: one flag does not affect another
# ---------------------------------------------------------------------------

def test_flags_are_independent():
    """Enabling one flag does not enable others."""
    from utils.feature_flags import flags
    os.environ["SUBREP_FLAG_MINECRAFT_ENV_ENABLED"] = "1"
    other_flags = [n for n in _EXPECTED_FLAGS if n != "minecraft_env_enabled"]
    for name in other_flags:
        assert flags.is_enabled(name) is False, f"{name} should still be OFF"


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr_shows_enabled_flags():
    """repr() lists the names of enabled flags."""
    from utils.feature_flags import flags
    os.environ["SUBREP_FLAG_MINECRAFT_ENV_ENABLED"] = "1"
    r = repr(flags)
    assert "minecraft_env_enabled" in r


def test_repr_empty_when_all_off():
    """repr() shows no enabled flags when all are OFF."""
    from utils.feature_flags import flags
    r = repr(flags)
    assert "enabled=[]" in r
