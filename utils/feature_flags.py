"""
utils/feature_flags.py
ADR-007 — Feature Flags for Minecraft Integration.

Eight flags, all default OFF, overridable via ``SUBREP_FLAG_<NAME>`` environment
variables. When a Minecraft flag is OFF, no Minecraft import is attempted, so
LunarLander and Safety-Gymnasium paths are never affected.

Usage:
    from utils.feature_flags import flags

    if flags.minecraft_env_enabled:
        from env.minecraft_adapter import MinecraftEnv

    # Or check at call time (re-reads env var on every call):
    if flags.is_enabled("minecraft_env_enabled"):
        ...

Overriding in tests or CI:
    import os
    os.environ["SUBREP_FLAG_MINECRAFT_ENV_ENABLED"] = "1"

    # Or use the provided test helper:
    from utils.feature_flags import override_flag
    with override_flag("minecraft_env_enabled", True):
        ...

Flag definitions (all default OFF):
    minecraft_env_enabled       — Python Minecraft env wrapper may be imported
    real_server_enabled         — live Paper/Mineflayer server connection allowed
    paired_rollout_enabled      — snapshot-restore paired rollout collector active
    minecraft_certification_enabled — 6D Minecraft cert path in admission pipeline
    skill_generator_enabled     — SkillGenerator pre-filter active at runtime
    mdn_ranking_enabled         — MDN-based runtime skill ranking active
    rl_options_enabled          — trained SMDP options replace scripted candidates
    contextual_geometry_enabled — 6D W_x contextual geometry (LP-solver) active

Fallback guarantee (from ADR-007):
    When a learned-component flag is OFF, the system selects from certified
    scripted options — never from uncertified actions.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Generator


# ---------------------------------------------------------------------------
# Flag names — single source of truth
# ---------------------------------------------------------------------------

_ALL_FLAGS: tuple[str, ...] = (
    "minecraft_env_enabled",
    "real_server_enabled",
    "paired_rollout_enabled",
    "minecraft_certification_enabled",
    "skill_generator_enabled",
    "mdn_ranking_enabled",
    "rl_options_enabled",
    "contextual_geometry_enabled",
)


def _env_var(name: str) -> str:
    """Return the environment variable name for a flag."""
    return f"SUBREP_FLAG_{name.upper()}"


def _read(name: str) -> bool:
    """Read a flag from the environment. Any non-empty truthy value enables it."""
    val = os.environ.get(_env_var(name), "").strip().lower()
    return val in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# FeatureFlags class
# ---------------------------------------------------------------------------

@dataclass(frozen=False)
class FeatureFlags:
    """
    Runtime feature flag accessor.

    Each flag is a property that reads its ``SUBREP_FLAG_<NAME>`` environment
    variable dynamically, so flag changes (e.g. in tests) take effect without
    restarting the process.
    """

    # ---------- public properties (dynamic env-var reads) ----------

    @property
    def minecraft_env_enabled(self) -> bool:
        """Python Minecraft environment wrapper may be imported."""
        return _read("minecraft_env_enabled")

    @property
    def real_server_enabled(self) -> bool:
        """Live Paper/Mineflayer server connection is allowed."""
        return _read("real_server_enabled")

    @property
    def paired_rollout_enabled(self) -> bool:
        """Snapshot-restore paired rollout collector is active."""
        return _read("paired_rollout_enabled")

    @property
    def minecraft_certification_enabled(self) -> bool:
        """6D Minecraft certification path in admission pipeline is active."""
        return _read("minecraft_certification_enabled")

    @property
    def skill_generator_enabled(self) -> bool:
        """SkillGenerator pre-filter is active at runtime."""
        return _read("skill_generator_enabled")

    @property
    def mdn_ranking_enabled(self) -> bool:
        """MDN-based runtime skill ranking is active."""
        return _read("mdn_ranking_enabled")

    @property
    def rl_options_enabled(self) -> bool:
        """Trained SMDP options replace scripted candidates."""
        return _read("rl_options_enabled")

    @property
    def contextual_geometry_enabled(self) -> bool:
        """6D W_x contextual geometry (LP-solver path) is active."""
        return _read("contextual_geometry_enabled")

    # ---------- helpers ----------

    def is_enabled(self, name: str) -> bool:
        """
        Check a flag by name at call time.

        Args:
            name: One of the flag names in _ALL_FLAGS.

        Returns:
            True if the flag is enabled.

        Raises:
            ValueError: If name is not a known flag.
        """
        if name not in _ALL_FLAGS:
            raise ValueError(
                f"Unknown feature flag: {name!r}. "
                f"Known flags: {sorted(_ALL_FLAGS)}"
            )
        return _read(name)

    def all_flags(self) -> dict[str, bool]:
        """Return current state of all flags as a dict."""
        return {name: _read(name) for name in _ALL_FLAGS}

    def __repr__(self) -> str:
        enabled = [n for n in _ALL_FLAGS if _read(n)]
        return f"FeatureFlags(enabled={enabled})"


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

flags = FeatureFlags()
"""Module-level singleton. Import and use: ``from utils.feature_flags import flags``"""


# ---------------------------------------------------------------------------
# Test helper: context-manager override
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def override_flag(name: str, value: bool) -> Generator[None, None, None]:
    """
    Context manager to temporarily set a feature flag for testing.

    Usage::

        from utils.feature_flags import override_flag, flags

        with override_flag("minecraft_env_enabled", True):
            assert flags.minecraft_env_enabled is True
        assert flags.minecraft_env_enabled is False  # restored

    Args:
        name: Flag name (must be in _ALL_FLAGS).
        value: Desired value while inside the context.
    """
    if name not in _ALL_FLAGS:
        raise ValueError(
            f"Unknown feature flag: {name!r}. "
            f"Known flags: {sorted(_ALL_FLAGS)}"
        )
    env_key = _env_var(name)
    old = os.environ.get(env_key)
    try:
        os.environ[env_key] = "1" if value else "0"
        yield
    finally:
        if old is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = old


def list_flags() -> list[str]:
    """Return all known flag names."""
    return list(_ALL_FLAGS)
