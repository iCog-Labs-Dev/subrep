"""
Environment factory/registry for SubRep.

Provides a central ``EnvRegistry`` that maps environment names to factory
callables, so any component can create an environment by name rather than
importing concrete wrapper classes directly.

Usage::

    from env.registry import make_env, register_env

    env = make_env("lunarlander", seed=7)
    env = make_env("village", seed=42)
    env = make_env("safety_gymnasium", env=my_fake_backend)

Custom environments can be added at runtime::

    register_env("my_env", MyEnvClass)
"""

from __future__ import annotations

from typing import Any, Callable, Dict


class EnvRegistry:
    """
    Lightweight environment factory/registry.

    Maps string names to callables (classes or functions) that construct
    SubRepBaseEnv-conforming environment instances.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        """
        Register an environment factory under *name*.

        Args:
            name: Case-insensitive identifier used in ``make()``.
            factory: Callable (class or function) that returns a new env
                     instance when called.  All positional and keyword
                     arguments passed to ``make()`` are forwarded here.

        Raises:
            ValueError: If *name* is empty or *factory* is not callable.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if not callable(factory):
            raise ValueError(f"factory must be callable, got {type(factory).__name__}")
        self._registry[name.strip().lower()] = factory

    def make(self, name: str, **kwargs: Any) -> Any:
        """
        Instantiate a registered environment by *name*.

        Args:
            name: The environment identifier (case-insensitive).
            **kwargs: Forwarded verbatim to the registered factory callable.

        Returns:
            A new environment instance.

        Raises:
            KeyError: If *name* is not registered.
        """
        key = name.strip().lower()
        if key not in self._registry:
            available = sorted(self._registry.keys())
            raise KeyError(
                f"Environment {name!r} is not registered. "
                f"Available: {available}"
            )
        return self._registry[key](**kwargs)

    def list_envs(self) -> list[str]:
        """Return a sorted list of all registered environment names."""
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name.strip().lower() in self._registry


# ---------------------------------------------------------------------------
# Global singleton registry
# ---------------------------------------------------------------------------

_REGISTRY = EnvRegistry()


def register_env(name: str, factory: Callable[..., Any]) -> None:
    """Register an environment factory in the global registry."""
    _REGISTRY.register(name, factory)


def make_env(name: str, **kwargs: Any) -> Any:
    """Instantiate a registered environment from the global registry."""
    return _REGISTRY.make(name, **kwargs)


def list_envs() -> list[str]:
    """Return all registered environment names."""
    return _REGISTRY.list_envs()


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------

def _make_lunarlander(**kwargs):
    from env.lunar_lander_wrapper import SubRepEnv
    return SubRepEnv(**kwargs)


def _make_village(**kwargs):
    from village_sim.env import VillageEnv
    return VillageEnv(**kwargs)


def _make_safety_gymnasium(**kwargs):
    from env.safety_gymnasium_wrapper import SafetyGymnasiumEnv
    return SafetyGymnasiumEnv(**kwargs)


_REGISTRY.register("lunarlander", _make_lunarlander)
_REGISTRY.register("village", _make_village)
_REGISTRY.register("safety_gymnasium", _make_safety_gymnasium)
