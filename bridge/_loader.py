"""Locate the MetaMo checkout and put it on sys.path.

This is the ONLY place in SubRep that manipulates sys.path for MetaMo.

MetaMo is not pip-installable (no pyproject.toml / setup.py) and its modules
use root-relative absolute imports -- `core/state.py:6` does
`from core.config import ...`. So its repository root must be importable.

Resolution order:
  1. $SUBREP_METAMO_PATH             -- explicit override
  2. <subrep>/external/metamo        -- the pinned git submodule
  3. <workspace>/MetaMo-Python       -- sibling checkout (current dev layout)

Putting the MetaMo root on sys.path makes its top-level packages importable
by their own names (core, category, dynamics, openpsi, magus). Those do not
collide with SubRep's top-level packages today; `assert_no_import_collision`
turns a future collision into a loud error instead of a silent shadowing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

# Packages MetaMo exposes at its repository root that we actually import.
_METAMO_PACKAGES = ("core", "category", "dynamics", "openpsi", "magus")

# A file that must exist for a directory to be a plausible MetaMo root.
_METAMO_SENTINEL = Path("core") / "config.py"

_SUBREP_ROOT = Path(__file__).resolve().parent.parent
_WORKSPACE_ROOT = _SUBREP_ROOT.parent

_resolved_root: Optional[Path] = None


def candidate_roots() -> List[Path]:
    """Return the MetaMo root candidates, in resolution order."""
    candidates: List[Path] = []

    override = os.environ.get("SUBREP_METAMO_PATH")
    if override:
        candidates.append(Path(override).expanduser())

    candidates.append(_SUBREP_ROOT / "external" / "metamo")
    candidates.append(_WORKSPACE_ROOT / "MetaMo-Python")
    return candidates


def _is_metamo_root(path: Path) -> bool:
    return (path / _METAMO_SENTINEL).is_file()


def find_metamo_root() -> Optional[Path]:
    """Return the first candidate that looks like a MetaMo checkout."""
    for candidate in candidate_roots():
        try:
            if _is_metamo_root(candidate):
                return candidate.resolve()
        except OSError:
            # An unreadable candidate is simply not a match.
            continue
    return None


def is_available() -> bool:
    """True when MetaMo can be located. Used to skip adapter tests."""
    return find_metamo_root() is not None


def assert_no_import_collision(metamo_root: Path) -> None:
    """Fail loudly if MetaMo would shadow a SubRep top-level package.

    MetaMo's package names are generic (`core`, `dynamics`). If SubRep ever
    grows a package with the same name, silent shadowing would be very hard
    to diagnose -- so refuse to load instead.
    """
    collisions = [
        name for name in _METAMO_PACKAGES
        if (_SUBREP_ROOT / name).is_dir() or (_SUBREP_ROOT / f"{name}.py").is_file()
    ]
    if collisions:
        raise ImportError(
            "MetaMo package name(s) collide with SubRep top-level modules: "
            f"{', '.join(sorted(collisions))}. Adding {metamo_root} to sys.path "
            "would shadow them. Rename the SubRep module or vendor MetaMo under "
            "a namespace package before continuing."
        )


def ensure_metamo_on_path() -> Path:
    """Put the MetaMo root on sys.path (idempotent) and return it.

    Raises ImportError with actionable guidance when MetaMo is absent.
    """
    global _resolved_root
    if _resolved_root is not None:
        return _resolved_root

    root = find_metamo_root()
    if root is None:
        searched = "\n  ".join(str(p) for p in candidate_roots())
        raise ImportError(
            "Could not locate a MetaMo checkout. Searched:\n  "
            f"{searched}\n\n"
            "Fix by either:\n"
            "  git submodule add https://github.com/kirubel-Nigussie/MetaMo-Python.git "
            "external/metamo\n"
            "or by setting SUBREP_METAMO_PATH to an existing checkout."
        )

    assert_no_import_collision(root)

    root_str = str(root)
    if root_str not in sys.path:
        # Append rather than prepend: SubRep's own packages keep priority.
        sys.path.append(root_str)

    _resolved_root = root
    return root
