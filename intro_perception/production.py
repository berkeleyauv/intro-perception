"""Import the production package without its legacy eager top-level imports."""

from __future__ import annotations

from importlib.util import find_spec
from types import ModuleType
import sys


def prepare_production_namespace():
    """Expose the installed ``perception`` package as a lazy namespace.

    The pinned production package's top-level ``__init__`` imports an optional
    Cython extension eagerly. Intro algorithms only need its task interfaces and
    visualizer, so bypassing that unrelated import keeps classical-only setup
    usable while preserving normal submodule imports.
    """
    if "perception" in sys.modules:
        return
    spec = find_spec("perception")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(
            "production perception package is unavailable; run ./scripts/setup.sh"
        )
    package = ModuleType("perception")
    package.__path__ = list(spec.submodule_search_locations)
    package.__package__ = "perception"
    package.__spec__ = spec
    sys.modules["perception"] = package
