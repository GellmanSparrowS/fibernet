"""Pytest configuration - cross-platform Taichi test management.

IMPORTANT: Taichi version check monkey-patch MUST happen at module level,
before any test file imports taichi (which starts a background thread that
crashes on macOS CI).
"""
import os
import sys

collect_ignore_glob = ["_archived/**"]

IS_WINDOWS = sys.platform == "win32"

# Disable Taichi version check env var
os.environ["TI_DISABLE_VERSION_CHECK"] = "1"

# Monkey-patch Taichi version check IMMEDIATELY (before any import taichi)
try:
    import taichi._version_check as _vc
    _vc.try_check_version = lambda: None
    _vc.check_version = lambda: None
except (ImportError, AttributeError):
    pass

# ── Pytest fixtures and hooks below ──
import pytest


@pytest.fixture(autouse=True, scope="class")
def _clear_taichi_cache():
    """Clear TaichiEngine field cache between test classes."""
    yield
    try:
        from fibernet.sim.accelerated import TaichiEngine
        TaichiEngine.clear_field_cache()
    except ImportError:
        pass


def pytest_collection_modifyitems(config, items):
    """Skip Taichi simulation tests on Windows (no fork support)."""
    if not IS_WINDOWS:
        return

    skip_taichi = pytest.mark.skip(
        reason="Taichi SNode exhaustion on Windows (no fork support)"
    )

    taichi_classes = {
        "TestSimulation3D", "TestVisualization3D",
        "TestEnergyComputation", "TestOOMGuard",
    }

    for item in items:
        if hasattr(item, "cls") and item.cls and item.cls.__name__ in taichi_classes:
            item.add_marker(skip_taichi)
