"""Pytest configuration - cross-platform Taichi test management."""
import os
import sys
import pytest

collect_ignore_glob = ["_archived/**"]

IS_WINDOWS = sys.platform == "win32"

# Disable Taichi version check (causes abort on macOS CI)
os.environ["TI_DISABLE_VERSION_CHECK"] = "1"


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
    """Skip Taichi simulation tests on Windows (SNode exhaustion + no fork support)."""
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
