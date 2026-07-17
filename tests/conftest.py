"""Pytest configuration - exclude archived tests + Taichi cache management."""
import os
import pytest

collect_ignore_glob = ["_archived/**"]


@pytest.fixture(autouse=True, scope="class")
def _clear_taichi_cache():
    """Clear TaichiEngine field cache between test classes to prevent SNode accumulation."""
    yield
    try:
        from fibernet.sim.accelerated import TaichiEngine
        TaichiEngine.clear_field_cache()
    except ImportError:
        pass
