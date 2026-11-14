"""Optional native manufacturing APIs fail before geometry work."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest

from test_typed_package_data import _literal_list_declaration

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("extra", ["manufacturing", "full"])
def test_manufacturing_extras_declare_simplify_floor(extra):
    requirements = _literal_list_declaration(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        "project.optional-dependencies",
        extra,
    )
    assert [item for item in requirements if item.startswith("manifold3d")] == [
        "manifold3d>=3.2.0"
    ]


def test_native_provider_remains_optional_for_core():
    requirements = _literal_list_declaration(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        "project",
        "dependencies",
    )
    assert not any(item.startswith("manifold3d") for item in requirements)


@pytest.mark.parametrize("name", ["to_mesh64", "simplify"])
@pytest.mark.parametrize("value", ["missing", None, 7, False])
def test_missing_native_capability_fails_before_geometry(name, value):
    spec = importlib.util.spec_from_file_location(
        "provider_contract_tube", ROOT / "fibernet/gen/tube_solid.py"
    )
    tube = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tube)

    class Manifold:
        def to_mesh64(self):
            raise AssertionError("geometry must not start")

        def simplify(self, tolerance):
            raise AssertionError("geometry must not start")

        @staticmethod
        def sphere(*args, **kwargs):
            raise AssertionError("geometry must not start")

    if value == "missing":
        delattr(Manifold, name)
    else:
        setattr(Manifold, name, value)
    provider = ModuleType("manifold3d")
    provider.Manifold = Manifold
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edges = np.array([[0, 1]])
    with (
        patch.dict(sys.modules, {"manifold3d": provider}),
        patch.object(
            tube.np, "unique", side_effect=AssertionError("geometry must not start")
        ),
    ):
        with pytest.raises(RuntimeError, match=rf"Manifold\.{name}.*>=3\.2\.0"):
            tube.tube_mesh(points, edges, 0.25)
