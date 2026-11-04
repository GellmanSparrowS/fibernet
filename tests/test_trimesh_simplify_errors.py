"""Provider-free regression for mesh simplification dependency diagnostics."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


_SCRIPT = r'''
import importlib.abc
from pathlib import Path
import sys

missing_name, package_parent = sys.argv[1:]
missing_name = None if missing_name == "<none>" else missing_name
blocked = {
    "trimesh", "fast_simplification", "open3d", "taichi", "pyvista",
    "manifold3d", "torch", "torch_geometric", "sklearn", "gymnasium",
    "gym", "stable_baselines3", "lammps", "exudyn",
}
attempts = []


class BlockProviders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            attempts.append(fullname)
            if fullname == "trimesh":
                raise ModuleNotFoundError(
                    "test blocks real trimesh", name="trimesh"
                )
            raise AssertionError("unexpected optional provider import: " + fullname)


assert not any(name.split(".")[0] in blocked for name in sys.modules)
sys.meta_path.insert(0, BlockProviders())

# Exercise the actual adapter, not a copied function or a provider implementation.
import fibernet.trimesh_integration as subject

assert Path(subject.__file__).resolve() == (
    Path(package_parent) / "fibernet" / "trimesh_integration.py"
).resolve()
assert subject.TRIMESH_AVAILABLE is False
assert attempts == ["trimesh"]

sentinel = ModuleNotFoundError("dependency sentinel", name=missing_name)
calls = []


def raise_missing_dependency():
    raise sentinel


class MissingDependencyMesh:
    # A stand-in for a provider failure, not a real Trimesh/Open3D execution.
    # Supported older Trimesh delegates simplification through Open3D.
    def simplify_quadric_decimation(self, face_count):
        calls.append(face_count)
        raise_missing_dependency()


try:
    subject.simplify_mesh(MissingDependencyMesh(), face_count=1)
except ImportError as error:
    if missing_name == "fast_simplification":
        assert type(error) is ImportError
        assert str(error) == (
            "fast_simplification is required for mesh simplification. "
            "Install with: pip install fast-simplification"
        )
        assert error.__context__ is sentinel
    else:
        assert error is sentinel, (type(error), str(error))
        assert error.name == missing_name
        assert str(error) == "dependency sentinel"
        codes = []
        tb = error.__traceback__
        while tb is not None:
            codes.append(tb.tb_frame.f_code)
            tb = tb.tb_next
        # Bare re-raise preserves the provider frames without inserting an
        # additional adapter frame as `raise error` would do.
        assert codes[1:] == [
            subject.simplify_mesh.__code__,
            MissingDependencyMesh.simplify_quadric_decimation.__code__,
            raise_missing_dependency.__code__,
        ], [code.co_name for code in codes]
else:
    raise AssertionError("missing dependency did not raise")

assert calls == [1]
assert attempts == ["trimesh"]
assert not any(name.split(".")[0] in blocked for name in sys.modules)
'''


@pytest.mark.parametrize("missing_name", [
    "fast_simplification", "open3d", "fast_simplification._simplify",
    "unrelated_dependency", "<none>",
])
def test_simplify_mesh_preserves_missing_dependency(missing_name, tmp_path):
    spec = importlib.util.find_spec("fibernet")
    assert spec is not None and spec.origin is not None
    package_parent = str(Path(spec.origin).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [package_parent, env.get("PYTHONPATH", "")]
    )
    for key in ("PYTHONSTARTUP", "PYTHONHOME", "PYTHONUSERBASE"):
        env.pop(key, None)
    result = subprocess.run(
        [sys.executable, "-B", "-c", _SCRIPT, missing_name, package_parent],
        cwd=tmp_path, env=env, capture_output=True, text=True,
        encoding="utf-8", timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
