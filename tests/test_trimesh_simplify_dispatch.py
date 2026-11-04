"""Provider-free regression for Trimesh face-count keyword delegation."""
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

provider_style, call_style, requested_text, package_parent = sys.argv[1:]
requested = int(requested_text)
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

# Import the actual complete package/module; do not copy or extract its function.
import fibernet.trimesh_integration as subject

assert Path(subject.__file__).resolve() == (
    Path(package_parent) / "fibernet" / "trimesh_integration.py"
).resolve()
assert subject.TRIMESH_AVAILABLE is False
assert attempts == ["trimesh"]

returned = object()
calls = []


class LegacyMesh:
    # Official Trimesh 3.20.0/4.0.0 signature: face_count is first.
    def simplify_quadric_decimation(self, face_count):
        calls.append((None, face_count, None))
        return returned


class ModernMesh:
    # Official Trimesh 4.11.0 signature: percent is first.
    def simplify_quadric_decimation(
        self, percent=None, face_count=None, aggression=None
    ):
        calls.append((percent, face_count, aggression))
        return returned


mesh = LegacyMesh() if provider_style == "legacy" else ModernMesh()
if call_style == "positional":
    actual = subject.simplify_mesh(mesh, requested)
else:
    actual = subject.simplify_mesh(mesh, face_count=requested)

# These assertions observe Python's real method argument binding through the
# production wrapper. Calling either spy directly would not exercise the bug.
assert calls == [(None, requested, None)], calls
assert actual is returned
assert attempts == ["trimesh"]
assert not any(name.split(".")[0] in blocked for name in sys.modules)
'''


@pytest.mark.parametrize("provider_style", ["legacy", "modern"])
@pytest.mark.parametrize("call_style", ["positional", "keyword"])
@pytest.mark.parametrize("face_count", [1, 1000])
def test_simplify_mesh_delegates_face_count(
    provider_style, call_style, face_count, tmp_path
):
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
        [
            sys.executable, "-B", "-c", _SCRIPT,
            provider_style, call_style, str(face_count), package_parent,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
