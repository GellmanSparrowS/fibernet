"""Provider-free checks of the real MDAnalysis bridge constructor."""
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
from types import ModuleType

provider_state, package_parent = sys.argv[1:]
blocked = {
    "MDAnalysis", "ovito", "trimesh", "fast_simplification", "open3d",
    "taichi", "pyvista", "manifold3d", "torch", "torch_geometric",
    "sklearn", "gymnasium", "gym", "stable_baselines3", "lammps", "exudyn",
}
assert not any(name.split(".")[0] in blocked for name in sys.modules)
attempts = []
missing = ModuleNotFoundError("test blocks real MDAnalysis", name="MDAnalysis")


class BlockProviders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            attempts.append(fullname)
            if provider_state == "missing" and fullname == "MDAnalysis":
                raise missing
            raise AssertionError("unexpected real provider import: " + fullname)


sys.meta_path.insert(0, BlockProviders())
standins = {}
if provider_state == "present":
    # Match the relevant namespace in official MDAnalysis 2.10.0: base is
    # importable and exports TopologyReaderBase, but does not export Topology.
    # ModuleType is deliberate: mocks must not invent missing attributes.
    provider = ModuleType("MDAnalysis")
    provider.__path__ = []
    topology = ModuleType("MDAnalysis.topology")
    topology.__path__ = []
    base = ModuleType("MDAnalysis.topology.base")

    class TopologyReaderBase:
        pass

    base.TopologyReaderBase = TopologyReaderBase
    provider.topology = topology
    topology.base = base
    standins = {module.__name__: module for module in (provider, topology, base)}
    sys.modules.update(standins)
    assert "Topology" not in vars(base)

# Import the complete production module, not an extracted/copied constructor.
import fibernet.integrations.mdanalysis_integration as subject
from fibernet.core.network import FiberNetwork

assert Path(subject.__file__).resolve() == (
    Path(package_parent) / "fibernet" / "integrations" / "mdanalysis_integration.py"
).resolve()
assert attempts == []
network = FiberNetwork()

if provider_state == "present":
    bridge = subject.MDAnalysisBridge(network)
    assert bridge.network is network
    assert bridge.mda is provider
    assert "Topology" not in vars(base)
    assert attempts == []
else:
    try:
        subject.MDAnalysisBridge(network)
    except ImportError as exc:
        assert type(exc) is ImportError
        assert str(exc) == "MDAnalysis required. Install with: pip install MDAnalysis"
        assert exc.__context__ is missing
    else:
        raise AssertionError("missing MDAnalysis must keep its installation error")
    assert attempts == ["MDAnalysis"]

assert {
    name: module for name, module in sys.modules.items()
    if name.split(".")[0] in blocked
} == standins
# Each case has its own child process: no provider sys.modules state leaks.
'''


@pytest.mark.parametrize("provider_state", ["present", "missing"])
def test_mdanalysis_bridge_constructor(provider_state, tmp_path):
    spec = importlib.util.find_spec("fibernet")
    assert spec is not None and spec.origin is not None
    package_parent = str(Path(spec.origin).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [package_parent, env.get("PYTHONPATH", "")]
    )
    for key in (
        "PYTHONSTARTUP", "PYTHONHOME", "PYTHONUSERBASE",
        "PYTHONOPTIMIZE", "PYTHONPYCACHEPREFIX",
    ):
        env.pop(key, None)
    result = subprocess.run(
        [sys.executable, "-B", "-c", _SCRIPT, provider_state, package_parent],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
