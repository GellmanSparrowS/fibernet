"""Frozen pre-optimization small-branch outputs and ownership regressions."""
import copy
import hashlib
import json
import numpy as np
import pytest
from fibernet import Material, StructureGraph
from fibernet.core.tiling import tile_2d
import fibernet.gen.pattern as pattern

# Frozen from unmodified source, before applying the owned-helper change.
BASELINES = [({'unit': 'voronoi', 'box': [3, 2], 'grid': [2, 1], 'unit_kwargs': {'n_seeds': 5}}, '86a0abcba247ff575115882f84270c04ea80bbcca10270286004209b109f2906'), ({'unit': 'voronoi', 'box': [3, 2], 'grid': [2, 1], 'unit_kwargs': {'n_seeds': 5}, 'seed': None}, '86a0abcba247ff575115882f84270c04ea80bbcca10270286004209b109f2906'), ({'unit': 'voronoi', 'box': [3, 2], 'grid': [2, 1], 'unit_kwargs': {'n_seeds': 5}, 'seed': 7}, 'e880b51982601abac395438ea41b6732c96a5552242473d688e97ac245882be5'), ({'unit': 'voronoi', 'box': [3, 2], 'grid': [2, 1], 'unit_kwargs': {'n_seeds': 5}, 'mirror_x': True}, 'ebc293463f5192b5348b337dcddf334831038ca7f82d8a80bf58322186d20b58'), ({'unit': 'voronoi', 'box': [3, 2], 'grid': [2, 1], 'unit_kwargs': {'n_seeds': 5}, 'mirror_y': True}, 'c72f3a44abf03d6469689b571c5e7e6c9eb314b04929e8d3fdebe1964ce2110e'), ({'unit': 'voronoi', 'box': [3, 2], 'grid': [2, 1], 'unit_kwargs': {'n_seeds': 5}, 'rotation': 27}, '42ae8ccf4ef9c6d7a9b22bf66e66870d98d8d51c42d2816180882f38c3c16d8f'), ({'unit': 'voronoi', 'box': [3, 2], 'grid': [2, 1], 'unit_kwargs': {'n_seeds': 5}, 'boundary_mode': 'extend'}, 'da2c3ad2f5934e476593826f55ef0449b61a8155ddbac8687a787241fd600466'), ({'unit': 'voronoi', 'box': [3, 2], 'grid': [2, 1], 'unit_kwargs': {'n_seeds': 5}, 'boundary_mode': 'error'}, '86a0abcba247ff575115882f84270c04ea80bbcca10270286004209b109f2906'), ({'unit': 'voronoi', 'box': [3, 2], 'grid': [2, 1], 'unit_kwargs': {'n_seeds': 5}, 'n_internal': 2}, '9b72705caaddb723066de7fc22b0fa7c4415fac0a0c1fb5b48c7f57ef0a1ab4b'), ({'unit': 'voronoi', 'box': [3, 2], 'grid': [2, 1], 'unit_kwargs': {'n_seeds': 5}, 'n_pts_per_side': 1, 'perturbation': 0.04}, '20ad7d3d7d534198430e5cf23eba5d2c0f7ea9c818c34209b7dd147ffb639265'), ({'unit': 'voronoi', 'box': [3, 2], 'grid': [2, 1], 'unit_kwargs': {'n_seeds': 5, 'seed': None, 'radius': 0.2, 'n_internal': 1}, 'seed': 7}, 'a11895d103f238d15548e63520dabc80ca255ff5e49833736da2aef3f8831c3d'), ({'unit': 'Voronoi', 'box': [3, 2], 'grid': [2, 1], 'unit_kwargs': {'n_seeds': 5}}, '186730ea2777dab124093f883a0d0f1a20b9e01755b30eb09db5aa771f0fa302'), ({'unit': None, 'box': [3, 2], 'grid': [2, 1], 'unit_kwargs': {'n_seeds': 5}, 'points': [[0, 0], [1, 0], [1, 1]], 'fit_to_box': True}, 'ed24c62d5f0a8dd07477faf1e2fab4c41dda1e00f682529017a2c6c715ab89c2')]

# Historical JSON remains byte-for-byte unchanged. Hashes are only a contract
# for the verified reference numerical stack, not every SciPy/QHull build.
from pathlib import Path
import platform
import runpy
import scipy

_FIXTURES = Path(__file__).parent / "fixtures"
_FROZEN = json.loads((_FIXTURES / "voronoi_pre_owned_baselines.json").read_text())
_ORIGINAL_BRIDGE = runpy.run_path(str(_FIXTURES / "voronoi_pre_owned_bridge.py"))["_bridge_small_components"]
_REFERENCE_STACK = (platform.system(), platform.machine(), platform.python_version(),
                    np.__version__, scipy.__version__) == (
                        "Linux", "x86_64", "3.12.14", "2.3.5", "1.17.0")


def _value(g):
    return {"graph": g.to_dict(), "node_order": list(g.nodes), "edge_order": list(g.edges)}


def _raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _compare_frozen(actual, expected, path="$", differences=None):
    """Exact schema, IDs, order, metadata; bounded roundoff for float leaves."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _compare_frozen(actual[key], expected[key], path + "." + key, differences)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)):
            _compare_frozen(a, b, f"{path}[{i}]", differences)
    elif isinstance(expected, float):
        assert isinstance(actual, float), path
        # This does not replace the zero-tolerance original-vs-owned comparison.
        assert np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-12, atol=1e-12), (path, actual, expected)
        if actual != expected:
            differences.append((path, actual, expected, abs(actual - expected)))
    else:
        assert type(actual) is type(expected) and actual == expected, (path, actual, expected)


@pytest.mark.parametrize("kwargs,expected", BASELINES)
def test_public_options_frozen_equivalence(kwargs, expected, monkeypatch):
    g = pattern.pattern_2d(**kwargs)
    value = _value(g)
    # Force the actual pre-optimization copy implementation, independent of the
    # refactored common in-place helper. Never change seed, IDs or factory.
    calls = []
    def original(*args, **options):
        calls.append(True)
        return _ORIGINAL_BRIDGE(*args, **options)
    monkeypatch.setattr(pattern, "_bridge_owned_components", original)
    monkeypatch.setattr(pattern, "_bridge_small_components", original)
    reference = _value(pattern.pattern_2d(**kwargs))
    assert bool(calls) == (kwargs["unit"] == "voronoi")
    assert value == reference  # Every parsed field, including numerical leaves.
    assert _raw(value) == _raw(reference)  # Strict same-environment bytes.
    assert _raw(value) == _raw(_value(pattern.pattern_2d(**kwargs)))
    record = next(r for r in _FROZEN if r["kwargs"] == kwargs)
    assert record["sha256"] == expected
    assert hashlib.sha256(_raw(record["value"])).hexdigest() == expected
    differences = []
    _compare_frozen(value, record["value"], differences=differences)
    print("Voronoi reference comparison:", platform.platform(), platform.python_version(),
          "numpy", np.__version__, "scipy", scipy.__version__,
          "float differences", len(differences),
          "max absolute delta", max((d[3] for d in differences), default=0.),
          "first differences", differences[:3])
    if _REFERENCE_STACK or kwargs["unit"] is None:
        assert hashlib.sha256(_raw(value)).hexdigest() == expected


class CountingMaterial(Material):
    calls = 0
    def __deepcopy__(self, memo):
        type(self).calls += 1
        result = type(self)()
        memo[id(self)] = result
        result.__dict__ = copy.deepcopy(self.__dict__, memo)
        return result

@pytest.mark.parametrize("choice", ["material", "subclass", "kwargs_none", "kwargs_material"])
def test_public_material_falls_back(monkeypatch, choice):
    calls = []
    original = pattern._bridge_small_components
    def recording(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(pattern, "_bridge_small_components", recording)
    def forbidden(*args, **kwargs):
        pytest.fail("caller material must not use owned path")
    monkeypatch.setattr(pattern, "_bridge_owned_components", forbidden)
    mat = CountingMaterial(extra={"nested": [1]}) if choice == "subclass" else Material(extra={"nested": [1]})
    kwargs = {"unit": "voronoi", "grid": 1, "unit_kwargs": {"n_seeds": 4}}
    if choice.startswith("kwargs"):
        kwargs["unit_kwargs"]["material"] = None if choice == "kwargs_none" else mat
    else:
        kwargs["material"] = mat
    before = copy.deepcopy(mat.to_dict())
    CountingMaterial.calls = 0
    g = pattern.pattern_2d(**kwargs)
    assert calls == [True]
    assert mat.to_dict() == before
    if choice == "subclass":
        assert CountingMaterial.calls >= 2 * len(g.edges)
        assert all(isinstance(e.material, CountingMaterial) for e in g.edges.values())


def test_registered_voronoi_factory_falls_back_and_registry_restored(monkeypatch):
    registry = dict(pattern._UNIT_FACTORIES)
    mat = CountingMaterial(extra={"shared": [1]})
    def custom(**kwargs):
        g = StructureGraph(dimension=2, box_size=[10,10])
        a = g.add_node([0,0], nested={"v": [1]})
        b = g.add_node([1,0])
        g.add_edge(a,b,material=mat)
        return g
    def forbidden(*args, **kwargs):
        pytest.fail("registered factory must use copy path")
    monkeypatch.setattr(pattern, "_bridge_owned_components", forbidden)
    try:
        pattern.register_unit("voronoi", custom)
        CountingMaterial.calls = 0
        result = pattern.pattern_2d(unit="voronoi", grid=1)
        assert CountingMaterial.calls == 2
        assert next(iter(result.edges.values())).material is not mat
    finally:
        pattern._UNIT_FACTORIES.clear()
        pattern._UNIT_FACTORIES.update(registry)
    assert pattern._UNIT_FACTORIES == registry


def fixture():
    mat = Material(extra={"nested": {"values": [1.,2.]}})
    unit = StructureGraph(dimension=2, box_size=[2,2])
    a = unit.add_node([0,0], nested={"values": [3]})
    b = unit.add_node([1,0], nested={"values": [4]})
    unit.add_edge(a,b,material=mat,n_internal=1,nested={"values": [5]})
    unit.metadata["nested"] = {"values": [6]}
    return unit, mat, tile_2d(unit, grid=(2,1))


def alias_matrix(g, unit, mat):
    objects = [mat, mat.extra, mat.extra["nested"], unit.metadata["nested"]]
    for node in unit.nodes.values():
        objects.append(node.metadata["nested"])
    for edge in unit.edges.values():
        objects.extend([edge.material,edge.material.extra,edge.metadata["nested"]])
    objects.append(g.metadata["nested"])
    for node in g.nodes.values():
        objects.append(node.metadata["nested"])
    for edge in g.edges.values():
        objects.extend([edge.material,edge.material.extra,edge.metadata.get("nested")])
    return [[x is y for y in objects] for x in objects]


def test_owned_nested_alias_topology_and_mutations():
    ua, ma, ga = fixture()
    ub, mb, gb = fixture()
    original = copy.deepcopy(ga.to_dict())
    a = pattern._bridge_small_components(ga, material=ma)
    b = pattern._bridge_owned_components(gb, material=mb)
    assert a is not ga and ga.to_dict() == original
    assert b is gb
    assert a.to_dict() == b.to_dict()
    assert alias_matrix(a,ua,ma) == alias_matrix(b,ub,mb)
    for graph, unit in [(a,ua),(b,ub)]:
        for node in graph.nodes.values():
            assert all(not np.shares_memory(node.position,u.position) for u in unit.nodes.values())
        for edge in graph.edges.values():
            if edge.internal_points is not None:
                assert all(not np.shares_memory(edge.internal_points,u.internal_points) for u in unit.edges.values())
        next(iter(graph.nodes.values())).metadata["nested"]["values"].append(99)
        next(iter(graph.edges.values())).metadata["nested"]["values"].append(99)
        next(iter(graph.edges.values())).material.extra["nested"]["values"].append(99)
        next(iter(graph.nodes.values())).position[0] += .123
        next(iter(graph.edges.values())).internal_points[0,0] += .456
        graph.metadata["nested"]["values"].append(99)
    assert a.to_dict() == b.to_dict()
    assert ua.to_dict() == ub.to_dict() and ma.to_dict() == mb.to_dict()

@pytest.mark.parametrize("kind", ["empty", "connected", "small", "parallel"])
def test_general_helper_copy_semantics(kind):
    g = StructureGraph(dimension=2)
    if kind != "empty":
        for pos in [[0,0],[1,0],[2,0]]:
            g.add_node(pos, nested={"v": [1]})
        mat = CountingMaterial(extra={"v": [1]})
        g.add_edge(0,1,material=mat)
        if kind in ("connected", "parallel"):
            g.add_edge(1,2,material=mat)
        if kind == "parallel":
            g.add_edge(1,0,material=mat, allow_parallel=True)
    before = copy.deepcopy(g.to_dict())
    CountingMaterial.calls = 0
    out = pattern._bridge_small_components(g)
    assert out is not g and g.to_dict() == before
    assert CountingMaterial.calls == len(g.edges)
    assert len(out.edges) == len(g.edges) + (kind == "small")
    for eid, edge in g.edges.items():
        assert out.edges[eid].material is not edge.material
    if len(g.edges) > 1:
        assert out.edges[0].material is not out.edges[1].material
