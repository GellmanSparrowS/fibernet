"""Proposed bounded acceptance only. Not executed in the contract phase."""
import copy
import itertools
import json

import numpy as np
import pytest

from fibernet.core.material import Material
from fibernet.core.structure_graph import StructureGraph
from fibernet.gen import standard_lattice as subject


def _oracle(grid):
    # Independent induced integer lattice: no production tables or helpers.
    nodes = {q for q in itertools.product(*(range(2 * n + 1) for n in grid))
             if sum(q) % 2 == 0}
    edges = {tuple(sorted((u, v))) for u, v in itertools.combinations(nodes, 2)
             if sum((x - y) ** 2 for x, y in zip(u, v)) == 2}
    return nodes, edges


def _generate(grid=(1, 1, 1), a=1e-6, radius=1e-8, material=None):
    return subject.octet_fcc_v1(a=a, grid=grid, radius=radius,
                                material=material if material is not None else Material())


@pytest.mark.parametrize('grid,counts', [
    ((1, 1, 1), (14, 36)), ((2, 1, 1), (23, 68)),
    ((2, 2, 2), (63, 240)), ((2, 3, 1), (53, 188)),
    ((3, 3, 3), (172, 756)),
])
def test_exact_finite_geometry(grid, counts):
    a, radius = 1e-6, 7e-9
    graph = _generate(grid, a, radius)
    expected_nodes, expected_edges = _oracle(grid)
    labels = {nid: tuple(n.metadata['lattice_index']) for nid, n in graph.nodes.items()}
    pairs = [tuple(sorted((labels[e.node_i], labels[e.node_j]))) for e in graph.edges.values()]
    assert (graph.num_nodes, graph.num_edges) == counts
    assert set(labels.values()) == expected_nodes
    assert len(pairs) == len(set(pairs))
    assert set(pairs) == expected_edges
    degrees = dict.fromkeys(labels, 0)
    for node_id, node in graph.nodes.items():
        q = labels[node_id]
        np.testing.assert_allclose(node.position, np.array(q) * a / 2, rtol=1e-14, atol=0)
        assert node.boundary == tuple(flag for x, n in zip(q, grid) for flag in (x == 0, x == 2*n))
        assert node.metadata['canonical_node_key'] == 'fcc:' + ','.join(map(str, q))
    for edge in graph.edges.values():
        lo, hi = sorted((labels[edge.node_i], labels[edge.node_j]))
        assert edge.metadata['physical_edge_key'] == ['octet_fcc_v1', list(lo), list(hi)]
        assert edge.radius == radius
        assert edge.internal_points is None
        assert edge.segments == 4
        degrees[edge.node_i] += 1
        degrees[edge.node_j] += 1
    for nid, q in labels.items():
        if all(0 < x < 2*n for x, n in zip(q, grid)):
            assert degrees[nid] == 12
    assert graph.box_size.tolist() == [a*n for n in grid]
    assert graph.metadata['length_unit'] == 'm'


@pytest.mark.parametrize('a', [1., 1e-3, 1e-6, 1e-9])
def test_scale_native_ids_and_lookup(a):
    grid = (2, 1, 1)
    requested = Material(name='scale-invariant', youngs_modulus=2e8,
                         extra={'unit_policy': 'SI unchanged'})
    graph = _generate(grid, a=a, radius=3e-10, material=requested)
    assert graph.tolerance == a * 1e-6
    expected_nodes, expected_edges = _oracle(grid)
    labels = {nid: tuple(n.metadata['lattice_index']) for nid, n in graph.nodes.items()}
    assert set(labels.values()) == expected_nodes
    pairs = [tuple(sorted((labels[e.node_i], labels[e.node_j]))) for e in graph.edges.values()]
    assert len(pairs) == len(set(pairs)) == len(expected_edges)
    assert set(pairs) == expected_edges
    for nid, node in graph.nodes.items():
        q = labels[nid]
        np.testing.assert_allclose(node.position, np.array(q)*a/2, rtol=1e-14, atol=0)
        assert node.metadata['canonical_node_key'] == 'fcc:' + ','.join(map(str, q))
    for edge in graph.edges.values():
        lo, hi = sorted((labels[edge.node_i], labels[edge.node_j]))
        assert edge.metadata['physical_edge_key'] == ['octet_fcc_v1', list(lo), list(hi)]
        assert edge.radius == 3e-10
        assert edge.material.to_dict() == requested.to_dict()
    extent = [a*n for n in grid]
    assert graph.box_size.tolist() == extent
    assert graph.metadata['nominal_extent_m'] == extent
    assert graph.metadata['geometric_bounds_m'] == [[0., 0., 0.], extent]
    assert graph.metadata['construction_normalized']['tiling']['unit_box'] == [1., 1., 1.]
    record = graph.to_dict()
    restored = StructureGraph.from_dict(json.loads(json.dumps(record, allow_nan=False)))
    assert restored.to_dict() == record
    assert all(edge.radius == 3e-10 for edge in restored.edges.values())
    origin = next(nid for nid, node in restored.nodes.items()
                  if node.metadata['lattice_index'] == [0, 0, 0])
    assert restored.add_node([0, 0, 0], merge=True) == origin
    assert restored.add_node([restored.tolerance / 8, 0, 0], merge=True) == origin
    fresh = restored.add_node([-a, -a, -a], merge=True)
    assert fresh == record['next_node_id']
    restored.set_node_position(fresh, [-2*a, -2*a, -2*a])
    assert restored.add_node([-2*a, -2*a, -2*a], merge=True) == fresh
    assert restored.add_node([-a, -a, -a], merge=True) != fresh
    assert graph.to_dict() == record


@pytest.mark.parametrize('subclass', [False, True])
def test_deep_material_shared_edge_collapse_and_isolation(subclass):
    class NativeSubclass(Material):
        pass
    material_type = NativeSubclass if subclass else Material
    nested = {'leaf': ['value', None, True]}
    for _ in range(40):
        nested = {'next': nested}
    material = material_type(name='requested', youngs_modulus=7e8,
                             thermal_conductivity=2., stiffness_tensor=np.eye(6),
                             extra={'deep': nested, 'editable': ['original']})
    before = copy.deepcopy(material.to_dict())
    graph = _generate((2, 1, 1), material=material)
    assert graph.num_edges == 68
    for edge in graph.edges.values():
        assert edge.material.to_dict() == before
        assert edge.material is not material
    restored = StructureGraph.from_dict(graph.to_dict())
    assert restored.to_dict() == graph.to_dict()
    edges = list(graph.edges.values())
    edges[0].material.extra['editable'][0] = 'changed'
    assert material.to_dict() == before
    assert edges[1].material.extra['editable'] == ['original']


@pytest.mark.parametrize('changes', [
    {'a': 0}, {'a': -1}, {'a': float('nan')}, {'a': float('inf')}, {'a': True},
    {'a': 1e-320}, {'a': 1e-160}, {'a': 1e200},
    {'radius': 0}, {'radius': False}, {'radius': float('inf')},
    {'grid': (0, 1, 1)}, {'grid': (1., 1, 1)}, {'grid': (True, 1, 1)},
    {'grid': [1, 1, 1]}, {'grid': (1000000, 1, 1)},
    {'material': None}, {'seed': 0}, {'n_internal': 1},
    {'box': (1, 2, 1)}, {'point_displacements': []},
])
def test_invalid_requests_never_call_factory(monkeypatch, changes):
    def forbidden(**kwargs):
        raise AssertionError('factory must not be entered')
    monkeypatch.setattr(subject, '_unit_fcc_3d', forbidden)
    args = dict(a=1e-6, grid=(1, 1, 1), radius=1e-8, material=Material())
    args.update(changes)
    with pytest.raises((ValueError, TypeError)):
        subject.octet_fcc_v1(**args)


def test_payload_work_and_occupancy_preflight(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError('factory must not be entered')
    monkeypatch.setattr(subject, '_unit_fcc_3d', forbidden)
    with pytest.raises(ValueError, match='payload'):
        _generate(material=Material(extra={'large': 'x' * 20000}))
    with pytest.raises(ValueError, match='peak-work'):
        _generate((10, 10, 10), material=Material(extra={'large': 'x' * 1000}))
    with pytest.raises(ValueError, match='spatial-bin'):
        _generate((3, 3, 3), a=1e-13)


def test_no_registration_or_legacy_changes():
    from fibernet.gen.pattern import list_units_3d, _UNIT_FACTORIES_3D
    before = list_units_3d()
    factories = dict(_UNIT_FACTORIES_3D)
    _generate()
    assert list_units_3d() == before
    assert _UNIT_FACTORIES_3D == factories
    assert 'octet_fcc_v1' not in before
