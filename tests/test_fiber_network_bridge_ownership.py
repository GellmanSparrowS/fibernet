"""Detached legacy bridge ownership, not curved manufacturing or solver tests.

Reuse the native interchange and merge fixtures instead of another graph model.
The existing endpoint-chord, dense-ID and small-chord omission rules are retained.
"""
import copy
import inspect
from itertools import combinations

import numpy as np
import pytest

from fibernet import Material, StructureGraph
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.io.mesh_export import export_obj
from test_merge_right_ownership import fixture
from test_networkx_lossless import _graph
from test_obj_export_indices import _parse_obj


def scene():
    """Eight nodes, six members, three curves, two parallel pairs, one omit."""
    _, right = fixture()
    right.metadata['units'] = 'mm'
    graph = _graph().merge(right)
    # This fixture establishes common raw geometry units before composition.
    graph.metadata.update(coordinate_frame='lab_xyz', nested={'values': [1]})
    a = graph.add_node([5., 0., 0.], merge=False)
    b = graph.add_node([5. + 5e-13, 0., 0.], merge=False)
    graph.add_edge(a, b, radius=.4, allow_parallel=True)
    assert list(graph.edges) == [90, 7, 50, 1000, 1001, 1002]
    return graph


def graph_state(graph):
    return copy.deepcopy(graph.to_dict())


def network_state(network):
    return copy.deepcopy(network.to_dict())


@pytest.mark.parametrize('direction', ['source_to_output', 'output_to_source'])
@pytest.mark.parametrize('field', ['modulus', 'nested', 'tensor'])
def test_material_mutations_are_detached(direction, field):
    graph = scene()
    network = graph.to_fiber_network()
    source = graph.edges[1000].material
    derived = network.fibers[3].material
    target = network if direction == 'source_to_output' else graph
    state = network_state if direction == 'source_to_output' else graph_state
    before = state(target)
    changed = source if direction == 'source_to_output' else derived
    if field == 'modulus':
        changed.youngs_modulus += 1
    elif field == 'nested':
        changed.extra['v'].append(999)
    else:
        changed.stiffness_tensor[0, 0] += 1
    assert state(target) == before
    assert source is not derived


@pytest.mark.parametrize('direction', ['source_to_output', 'output_to_source'])
def test_box_mutations_are_detached(direction):
    graph = scene()
    network = graph.to_fiber_network()
    before = network.box_size.copy() if direction == 'source_to_output' else graph.box_size
    # graph.box_size deliberately returns a copy; mutate real storage here.
    changed = graph._box_size if direction == 'source_to_output' else network.box_size
    changed[0] += 1
    target = network.box_size if direction == 'source_to_output' else graph.box_size
    np.testing.assert_array_equal(target, before)
    assert not np.shares_memory(graph._box_size, network.box_size)


@pytest.mark.parametrize('direction', ['source_to_output', 'output_to_source'])
def test_shared_source_material_does_not_link_sibling_fibers(direction):
    graph = scene()
    shared = graph.edges[1000].material
    graph.edges[1001].material = shared
    network = graph.to_fiber_network()
    a, b = network.fibers[3].material, network.fibers[4].material
    before = copy.deepcopy(shared.to_dict())
    if direction == 'source_to_output':
        shared.extra['v'].append(88)
        assert a.to_dict() == before == b.to_dict()
    else:
        a.extra['v'].append(88)
        assert shared.to_dict() == before == b.to_dict()
    assert a is not b and a is not shared and b is not shared


def test_separate_conversions_do_not_share_material_or_box():
    graph = scene()
    first, second = graph.to_fiber_network(), graph.to_fiber_network()
    source_before, second_before = graph_state(graph), network_state(second)
    first.box_size[0] += 1
    first.fibers[3].material.extra['v'].append(99)
    assert graph_state(graph) == source_before
    assert network_state(second) == second_before


@pytest.mark.parametrize('direction', ['source_to_output', 'output_to_source'])
def test_existing_coordinate_and_metadata_detachment(direction):
    graph = scene()
    network = graph.to_fiber_network()
    source_before, derived_before = graph_state(graph), network_state(network)
    if direction == 'source_to_output':
        graph.nodes[10].position[0] += 10
        graph.edges[90].internal_points[0, 0] += 10
        graph.metadata['nested']['values'].append(9)
        assert network_state(network) == derived_before
    else:
        network.fibers[0].centerline[0, 0] += 10
        network.crosslinks[0].position[0] += 10
        network.metadata['nested']['values'].append(9)
        assert graph_state(graph) == source_before


def test_legacy_geometry_ids_omission_radii_metadata_and_crosslinks():
    graph = scene()
    before = graph_state(graph)
    network = graph.to_fiber_network()
    assert list(inspect.signature(StructureGraph.to_fiber_network).parameters) == ['self']
    assert (graph.num_nodes, graph.num_edges, network.num_fibers) == (8, 6, 5)
    assert sum(e.internal_points is not None and len(e.internal_points) > 0
               for e in graph.edges.values()) == 3
    assert [f.fiber_id for f in network.fibers] == list(range(5))
    assert network.dimension == graph.dimension == 3
    assert network.metadata == graph.metadata
    np.testing.assert_array_equal(network.box_size, graph.box_size)
    retained = list(graph.edges.values())[:5]
    for fiber, edge in zip(network.fibers, retained):
        a, b = graph.nodes[edge.node_i].position, graph.nodes[edge.node_j].position
        segments = max(edge.segments, max(4, int(np.linalg.norm(b-a) / max(edge.radius*2, .05))))
        np.testing.assert_array_equal(fiber.centerline, np.linspace(a, b, segments+1))
        assert (fiber.radius, fiber.segments) == (edge.radius, segments)
        assert fiber.material.to_dict() == edge.material.to_dict()
    expected = []
    for nid in graph.nodes:
        endpoints = [(i, value) for i, e in enumerate(retained)
                     for endpoint, value in ((e.node_i, 0.), (e.node_j, 1.)) if endpoint == nid]
        for (a, ta), (b, tb) in combinations(endpoints, 2):
            expected.append((a, b, ta, tb, graph.nodes[nid].position.tolist(), 'welded'))
    actual = [(c.fiber_i, c.fiber_j, c.param_i, c.param_j, c.position.tolist(), c.crosslink_type)
              for c in network.crosslinks]
    assert actual == expected
    assert graph_state(graph) == before
    # The authoritative source, not the projected network, preserves the curves.
    assert StructureGraph.from_dict(before).to_dict() == before


def test_obj_bytes_match_existing_independent_chord_construction(tmp_path):
    graph = scene()
    before = graph_state(graph)
    network = graph.to_fiber_network()
    # OBJ consumes only endpoints and radii; this reference bypasses the bridge.
    expected = FiberNetwork(dimension=3, fibers=[
        Fiber(np.array([graph.nodes[e.node_i].position, graph.nodes[e.node_j].position]),
              radius=e.radius, fiber_id=i)
        for i, e in enumerate(list(graph.edges.values())[:5])])
    actual_path, expected_path = tmp_path/'actual.obj', tmp_path/'expected.obj'
    export_obj(network, str(actual_path))
    export_obj(expected, str(expected_path))
    assert actual_path.read_bytes() == expected_path.read_bytes()
    vertices, faces = _parse_obj(actual_path)
    assert vertices.shape == (5*18, 3) and faces.shape == (5*32, 3)
    assert faces.min() == 1 and faces.max() == len(vertices)
    assert graph_state(graph) == before


def test_none_box_and_material_keep_default_behavior():
    graph = StructureGraph()
    a, b = graph.add_node([0., 0.]), graph.add_node([1., 0.])
    eid = graph.add_edge(a, b)
    graph.edges[eid].material = None
    network = graph.to_fiber_network()
    assert network.box_size is None
    assert network.fibers[0].material.to_dict() == Material().to_dict()
    assert graph.edges[eid].material is None


def test_per_edge_custom_deepcopy_and_skipped_member_behavior():
    calls = []
    class CustomMaterial(Material):
        def __deepcopy__(self, memo):
            calls.append(self.name)
            return CustomMaterial(name=self.name, extra=copy.deepcopy(self.extra, memo))
    class UncopyableMaterial(Material):
        def __deepcopy__(self, memo):
            raise AssertionError('skipped member must not copy material')
    graph = scene()
    shared = CustomMaterial(name='shared', extra={'nested': [1]})
    graph.edges[1000].material = graph.edges[1001].material = shared
    graph.edges[1002].material = UncopyableMaterial()
    network = graph.to_fiber_network()
    assert calls == ['shared', 'shared']
    a, b = network.fibers[3].material, network.fibers[4].material
    assert isinstance(a, CustomMaterial) and isinstance(b, CustomMaterial)
    assert a is not b and a is not shared and b is not shared
    a.extra['nested'].append(2)
    assert shared.extra == b.extra == {'nested': [1]}
