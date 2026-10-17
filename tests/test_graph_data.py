"""Pure-data fixtures, not physical simulations or learning benchmarks."""
import math
import os
import struct
import subprocess
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from fibernet.core.structure_graph import StructureGraph, SNode, SEdge
from fibernet.ml.graph_data import graph_batch


def fixture(dimension=3):
    g = StructureGraph(dimension=dimension)
    g.nodes[20] = SNode(np.array([1., 0., 2.]), node_id=20,
                        metadata={'unit_id': 'unit-A', 'tags': ['end']})
    g.nodes[4] = SNode(np.array([1., 0., 0.]), node_id=4)
    g.edges[9] = SEdge(20, 4, edge_id=9, radius=.2,
                      internal_points=np.array([[2., 0., 1.]]),
                      metadata={'material_id': 'm1'})
    g.edges[2] = SEdge(4, 20, edge_id=2, radius=.3)
    g.metadata['origin'] = {'specimen': ['s1']}
    return g


def batch(*graphs, **kw):
    return graph_batch(graphs, graph_ids=[str(i) for i in range(len(graphs))],
                       length_unit='mm', coordinate_frame='specimen', **kw)


def test_xyz_sparse_parallel_orientation_and_lengths():
    g = fixture()
    b = batch(g, edge_features=['radius', 'chord_length', 'polyline_length'])
    assert b.schema_version == 'fibernet.graph_batch.v1'
    assert b.node_ids.tolist() == [4, 20]
    assert b.edge_ids.tolist() == [9, 2]
    assert b.edge_index.tolist() == [[1, 0], [0, 1]]
    assert b.message_index.tolist() == [[1, 0, 0, 1], [0, 1, 1, 0]]
    assert b.message_parent_rows.tolist() == [0, 0, 1, 1]
    assert b.message_parent_ids.tolist() == [9, 9, 2, 2]
    assert b.message_orientation.tolist() == [1, -1, 1, -1]
    np.testing.assert_allclose(b.edge_features, [[.2, 2, 2*np.sqrt(2)], [.3, 2, 2]])
    assert b.internal_ptr.tolist() == [0, 1, 1]
    assert b.internal_available.tolist() == [True, False]
    assert b.edge_feature_available.all()
    assert list(g.nodes) == [20, 4]


def test_2d_and_insertion_order():
    g = fixture(2)
    for node in g.nodes.values():
        node.position[2] = 0
    g.nodes[20].position[0] = 2
    g.edges[9].internal_points[:] = [[1.5, .5, 0]]
    a = batch(g)
    g._nodes = dict(reversed(list(g.nodes.items())))
    b = batch(g)
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.edge_index, b.edge_index)
    assert b.dimensions.tolist() == [2]
    assert b.positions.shape == (2, 3)


def test_multiple_graphs_local_ids_and_empty():
    b = batch(fixture(), StructureGraph(2), fixture())
    assert b.node_ptr.tolist() == [0, 2, 2, 4]
    assert b.edge_ptr.tolist() == [0, 2, 2, 4]
    assert b.node_ids.tolist() == [4, 20, 4, 20]
    assert b.edge_index.tolist() == [[1, 0, 3, 2], [0, 1, 2, 3]]
    assert b.message_parent_rows.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert b.graph_ids == ('0', '1', '2')
    assert b.edge_features.shape == (4, 0)
    for b in (batch(), batch(StructureGraph())):
        assert b.positions.shape == (0, 3)
        assert b.edge_index.shape == b.message_index.shape == (2, 0)
        assert b.internal_points.shape == (0, 3)
        assert b.internal_ptr.tolist() == [0]


def test_detached_and_normal_write_protection():
    g = fixture()
    b = batch(g)
    old = b.positions.copy()
    g.nodes[20].position[:] = 8
    g.edges[9].internal_points[:] = 9
    g.metadata['origin']['specimen'].append('s2')
    g.nodes[20].metadata['tags'].append('changed')
    g.edges[9].metadata['material_id'] = 'm2'
    np.testing.assert_equal(b.positions, old)
    np.testing.assert_equal(b.internal_points, [[2, 0, 1]])
    assert b.graph_metadata[0]['origin']['specimen'] == ('s1',)
    assert b.node_metadata[1]['tags'] == ('end',)
    assert b.edge_metadata[0]['material_id'] == 'm1'
    for value in vars(b).values():
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
    with pytest.raises(ValueError):
        b.positions[0, 0] = 3
    with pytest.raises(TypeError):
        b.graph_metadata[0]['other'] = 1
    with pytest.raises(FrozenInstanceError):
        b.length_unit = 'm'
    # Write protection is not a security boundary; even deliberate writes do not alias.
    b.positions.setflags(write=True)
    b.positions[:] = -10
    assert g.nodes[20].position.tolist() == [8, 8, 8]


def test_availability_no_material_or_bc_inference():
    g = fixture()
    g.edges[9].radius = None
    b = batch(g, edge_features=['radius'])
    assert b.edge_features[:, 0].tolist() == [0, .3]
    assert b.edge_feature_available[:, 0].tolist() == [False, True]
    assert not hasattr(b, 'materials') and not hasattr(b, 'targets')
    assert not hasattr(b, 'boundary_conditions')
    with pytest.raises(ValueError, match='unknown'):
        batch(g, edge_features=['youngs_modulus'])


@pytest.mark.parametrize('names', ['radius', ['radius', 'radius'], ['xyz'], [None]])
def test_invalid_features(names):
    with pytest.raises(ValueError):
        batch(fixture(), edge_features=names)


@pytest.mark.parametrize('where', ['node', 'internal', 'radius'])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite(where, bad):
    g = fixture()
    if where == 'node':
        g.nodes[4].position[0] = bad
    elif where == 'internal':
        g.edges[9].internal_points[0, 0] = bad
    else:
        g.edges[9].radius = bad
    with pytest.raises(ValueError, match='finite'):
        batch(g, edge_features=['radius'])


@pytest.mark.parametrize('bad', [-1, True, 1.5, 2**63])
def test_invalid_ids(bad):
    g = fixture()
    g.nodes[bad] = g.nodes.pop(4)
    with pytest.raises(ValueError):
        batch(g)


def test_bad_identity_shapes_and_metadata():
    g = fixture()
    g.edges[9].node_j = 999
    with pytest.raises(ValueError, match='missing node'):
        batch(g)
    g = fixture()
    g.nodes[4].node_id = 8
    with pytest.raises(ValueError, match='disagree'):
        batch(g)
    g = fixture()
    g.edges[9].edge_id = 8
    with pytest.raises(ValueError, match='disagree'):
        batch(g)
    g = fixture()
    g.nodes[4].position = [0, 1]
    with pytest.raises(ValueError):
        batch(g)
    g = fixture()
    g.edges[9].internal_points = [[1, 2]]
    with pytest.raises(ValueError, match='shape'):
        batch(g)
    for value in (object(), np.array([1]), float('nan')):
        g = fixture()
        g.metadata['bad'] = value
        with pytest.raises(ValueError):
            batch(g)
    g = fixture()
    g.metadata['cycle'] = g.metadata
    with pytest.raises(ValueError, match='cycle'):
        batch(g)


@pytest.mark.parametrize('kwargs', [dict(length_unit=''), dict(coordinate_frame=' '),
                                    dict(graph_ids=['a', 'a']), dict(graph_ids=[])])
def test_caller_declarations(kwargs):
    args = dict(graph_ids=['a', 'b'], length_unit='mm', coordinate_frame='lab')
    args.update(kwargs)
    with pytest.raises(ValueError):
        graph_batch([fixture(), fixture()], **args)


def test_lazy_exports_without_learning_backends():
    script = '''
import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_geometric', 'stable_baselines3', 'xgboost'}:
            raise AssertionError('unexpected optional backend import: ' + fullname)
sys.meta_path.insert(0, Block())
from fibernet.ml import graph_batch, GraphBatch
from fibernet.core.structure_graph import StructureGraph
view = graph_batch([StructureGraph()], graph_ids=['a'], length_unit='m', coordinate_frame='lab')
assert isinstance(view, GraphBatch)
assert not any(x in sys.modules for x in ['torch', 'torch_geometric', 'stable_baselines3', 'xgboost'])
'''
    subprocess.run([sys.executable, '-c', script], check=True, timeout=30,
                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))


@pytest.mark.parametrize('where', ['node', 'internal'])
@pytest.mark.parametrize('masked', [True, False])
def test_masked_geometry_rejected_before_conversion(where, masked):
    g = fixture()
    if where == 'node':
        g.nodes[4].position = np.ma.array([7., 0., 0.], mask=[masked, False, False])
    else:
        g.edges[9].internal_points = np.ma.array([[7., 0., 0.]],
                                                mask=[[masked, False, False]])
    with pytest.raises(ValueError, match='MaskedArray'):
        batch(g)


def test_radius_must_remain_finite_in_float64():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip('longdouble has no wider finite range on this platform')
    g = fixture()
    g.edges[9].radius = np.longdouble(np.finfo(np.float64).max) * 2
    assert np.isfinite(g.edges[9].radius)
    with pytest.raises(ValueError, match='finite in float64'):
        batch(g, edge_features=['radius'])


@pytest.mark.parametrize('scale', [1e-200, 1e200, struct.unpack('>d', bytes.fromhex('0000000000000001'))[0]])
def test_distance_preserves_representable_tiny_and_large_lengths(scale):
    # Change coordinates after construction: no generator tolerance/merging involved.
    g = StructureGraph(3)
    a, b = g.add_node([0, 0, 0]), g.add_node([1, 0, 0])
    g.add_edge(a, b)
    g.nodes[b].position[:] = [scale, 0, 0]
    packed = struct.pack('>d', scale)
    assert struct.pack('>d', g.nodes[b].position[0]) == packed
    # Probe these exact platform operations, not float equality (DAZ can hide loss).
    distance = math.dist(g.nodes[a].position, g.nodes[b].position)
    total = math.fsum([distance])
    if any(struct.pack('>d', x) in (bytes(8), bytes.fromhex('8000000000000000'))
           for x in (distance, total)):
        with pytest.raises(ValueError, match='lost a nonzero'):
            batch(g, edge_features=['chord_length', 'polyline_length'])
    else:
        v = batch(g, edge_features=['chord_length', 'polyline_length'])
        assert v.edge_feature_available.all()
        assert [struct.pack('>d', x) for x in v.edge_features[0]] == [packed, packed]
    assert struct.pack('>d', g.nodes[b].position[0]) == packed


@pytest.mark.parametrize('scale', [1e-200, 1e200])
def test_distance_uses_stable_xyz_and_polyline_sum(scale):
    g = StructureGraph(3)
    a, b = g.add_node([0, 0, 0]), g.add_node([1, 0, 0])
    eid = g.add_edge(a, b)
    g.nodes[b].position[:] = [3 * scale, 0, 4 * scale]
    g.edges[eid].internal_points = np.array([[0, 0, 4 * scale]])
    v = batch(g, edge_features=['chord_length', 'polyline_length'])
    np.testing.assert_allclose(v.edge_features / scale, [[5, 7]], rtol=1e-15, atol=0)


@pytest.mark.parametrize('feature', ['chord_length', 'polyline_length'])
def test_unrepresentable_endpoint_distance_rejected(feature):
    g = fixture()
    g.nodes[4].position[:] = [-np.finfo(float).max, 0, 0]
    g.nodes[20].position[:] = [np.finfo(float).max, 0, 0]
    g.edges[9].internal_points = None
    with pytest.raises(ValueError, match='finite'):
        batch(g, edge_features=[feature])


def test_polyline_sum_overflow_rejected_when_segments_are_finite():
    g = fixture()
    g.nodes[4].position[:] = [0, 0, 0]
    g.nodes[20].position[:] = [0, 0, 0]
    g.edges[9].internal_points = np.array([[1e308, 0, 0]])
    with pytest.raises(ValueError, match='finite float64 range'):
        batch(g, edge_features=['polyline_length'])


@pytest.mark.parametrize('where', ['node', 'internal', 'radius'])
def test_nonzero_source_must_not_underflow_during_float64_conversion(where):
    small = np.longdouble(np.nextafter(0., 1.)) / 4
    if small == 0:
        pytest.skip('longdouble has no wider subnormal range on this platform')
    assert float(small) == 0
    g = fixture()
    if where == 'node':
        g.nodes[4].position = np.array([small, 0, 0], dtype=np.longdouble)
    elif where == 'internal':
        g.edges[9].internal_points = np.array([[small, 0, 0]], dtype=np.longdouble)
    else:
        g.edges[9].radius = small
    with pytest.raises(ValueError, match='underflow to zero'):
        batch(g, edge_features=['radius'])
