"""Pure graph edit atomicity; legacy coordinate normalization is preserved."""
import copy


import numpy as np


import pytest


from fibernet import StructureGraph


def graph():
    g = StructureGraph(dimension=2)
    for position in ([99, 99], [0, 0], [88, 88], [2, 0]):
        g.add_node(position, merge=False)
    g.remove_node(0)
    g.remove_node(2)
    for radius in (.1, .2):
        g.add_edge(1, 3, radius=radius, allow_parallel=True,
                   internal_points=np.array([[1., .1, 0.]]))
    g.nodes[1].boundary = (True, False, False, False, False, False)
    g.metadata['units'] = {'length': 'mm'}
    return g


def snapshot(g):
    return copy.deepcopy((g.to_dict(), g._spatial_hash,
                          list(g.nodes), list(g.edges)))


def assert_unchanged(g, before):
    assert snapshot(g) == before
    # Public merge behavior exposes hash corruption that to_dict alone misses.
    for node_id, node in list(g.nodes.items()):
        assert g.add_node(node.position) == node_id
    assert snapshot(g) == before


@pytest.mark.parametrize('operation', ['add', 'set', 'displace'])
def test_unhashable_finite_position_preserves_graph(operation):
    g = graph()
    before = snapshot(g)
    value = [np.finfo(float).max, 0]
    with pytest.raises(ValueError, match='spatial hash coordinate range'):
        if operation == 'add':
            g.add_node(value, merge=False)
        elif operation == 'set':
            g.set_node_position(1, value)
        else:
            g.displace_node(1, value)
    assert_unchanged(g, before)


def test_displacement_arithmetic_overflow_preserves_graph():
    g = StructureGraph(tolerance=1.)
    g.add_node([np.finfo(float).max, 0], merge=False)
    before = snapshot(g)
    with pytest.raises(ValueError, match='finite position'):
        g.displace_node(0, [np.finfo(float).max, 0])
    assert snapshot(g) == before


@pytest.mark.parametrize('operation', ['set', 'displace'])
def test_unknown_node_preserves_graph(operation):
    g = graph()
    before = snapshot(g)
    with pytest.raises(KeyError, match='Node 999 not found'):
        getattr(g, 'set_node_position' if operation == 'set' else 'displace_node')(999, [1, 2])
    assert_unchanged(g, before)


@pytest.mark.parametrize('bad', [{999: [0, 0]}, {3: [0]}, {3: [np.nan, 0]},
                                {3: [np.finfo(float).max, 0]}])
def test_late_batch_failure_is_atomic(bad):
    g = graph()
    before = snapshot(g)
    with pytest.raises((ValueError, KeyError)):
        g.set_node_positions({1: [10, 20], **bad})
    assert_unchanged(g, before)


def test_success_preserves_identity_metadata_curves_and_roundtrip(tmp_path):
    g = graph()
    before = g.to_dict()
    positions = {1: np.array([2., 0.]), 3: np.array([0., 0.])}
    g.set_node_positions(positions)  # Swap hash buckets without merging nodes.
    positions[1][:] = 999  # Inputs are not retained as mutable aliases.
    assert g.add_node([2, 0]) == 1
    assert g.add_node([0, 0]) == 3
    g.displace_node(3, [.25, .5])
    np.testing.assert_array_equal(g.nodes[3].position, [.25, .5, 0.])
    g.set_node_position(1, (3, 4, 5))
    np.testing.assert_array_equal(g.nodes[1].position, [3, 4, 5])
    after = g.to_dict()
    assert before['edges'] == after['edges']
    assert before['metadata'] == after['metadata']
    assert [n['id'] for n in before['nodes']] == [n['id'] for n in after['nodes']]
    assert [n['boundary'] for n in before['nodes']] == [n['boundary'] for n in after['nodes']]
    assert before['next_node_id'] == after['next_node_id']
    assert before['next_edge_id'] == after['next_edge_id']
    path = tmp_path / 'graph.json'
    g.save_json(path)
    restored = StructureGraph.load_json(path)
    assert restored.to_dict() == after
    assert restored.add_node([3, 4, 5]) == 1
    assert restored.add_node([9, 8]) == 4


def test_rejected_edit_keeps_prepared_id_controls_usable():
    from fibernet.sim import prepare_nodal_controls
    g = graph()
    controls = prepare_nodal_controls(g, fixed_node_ids=[1],
                                      forces_by_id={3: [1., 0., 0.]})
    with pytest.raises(ValueError):
        g.set_node_positions({1: [10, 20], 3: [np.nan, 0]})
    assert controls.taichi_kwargs(g)['fixed_nodes'] == [0]
    g.displace_node(3, [.1, 0])
    with pytest.raises(ValueError, match='graph changed'):
        controls.taichi_kwargs(g)


def test_add_and_set_valid_vector_forms_are_detached():
    for value in ([1, 2], (1., 2., 3.), np.array([1., 2.]),
                  np.array([1., 2., 3.], dtype=np.float32)):
        g = StructureGraph()
        nid = g.add_node(value)
        g.set_node_position(nid, value)
        expected = [1., 2., 0.] if len(value) == 2 else [1., 2., 3.]
        np.testing.assert_array_equal(g.nodes[nid].position, expected)
        if isinstance(value, np.ndarray):
            value[:] = 99
            np.testing.assert_array_equal(g.nodes[nid].position, expected)


@pytest.mark.parametrize('value', [[[1, 2]], [[1], [2]], [[1, 2, 3]], [[1], [2], [3]]])
def test_add_preserves_legacy_flattened_coordinate_shapes(value):
    g = StructureGraph()
    array = np.array(value, dtype=float)
    nid = g.add_node(array)
    expected = np.r_[array.ravel(), 0.] if array.size == 2 else array.ravel().copy()
    np.testing.assert_array_equal(g.nodes[nid].position, expected)
    array[:] = 99
    np.testing.assert_array_equal(g.nodes[nid].position, expected)



def test_failed_displacement_preserves_original_merge_id():
    g = graph()
    before = snapshot(g)
    with pytest.raises(ValueError):
        g.displace_node(1, [1, 2, 3, 4])
    assert_unchanged(g, before)


def test_failed_nonfinite_add_does_not_consume_id():
    g = graph()
    before = snapshot(g)
    with pytest.raises(ValueError):
        g.add_node([np.nan, 0], merge=False)
    assert_unchanged(g, before)


def test_legacy_one_component_displacement_still_broadcasts():
    g = graph()
    g.displace_node(1, [.25])
    np.testing.assert_array_equal(g.nodes[1].position, [.25, .25, .25])
