"""Merge keeps imported metadata separate from structural constructor arguments."""
import copy

import numpy as np
import pytest

from fibernet import Material, StructureGraph


def rich_graph():
    material = Material(name='custom', youngs_modulus=1234.,
                        stiffness_tensor=np.eye(6), extra={'nested': [1]})
    return StructureGraph.from_dict({
        'version': '3.1', 'dimension': 3, 'tolerance': .002,
        'box_size': [4., 5., 6.], 'next_node_id': 100, 'next_edge_id': 1000,
        'metadata': {'units': {'length': 'mm'}, 'owner': 'left'},
        'nodes': [{'id': 70, 'position': [0., 0., 0.],
                   'boundary': [True, False, False, False, False, False]},
                  {'id': 10, 'position': [1., 0., 2.]}],
        'edges': [{'id': 90, 'node_i': 10, 'node_j': 70, 'radius': .2,
                   'segments': 7, 'material': material.to_dict(),
                   'internal_points': [[.5, .3, 1.]]},
                  {'id': 7, 'node_i': 70, 'node_j': 10, 'radius': .3,
                   'internal_points': []}]})


@pytest.mark.parametrize('kind,key', [
    *[('node', key) for key in ('position', 'boundary', 'merge')],
    *[('edge', key) for key in ('node_i', 'node_j', 'radius', 'material',
                               'segments', 'internal_points', 'allow_parallel', 'n_internal')],
])
def test_reserved_metadata_is_data_not_constructor_configuration(kind, key):
    source = rich_graph()
    if kind == 'node':
        source.nodes[70].metadata[key] = {'nested': [2]}
    else:
        source.edges[90].internal_points = None
        source.edges[90].metadata[key] = 2 if key == 'n_internal' else {'nested': [2]}
    before = copy.deepcopy(source.to_dict())
    merged = StructureGraph(dimension=3).merge(source)
    nodes = {node['id']: node for node in before['nodes']}
    edges = {edge['id']: edge for edge in before['edges']}
    for old, new in zip((nodes[nid] for nid in source.nodes), merged.to_dict()['nodes']):
        assert new['metadata'] == old['metadata']
    for old, new in zip((edges[eid] for eid in source.edges), merged.to_dict()['edges']):
        assert new['metadata'] == old['metadata']
        assert new['internal_points'] == old['internal_points']
    assert source.to_dict() == before


def test_native_roundtrip_merge_preserves_union_records_and_detaches_sources():
    left = rich_graph()
    right = rich_graph()
    right.metadata['owner'] = 'right'
    right.nodes[70].metadata = {'merge': False, 'boundary': 'label', 'nested': [2]}
    right.edges[90].metadata = {'radius': 'label', 'material': 'reference',
        'n_internal': 9, 'allow_parallel': False, 'nested': [3]}
    right.edges[7].material = Material(name='second', youngs_modulus=4567.)
    left_before, right_before = copy.deepcopy(left.to_dict()), copy.deepcopy(right.to_dict())
    right = StructureGraph.from_networkx_lossless(right.to_networkx_lossless())
    assert right.to_dict() == right_before
    merged = left.merge(right)
    record = merged.to_dict()
    assert record['nodes'][:2] == left_before['nodes']
    assert record['edges'][:2] == left_before['edges']
    assert record['metadata'] == left_before['metadata']
    assert record['box_size'] == left_before['box_size']
    assert record['dimension'] == 3 and record['tolerance'] == .002
    # Disjoint union, even at coincident positions; RHS IDs follow LHS counters.
    assert list(merged.nodes) == [70, 10, 100, 101]
    assert list(merged.edges) == [90, 7, 1000, 1001]
    assert (record['next_node_id'], record['next_edge_id']) == (102, 1002)
    mapping = {70: 100, 10: 101}
    nodes = {node['id']: node for node in right_before['nodes']}
    edges = {edge['id']: edge for edge in right_before['edges']}
    for index, old in enumerate(nodes[nid] for nid in right.nodes):
        expected = copy.deepcopy(old); expected['id'] = mapping[old['id']]
        assert record['nodes'][2 + index] == expected
    for index, old in enumerate(edges[eid] for eid in right.edges):
        expected = copy.deepcopy(old)
        expected.update(id=1000 + index, node_i=mapping[old['node_i']], node_j=mapping[old['node_j']])
        assert record['edges'][2 + index] == expected
    assert merged.edges[1001].internal_points.shape == (0, 3)
    assert StructureGraph.from_networkx_lossless(merged.to_networkx_lossless()).to_dict() == record
    merged.nodes[100].metadata['nested'].append(9)
    merged.nodes[100].position[0] = 9
    merged.edges[1000].metadata['nested'].append(9)
    merged.edges[1000].internal_points[0, 0] = 9
    merged.edges[1000].material.extra['nested'].append(9)
    merged.edges[1000].material.stiffness_tensor[0, 0] = 9
    merged.edges[90].material.extra['nested'].append(9)
    merged.metadata['units']['length'] = 'm'
    assert left.to_dict() == left_before and right.to_dict() == right_before


def test_generated_ordered_and_disordered_pair_roundtrip_merge():
    from fibernet.gen.pattern import pattern_2d
    left = pattern_2d(unit='square', grid=(1, 1), box=(2., 2.), n_internal=1)
    right = pattern_2d(unit='square', grid=(1, 1), box=(2., 2.),
                       perturbation=.1, seed=17, n_internal=2)
    right.nodes[next(iter(right.nodes))].metadata['merge'] = 'generator label'
    right.edges[next(iter(right.edges))].metadata['material'] = 'generator reference'
    records = [copy.deepcopy(graph.to_dict()) for graph in (left, right)]
    restored = [StructureGraph.from_networkx_lossless(graph.to_networkx_lossless())
                for graph in (left, right)]
    assert [graph.to_dict() for graph in restored] == records
    result = restored[0].merge(restored[1])
    assert result.num_nodes == left.num_nodes + right.num_nodes
    assert result.num_edges == left.num_edges + right.num_edges
    assert [graph.to_dict() for graph in restored] == records
    assert StructureGraph.from_networkx_lossless(result.to_networkx_lossless()).to_dict() == result.to_dict()


def test_representative_generated_pair_lossless_merge_and_topology():
    """Hundreds-scale graph composition only, not physics qualification."""
    from fibernet.gen.pattern import pattern_2d
    graphs = [pattern_2d(unit='square', grid=(10, 10), box=(20., 20.),
                         n_internal=internal) for internal in (1, 2)]
    # Square corner-only units ignore factory perturbation: use the existing
    # atomic position API to make the RHS genuinely disordered, reproducibly.
    rng = np.random.default_rng(17)
    right_positions = {nid: node.position + np.r_[rng.uniform(-.1, .1, 2), 0.]
                       for nid, node in list(graphs[1].nodes.items())[1:]}
    graphs[1].set_node_positions(right_positions)
    assert not np.array_equal(graphs[0].node_positions(), graphs[1].node_positions())
    for index, graph in enumerate(graphs):
        assert (graph.num_nodes, graph.num_edges) == (121, 220)
        assert len(graph.connected_components()) == 1
        graph.metadata['units'] = {'length': 'mm'}
        first_node = next(iter(graph.nodes))
        graph.nodes[first_node].metadata.update(merge='label', position='label', nested=[index])
        first_edge = next(iter(graph.edges.values()))
        first_edge.metadata.update(radius='label', material='reference', n_internal=99, nested=[index])
        endpoints = (first_edge.node_i, first_edge.node_j)
        midpoint = (graph.nodes[endpoints[0]].position + graph.nodes[endpoints[1]].position) / 2
        for member in range(3):
            a, b = endpoints if member % 2 == 0 else endpoints[::-1]
            material = Material(name=f'phase-{member}', youngs_modulus=1000. * (member + 1),
                                extra={'reference': f'material-{member}', 'nested': [index]})
            curve = (np.empty((0, 3)) if member == 2 else
                     np.array([midpoint + np.array([0., .05 * (member + 1), 0.])]))
            eid = graph.add_edge(a, b, radius=.1 + .02 * member, material=material,
                                 internal_points=curve, segments=member + 2, allow_parallel=True)
            graph.edges[eid].metadata.update(material=f'material-{member}',
                                             allow_parallel='label', nested=[index])
    records = [copy.deepcopy(graph.to_dict()) for graph in graphs]
    restored = [StructureGraph.from_networkx_lossless(graph.to_networkx_lossless()) for graph in graphs]
    assert [graph.to_dict() for graph in restored] == records
    left, right = restored
    result = left.merge(right)
    node_map = {nid: left._next_node_id + index for index, nid in enumerate(right.nodes)}
    edge_map = {eid: left._next_edge_id + index for index, eid in enumerate(right.edges)}
    expected = copy.deepcopy(records[0])
    for node in records[1]['nodes']:
        mapped = copy.deepcopy(node); mapped['id'] = node_map[node['id']]
        expected['nodes'].append(mapped)
    for edge in records[1]['edges']:
        mapped = copy.deepcopy(edge)
        mapped.update(id=edge_map[edge['id']], node_i=node_map[edge['node_i']],
                      node_j=node_map[edge['node_j']])
        expected['edges'].append(mapped)
    expected['nodes'].sort(key=lambda node: node['id'])
    expected['edges'].sort(key=lambda edge: edge['id'])
    expected['next_node_id'] += right.num_nodes
    expected['next_edge_id'] += right.num_edges
    # Whole-record comparison includes IDs, ordering, material references/properties,
    # unit metadata, boundary flags, straight/curved/empty discretizations and parallels.
    assert result.to_dict() == expected
    assert (result.num_nodes, result.num_edges) == (242, 446)
    assert list(result.nodes) == list(left.nodes) + list(node_map.values())
    assert list(result.edges) == list(left.edges) + list(edge_map.values())
    first_node = next(iter(right.nodes))
    np.testing.assert_array_equal(result.nodes[first_node].position,
                                  result.nodes[node_map[first_node]].position)
    assert first_node != node_map[first_node]
    assert result.num_nodes == sum(graph.num_nodes for graph in restored)
    assert result.num_edges == sum(graph.num_edges for graph in restored)
    assert set(result.nodes) == set(left.nodes) | set(node_map.values())
    assert set(left.nodes).isdisjoint(node_map.values())
    left_components = {frozenset(component) for component in left.connected_components()}
    right_components = {frozenset(node_map[nid] for nid in component)
                        for component in right.connected_components()}
    assert {frozenset(component) for component in result.connected_components()} == left_components | right_components
    assert len(result.connected_components()) == len(left_components) + len(right_components) == 2
    assert all((edge.node_i in left.nodes) == (edge.node_j in left.nodes) for edge in result.edges.values())
    assert StructureGraph.from_networkx_lossless(result.to_networkx_lossless()).to_dict() == expected
    assert [graph.to_dict() for graph in restored] == records
    assert [graph.to_dict() for graph in graphs] == records
    # Mutation checks in both directions, including generated and explicit curves.
    rhs_node, rhs_edge = next(iter(right.nodes)), next(iter(right.edges))
    result.nodes[node_map[rhs_node]].metadata['nested'].append(99)
    result.edges[edge_map[rhs_edge]].internal_points[0, 0] += .01
    last_rhs_edge = next(reversed(right.edges))
    result.edges[edge_map[last_rhs_edge]].material.extra['nested'].append(99)
    assert [graph.to_dict() for graph in restored] == records
    result_before_source_edit = copy.deepcopy(result.to_dict())
    right.nodes[rhs_node].position[0] += 1.
    right.edges[last_rhs_edge].material.extra['nested'].append(88)
    left.edges[next(iter(left.edges))].internal_points[0, 0] += .01
    assert result.to_dict() == result_before_source_edit
