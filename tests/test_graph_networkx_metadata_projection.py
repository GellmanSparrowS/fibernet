import copy

import networkx as nx

from fibernet import StructureGraph


def test_projection_preserves_reserved_metadata_without_overriding_structure():
    graph = StructureGraph()
    a = graph.add_node((0, 0))
    b = graph.add_node((1, 0))
    e = graph.add_edge(a, b, radius=0.2, segments=7)
    graph.nodes[a].metadata.update({'pos': 'label', 'boundary': 'label',
                                    '_fibernet_metadata': 'original', 'tag': 'node'})
    graph.edges[e].metadata.update({'radius': 'label', 'edge_id': 'label',
                                    'segments': 'label', '_fibernet_metadata': 'edge',
                                    'tag': 'edge'})
    before = copy.deepcopy(graph.to_dict())
    result = graph.to_networkx()
    assert result.nodes[a]['pos'] == (0., 0.)
    assert result.nodes[a]['boundary'] == graph.nodes[a].boundary
    assert result.nodes[a]['_fibernet_metadata'] == graph.nodes[a].metadata
    assert result.nodes[a]['tag'] == 'node'
    assert result.edges[a, b]['radius'] == 0.2
    assert result.edges[a, b]['edge_id'] == e
    assert result.edges[a, b]['segments'] == 7
    assert result.edges[a, b]['_fibernet_metadata'] == graph.edges[e].metadata
    assert result.edges[a, b]['tag'] == 'edge'
    assert graph.to_dict() == before


def test_projection_retains_parallel_edges_and_signature_named_metadata():
    graph = StructureGraph()
    a, b = graph.add_node((0, 0)), graph.add_node((1, 0))
    first = graph.add_edge(a, b)
    second = graph.add_edge(a, b, allow_parallel=True)
    graph.nodes[a].metadata['node_for_adding'] = 'node'
    graph.edges[first].metadata['key'] = 'same'
    graph.edges[second].metadata.update({'radius': 'second', 'key': 'same'})
    result = graph.to_networkx()
    assert isinstance(result, nx.MultiGraph)
    assert result.nodes[a]['node_for_adding'] == 'node'
    by_id = {data['edge_id']: data for _, _, data in result.edges(data=True)}
    assert len(by_id) == 2 and set(by_id) == {first, second}
    assert by_id[first]['key'] == 'same'
    assert '_fibernet_metadata' not in by_id[first]
    assert by_id[second]['key'] == 'same'
    assert by_id[second]['_fibernet_metadata']['radius'] == 'second'


def test_projection_preserves_namespace_only_collision():
    graph = StructureGraph()
    a, b = graph.add_node((0, 0)), graph.add_node((1, 0))
    edge = graph.add_edge(a, b)
    graph.nodes[a].metadata['_fibernet_metadata'] = 'node value'
    graph.edges[edge].metadata['_fibernet_metadata'] = 'edge value'
    result = graph.to_networkx()
    assert result.nodes[a]['_fibernet_metadata'] == {'_fibernet_metadata': 'node value'}
    assert result.edges[a, b]['_fibernet_metadata'] == {'_fibernet_metadata': 'edge value'}
