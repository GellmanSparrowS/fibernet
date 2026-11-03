"""NetworkX projection import can retain user metadata without aliasing."""
import unittest
import networkx as nx
from fibernet.core.structure_graph import StructureGraph


class NetworkXMetadataImport(unittest.TestCase):
    def test_roundtrip_collisions_parallel_edges_and_detachment(self):
        graph = StructureGraph()
        a, b = graph.add_node((0, 0)), graph.add_node((1, 0))
        e0 = graph.add_edge(a, b, radius=0.2, segments=7)
        e1 = graph.add_edge(a, b, radius=0.4, allow_parallel=True)
        graph.nodes[a].metadata.update({'pos': 'node label', 'boundary': 'b',
                                        '_fibernet_metadata': 'original', 'tag': {'n': [1]}})
        graph.edges[e0].metadata.update({'radius': 'label', 'edge_id': 'id',
                                         'segments': 'count', 'tag': {'e': [2]}})
        graph.edges[e1].metadata.update({'tag': 'second'})
        nxgraph = graph.to_networkx()
        nxgraph.graph['experiment'] = {'id': [3]}
        nxgraph.nodes[a]['tag'] = {'n': [4]}  # edited flat metadata wins
        result = StructureGraph.from_networkx(nxgraph, preserve_metadata=True)
        self.assertEqual(result.metadata, {'experiment': {'id': [3]}})
        self.assertEqual(result.nodes[a].metadata,
                         {'pos': 'node label', 'boundary': 'b',
                          '_fibernet_metadata': 'original', 'tag': {'n': [4]}})
        edges = list(result.edges.values())
        self.assertEqual(len(edges), 2)
        self.assertEqual((edges[0].radius, edges[0].segments), (0.2, 7))
        self.assertEqual(edges[0].metadata,
                         {'radius': 'label', 'edge_id': 'id', 'segments': 'count',
                          'tag': {'e': [2]}})
        self.assertEqual(edges[1].metadata, {'tag': 'second'})
        nxgraph.graph['experiment']['id'].append(9)
        nxgraph.nodes[a]['tag']['n'].append(9)
        nxgraph.edges[a, b, e0]['tag']['e'].append(9)
        self.assertEqual(result.metadata['experiment']['id'], [3])
        self.assertEqual(result.nodes[a].metadata['tag']['n'], [4])
        self.assertEqual(edges[0].metadata['tag']['e'], [2])

    def test_defaults_and_external_attributes(self):
        g = nx.Graph(project={'number': 1})
        g.add_node('a', pos=(0, 0), color=['red'], _fibernet_metadata={'note': 'literal'})
        g.add_node('b', pos=(1, 0))
        g.add_edge('a', 'b', radius=0.3, custom={'run': [1]})
        old = StructureGraph.from_networkx(g)
        self.assertEqual(old.metadata, {})
        self.assertEqual(old.nodes[0].metadata, {})
        self.assertEqual(old.edges[0].metadata, {})
        new = StructureGraph.from_networkx(g, preserve_metadata=True)
        self.assertEqual(new.metadata, {'project': {'number': 1}})
        self.assertEqual(new.nodes[0].metadata,
                         {'color': ['red'], '_fibernet_metadata': {'note': 'literal'}})
        self.assertEqual(new.edges[0].metadata, {'custom': {'run': [1]}})
        g.nodes['a']['_fibernet_metadata'] = {'pos': 'label', 'note': 'kept',
                                               'color': ['old']}
        restored = StructureGraph.from_networkx(g, preserve_metadata=True)
        self.assertEqual(restored.nodes[0].metadata,
                         {'pos': 'label', 'note': 'kept', 'color': ['red']})

    def test_self_loop_and_collapsed_directed_edges(self):
        g = nx.DiGraph()
        g.add_node('a', pos=(0, 0))
        g.add_node('b', pos=(1, 0))
        g.add_edge('a', 'a', label='loop')
        g.add_edge('a', 'b', label='first')
        g.add_edge('b', 'a', label='second')
        result = StructureGraph.from_networkx(g, preserve_metadata=True)
        self.assertEqual(result.num_edges, 1)
        self.assertEqual(result.edges[0].metadata, {'label': 'first'})


if __name__ == '__main__':
    unittest.main()
