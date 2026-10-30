import numpy as np
import networkx as nx
from fibernet.analysis.graph_features_3d import GraphFeatureExtractor3D
from fibernet.core.structure_graph import StructureGraph

def assert_empty(graph, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('empty graph must not query connectivity')
    monkeypatch.setattr(nx, 'is_connected', forbidden)
    ext=GraphFeatureExtractor3D()
    assert ext.extract(graph) == dict.fromkeys(ext.FEATURE_NAMES, 0.0)
    vector=ext.extract_vector(graph)
    assert vector.shape == (len(ext.FEATURE_NAMES),)
    assert np.array_equal(vector, np.zeros(len(ext.FEATURE_NAMES)))

def test_empty_networkx(monkeypatch):
    assert_empty(nx.Graph(), monkeypatch)

def test_empty_structure_graph(monkeypatch):
    assert_empty(StructureGraph(), monkeypatch)

def test_singleton_preserves_counts():
    graph=nx.Graph(); graph.add_node(7,pos=(1.,2.,3.))
    result=GraphFeatureExtractor3D().extract(graph)
    assert result['n_node']==1 and result['n_edge']==0
    assert result['n_components']==1 and result['largest_component_frac']==1
    assert result['graph_diameter']==0 and result['mean_eccentricity']==0

def test_two_components_preserve_counts():
    graph=nx.Graph()
    for node,pos in [(3,(0.,0.,0.)),(9,(1.,0.,0.)),(21,(0.,2.,0.)),(27,(1.,2.,0.))]:
        graph.add_node(node,pos=pos)
    graph.add_edges_from([(3,9),(21,27)])
    result=GraphFeatureExtractor3D().extract(graph)
    assert result['n_node']==4 and result['n_edge']==2
    assert result['n_components']==2 and result['largest_component_frac']==.5
    assert result['total_length']==2 and result['graph_diameter']==0
