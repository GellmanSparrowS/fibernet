"""
Comprehensive tests for the graph-based API:
- Graph I/O (JSON roundtrip, NetworkX conversion)
- Weld graph (intersection detection, node insertion)
- RegularNetworkGenerator (structure, tiling, perturbations)
- ZigZagGenerator (structure, mirroring)
- GraphFeatureExtractor (94-dim features)
- Visualization (plot_graph)
"""

import json
import math
import os
import tempfile

import numpy as np
import pytest

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

from fibernet.graph.io import to_networkx, from_networkx, save_graph_json, load_graph_json
from fibernet.graph.weld import weld_graph, find_intersections, merge_coincident_nodes
from fibernet.gen.regular import RegularNetworkGenerator
from fibernet.gen.zigzag import ZigZagGenerator
from fibernet.analysis.graph_features import GraphFeatureExtractor


# ============================================================
# Graph I/O Tests
# ============================================================

class TestGraphIO:
    """Tests for JSON and NetworkX graph I/O."""

    def _make_simple_graph(self):
        G = nx.Graph()
        G.add_node(0, pos=(0.0, 0.0, 0.0))
        G.add_node(1, pos=(1.0, 0.0, 0.0))
        G.add_node(2, pos=(1.0, 1.0, 0.0))
        G.add_node(3, pos=(0.0, 1.0, 0.0))
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 0)
        G.add_edge(0, 2)
        return G

    def test_json_roundtrip(self, tmp_path):
        G = self._make_simple_graph()
        path = tmp_path / "test_graph.json"
        save_graph_json(G, path)
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "nodes" in data
        assert "links" in data
        assert len(data["nodes"]) == 4
        assert len(data["links"]) == 5

    def test_json_load_returns_fiber_network(self, tmp_path):
        G = self._make_simple_graph()
        path = tmp_path / "test_graph.json"
        save_graph_json(G, path)
        net = load_graph_json(path)
        assert hasattr(net, 'fibers')
        assert len(net.fibers) > 0

    def test_from_networkx_basic(self):
        G = self._make_simple_graph()
        net = from_networkx(G)
        assert hasattr(net, 'fibers')
        assert len(net.fibers) > 0


# ============================================================
# Weld Graph Tests
# ============================================================

class TestWeldGraph:

    def test_single_intersection(self):
        G = nx.Graph()
        G.add_node(0, pos=(0, 0))
        G.add_node(1, pos=(10, 10))
        G.add_node(2, pos=(0, 10))
        G.add_node(3, pos=(10, 0))
        G.add_edge(0, 1)
        G.add_edge(2, 3)
        Gw = weld_graph(G)
        assert Gw.number_of_nodes() == 5
        assert Gw.number_of_edges() == 4

    def test_no_intersection(self):
        G = nx.Graph()
        G.add_node(0, pos=(0, 0))
        G.add_node(1, pos=(10, 0))
        G.add_node(2, pos=(0, 5))
        G.add_node(3, pos=(10, 5))
        G.add_edge(0, 1)
        G.add_edge(2, 3)
        Gw = weld_graph(G)
        assert Gw.number_of_nodes() == 4
        assert Gw.number_of_edges() == 2

    def test_adjacent_edges_no_intersection(self):
        G = nx.Graph()
        G.add_node(0, pos=(0, 0))
        G.add_node(1, pos=(5, 5))
        G.add_node(2, pos=(10, 0))
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        Gw = weld_graph(G)
        assert Gw.number_of_nodes() == 3

    def test_find_intersections(self):
        G = nx.Graph()
        G.add_node(0, pos=(0, 0))
        G.add_node(1, pos=(10, 10))
        G.add_node(2, pos=(0, 10))
        G.add_node(3, pos=(10, 0))
        G.add_edge(0, 1)
        G.add_edge(2, 3)
        ix = find_intersections(G)
        assert len(ix) > 0

    def test_merge_coincident_nodes(self):
        G = nx.Graph()
        G.add_node(0, pos=(0, 0))
        G.add_node(1, pos=(0.1, 0.1))
        G.add_node(2, pos=(10, 10))
        G.add_edge(0, 2)
        G.add_edge(1, 2)
        Gm = merge_coincident_nodes(G, tolerance=0.5)
        assert Gm.number_of_nodes() == 2

    def test_weld_preserves_original(self):
        G = nx.Graph()
        G.add_node(0, pos=(0, 0))
        G.add_node(1, pos=(10, 10))
        G.add_node(2, pos=(0, 10))
        G.add_node(3, pos=(10, 0))
        G.add_edge(0, 1)
        G.add_edge(2, 3)
        orig_nodes = G.number_of_nodes()
        weld_graph(G)
        assert G.number_of_nodes() == orig_nodes


# ============================================================
# RegularNetworkGenerator Tests
# ============================================================

class TestRegularNetworkGenerator:

    def test_basic_generation(self):
        gen = RegularNetworkGenerator(side_length=10, num_points_per_side=0, tiling=1)
        G = gen.generate()
        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 4

    def test_midpoints(self):
        gen = RegularNetworkGenerator(side_length=10, num_points_per_side=1, tiling=1)
        G = gen.generate()
        assert G.number_of_nodes() == 8

    def test_two_midpoints(self):
        gen = RegularNetworkGenerator(side_length=10, num_points_per_side=2, tiling=1)
        G = gen.generate()
        assert G.number_of_nodes() == 12

    def test_tiling(self):
        gen1 = RegularNetworkGenerator(side_length=10, num_points_per_side=1, tiling=1)
        gen3 = RegularNetworkGenerator(side_length=10, num_points_per_side=1, tiling=3)
        G1 = gen1.generate()
        G3 = gen3.generate()
        assert G3.number_of_nodes() > G1.number_of_nodes()

    def test_random_perturbations(self):
        perts = RegularNetworkGenerator.random_perturbations(num_points=5)
        assert len(perts) == 5
        for dx, dy in perts:
            assert -0.5 <= dx <= 0.5
            assert -0.5 <= dy <= 0.5

    def test_scale_to_unit(self):
        gen = RegularNetworkGenerator(side_length=100, num_points_per_side=1,
                                      tiling=3, scale_to_unit=True)
        G = gen.generate()
        pos = nx.get_node_attributes(G, 'pos')
        coords = np.array(list(pos.values()))
        assert coords.max() <= 1.0 + 1e-6
        assert coords.min() >= -1e-6

    def test_pos_attribute_exists(self):
        gen = RegularNetworkGenerator(side_length=10, num_points_per_side=2, tiling=2)
        G = gen.generate()
        pos = nx.get_node_attributes(G, 'pos')
        assert len(pos) == G.number_of_nodes()


# ============================================================
# ZigZagGenerator Tests
# ============================================================

class TestZigZagGenerator:

    def test_basic_generation(self):
        gen = ZigZagGenerator(n_cols=2, n_rows=2)
        G = gen.generate()
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_custom_base_points(self):
        points = [(0, 0), (5, 10), (10, 0)]
        gen = ZigZagGenerator(base_points=points, n_cols=3, n_rows=1)
        G = gen.generate()
        assert G.number_of_nodes() > 0

    def test_no_mirror(self):
        gen = ZigZagGenerator(n_cols=3, n_rows=3, mirror_x=False, mirror_y=False)
        G = gen.generate()
        assert G.number_of_nodes() > 0

    def test_simple_zigzag_factory(self):
        gen = ZigZagGenerator.simple_zigzag(amplitude=30, wavelength=60, n_periods=5)
        G = gen.generate()
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_pos_attribute_3d(self):
        gen = ZigZagGenerator(n_cols=2, n_rows=2)
        G = gen.generate()
        pos = nx.get_node_attributes(G, 'pos')
        for node, p in pos.items():
            assert len(p) == 3


# ============================================================
# GraphFeatureExtractor Tests
# ============================================================

class TestGraphFeatureExtractor:

    def _make_test_graph(self, n=30, seed=42):
        rng = np.random.default_rng(seed)
        G = nx.Graph()
        points = rng.random((n, 2))
        for i, p in enumerate(points):
            G.add_node(i, pos=tuple(p))
        for i in range(n):
            for j in range(i + 1, n):
                pi = np.array(points[i])
                pj = np.array(points[j])
                if np.linalg.norm(pi - pj) < 0.4:
                    G.add_edge(i, j)
        return G

    def test_feature_count(self):
        G = self._make_test_graph()
        ext = GraphFeatureExtractor(canvas_size=128)
        feat = ext.extract(G)
        assert len(feat) == 94

    def test_feature_cols_match(self):
        G = self._make_test_graph()
        ext = GraphFeatureExtractor(canvas_size=128)
        feat = ext.extract(G)
        for col in GraphFeatureExtractor.FEATURE_COLS:
            assert col in feat, f"Missing feature: {col}"

    def test_structural_features_nonzero(self):
        G = self._make_test_graph(n=50)
        ext = GraphFeatureExtractor(canvas_size=128)
        feat = ext.extract(G)
        assert feat['n_node'] > 0
        assert feat['n_edge'] > 0
        assert feat['total_length'] > 0
        assert feat['mean_edge_len'] > 0
        assert feat['radius_gyration'] > 0

    def test_vector_form(self):
        G = self._make_test_graph()
        ext = GraphFeatureExtractor(canvas_size=128)
        vec = ext.extract_vector(G)
        assert vec.shape == (94,)
        assert isinstance(vec, np.ndarray)

    def test_batch_extract(self):
        graphs = [self._make_test_graph(n=20, seed=i) for i in range(3)]
        ext = GraphFeatureExtractor(canvas_size=128)
        results = ext.batch_extract(graphs)
        assert len(results) == 3
        for r in results:
            assert len(r) == 94

    def test_batch_matrix(self):
        graphs = [self._make_test_graph(n=20, seed=i) for i in range(3)]
        ext = GraphFeatureExtractor(canvas_size=128)
        mat = ext.batch_extract_matrix(graphs)
        assert mat.shape == (3, 94)

    def test_reproducibility(self):
        G = self._make_test_graph()
        ext = GraphFeatureExtractor(canvas_size=128)
        f1 = ext.extract(G)
        f2 = ext.extract(G)
        for k in f1:
            if isinstance(f1[k], float) and isinstance(f2[k], float):
                assert math.isclose(f1[k], f2[k], rel_tol=1e-10), f"Non-reproducible: {k}"
            else:
                assert f1[k] == f2[k], f"Non-reproducible: {k}"

    def test_empty_graph(self):
        G = nx.Graph()
        ext = GraphFeatureExtractor(canvas_size=128)
        feat = ext.extract(G)
        assert len(feat) == 94
        assert feat['n_node'] == 0
        assert feat['n_edge'] == 0

    def test_single_edge(self):
        G = nx.Graph()
        G.add_node(0, pos=(0, 0))
        G.add_node(1, pos=(1, 0))
        G.add_edge(0, 1)
        ext = GraphFeatureExtractor(canvas_size=128)
        feat = ext.extract(G)
        assert feat['n_node'] == 2
        assert feat['n_edge'] == 1

    def test_anisotropy_range(self):
        G = self._make_test_graph(n=50)
        ext = GraphFeatureExtractor(canvas_size=128)
        feat = ext.extract(G)
        assert -1.0 <= feat['anisotropy'] <= 1.0

    def test_convenience_function(self):
        import fibernet as fn
        G = self._make_test_graph()
        feat = fn.extract_features(G, canvas_size=128)
        assert len(feat) == 94


# ============================================================
# Visualization Tests
# ============================================================

class TestVisualization:

    def test_plot_graph_no_crash(self):
        import matplotlib
        matplotlib.use('Agg')
        gen = RegularNetworkGenerator(side_length=10, tiling=2)
        G = gen.generate()
        from fibernet.viz.graph_plot import plot_graph
        fig, ax = plot_graph(G)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_plot_graph_save(self, tmp_path):
        import matplotlib
        matplotlib.use('Agg')
        gen = RegularNetworkGenerator(side_length=10, tiling=2)
        G = gen.generate()
        from fibernet.viz.graph_plot import plot_graph
        save_path = tmp_path / "test_plot.png"
        fig, ax = plot_graph(G, save_path=str(save_path))
        assert save_path.exists()
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_plot_with_nodes(self):
        import matplotlib
        matplotlib.use('Agg')
        gen = RegularNetworkGenerator(side_length=10, tiling=2)
        G = gen.generate()
        from fibernet.viz.graph_plot import plot_graph
        fig, ax = plot_graph(G, show_nodes=True)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_plot_comparison(self):
        import matplotlib
        matplotlib.use('Agg')
        G1 = RegularNetworkGenerator(side_length=10, tiling=2).generate()
        G2 = ZigZagGenerator(n_cols=3, n_rows=3).generate()
        from fibernet.viz.graph_plot import plot_graph_comparison
        fig, axes = plot_graph_comparison([G1, G2], labels=["Regular", "ZigZag"])
        assert len(axes) == 2
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_plot_structure_stats(self):
        import matplotlib
        matplotlib.use('Agg')
        G = RegularNetworkGenerator(side_length=10, tiling=3).generate()
        from fibernet.viz.graph_plot import plot_structure_stats
        fig, axes = plot_structure_stats(G)
        assert len(axes) == 3
        import matplotlib.pyplot as plt
        plt.close('all')


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:

    def test_regular_weld_feature_pipeline(self):
        gen = RegularNetworkGenerator(side_length=10, tiling=3, num_points_per_side=1)
        G = gen.generate()
        Gw = weld_graph(G)
        ext = GraphFeatureExtractor(canvas_size=256)
        feat = ext.extract(Gw)
        assert feat['n_node'] >= G.number_of_nodes()

    def test_zigzag_feature_pipeline(self):
        gen = ZigZagGenerator(n_cols=4, n_rows=4)
        G = gen.generate()
        ext = GraphFeatureExtractor(canvas_size=256)
        feat = ext.extract(G)
        assert feat['n_node'] == G.number_of_nodes()

    def test_json_roundtrip_with_features(self, tmp_path):
        gen = RegularNetworkGenerator(side_length=10, tiling=2)
        G = gen.generate()
        path = tmp_path / "graph.json"
        save_graph_json(G, path)
        net = load_graph_json(path)
        ext = GraphFeatureExtractor(canvas_size=128)
        feat = ext.extract(net)
        assert len(feat) == 94

    def test_parametric_sweep(self):
        results = []
        for dx in [0.0, 0.1, 0.2, 0.3]:
            gen = RegularNetworkGenerator(
                side_length=10, num_points_per_side=1,
                tiling=3, perturbations=[(dx, 0.0)]
            )
            G = gen.generate()
            ext = GraphFeatureExtractor(canvas_size=128)
            feat = ext.extract(G)
            results.append(feat)
        n_nodes = [r['n_node'] for r in results]
        assert len(set(n_nodes)) == 1

    def test_top_level_api_imports(self):
        import fibernet as fn
        assert hasattr(fn, 'RegularNetworkGenerator')
        assert hasattr(fn, 'ZigZagGenerator')
        assert hasattr(fn, 'GraphFeatureExtractor')
        assert hasattr(fn, 'weld_graph')
        assert hasattr(fn, 'find_intersections')
        assert hasattr(fn, 'merge_coincident_nodes')
        assert hasattr(fn, 'to_networkx')
        assert hasattr(fn, 'from_networkx')
        assert hasattr(fn, 'load_graph_json')
        assert hasattr(fn, 'save_graph_json')
        assert hasattr(fn, 'plot_graph')
        assert hasattr(fn, 'plot_graph_comparison')
        assert hasattr(fn, 'plot_structure_stats')
        assert hasattr(fn, 'extract_features')
