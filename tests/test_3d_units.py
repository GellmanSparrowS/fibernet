"""
Integration tests for 3D unit types and GraphFeatureExtractor3D.
Tests generation, tiling, simulation, feature extraction, and visualization.
"""

import pytest
import numpy as np
import fibernet as fn
from fibernet.analysis.graph_features_3d import GraphFeatureExtractor3D


# All available 3D unit types
ALL_3D_UNITS = fn.list_units_3d()

# Units that need special kwargs
SPECIAL_KWARGS = {
    'chiral_3d': {'chirality': 0.3},
    'reentrant_3d': {'angle': 15.0},
}


class TestListUnits3D:
    """Test list_units_3d API."""

    def test_returns_sorted_list(self):
        units = fn.list_units_3d()
        assert isinstance(units, list)
        assert units == sorted(units)

    def test_contains_expected_types(self):
        units = fn.list_units_3d()
        expected = ['cubic', 'octet', 'diamond_3d', 'bcc', 'fcc', 'hcp',
                    'gyroid', 'schwarz_p', 'schwarz_d', 'iwp', 'neovius', 'lidinoid',
                    'chiral_3d', 'reentrant_3d']
        for u in expected:
            assert u in units, f"Missing unit: {u}"

    def test_minimum_count(self):
        assert len(fn.list_units_3d()) >= 14


class TestPattern3DGeneration:
    """Test pattern_3d generates valid StructureGraph for each unit type."""

    @pytest.mark.parametrize("unit", ALL_3D_UNITS)
    def test_generates_3d_graph(self, unit):
        ukw = SPECIAL_KWARGS.get(unit, {})
        g = fn.pattern_3d(unit=unit, box=(10, 10, 10), grid=(1, 1, 1), unit_kwargs=ukw)
        assert g.dimension == 3
        assert g.num_nodes > 0
        assert g.num_edges > 0

    @pytest.mark.parametrize("unit", ALL_3D_UNITS)
    def test_tiling(self, unit):
        ukw = SPECIAL_KWARGS.get(unit, {})
        g1 = fn.pattern_3d(unit=unit, box=(10, 10, 10), grid=(1, 1, 1), unit_kwargs=ukw)
        g2 = fn.pattern_3d(unit=unit, box=(10, 10, 10), grid=(2, 2, 2), unit_kwargs=ukw)
        assert g2.num_nodes > g1.num_nodes
        assert g2.num_edges > g1.num_edges

    @pytest.mark.parametrize("unit", ALL_3D_UNITS)
    def test_n_pts_per_side(self, unit):
        ukw = SPECIAL_KWARGS.get(unit, {})
        g0 = fn.pattern_3d(unit=unit, box=(10, 10, 10), grid=(1, 1, 1), unit_kwargs=ukw)
        g2 = fn.pattern_3d(unit=unit, box=(10, 10, 10), grid=(1, 1, 1),
                           n_pts_per_side=1, unit_kwargs=ukw)
        # With intermediate points, should have more nodes
        assert g2.num_nodes >= g0.num_nodes

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown 3D unit"):
            fn.pattern_3d(unit="nonexistent_unit")

    def test_integer_grid(self):
        g = fn.pattern_3d(unit="cubic", box=(10, 10, 10), grid=2)
        assert g.num_nodes > 8  # 2x2x2 should have more than single cell


class TestSimulation3D:
    """Test simulation interfaces with 3D structures."""

    @pytest.mark.parametrize("unit", ["cubic", "octet", "bcc", "gyroid"])
    def test_stretch_test(self, unit):
        ukw = SPECIAL_KWARGS.get(unit, {})
        g = fn.pattern_3d(unit=unit, box=(10, 10, 10), grid=(2, 2, 2), unit_kwargs=ukw)
        engine = fn.TaichiEngine()
        result = engine.stretch_test(g, target_stretch=1.3, num_steps=50)
        assert result.displacements is not None
        assert result.displacements.shape[0] == g.num_nodes

    def test_dynamics_simulation(self):
        g = fn.pattern_3d(unit="cubic", box=(10, 10, 10), grid=(2, 2, 2))
        engine = fn.TaichiEngine()
        fixed = [nid for nid in sorted(g.nodes.keys())
                 if g.nodes[nid].position[0] < 1.0]
        result = engine.dynamics(g, fixed_nodes=fixed, num_steps=50)
        assert result.displacements is not None

    def test_node_manipulation(self):
        g = fn.pattern_3d(unit="cubic", box=(10, 10, 10), grid=(2, 2, 2))
        internal = g.get_internal_nodes()
        if len(internal) > 0:
            nid = internal[0]
            orig = g.nodes[nid].position.copy()
            g.displace_node(nid, [0.1, 0.2, 0.3])
            np.testing.assert_allclose(g.nodes[nid].position - orig, [0.1, 0.2, 0.3])


class TestGraphFeatureExtractor3D:
    """Test 3D feature extraction."""

    def test_feature_count(self):
        ext = GraphFeatureExtractor3D()
        assert len(ext.get_feature_names()) == 60

    @pytest.mark.parametrize("unit", ["cubic", "bcc", "gyroid", "chiral_3d"])
    def test_extract_returns_dict(self, unit):
        ukw = SPECIAL_KWARGS.get(unit, {})
        g = fn.pattern_3d(unit=unit, box=(10, 10, 10), grid=(2, 2, 2), unit_kwargs=ukw)
        ext = GraphFeatureExtractor3D()
        features = ext.extract(g)
        assert isinstance(features, dict)
        assert len(features) == 60

    @pytest.mark.parametrize("unit", ["cubic", "bcc", "gyroid", "chiral_3d"])
    def test_extract_no_nan(self, unit):
        ukw = SPECIAL_KWARGS.get(unit, {})
        g = fn.pattern_3d(unit=unit, box=(10, 10, 10), grid=(2, 2, 2), unit_kwargs=ukw)
        ext = GraphFeatureExtractor3D()
        vec = ext.extract_vector(g)
        assert not np.any(np.isnan(vec))
        assert not np.any(np.isinf(vec))

    def test_batch_extraction(self):
        graphs = [
            fn.pattern_3d(unit=u, box=(10, 10, 10), grid=(2, 2, 2))
            for u in ["cubic", "bcc"]
        ]
        ext = GraphFeatureExtractor3D()
        matrix = ext.batch_extract_matrix(graphs)
        assert matrix.shape == (2, 60)

    def test_features_distinguish_structures(self):
        """Verify that different structures produce different feature vectors."""
        ext = GraphFeatureExtractor3D()
        g1 = fn.pattern_3d(unit="cubic", box=(10, 10, 10), grid=(2, 2, 2))
        g2 = fn.pattern_3d(unit="bcc", box=(10, 10, 10), grid=(2, 2, 2))
        v1 = ext.extract_vector(g1)
        v2 = ext.extract_vector(g2)
        assert not np.allclose(v1, v2), "cubic and bcc should have different features"


class TestVisualization3D:
    """Test 3D visualization functions."""

    def test_render_graph_3d(self):
        import matplotlib
        matplotlib.use('Agg')
        g = fn.pattern_3d(unit="cubic", box=(10, 10, 10), grid=(2, 2, 2))
        fig = fn.render_graph_3d(g)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_render_deformation_3d(self):
        import matplotlib
        matplotlib.use('Agg')
        g = fn.pattern_3d(unit="cubic", box=(10, 10, 10), grid=(2, 2, 2))
        engine = fn.TaichiEngine()
        result = engine.stretch_test(g, target_stretch=1.3, num_steps=50)
        fig = fn.render_deformation_3d(g, result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_render_gallery_3d(self):
        import matplotlib
        matplotlib.use('Agg')
        graphs = [
            fn.pattern_3d(unit=u, box=(10, 10, 10), grid=(2, 2, 2))
            for u in ["cubic", "bcc", "fcc"]
        ]
        fig = fn.render_gallery_3d(graphs, titles=["cubic", "bcc", "fcc"])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_render_stress_3d(self):
        import matplotlib
        matplotlib.use('Agg')
        g = fn.pattern_3d(unit="cubic", box=(10, 10, 10), grid=(2, 2, 2))
        engine = fn.TaichiEngine()
        result = engine.stretch_test(g, target_stretch=1.2, num_steps=200)
        for color_by in ["force", "stretch", "displacement"]:
            fig = fn.render_stress_3d(g, result, color_by=color_by)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_render_comparison_3d(self):
        import matplotlib
        matplotlib.use('Agg')
        g = fn.pattern_3d(unit="cubic", box=(10, 10, 10), grid=(2, 2, 2))
        engine = fn.TaichiEngine()
        result = engine.stretch_test(g, target_stretch=1.2, num_steps=200)
        fig = fn.render_comparison_3d(g, result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_render_multi_angle_3d(self):
        import matplotlib
        matplotlib.use('Agg')
        g = fn.pattern_3d(unit="cubic", box=(10, 10, 10), grid=(2, 2, 2))
        fig = fn.render_multi_angle_3d(g)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_render_trajectory_3d_from_sim_result(self):
        import matplotlib
        matplotlib.use('Agg')
        g = fn.pattern_3d(unit="cubic", box=(10, 10, 10), grid=(2, 2, 2))
        engine = fn.TaichiEngine()
        fixed = [nid for nid in sorted(g.nodes.keys()) if g.nodes[nid].position[0] < 0.5]
        moved = {nid: [(0, [0.1, 0, 0])] for nid in sorted(g.nodes.keys()) if g.nodes[nid].position[0] > 19.5}
        result = engine.dynamics(g, fixed_nodes=fixed, displacement_schedule=moved,
                                  num_steps=500, save_interval=100)
        # Now returns ONE combined Figure
        import matplotlib.pyplot as plt
        fig = fn.render_trajectory_3d(g, sim_result=result, n_frames=3)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 3
        plt.close(fig)


class TestEnergyComputation:
    """Test that elastic energy is computed correctly."""

    def test_stretch_energy_nonzero(self):
        g = fn.pattern_3d(unit="cubic", box=(10, 10, 10), grid=(2, 2, 2))
        engine = fn.TaichiEngine()
        result = engine.stretch_test(g, target_stretch=1.2, num_steps=500)
        assert result.energy > 0, "Energy should be positive after stretch"

    def test_energy_scales_with_stretch(self):
        g = fn.pattern_3d(unit="cubic", box=(10, 10, 10), grid=(2, 2, 2))
        engine = fn.TaichiEngine()
        r1 = engine.stretch_test(g, target_stretch=1.1, num_steps=500)
        r2 = engine.stretch_test(g, target_stretch=1.3, num_steps=500)
        assert r2.energy > r1.energy, "More stretch should produce more energy"

    def test_edge_forces_computed(self):
        g = fn.pattern_3d(unit="cubic", box=(10, 10, 10), grid=(2, 2, 2))
        engine = fn.TaichiEngine()
        result = engine.stretch_test(g, target_stretch=1.2, num_steps=500)
        assert result.edge_forces is not None
        assert result.edge_stretches is not None
        assert result.max_force > 0



class TestOOMGuard:
    """Test memory guard behavior."""

    def test_warn_by_default(self):
        import warnings
        g = fn.pattern_3d(unit="gyroid", box=(10, 10, 10), grid=(3, 3, 3))
        engine = fn.TaichiEngine()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = engine.dynamics(g, num_steps=10, max_nodes=100)
            assert len(w) > 0
            assert "nodes" in str(w[-1].message)

    def test_strict_mode_blocks(self):
        import os
        g = fn.pattern_3d(unit="gyroid", box=(10, 10, 10), grid=(3, 3, 3))
        engine = fn.TaichiEngine()
        os.environ["FIBERNET_STRICT_MEMORY"] = "1"
        try:
            with pytest.raises(ValueError, match="nodes"):
                engine.dynamics(g, num_steps=10, max_nodes=100)
        finally:
            del os.environ["FIBERNET_STRICT_MEMORY"]

    def test_no_warn_under_limit(self):
        import warnings
        g = fn.pattern_3d(unit="cubic", box=(10, 10, 10), grid=(2, 2, 2))
        engine = fn.TaichiEngine()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = engine.dynamics(g, num_steps=10, max_nodes=50000)
            resource_warns = [x for x in w if issubclass(x.category, ResourceWarning)]
            assert len(resource_warns) == 0
