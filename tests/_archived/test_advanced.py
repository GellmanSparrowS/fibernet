"""Tests for advanced generators and variants."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fibernet.gen import advanced, variants
from fibernet.gen.ordered import square_lattice_2d, honeycomb_lattice_2d
from fibernet.gen.disordered import random_straight_2d


class TestAdvancedGenerators:
    def test_voronoi_2d(self):
        net = advanced.voronoi_network_2d(num_seeds=30, box_size=(30, 30), seed=42)
        assert net.num_fibers > 0
        assert net.dimension == 2
    
    def test_voronoi_3d(self):
        net = advanced.voronoi_network_3d(num_seeds=20, box_size=(15, 15, 15), seed=42)
        assert net.num_fibers > 0
        assert net.dimension == 3
    
    def test_electrospun(self):
        net = advanced.electrospun_network(num_fibers=50, box_size=(50, 50), seed=42)
        assert net.num_fibers == 50
    
    def test_electrospun_aligned(self):
        net = advanced.electrospun_network(
            num_fibers=30, deposition_pattern="aligned", seed=42
        )
        assert net.num_fibers == 30
    
    def test_meltblown(self):
        net = advanced.meltblown_network(num_fibers=50, seed=42)
        assert net.num_fibers == 50
    
    def test_biomimetic_collagen(self):
        net = advanced.biomimetic_collagen(num_fibers=30, seed=42)
        assert net.num_fibers == 30
    
    def test_biomimetic_fibrin(self):
        net = advanced.biomimetic_fibrin(num_fibers=20, seed=42)
        assert net.num_fibers >= 20
    
    def test_defected_lattice_vacancy(self):
        net = advanced.defected_lattice(
            square_lattice_2d, {"spacing": 5, "grid_size": (4, 4)},
            defect_type="vacancy", defect_fraction=0.2, seed=42,
        )
        base = square_lattice_2d(spacing=5, grid_size=(4, 4))
        assert net.num_fibers < base.num_fibers
    
    def test_defected_lattice_displacement(self):
        net = advanced.defected_lattice(
            square_lattice_2d, {"spacing": 5, "grid_size": (3, 3)},
            defect_type="displacement", defect_fraction=0.3, seed=42,
        )
        assert net.num_fibers > 0
    
    def test_auxetic_structure(self):
        net = advanced.auxetic_structure(reentrant_angle=np.pi/4, grid_size=(3, 3))
        assert net.num_fibers > 0
    
    def test_kirigami_structure(self):
        net = advanced.kirigami_structure(num_cuts=3)
        assert net.num_fibers > 0
    
    def test_composite_network(self):
        net1 = random_straight_2d(20, 10, (30, 30), seed=42)
        net2 = square_lattice_2d(spacing=5, grid_size=(2, 2))
        composite = advanced.composite_network([net1, net2])
        assert composite.num_fibers > 0
    
    def test_graded_network(self):
        net = advanced.graded_network(
            random_straight_2d,
            {"num_fibers": 30, "fiber_length": 10, "box_size": (30, 30), "seed": 42},
            gradient_axis=0, property_name="radius",
        )
        assert net.num_fibers > 0


class TestVariants:
    def test_lattice_2d_to_3d(self):
        net_2d = square_lattice_2d(spacing=5, grid_size=(3, 3))
        net_3d = variants.lattice_2d_to_3d(net_2d, num_layers=3, layer_spacing=5)
        assert net_3d.dimension == 3
        assert net_3d.num_fibers >= net_2d.num_fibers * 3
    
    def test_curved_lattice(self):
        net = variants.curved_lattice(
            square_lattice_2d,
            {"spacing": 5, "grid_size": (3, 3)},
            curvature=0.05,
        )
        assert net.num_fibers > 0
    
    def test_multi_radius_bimodal(self):
        net = variants.multi_radius_network(
            random_straight_2d,
            {"num_fibers": 30, "fiber_length": 10, "box_size": (30, 30), "seed": 42},
            radius_distribution="bimodal",
            seed=42,
        )
        radii = [f.radius for f in net.fibers]
        assert len(set(radii)) >= 2  # At least 2 different radii
    
    def test_multi_radius_uniform(self):
        net = variants.multi_radius_network(
            random_straight_2d,
            {"num_fibers": 30, "fiber_length": 10, "box_size": (30, 30), "seed": 42},
            radius_distribution="uniform",
            seed=42,
        )
        assert net.num_fibers > 0
    
    def test_variable_stiffness(self):
        net = variants.variable_stiffness_network(
            square_lattice_2d,
            {"spacing": 5, "grid_size": (3, 3)},
            stiffness_func="linear",
        )
        assert net.num_fibers > 0
    
    def test_diamond_lattice_3d(self):
        net = variants.diamond_lattice_3d(spacing=5, grid_size=(2, 2, 2))
        assert net.num_fibers > 0
        assert net.dimension == 3
    
    def test_foam_like_3d(self):
        net = variants.foam_like_3d(box_size=(15, 15, 15), num_cells=20, seed=42)
        assert net.num_fibers > 0
    
    def test_gyroid_infill(self):
        net = variants.gyroid_infill(box_size=(10, 10, 10), cell_size=5, resolution=10)
        assert net.dimension == 3


class TestAdvancedAnalysis:
    def test_spectral_analyzer(self):
        from fibernet.analysis.advanced import SpectralAnalyzer
        net = square_lattice_2d(spacing=5, grid_size=(3, 3))
        spec = SpectralAnalyzer(net)
        eigs = spec.laplacian_eigenvalues(k=5)
        assert len(eigs) > 0
        gap = spec.spectral_gap()
        assert gap >= 0
    
    def test_pore_analyzer(self):
        from fibernet.analysis.advanced import PoreAnalyzer
        net = random_straight_2d(50, 10, (30, 30), seed=42)
        pore = PoreAnalyzer(net)
        mean_pore = pore.mean_pore_size()
        assert mean_pore >= 0
        stats = pore.pore_size_statistics()
        assert "mean" in stats
    
    def test_anisotropy_analyzer(self):
        from fibernet.analysis.advanced import AnisotropyAnalyzer
        net = random_straight_2d(50, 10, (30, 30), seed=42)
        aniso = AnisotropyAnalyzer(net)
        A = aniso.orientation_tensor()
        assert A.shape == (3, 3)
        idx = aniso.anisotropy_index()
        assert 0 <= idx <= 1
    
    def test_structural_fingerprint(self):
        from fibernet.analysis.advanced import StructuralFingerprint
        net1 = random_straight_2d(50, 10, (30, 30), seed=42)
        net2 = random_straight_2d(50, 10, (30, 30), seed=43)
        fp1 = StructuralFingerprint(net1)
        fp2 = StructuralFingerprint(net2)
        dist = fp1.distance_to(fp2)
        assert dist >= 0
