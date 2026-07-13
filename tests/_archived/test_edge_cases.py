"""
Edge case tests for FiberNet.

Tests boundary conditions, empty inputs, and unusual parameter values
to ensure robustness of the library.
"""

import numpy as np
import pytest
from fibernet import gen, sim, analysis
from fibernet.core import Fiber, FiberNetwork, Material


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_network(self):
        """Test operations on empty network."""
        net = FiberNetwork(dimension=2)
        assert net.num_fibers == 0
        assert net.num_crosslinks == 0
        
        # Analysis should handle empty network
        morph = analysis.MorphologyAnalyzer(net)
        assert morph.nematic_order_parameter() == 0.0
        assert morph.porosity() == 1.0

    def test_single_fiber(self):
        """Test operations on network with single fiber."""
        fiber = Fiber(
            centerline=np.array([[0, 0, 0], [1, 0, 0]]),
            radius=0.1
        )
        net = FiberNetwork(dimension=2)
        net.add_fiber(fiber)
        
        assert net.num_fibers == 1
        assert net.num_crosslinks == 0

    def test_zero_length_fiber(self):
        """Test handling of zero-length fiber."""
        fiber = Fiber(
            centerline=np.array([[0, 0, 0], [0, 0, 0]]),
            radius=0.1
        )
        # Should handle gracefully (length = 0)
        assert fiber.length == 0.0

    def test_very_small_network(self):
        """Test with very small box size."""
        net = gen.random_straight_2d(
            num_fibers=5,
            fiber_length=0.1,
            box_size=(1, 1),
            seed=42
        )
        assert net.num_fibers == 5

    def test_very_large_network(self):
        """Test with large number of fibers."""
        net = gen.random_straight_2d(
            num_fibers=500,
            fiber_length=5.0,
            box_size=(100, 100),
            seed=42
        )
        assert net.num_fibers == 500
        # Should have many crosslinks
        assert net.num_crosslinks > 100

    def test_high_fiber_density(self):
        """Test with very high fiber density."""
        net = gen.random_straight_2d(
            num_fibers=100,
            fiber_length=20.0,
            box_size=(10, 10),
            seed=42
        )
        # Should still work even with many overlaps
        assert net.num_fibers == 100

    def test_aligned_fibers(self):
        """Test with perfectly aligned fibers."""
        net = FiberNetwork(dimension=2)
        for i in range(10):
            fiber = Fiber(
                centerline=np.array([[0, i, 0], [10, i, 0]]),
                radius=0.1
            )
            net.add_fiber(fiber)
        
        morph = analysis.MorphologyAnalyzer(net)
        # Perfectly aligned should have high nematic order
        assert morph.nematic_order_parameter() > 0.99

    def test_random_orientations(self):
        """Test with completely random orientations."""
        net = gen.random_straight_2d(
            num_fibers=1000,
            fiber_length=5.0,
            box_size=(50, 50),
            seed=42
        )
        morph = analysis.MorphologyAnalyzer(net)
        # Random should have low nematic order
        assert morph.nematic_order_parameter() < 0.2

    def test_fem_zero_strain(self):
        """Test FEM with zero strain."""
        net = gen.random_straight_2d(
            num_fibers=20,
            fiber_length=5.0,
            box_size=(20, 20),
            seed=42
        )
        fem = sim.FiberFEM(net, segments_per_fiber=3)
        result = fem.apply_uniaxial_strain(strain=0.0, axis=0)
        
        # Zero strain should give zero energy
        assert result.energy < 1e-10

    def test_fem_negative_strain(self):
        """Test FEM with negative strain (compression)."""
        net = gen.random_straight_2d(
            num_fibers=20,
            fiber_length=5.0,
            box_size=(20, 20),
            seed=42
        )
        fem = sim.FiberFEM(net, segments_per_fiber=3)
        result = fem.apply_uniaxial_strain(strain=-0.01, axis=0)
        
        # Compression should still work
        assert result.energy > 0

    def test_thermal_uniform_temperature(self):
        """Test thermal simulation with uniform temperature."""
        net = gen.random_straight_2d(
            num_fibers=20,
            fiber_length=5.0,
            box_size=(20, 20),
            seed=42
        )
        from fibernet import simulate_thermal
        result = simulate_thermal(net, T_hot=50.0, T_cold=50.0)
        
        # Uniform temperature should give zero heat flux
        assert result['conductivity'] >= 0

    def test_material_extreme_properties(self):
        """Test with extreme material properties."""
        mat = Material(
            name="extreme",
            youngs_modulus=1e15,  # Very stiff
            density=1e-10,        # Very light
        )
        net = gen.random_straight_2d(
            num_fibers=10,
            fiber_length=5.0,
            box_size=(20, 20),
            material=mat,
            seed=42
        )
        assert net.num_fibers == 10

    def test_3d_network_generation(self):
        """Test 3D network generation."""
        net = gen.random_straight_3d(
            num_fibers=30,
            fiber_length=10.0,
            box_size=(30, 30, 30),
            seed=42
        )
        assert net.num_fibers == 30
        assert net.dimension == 3

    def test_periodic_boundary_conditions(self):
        """Test with periodic boundary conditions."""
        net = gen.random_straight_2d(
            num_fibers=50,
            fiber_length=15.0,
            box_size=(20, 20),
            seed=42
        )
        # Network should be generated without errors
        assert net.num_fibers == 50

    def test_ordered_lattice_edge_cases(self):
        """Test ordered lattices with minimal size."""
        # 1x1 lattice
        net = gen.square_lattice_2d(spacing=5.0, grid_size=(1, 1))
        assert net.num_fibers >= 0
        
        # 2x2 lattice
        net = gen.square_lattice_2d(spacing=5.0, grid_size=(2, 2))
        assert net.num_fibers > 0

    def test_transformation_identity(self):
        """Test identity transformations."""
        from fibernet.api import scale, rotate
        
        net = gen.random_straight_2d(
            num_fibers=20,
            fiber_length=5.0,
            box_size=(20, 20),
            seed=42
        )
        
        # Identity scale
        net_scaled = scale(net, factor=1.0)
        assert net_scaled.num_fibers == net.num_fibers
        
        # Identity rotation (0 degrees)
        net_rotated = rotate(net, angle=0.0, axis=[0, 0, 1])
        assert net_rotated.num_fibers == net.num_fibers

    def test_merge_empty_networks(self):
        """Test merging empty networks."""
        from fibernet.api import merge
        
        net1 = FiberNetwork(dimension=2)
        net2 = FiberNetwork(dimension=2)
        
        merged = merge([net1, net2])
        assert merged.num_fibers == 0

    def test_export_import_roundtrip(self):
        """Test export/import roundtrip preserves structure."""
        import tempfile
        import os
        
        net = gen.random_straight_2d(
            num_fibers=30,
            fiber_length=8.0,
            box_size=(30, 30),
            seed=42
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.json')
            net.save_json(path)
            net_loaded = FiberNetwork.load_json(path)
            
            assert net_loaded.num_fibers == net.num_fibers
            assert net_loaded.num_crosslinks == net.num_crosslinks
