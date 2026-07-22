"""Tests for fiber bundle generators."""

import pytest
import numpy as np
from fibernet.gen.bundles import (
    parallel_bundle_2d,
    twisted_bundle_2d,
    random_bundle_3d,
    braided_bundle_3d,
    tendon_like_bundle_3d,
)


class TestParallelBundle2D:
    """Test parallel bundle generator."""
    
    def test_basic_generation(self):
        """Test basic bundle creation."""
        net = parallel_bundle_2d(num_fibers=10, bundle_length=50.0, seed=42)
        assert len(net.fibers) == 10
        assert net.dimension == 2
    
    def test_bundle_dimensions(self):
        """Test bundle has correct dimensions."""
        net = parallel_bundle_2d(
            num_fibers=10,
            bundle_length=50.0,
            bundle_width=5.0,
            seed=42
        )
        
        # Check fibers are roughly the right length
        lengths = [f.length for f in net.fibers]
        assert np.mean(lengths) == pytest.approx(50.0, rel=0.1)
    
    def test_orientation(self):
        """Test bundle orientation."""
        net = parallel_bundle_2d(
            num_fibers=10,
            bundle_length=50.0,
            orientation=np.pi/4,  # 45 degrees
            seed=42
        )
        
        # Fibers should be oriented at ~45 degrees
        for fiber in net.fibers[:5]:  # Check first few
            direction = fiber.end_point - fiber.start_point
            angle = np.arctan2(direction[1], direction[0])
            # Allow some tolerance
            assert abs(angle - np.pi/4) < 0.2
    
    def test_reproducibility(self):
        """Test seed produces reproducible results."""
        net1 = parallel_bundle_2d(num_fibers=10, seed=42)
        net2 = parallel_bundle_2d(num_fibers=10, seed=42)
        
        for f1, f2 in zip(net1.fibers, net2.fibers):
            assert np.allclose(f1.start_point, f2.start_point)
            assert np.allclose(f1.end_point, f2.end_point)


class TestTwistedBundle2D:
    """Test twisted bundle generator."""
    
    def test_basic_generation(self):
        """Test basic twisted bundle creation."""
        net = twisted_bundle_2d(num_fibers=8, bundle_length=50.0, seed=42)
        assert len(net.fibers) == 8
        assert net.dimension == 2
    
    def test_twist_pitch(self):
        """Test different twist pitches."""
        net1 = twisted_bundle_2d(num_fibers=8, twist_pitch=20.0, seed=42)
        net2 = twisted_bundle_2d(num_fibers=8, twist_pitch=40.0, seed=42)
        
        # Both should generate networks
        assert len(net1.fibers) == 8
        assert len(net2.fibers) == 8


class TestRandomBundle3D:
    """Test 3D random bundle generator."""
    
    def test_basic_generation(self):
        """Test basic 3D bundle creation."""
        net = random_bundle_3d(num_fibers=20, bundle_length=50.0, seed=42)
        assert len(net.fibers) == 20
        assert net.dimension == 3
    
    def test_orientation_variance(self):
        """Test orientation spread parameter."""
        # Low variance = well aligned
        net_low = random_bundle_3d(
            num_fibers=50,
            orientation_variance=0.05,
            seed=42
        )
        
        # High variance = more spread
        net_high = random_bundle_3d(
            num_fibers=50,
            orientation_variance=0.5,
            seed=42
        )
        
        # Both should generate valid networks
        assert len(net_low.fibers) == 50
        assert len(net_high.fibers) == 50
    
    def test_3d_positions(self):
        """Test fibers are in 3D space."""
        net = random_bundle_3d(num_fibers=20, seed=42)
        
        for fiber in net.fibers:
            # Should have 3D coordinates
            assert len(fiber.start_point) == 3
            assert len(fiber.end_point) == 3


class TestBraidedBundle3D:
    """Test braided bundle generator."""
    
    def test_basic_generation(self):
        """Test basic braided bundle creation."""
        net = braided_bundle_3d(num_strands=6, bundle_length=50.0, seed=42)
        assert net.dimension == 3
        assert len(net.fibers) > 0  # Should have fibers from all strands
    
    def test_strand_count(self):
        """Test different strand counts."""
        net = braided_bundle_3d(
            num_strands=8,
            fibers_per_strand=3,
            seed=42
        )
        
        # Should have roughly num_strands * fibers_per_strand fibers
        assert len(net.fibers) == 8 * 3
    
    def test_braid_structure(self):
        """Test braid has proper 3D structure."""
        net = braided_bundle_3d(
            num_strands=6,
            bundle_length=50.0,
            braid_radius=5.0,
            seed=42
        )
        
        # Fibers should be distributed in 3D
        positions = np.array([f.start_point for f in net.fibers])
        assert positions.shape[1] == 3


class TestTendonLikeBundle3D:
    """Test tendon-like (crimped) bundle generator."""
    
    def test_basic_generation(self):
        """Test basic tendon-like bundle creation."""
        net = tendon_like_bundle_3d(num_fibers=30, bundle_length=80.0, seed=42)
        assert len(net.fibers) == 30
        assert net.dimension == 3
    
    def test_crimp_parameters(self):
        """Test crimp amplitude and wavelength."""
        net = tendon_like_bundle_3d(
            num_fibers=20,
            crimp_amplitude=2.0,
            crimp_wavelength=15.0,
            seed=42
        )
        
        assert len(net.fibers) == 20
    
    def test_bundle_geometry(self):
        """Test bundle has proper cylindrical geometry."""
        net = tendon_like_bundle_3d(
            num_fibers=50,
            bundle_length=80.0,
            bundle_radius=8.0,
            seed=42
        )
        
        # Check that fibers are roughly aligned with z-axis
        directions = []
        for fiber in net.fibers:
            direction = fiber.end_point - fiber.start_point
            direction = direction / np.linalg.norm(direction)
            directions.append(direction)
        
        directions = np.array(directions)
        
        # Mean direction should be close to z-axis
        mean_dir = np.mean(directions, axis=0)
        z_alignment = abs(mean_dir[2])
        assert z_alignment > 0.8  # Should be mostly aligned with z


class TestBundleIntegration:
    """Test bundle generators work together."""
    
    def test_all_bundle_types(self):
        """Test all bundle types can be generated."""
        bundles = [
            parallel_bundle_2d(num_fibers=10, seed=42),
            twisted_bundle_2d(num_fibers=8, seed=42),
            random_bundle_3d(num_fibers=20, seed=42),
            braided_bundle_3d(num_strands=6, seed=42),
            tendon_like_bundle_3d(num_fibers=30, seed=42),
        ]
        
        # All should be valid networks
        for net in bundles:
            assert len(net.fibers) > 0
            assert net.dimension in [2, 3]
    
    def test_with_materials(self):
        """Test bundles can use material properties."""
        from fibernet.core.material import Material
        
        mat = Material(
            name="test_fiber",
            youngs_modulus=1e9,
            density=1000.0,
            poissons_ratio=0.3
        )
        
        net = parallel_bundle_2d(num_fibers=10, material=mat, seed=42)
        assert len(net.fibers) == 10


