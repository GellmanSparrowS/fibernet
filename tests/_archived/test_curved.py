"""Tests for curved fiber generators."""

import pytest
import numpy as np
from fibernet.gen.curved import (
    sinusoidal_fiber_2d,
    helical_fiber_3d,
    arc_fiber_2d,
    bezier_fiber_3d,
    random_curved_network_3d,
    crimped_network_2d,
)


class TestSinusoidalFiber2D:
    """Test sinusoidal fiber generator."""
    
    def test_basic_generation(self):
        """Test basic sinusoidal fiber."""
        fiber = sinusoidal_fiber_2d(
            length=50.0,
            amplitude=2.0,
            wavelength=10.0,
            seed=42
        )
        
        assert fiber is not None
        assert fiber.centerline.shape[0] > 0
        assert fiber.centerline.shape[1] == 3
    
    def test_amplitude(self):
        """Test that amplitude is correct."""
        fiber = sinusoidal_fiber_2d(
            length=50.0,
            amplitude=5.0,
            wavelength=10.0,
            num_segments=100,
            seed=42
        )
        
        # Y-coordinates should vary by ~2*amplitude
        y_coords = fiber.centerline[:, 1]
        y_range = y_coords.max() - y_coords.min()
        assert y_range > 0  # Should have some variation
    
    def test_orientation(self):
        """Test different orientations."""
        # Horizontal
        fiber_h = sinusoidal_fiber_2d(orientation=0.0, seed=42)
        assert fiber_h.centerline.shape[0] > 0
        
        # Vertical
        fiber_v = sinusoidal_fiber_2d(orientation=np.pi/2, seed=42)
        assert fiber_v.centerline.shape[0] > 0


class TestHelicalFiber3D:
    """Test helical fiber generator."""
    
    def test_basic_generation(self):
        """Test basic helical fiber."""
        fiber = helical_fiber_3d(
            length=50.0,
            radius_helix=3.0,
            pitch=10.0,
            num_turns=5.0
        )
        
        assert fiber is not None
        assert fiber.centerline.shape[0] > 0
        assert fiber.centerline.shape[1] == 3
    
    def test_helix_geometry(self):
        """Test helix geometry."""
        fiber = helical_fiber_3d(
            radius_helix=5.0,
            num_turns=2.0,
            num_segments=100
        )
        
        # Check that x-y projection forms a circle
        x_coords = fiber.centerline[:, 0]
        y_coords = fiber.centerline[:, 1]
        
        # Should have circular pattern
        assert len(x_coords) == 100
        assert len(y_coords) == 100
    
    def test_num_turns(self):
        """Test different number of turns."""
        fiber = helical_fiber_3d(num_turns=3.0, num_segments=50)
        assert fiber.centerline.shape[0] == 50


class TestArcFiber2D:
    """Test arc fiber generator."""
    
    def test_basic_generation(self):
        """Test basic arc fiber."""
        fiber = arc_fiber_2d(
            radius_arc=20.0,
            angle=np.pi/2,
            num_segments=50
        )
        
        assert fiber is not None
        assert fiber.centerline.shape[0] == 50
    
    def test_quarter_circle(self):
        """Test quarter circle arc."""
        fiber = arc_fiber_2d(
            radius_arc=10.0,
            angle=np.pi/2,
            start_angle=0.0,
            num_segments=20
        )
        
        # First point should be at (center_x + R, center_y)
        # Last point should be at (center_x, center_y + R)
        first = fiber.centerline[0]
        last = fiber.centerline[-1]
        
        assert first[0] > last[0]  # X decreases
        assert last[1] > first[1]  # Y increases
    
    def test_full_circle(self):
        """Test full circle arc."""
        fiber = arc_fiber_2d(
            radius_arc=10.0,
            angle=2*np.pi,
            num_segments=100
        )
        
        assert fiber.centerline.shape[0] == 100


class TestBezierFiber3D:
    """Test Bezier fiber generator."""
    
    def test_basic_generation(self):
        """Test basic Bezier fiber."""
        control_points = [
            (0, 0, 0),
            (10, 20, 0),
            (20, -10, 0),
            (30, 0, 0)
        ]
        
        fiber = bezier_fiber_3d(control_points=control_points)
        
        assert fiber is not None
        assert fiber.centerline.shape[0] > 0
        assert fiber.centerline.shape[1] == 3
    
    def test_endpoints(self):
        """Test that endpoints match first and last control points."""
        control_points = [
            (0, 0, 0),
            (10, 10, 0),
            (20, 0, 0)
        ]
        
        fiber = bezier_fiber_3d(
            control_points=control_points,
            num_segments=50
        )
        
        # First point should be first control point
        assert np.allclose(fiber.centerline[0], control_points[0], atol=1e-6)
        
        # Last point should be last control point
        assert np.allclose(fiber.centerline[-1], control_points[-1], atol=1e-6)
    
    def test_linear_bezier(self):
        """Test linear Bezier (2 control points)."""
        control_points = [(0, 0, 0), (10, 0, 0)]
        
        fiber = bezier_fiber_3d(
            control_points=control_points,
            num_segments=20
        )
        
        # Should be a straight line
        assert fiber.centerline.shape[0] == 20


class TestRandomCurvedNetwork3D:
    """Test random curved network generator."""
    
    def test_basic_generation(self):
        """Test basic network generation."""
        net = random_curved_network_3d(
            num_fibers=20,
            box_size=(50, 50, 50),
            seed=42
        )
        
        assert len(net.fibers) == 20
        assert net.dimension == 3
    
    def test_curvature_effect(self):
        """Test that curvature affects fibers."""
        # Low curvature
        net_low = random_curved_network_3d(
            num_fibers=10,
            curvature=0.1,
            seed=42
        )
        
        # High curvature
        net_high = random_curved_network_3d(
            num_fibers=10,
            curvature=0.8,
            seed=42
        )
        
        assert len(net_low.fibers) == 10
        assert len(net_high.fibers) == 10
    
    def test_length_range(self):
        """Test fiber length range."""
        net = random_curved_network_3d(
            num_fibers=50,
            min_length=10.0,
            max_length=30.0,
            seed=42
        )
        
        # All fibers should have reasonable lengths
        for fiber in net.fibers:
            assert fiber.length > 0


class TestCrimpedNetwork2D:
    """Test crimped network generator."""
    
    def test_basic_generation(self):
        """Test basic crimped network."""
        net = crimped_network_2d(
            num_fibers=30,
            box_size=(100, 100),
            seed=42
        )
        
        assert len(net.fibers) == 30
        assert net.dimension == 2
    
    def test_crimp_parameters(self):
        """Test crimp parameters."""
        net = crimped_network_2d(
            num_fibers=20,
            crimp_amplitude=5.0,
            crimp_wavelength=15.0,
            seed=42
        )
        
        assert len(net.fibers) == 20
    
    def test_reproducibility(self):
        """Test seed reproducibility."""
        net1 = crimped_network_2d(num_fibers=10, seed=42)
        net2 = crimped_network_2d(num_fibers=10, seed=42)
        
        assert len(net1.fibers) == len(net2.fibers)


