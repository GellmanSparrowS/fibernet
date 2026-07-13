"""Tests for periodic boundary conditions."""

import pytest
import numpy as np
from fibernet import gen
from fibernet.sim.periodic import (
    PeriodicBoundary, create_periodic_network, 
    apply_periodic_strain, homogenize_properties
)


class TestPeriodicBoundary:
    def test_creation_2d(self):
        """Test creating 2D periodic boundary."""
        pb = PeriodicBoundary(box_size=(10, 10), dimension=2)
        assert pb.dimension == 2
        assert np.allclose(pb.box_size[:2], [10, 10])
    
    def test_creation_3d(self):
        """Test creating 3D periodic boundary."""
        pb = PeriodicBoundary(box_size=(10, 10, 10), dimension=3)
        assert pb.dimension == 3
        assert np.allclose(pb.box_size, [10, 10, 10])
    
    def test_wrap_position(self):
        """Test position wrapping."""
        pb = PeriodicBoundary(box_size=(10, 10), dimension=2)
        
        pos = np.array([12, -3, 0])
        wrapped = pb.wrap_position(pos)
        
        assert 0 <= wrapped[0] <= 10
        assert 0 <= wrapped[1] <= 10
    
    def test_wrap_position_3d(self):
        """Test 3D position wrapping."""
        pb = PeriodicBoundary(box_size=(10, 10, 10), dimension=3)
        
        pos = np.array([15, -5, 25])
        wrapped = pb.wrap_position(pos)
        
        assert 0 <= wrapped[0] <= 10
        assert 0 <= wrapped[1] <= 10
        assert 0 <= wrapped[2] <= 10
    
    def test_minimum_image_distance(self):
        """Test minimum image distance."""
        pb = PeriodicBoundary(box_size=(10, 10), dimension=2)
        
        pos1 = np.array([1, 1, 0])
        pos2 = np.array([9, 9, 0])
        
        delta = pb.minimum_image_distance(pos1, pos2)
        dist = pb.minimum_image_norm(pos1, pos2)
        
        # Should go through boundary: sqrt(2^2 + 2^2) = 2.828
        assert abs(dist - 2.828) < 0.1
    
    def test_minimum_image_direct(self):
        """Test minimum image when direct is shorter."""
        pb = PeriodicBoundary(box_size=(10, 10), dimension=2)
        
        pos1 = np.array([4, 4, 0])
        pos2 = np.array([6, 6, 0])
        
        dist = pb.minimum_image_norm(pos1, pos2)
        
        # Direct: sqrt(2^2 + 2^2) = 2.828
        assert abs(dist - 2.828) < 0.1
    
    def test_get_images_2d(self):
        """Test getting periodic images in 2D."""
        pb = PeriodicBoundary(box_size=(10, 10), dimension=2)
        
        images = pb.get_images(np.array([5, 5, 0]), n_images=1)
        
        # Should have 8 images (3x3 - 1)
        assert len(images) == 8
    
    def test_get_images_3d(self):
        """Test getting periodic images in 3D."""
        pb = PeriodicBoundary(box_size=(10, 10, 10), dimension=3)
        
        images = pb.get_images(np.array([5, 5, 5]), n_images=1)
        
        # Should have 26 images (3x3x3 - 1)
        assert len(images) == 26


class TestPeriodicNetwork:
    def test_create_periodic_network(self):
        """Test creating periodic network."""
        net = gen.random_straight_2d(num_fibers=20, fiber_length=5.0, box_size=(20, 20), seed=42)
        
        periodic_net, pb = create_periodic_network(net, box_size=(20, 20))
        
        assert periodic_net.num_fibers == net.num_fibers
        assert isinstance(pb, PeriodicBoundary)
    
    def test_find_cross_boundary_crosslinks(self):
        """Test finding cross-boundary crosslinks."""
        net = gen.random_straight_2d(num_fibers=30, fiber_length=5.0, box_size=(20, 20), seed=42)
        pb = PeriodicBoundary(box_size=(20, 20), dimension=2)
        
        cross_boundary = pb.find_cross_boundary_crosslinks(net)
        
        # Should be a list (may be empty or not)
        assert isinstance(cross_boundary, list)


class TestHomogenization:
    def test_homogenize_mechanical(self):
        """Test homogenizing mechanical properties."""
        homo = homogenize_properties(
            gen.random_straight_2d,
            box_size=(30, 30),
            num_realizations=2,
            property_type='mechanical',
            base_seed=42,
            num_fibers=20,
            fiber_length=10.0,
        )
        
        assert homo['num_realizations'] == 2
        assert 'box_size' in homo
    
    def test_homogenize_thermal(self):
        """Test homogenizing thermal properties."""
        homo = homogenize_properties(
            gen.random_straight_2d,
            box_size=(30, 30),
            num_realizations=2,
            property_type='thermal',
            base_seed=42,
            num_fibers=20,
            fiber_length=10.0,
        )
        
        assert homo['num_realizations'] == 2
