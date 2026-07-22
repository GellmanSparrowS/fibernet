"""Tests for gradient network generators."""

import pytest
import numpy as np
from fibernet.gen.gradient import (
    density_gradient_2d,
    property_gradient_2d,
    multi_zone_2d,
)


class TestDensityGradient2D:
    def test_basic(self):
        net = density_gradient_2d(num_fibers=50, seed=42)
        assert net.num_fibers > 0
        assert net.dimension == 2
    
    def test_direction_x(self):
        net = density_gradient_2d(num_fibers=50, gradient_direction='x', seed=42)
        assert net.num_fibers > 0
    
    def test_direction_y(self):
        net = density_gradient_2d(num_fibers=50, gradient_direction='y', seed=42)
        assert net.num_fibers > 0
    
    def test_direction_radial(self):
        net = density_gradient_2d(num_fibers=50, gradient_direction='radial', seed=42)
        assert net.num_fibers > 0
    
    def test_profile_linear(self):
        net = density_gradient_2d(num_fibers=50, gradient_profile='linear', seed=42)
        assert net.num_fibers > 0
    
    def test_profile_exponential(self):
        net = density_gradient_2d(num_fibers=50, gradient_profile='exponential', seed=42)
        assert net.num_fibers > 0
    
    def test_profile_sinusoidal(self):
        net = density_gradient_2d(num_fibers=50, gradient_profile='sinusoidal', seed=42)
        assert net.num_fibers > 0
    
    def test_reproducibility(self):
        net1 = density_gradient_2d(num_fibers=50, seed=42)
        net2 = density_gradient_2d(num_fibers=50, seed=42)
        assert net1.num_fibers == net2.num_fibers


class TestPropertyGradient2D:
    def test_basic(self):
        net = property_gradient_2d(num_fibers=50, seed=42)
        assert net.num_fibers > 0
        assert net.dimension == 2
    
    def test_different_properties(self):
        net = property_gradient_2d(
            num_fibers=50,
            min_property=1e9,
            max_property=1e10,
            seed=42
        )
        assert net.num_fibers > 0
    
    def test_direction_x(self):
        net = property_gradient_2d(num_fibers=50, gradient_direction='x', seed=42)
        assert net.num_fibers > 0
    
    def test_direction_radial(self):
        net = property_gradient_2d(num_fibers=50, gradient_direction='radial', seed=42)
        assert net.num_fibers > 0


class TestMultiZone2D:
    def test_basic(self):
        zones = [
            {'region': (0, 0, 25, 50), 'num_fibers': 30, 'fiber_length': 8.0},
            {'region': (25, 0, 25, 50), 'num_fibers': 30, 'fiber_length': 12.0},
        ]
        net = multi_zone_2d(zones, box_size=(50, 50), seed=42)
        assert net.num_fibers == 60
    
    def test_three_zones(self):
        zones = [
            {'region': (0, 0, 20, 50), 'num_fibers': 20, 'fiber_length': 5.0},
            {'region': (20, 0, 20, 50), 'num_fibers': 30, 'fiber_length': 10.0},
            {'region': (40, 0, 20, 50), 'num_fibers': 40, 'fiber_length': 15.0},
        ]
        net = multi_zone_2d(zones, box_size=(60, 50), seed=42)
        assert net.num_fibers == 90
    
    def test_custom_radius(self):
        zones = [
            {'region': (0, 0, 50, 50), 'num_fibers': 20, 'fiber_length': 10.0, 'radius': 0.5},
        ]
        net = multi_zone_2d(zones, seed=42)
        assert net.num_fibers == 20
    
    def test_reproducibility(self):
        zones = [
            {'region': (0, 0, 25, 50), 'num_fibers': 30, 'fiber_length': 8.0},
            {'region': (25, 0, 25, 50), 'num_fibers': 30, 'fiber_length': 12.0},
        ]
        net1 = multi_zone_2d(zones, seed=42)
        net2 = multi_zone_2d(zones, seed=42)
        assert net1.num_fibers == net2.num_fibers
