"""Tests for copy utilities."""

import numpy as np
import pytest
from fibernet import gen
from fibernet.core.copy_utils import copy_network, copy_fiber, copy_material


class TestCopyNetwork:
    def test_deep_copy(self):
        net = gen.random_straight_2d(num_fibers=10, fiber_length=8, box_size=(25, 25), seed=42)
        net_copy = copy_network(net)
        
        assert net_copy.num_fibers == net.num_fibers
        assert net_copy.num_crosslinks == net.num_crosslinks
        
        # Verify deep copy
        net_copy.fibers[0].centerline[0, 0] = 999
        assert net.fibers[0].centerline[0, 0] != 999
    
    def test_empty_network(self):
        from fibernet.core.network import FiberNetwork
        net = FiberNetwork()
        net_copy = copy_network(net)
        assert net_copy.num_fibers == 0
    
    def test_3d_network(self):
        net = gen.random_straight_3d(num_fibers=5, fiber_length=5, box_size=(20, 20, 20), seed=42)
        net_copy = copy_network(net)
        assert net_copy.num_fibers == net.num_fibers
        assert net_copy.dimension == 3


class TestCopyFiber:
    def test_deep_copy(self):
        net = gen.random_straight_2d(num_fibers=3, fiber_length=8, box_size=(20, 20), seed=42)
        fiber = net.fibers[0]
        
        fiber_copy = copy_fiber(fiber)
        assert fiber_copy.fiber_id == fiber.fiber_id
        assert fiber_copy.radius == fiber.radius
        np.testing.assert_array_equal(fiber_copy.centerline, fiber.centerline)
        
        # Verify deep copy
        fiber_copy.centerline[0, 0] = 999
        assert fiber.centerline[0, 0] != 999
    
    def test_new_id(self):
        net = gen.random_straight_2d(num_fibers=3, fiber_length=8, box_size=(20, 20), seed=42)
        fiber_copy = copy_fiber(net.fibers[0], new_id=99)
        assert fiber_copy.fiber_id == 99


class TestCopyMaterial:
    def test_deep_copy(self):
        net = gen.random_straight_2d(num_fibers=3, fiber_length=8, box_size=(20, 20), seed=42)
        mat = net.fibers[0].material
        
        mat_copy = copy_material(mat)
        assert mat_copy.name == mat.name
        assert mat_copy.youngs_modulus == mat.youngs_modulus
