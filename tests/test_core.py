"""Tests for core data structures."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fibernet.core.material import Material, get_material, list_materials
from fibernet.core.fiber import Fiber, CrossSection
from fibernet.core.network import FiberNetwork, Crosslink


class TestMaterial:
    def test_create_basic(self):
        m = Material(name="test", density=1000, youngs_modulus=1e9)
        assert m.name == "test"
        assert m.density == 1000
        assert m.shear_modulus is not None
    
    def test_get_material(self):
        m = get_material("steel")
        assert m.name == "steel"
        assert m.density == 7800
    
    def test_list_materials(self):
        mats = list_materials()
        assert len(mats) > 10
        assert "carbon_fiber" in mats
    
    def test_serialization(self):
        m = Material(name="test", density=1234, youngs_modulus=5e9)
        d = m.to_dict()
        m2 = Material.from_dict(d)
        assert m2.density == 1234


class TestFiber:
    def test_straight_fiber(self):
        f = Fiber.straight([0, 0, 0], [10, 0, 0], radius=0.5)
        assert abs(f.length - 10.0) < 1e-6
        assert f.radius == 0.5
    
    def test_helical_fiber(self):
        f = Fiber.helical([0, 0, 1], [0, 0, 0], helix_radius=2, pitch=1, num_turns=3)
        assert f.num_points > 10
        assert f.length > 0
    
    def test_bezier_fiber(self):
        cps = np.array([[0, 0, 0], [5, 5, 0], [10, 0, 0]])
        f = Fiber.bezier(cps, radius=0.1)
        assert f.num_points == 50
    
    def test_wavy_fiber(self):
        f = Fiber.sine_wave([0, 0, 0], [10, 0, 0], amplitude=1, num_waves=3)
        assert f.tortuosity() > 1.0
    
    def test_cross_section_area(self):
        f = Fiber.straight([0, 0, 0], [1, 0, 0], radius=1.0)
        assert abs(f.cross_section_area - np.pi) < 1e-6
    
    def test_resample(self):
        f = Fiber.straight([0, 0, 0], [10, 0, 0], segments=5)
        f2 = f.resample(20)
        assert f2.num_points == 20
        assert abs(f2.length - f.length) < 0.1
    
    def test_curvature(self):
        f = Fiber.straight([0, 0, 0], [10, 0, 0])
        k = f.curvature()
        assert np.all(k < 1e-6)
    
    def test_serialization(self):
        f = Fiber.straight([0, 0, 0], [5, 5, 0], radius=0.3, fiber_id=7)
        d = f.to_dict()
        f2 = Fiber.from_dict(d)
        assert f2.fiber_id == 7
        assert abs(f2.radius - 0.3) < 1e-10
    
    def test_translate(self):
        f = Fiber.straight([0, 0, 0], [10, 0, 0])
        f2 = f.translate([5, 5, 5])
        assert abs(f2.start_point[0] - 5.0) < 1e-10


class TestFiberNetwork:
    def test_empty_network(self):
        net = FiberNetwork()
        assert net.num_fibers == 0
        assert net.total_length == 0
    
    def test_add_fibers(self):
        net = FiberNetwork()
        f1 = Fiber.straight([0, 0, 0], [10, 0, 0])
        f2 = Fiber.straight([5, -5, 0], [5, 5, 0])
        net.add_fiber(f1)
        net.add_fiber(f2)
        assert net.num_fibers == 2
        assert abs(net.total_length - 20.0) < 1e-6
    
    def test_crosslinks(self):
        net = FiberNetwork()
        net.add_fiber(Fiber.straight([0, 0, 0], [10, 0, 0]))
        net.add_fiber(Fiber.straight([5, -5, 0], [5, 5, 0]))
        net.add_crosslink(Crosslink(
            fiber_i=0, fiber_j=1, param_i=0.5, param_j=0.5,
            position=np.array([5, 0, 0]), crosslink_type="welded"
        ))
        assert net.num_crosslinks == 1
    
    def test_auto_crosslink(self):
        net = FiberNetwork()
        net.add_fiber(Fiber.straight([0, 0, 0], [10, 0, 0], radius=0.5))
        net.add_fiber(Fiber.straight([5, -5, 0], [5, 5, 0], radius=0.5))
        net.auto_crosslink(threshold=2.0)
        assert net.num_crosslinks >= 1
    
    def test_serialization(self):
        net = FiberNetwork(dimension=2)
        net.add_fiber(Fiber.straight([0, 0, 0], [10, 0, 0]))
        net.add_fiber(Fiber.straight([5, -5, 0], [5, 5, 0]))
        d = net.to_dict()
        net2 = FiberNetwork.from_dict(d)
        assert net2.num_fibers == 2
    
    def test_json_io(self):
        net = FiberNetwork(dimension=3, box_size=np.array([10, 10, 10]))
        net.add_fiber(Fiber.straight([0, 0, 0], [10, 0, 0], fiber_id=0))
        net.save_json("/tmp/test_fibernet.json")
        net2 = FiberNetwork.load_json("/tmp/test_fibernet.json")
        assert net2.num_fibers == 1
    
    def test_connectivity(self):
        net = FiberNetwork()
        net.add_fiber(Fiber.straight([0, 0, 0], [10, 0, 0]))
        net.add_fiber(Fiber.straight([5, -5, 0], [5, 5, 0]))
        net.add_fiber(Fiber.straight([0, 5, 0], [10, 5, 0]))
        net.add_crosslink(Crosslink(0, 1, 0.5, 0.5, np.array([5, 0, 0])))
        net.add_crosslink(Crosslink(0, 2, 0.5, 0.5, np.array([5, 5, 0])))
        conn = net.connectivity_matrix()
        assert conn[0, 1] == 1
        assert conn[0, 2] == 1
