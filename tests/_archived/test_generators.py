"""Tests for fiber network generators."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fibernet.gen import disordered, ordered, chiral, woven, hierarchical


class TestDisordered:
    def test_random_2d(self):
        net = disordered.random_straight_2d(num_fibers=50, fiber_length=8, box_size=(30, 30), seed=42)
        assert net.num_fibers == 50
        assert net.dimension == 2
    
    def test_random_3d(self):
        net = disordered.random_straight_3d(num_fibers=30, fiber_length=10, box_size=(20, 20, 20), seed=42)
        assert net.num_fibers == 30
        assert net.dimension == 3
    
    def test_random_3d_oriented(self):
        net = disordered.random_straight_3d(
            num_fibers=20, orientation_bias=[1, 0, 0], orientation_spread=0.1, seed=42
        )
        assert net.num_fibers == 20
    
    def test_random_walk(self):
        net = disordered.random_walk_fibers(num_fibers=10, num_steps=50, seed=42)
        assert net.num_fibers == 10
    
    def test_oriented_2d(self):
        net = disordered.oriented_random_2d(num_fibers=30, preferred_angle=0.0, angular_spread=0.1, seed=42)
        assert net.num_fibers == 30


class TestOrdered:
    def test_square_lattice(self):
        net = ordered.square_lattice_2d(spacing=5, grid_size=(4, 4))
        assert net.num_fibers > 0
        assert net.num_crosslinks > 0
    
    def test_honeycomb(self):
        net = ordered.honeycomb_lattice_2d(cell_size=5, grid_size=(3, 3))
        assert net.num_fibers > 0
    
    def test_cubic_3d(self):
        net = ordered.cubic_lattice_3d(spacing=5, grid_size=(2, 2, 2))
        assert net.num_fibers > 0
    
    def test_octet_truss(self):
        net = ordered.octet_truss_3d(spacing=5, grid_size=(2, 2, 2))
        assert net.num_fibers > 0


class TestChiral:
    def test_single_helix(self):
        net = chiral.single_helix(helix_radius=3, pitch=2, num_turns=3)
        assert net.num_fibers == 1
        assert net.fibers[0].tortuosity() > 1.0
    
    def test_double_helix(self):
        net = chiral.double_helix(helix_radius=3, pitch=2, num_turns=3)
        assert net.num_fibers == 2
        assert net.num_crosslinks > 0
    
    def test_braided_rope(self):
        net = chiral.braided_rope(num_strands=3, rope_radius=2, num_turns=2)
        assert net.num_fibers == 3
    
    def test_twisted_bundle(self):
        net = chiral.twisted_bundle(num_fibers=7, twist_angle=np.pi/4, total_length=20)
        assert net.num_fibers == 7


class TestWoven:
    def test_plain_weave(self):
        net = woven.plain_weave_2d(spacing=2, grid_size=(5, 5), radius=0.05)
        assert net.num_fibers == 10
    
    def test_twill_weave(self):
        net = woven.twill_weave_2d(spacing=2, grid_size=(5, 5))
        assert net.num_fibers == 10
    
    def test_3d_woven(self):
        net = woven.woven_3d_orthogonal(spacing=3, grid_size=(2, 2, 2))
        assert net.num_fibers > 0


class TestHierarchical:
    def test_gradient_density(self):
        net = hierarchical.gradient_density_network(num_fibers=50, seed=42)
        assert net.num_fibers > 0
    
    def test_core_shell(self):
        net = hierarchical.core_shell_fiber(num_shell_fibers=6)
        assert net.num_fibers == 7
    
    def test_fractal(self):
        net = hierarchical.fractal_network(iterations=2, branch_factor=2)
        assert net.num_fibers > 0
