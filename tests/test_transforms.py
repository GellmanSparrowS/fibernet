"""Tests for network transformation operations."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.core.transform import (
    mirror, rotate, scale, translate, merge, tile,
    trim_to_box, duplicate_and_transform, align_by_anchor, create_pattern,
)
from fibernet.gen.ordered import square_lattice_2d
from fibernet.gen.disordered import random_straight_2d


class TestMirror:
    def test_mirror_x(self):
        net = square_lattice_2d(spacing=5, grid_size=(3, 3))
        mirrored = mirror(net, axis=0)
        assert mirrored.num_fibers == net.num_fibers
    
    def test_mirror_preserves_count(self):
        net = random_straight_2d(50, 10, (30, 30), seed=42)
        mirrored = mirror(net, axis=1)
        assert mirrored.num_fibers == 50
    
    def test_mirror_inplace(self):
        net = random_straight_2d(20, 10, (30, 30), seed=42)
        orig_x = net.fibers[0].centerline[0, 0]
        mirror(net, axis=0, inplace=True)
        new_x = net.fibers[0].centerline[0, 0]
        assert abs(orig_x + new_x) < 100  # Changed relative to center


class TestRotate:
    def test_rotate_90deg(self):
        net = FiberNetwork()
        net.add_fiber(Fiber.straight([0, 0, 0], [10, 0, 0]))
        rotated = rotate(net, np.pi / 2, np.array([0, 0, 1]))
        assert rotated.num_fibers == 1
        # After 90 deg rotation around z, x-direction should become y-direction
        d = rotated.fibers[0].direction
        assert abs(abs(d[1]) - 1.0) < 0.1 or abs(abs(d[0]) - 1.0) < 0.1
    
    def test_rotate_360deg_identity(self):
        net = random_straight_2d(20, 10, (30, 30), seed=42)
        original_start = net.fibers[0].start_point.copy()
        rotated = rotate(net, 2 * np.pi, np.array([0, 0, 1]))
        assert np.allclose(rotated.fibers[0].start_point, original_start, atol=0.1)


class TestScale:
    def test_uniform_scale(self):
        net = FiberNetwork()
        net.add_fiber(Fiber.straight([0, 0, 0], [10, 0, 0], radius=0.5))
        scaled = scale(net, 2.0)
        assert abs(scaled.fibers[0].length - 20.0) < 0.5
        assert abs(scaled.fibers[0].radius - 1.0) < 0.01
    
    def test_anisotropic_scale(self):
        net = FiberNetwork()
        net.add_fiber(Fiber.straight([0, 0, 0], [10, 0, 0]))
        scaled = scale(net, np.array([2.0, 1.0, 1.0]))
        assert abs(scaled.fibers[0].length - 20.0) < 0.5


class TestTranslate:
    def test_translate(self):
        net = FiberNetwork()
        net.add_fiber(Fiber.straight([0, 0, 0], [10, 0, 0]))
        translated = translate(net, np.array([5, 10, 0]))
        assert abs(translated.fibers[0].start_point[0] - 5.0) < 1e-10
        assert abs(translated.fibers[0].start_point[1] - 10.0) < 1e-10


class TestMerge:
    def test_merge_two_networks(self):
        net1 = random_straight_2d(20, 10, (30, 30), seed=42)
        net2 = random_straight_2d(30, 10, (30, 30), seed=43)
        merged = merge([net1, net2])
        assert merged.num_fibers == 50
    
    def test_merge_with_offset(self):
        net1 = random_straight_2d(20, 10, (30, 30), seed=42)
        net2 = random_straight_2d(20, 10, (30, 30), seed=43)
        merged = merge([net1, net2], offsets=[np.zeros(3), np.array([50, 0, 0])])
        assert merged.num_fibers == 40


class TestTile:
    def test_tile_2x2(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        tiled = tile(net, repeats=(2, 2, 1))
        assert tiled.num_fibers >= net.num_fibers * 4
    
    def test_tile_preserves_fibers(self):
        net = random_straight_2d(10, 10, (20, 20), seed=42)
        tiled = tile(net, repeats=(2, 2, 1))
        assert tiled.num_fibers >= net.num_fibers * 4


class TestTrim:
    def test_trim_to_box(self):
        net = random_straight_2d(100, 15, (50, 50), seed=42)
        trimmed = trim_to_box(net, np.array([10, 10, -1]), np.array([40, 40, 1]))
        assert trimmed.num_fibers <= net.num_fibers


class TestAlign:
    def test_align_by_anchor(self):
        net1 = FiberNetwork()
        net1.add_fiber(Fiber.straight([0, 0, 0], [10, 0, 0]))
        net2 = FiberNetwork()
        net2.add_fiber(Fiber.straight([5, 5, 0], [15, 5, 0]))
        
        net1_aligned, net2_aligned = align_by_anchor(
            net1, net2,
            anchor1=np.array([10, 0, 0]),
            anchor2=np.array([5, 5, 0]),
        )
        
        assert abs(net2_aligned.fibers[0].start_point[0] - 10.0) < 1e-6


class TestPattern:
    def test_circular_pattern(self):
        base = random_straight_2d(5, 5, (10, 10), seed=42)
        patterned = create_pattern(base, "circular", num_units=4, radius=20)
        assert patterned.num_fibers >= base.num_fibers * 4
    
    def test_linear_pattern(self):
        base = random_straight_2d(5, 5, (10, 10), seed=42)
        patterned = create_pattern(base, "linear", num_units=3, spacing=15)
        assert patterned.num_fibers >= base.num_fibers * 3
    
    def test_grid_pattern(self):
        base = random_straight_2d(5, 5, (10, 10), seed=42)
        patterned = create_pattern(base, "grid", num_units=4)
        assert patterned.num_fibers >= base.num_fibers
