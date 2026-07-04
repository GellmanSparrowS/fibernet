"""Tests for fractal network generators."""

import pytest
import numpy as np
from fibernet.gen.fractal import (
    sierpinski_triangle,
    koch_curve,
    fractal_tree,
    hilbert_curve,
)


class TestSierpinskiTriangle:
    def test_basic(self):
        net = sierpinski_triangle(iterations=2, size=10.0)
        assert net.num_fibers > 0
        assert net.dimension == 2
    
    def test_iteration_count(self):
        # 3^0 = 1 triangle -> 3 fibers at depth 0
        # 3^1 = 3 triangles -> 9 fibers at depth 1
        # 3^2 = 9 triangles -> 27 fibers at depth 2
        net0 = sierpinski_triangle(iterations=0)
        assert net0.num_fibers == 3
        net1 = sierpinski_triangle(iterations=1)
        assert net1.num_fibers == 9
        net2 = sierpinski_triangle(iterations=2)
        assert net2.num_fibers == 27
    
    def test_different_sizes(self):
        net_small = sierpinski_triangle(iterations=2, size=5.0)
        net_large = sierpinski_triangle(iterations=2, size=20.0)
        assert net_small.num_fibers == net_large.num_fibers
    
    def test_with_radius(self):
        net = sierpinski_triangle(iterations=1, radius=0.5)
        assert all(f.radius == 0.5 for f in net.fibers)


class TestKochCurve:
    def test_basic(self):
        net = koch_curve(iterations=2)
        assert net.num_fibers > 0
        assert net.dimension == 2
    
    def test_iteration_count(self):
        # 4^0 = 1 segment at depth 0
        # 4^1 = 4 segments at depth 1
        # 4^2 = 16 segments at depth 2
        # 4^3 = 64 segments at depth 3
        net3 = koch_curve(iterations=3)
        assert net3.num_fibers == 64
    
    def test_custom_endpoints(self):
        net = koch_curve(iterations=1, start=(0, 0), end=(5, 5))
        assert net.num_fibers > 0


class TestFractalTree:
    def test_basic(self):
        net = fractal_tree(iterations=3, trunk_length=10.0)
        assert net.num_fibers > 0
        assert net.dimension == 2
    
    def test_iteration_count(self):
        # Binary tree: 2^n - 1 nodes (edges)
        net3 = fractal_tree(iterations=3)
        assert net3.num_fibers == 7  # 2^3 - 1 = 7
        net4 = fractal_tree(iterations=4)
        assert net4.num_fibers == 15  # 2^4 - 1 = 15
    
    def test_branch_ratio(self):
        net = fractal_tree(iterations=3, branch_ratio=0.5)
        assert net.num_fibers == 7
    
    def test_different_angles(self):
        net = fractal_tree(iterations=3, branch_angle=np.pi / 4)
        assert net.num_fibers == 7


class TestHilbertCurve:
    def test_basic(self):
        net = hilbert_curve(order=2, size=10.0)
        assert net.num_fibers > 0
        assert net.dimension == 2
    
    def test_order_count(self):
        # n=2^order, fibers = n^2 - 1
        net1 = hilbert_curve(order=1)
        assert net1.num_fibers == 3  # 2^2 - 1
        net2 = hilbert_curve(order=2)
        assert net2.num_fibers == 15  # 4^2 - 1
        net3 = hilbert_curve(order=3)
        assert net3.num_fibers == 63  # 8^2 - 1
