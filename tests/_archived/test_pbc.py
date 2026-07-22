"""Tests for periodic boundary conditions."""

import numpy as np
import pytest
from fibernet.core.pbc import PeriodicBox, apply_pbc, compute_rdf
from fibernet.core.network import FiberNetwork
from fibernet.core.fiber import Fiber


class TestPeriodicBox:
    def test_wrap_point(self):
        box = PeriodicBox([10, 10, 10])
        
        # Point inside box
        p1 = np.array([5.0, 5.0, 5.0])
        assert np.allclose(box.wrap_point(p1), p1)
        
        # Point outside box
        p2 = np.array([12.0, -3.0, 15.0])
        wrapped = box.wrap_point(p2)
        assert np.allclose(wrapped, [2.0, 7.0, 5.0])
    
    def test_minimum_image(self):
        box = PeriodicBox([10, 10, 10])
        
        # Short distance
        r1 = np.array([1.0, 1.0, 1.0])
        assert np.allclose(box.minimum_image(r1), r1)
        
        # Long distance should wrap
        r2 = np.array([8.0, 0.0, 0.0])
        assert np.allclose(box.minimum_image(r2), [-2.0, 0.0, 0.0])
    
    def test_distance(self):
        box = PeriodicBox([10, 10, 10])
        
        p1 = np.array([1.0, 1.0, 1.0])
        p2 = np.array([9.0, 1.0, 1.0])
        
        # Direct distance = 8, but minimum image = 2
        assert np.isclose(box.distance(p1, p2), 2.0)
    
    def test_volume(self):
        box = PeriodicBox([5, 10, 20])
        assert np.isclose(box.volume(), 1000.0)


class TestPBCNetwork:
    def test_apply_pbc(self):
        net = FiberNetwork(dimension=2)
        net.add_fiber(Fiber(centerline=[[0, 0, 0], [5, 0, 0]], radius=0.1))
        net.add_fiber(Fiber(centerline=[[3, 3, 0], [8, 3, 0]], radius=0.1))
        
        wrapped, box = apply_pbc(net, box_size=[6, 6, 6])
        
        assert wrapped.num_fibers == 2
        assert 'periodic' in wrapped.metadata
        assert wrapped.metadata['periodic'] is True
    
    def test_replicate(self):
        net = FiberNetwork(dimension=2)
        net.add_fiber(Fiber(centerline=[[0, 0, 0], [5, 0, 0]], radius=0.1))
        
        box = PeriodicBox([10, 10, 10])
        replicated = box.replicate(net, repeats=(2, 2, 1))
        
        # Should have 4 copies (2x2x1)
        assert replicated.num_fibers == 4


class TestRDF:
    def test_compute_rdf(self):
        net = FiberNetwork(dimension=3)
        # Add some fibers
        for i in range(5):
            net.add_fiber(Fiber(
                centerline=[[i, 0, 0], [i, 5, 0]],
                radius=0.1
            ))
        
        box = PeriodicBox([10, 10, 10])
        r, g = compute_rdf(net, box, r_max=5, num_bins=20)
        
        assert len(r) == 20
        assert len(g) == 20
        assert np.all(r >= 0)
        assert np.all(r <= 5)
