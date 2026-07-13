"""Tests for high-level convenience API."""

import numpy as np
import pytest
import tempfile
from fibernet import api
from fibernet.core.network import FiberNetwork


class TestCreate:
    def test_create_random_2d(self):
        net = api.create("random_2d", num_fibers=10, fiber_length=5, box_size=(20, 20))
        assert isinstance(net, FiberNetwork)
        assert net.num_fibers == 10
    
    def test_create_square_2d(self):
        net = api.create("square_2d", grid_size=(3, 3))
        assert isinstance(net, FiberNetwork)
        assert net.num_fibers > 0
    
    def test_create_helix(self):
        net = api.create("helix", num_turns=5)
        assert isinstance(net, FiberNetwork)
        assert net.num_fibers > 0
    
    def test_invalid_generator(self):
        with pytest.raises(ValueError):
            api.create("invalid_generator")


class TestTransforms:
    def test_mirror(self):
        net = api.create("square_2d", grid_size=(2, 2))
        mirrored = api.mirror(net, axis=0)
        assert mirrored.num_fibers == net.num_fibers
    
    def test_rotate(self):
        net = api.create("square_2d", grid_size=(2, 2))
        rotated = api.rotate(net, angle=np.pi/4)
        assert rotated.num_fibers == net.num_fibers
    
    def test_scale(self):
        net = api.create("square_2d", grid_size=(2, 2))
        scaled = api.scale(net, factor=2.0)
        assert scaled.num_fibers == net.num_fibers
    
    def test_translate(self):
        net = api.create("square_2d", grid_size=(2, 2))
        translated = api.translate(net, offset=[10, 10, 0])
        assert translated.num_fibers == net.num_fibers
    
    def test_merge(self):
        net1 = api.create("square_2d", grid_size=(2, 2))
        net2 = api.create("square_2d", grid_size=(2, 2))
        merged = api.merge([net1, net2])
        assert merged.num_fibers == net1.num_fibers + net2.num_fibers
    
    def test_tile(self):
        net = api.create("square_2d", grid_size=(2, 2))
        tiled = api.tile(net, repeats=(2, 2, 1))
        assert tiled.num_fibers == net.num_fibers * 4


class TestAnalyze:
    def test_analyze_basic(self):
        net = api.create("random_2d", num_fibers=20, fiber_length=5, box_size=(20, 20))
        results = api.analyze(net)
        
        assert 'num_fibers' in results
        assert 'num_crosslinks' in results
        assert 'nematic_order' in results
        assert 'mean_length' in results
        assert results['num_fibers'] == 20


class TestSimulate:
    def test_simulate_mechanics_linear(self):
        net = api.create("square_2d", grid_size=(3, 3))
        results = api.simulate_mechanics(net, strain=0.001, model="linear")
        
        assert 'modulus' in results
        assert 'max_stress' in results
        assert 'energy' in results
        assert results['modulus'] > 0
    
    def test_simulate_thermal(self):
        net = api.create("square_2d", grid_size=(3, 3))
        results = api.simulate_thermal(net, T_hot=100, T_cold=0)
        
        assert 'conductivity' in results
        assert 'temperatures' in results


class TestExportImport:
    def test_export_json(self):
        net = api.create("square_2d", grid_size=(2, 2))
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filename = f.name
        
        try:
            api.export(net, filename)
            loaded = api.load(filename)
            assert loaded.num_fibers == net.num_fibers
        finally:
            import os
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_export_lammps(self):
        net = api.create("square_2d", grid_size=(2, 2))
        
        with tempfile.NamedTemporaryFile(suffix='.lammps', delete=False) as f:
            filename = f.name
        
        try:
            api.export(net, filename)
            import os
            assert os.path.exists(filename)
        finally:
            import os
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_export_vtk(self):
        net = api.create("square_2d", grid_size=(2, 2))
        
        with tempfile.NamedTemporaryFile(suffix='.vtk', delete=False) as f:
            filename = f.name
        
        try:
            api.export(net, filename)
            import os
            assert os.path.exists(filename)
        finally:
            import os
            if os.path.exists(filename):
                os.remove(filename)
