"""Tests for mesh export functionality."""

import pytest
import numpy as np
import tempfile
import os
from fibernet import gen
from fibernet.io.mesh_export import export_stl, export_obj, export_ply


class TestExportSTL:
    """Test STL export."""
    
    def test_export_ascii_stl(self):
        """Test ASCII STL export."""
        net = gen.random_straight_2d(num_fibers=10, seed=42)
        
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            filename = f.name
        
        try:
            export_stl(net, filename, binary=False)
            assert os.path.exists(filename)
            assert os.path.getsize(filename) > 0
            
            # Check file starts with 'solid'
            with open(filename, 'r') as f:
                first_line = f.readline()
                assert first_line.startswith('solid')
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_export_binary_stl(self):
        """Test binary STL export."""
        net = gen.random_straight_2d(num_fibers=10, seed=42)
        
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            filename = f.name
        
        try:
            export_stl(net, filename, binary=True)
            assert os.path.exists(filename)
            assert os.path.getsize(filename) > 0
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_export_empty_network(self):
        """Test STL export with empty network."""
        from fibernet.core.network import FiberNetwork
        net = FiberNetwork(dimension=2)
        
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            filename = f.name
        
        try:
            export_stl(net, filename)
            # Should handle gracefully (may or may not create file)
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestExportOBJ:
    """Test OBJ export."""
    
    def test_export_obj(self):
        """Test OBJ export."""
        net = gen.random_straight_2d(num_fibers=10, seed=42)
        
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
            filename = f.name
        
        try:
            export_obj(net, filename)
            assert os.path.exists(filename)
            assert os.path.getsize(filename) > 0
            
            # Check file contains vertices and faces
            with open(filename, 'r') as f:
                content = f.read()
                assert 'v ' in content  # Vertex
                assert 'f ' in content  # Face
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_export_obj_3d(self):
        """Test OBJ export for 3D network."""
        net = gen.random_straight_3d(num_fibers=10, seed=42)
        
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
            filename = f.name
        
        try:
            export_obj(net, filename)
            assert os.path.exists(filename)
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestExportPLY:
    """Test PLY export."""
    
    def test_export_ply(self):
        """Test PLY export."""
        net = gen.random_straight_2d(num_fibers=10, seed=42)
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            filename = f.name
        
        try:
            export_ply(net, filename)
            assert os.path.exists(filename)
            assert os.path.getsize(filename) > 0
            
            # Check file starts with 'ply'
            with open(filename, 'r') as f:
                first_line = f.readline()
                assert first_line.startswith('ply')
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_export_ply_3d(self):
        """Test PLY export for 3D network."""
        net = gen.random_straight_3d(num_fibers=10, seed=42)
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            filename = f.name
        
        try:
            export_ply(net, filename)
            assert os.path.exists(filename)
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestDifferentNetworks:
    """Test mesh export with different network types."""
    
    def test_square_lattice(self):
        """Test with square lattice."""
        net = gen.square_lattice_2d(spacing=2.0, grid_size=(3, 3))
        
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            filename = f.name
        
        try:
            export_stl(net, filename)
            assert os.path.exists(filename)
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_honeycomb(self):
        """Test with honeycomb lattice."""
        net = gen.honeycomb_lattice_2d(cell_size=2.0, grid_size=(3, 3))
        
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
            filename = f.name
        
        try:
            export_obj(net, filename)
            assert os.path.exists(filename)
        finally:
            if os.path.exists(filename):
                os.remove(filename)
