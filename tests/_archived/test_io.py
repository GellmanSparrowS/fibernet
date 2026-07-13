"""Tests for I/O interoperability module."""

import numpy as np
import tempfile
import os
import pytest
from fibernet.gen import square_lattice_2d, random_straight_2d, single_helix
from fibernet.io import to_lammps, from_lammps, to_vtk, to_xyz, to_pdb, from_pdb, to_gmsh


class TestLAMMPS:
    def test_export_basic(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lammps', delete=False) as f:
            filename = f.name
        
        try:
            to_lammps(net, filename)
            assert os.path.exists(filename)
            
            with open(filename, 'r') as f:
                content = f.read()
                assert 'atoms' in content
                assert 'bonds' in content
                assert 'Atoms' in content
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_export_with_bead_spacing(self):
        net = random_straight_2d(20, 10, (30, 30), seed=42)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lammps', delete=False) as f:
            filename = f.name
        
        try:
            to_lammps(net, filename, bead_spacing=2.0)
            assert os.path.exists(filename)
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_import_basic(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lammps', delete=False) as f:
            filename = f.name
        
        try:
            to_lammps(net, filename, bead_spacing=3.0)
            imported = from_lammps(filename, bead_spacing=3.0)
            
            assert imported.num_fibers > 0
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestVTK:
    def test_export_legacy(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vtk', delete=False) as f:
            filename = f.name
        
        try:
            to_vtk(net, filename)
            assert os.path.exists(filename)
            
            with open(filename, 'r') as f:
                content = f.read()
                assert 'vtk DataFile Version' in content
                assert 'POINTS' in content
                assert 'LINES' in content
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_export_with_data(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vtk', delete=False) as f:
            filename = f.name
        
        try:
            # Create dummy point data
            num_points = sum(len(fiber.centerline) for fiber in net.fibers)
            point_data = {'temperature': np.random.rand(num_points)}
            
            to_vtk(net, filename, point_data=point_data)
            assert os.path.exists(filename)
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestXYZ:
    def test_export_basic(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            filename = f.name
        
        try:
            to_xyz(net, filename)
            assert os.path.exists(filename)
            
            with open(filename, 'r') as f:
                lines = f.readlines()
                assert len(lines) >= 3
                num_atoms = int(lines[0].strip())
                assert num_atoms > 0
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestPDB:
    def test_export_basic(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            filename = f.name
        
        try:
            to_pdb(net, filename)
            assert os.path.exists(filename)
            
            with open(filename, 'r') as f:
                content = f.read()
                assert 'HEADER' in content
                assert 'ATOM' in content
                assert 'END' in content
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_import_basic(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            filename = f.name
        
        try:
            to_pdb(net, filename, bead_spacing=3.0)
            imported = from_pdb(filename, bead_radius=0.5)
            
            assert imported.num_fibers > 0
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestGMSH:
    def test_export_basic(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.msh', delete=False) as f:
            filename = f.name
        
        try:
            to_gmsh(net, filename, segments_per_fiber=3)
            assert os.path.exists(filename)
            
            with open(filename, 'r') as f:
                content = f.read()
                assert '$MeshFormat' in content
                assert '$Nodes' in content
                assert '$Elements' in content
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestIOIntegration:
    def test_roundtrip_lammps(self):
        net = random_straight_2d(30, 10, (30, 30), seed=42)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lammps', delete=False) as f:
            filename = f.name
        
        try:
            to_lammps(net, filename, bead_spacing=2.5)
            imported = from_lammps(filename, bead_spacing=2.5)
            
            # Should have similar number of fibers
            assert abs(imported.num_fibers - net.num_fibers) < 10
        finally:
            if os.path.exists(filename):
                os.remove(filename)
