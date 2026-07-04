"""Tests for trimesh integration."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from fibernet import gen

try:
    import trimesh
    from fibernet.trimesh_integration import (
        TrimeshConverter,
        network_to_trimesh,
        analyze_mesh_properties,
        boolean_operation,
        repair_mesh,
        simplify_mesh,
        TRIMESH_AVAILABLE
    )
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


# Check for optional dependencies
try:
    import manifold3d
    MANIFOLD_AVAILABLE = True
except ImportError:
    MANIFOLD_AVAILABLE = False

try:
    import fast_simplification
    FAST_SIMPLIFICATION_AVAILABLE = True
except ImportError:
    FAST_SIMPLIFICATION_AVAILABLE = False


@pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="Trimesh not available")
class TestTrimeshConverter:
    """Test Trimesh converter."""
    
    def test_initialization(self):
        """Test converter initialization."""
        net = gen.random_straight_3d(num_fibers=10, box_size=(20, 20, 20), seed=42)
        converter = TrimeshConverter(net)
        
        assert converter.network == net
        assert converter.segments_per_fiber == 8
        assert converter.radial_segments == 6
    
    def test_to_mesh(self):
        """Test conversion to mesh."""
        net = gen.random_straight_3d(num_fibers=10, box_size=(20, 20, 20), seed=42)
        converter = TrimeshConverter(net)
        
        mesh = converter.to_mesh(merge=True)
        
        assert mesh is not None
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
    
    def test_to_mesh_unmerged(self):
        """Test conversion to unmerged meshes."""
        net = gen.random_straight_3d(num_fibers=10, box_size=(20, 20, 20), seed=42)
        converter = TrimeshConverter(net)
        
        meshes = converter.to_mesh(merge=False)
        
        assert isinstance(meshes, list)
        assert len(meshes) == 10
        assert all(isinstance(m, trimesh.Trimesh) for m in meshes)
    
    def test_to_scene(self):
        """Test conversion to scene."""
        net = gen.random_straight_3d(num_fibers=10, box_size=(20, 20, 20), seed=42)
        converter = TrimeshConverter(net)
        
        scene = converter.to_scene()
        
        assert isinstance(scene, trimesh.Scene)
        assert len(scene.geometry) == 10


@pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="Trimesh not available")
class TestNetworkToTrimesh:
    """Test network_to_trimesh function."""
    
    def test_basic_conversion(self):
        """Test basic conversion."""
        net = gen.random_straight_3d(num_fibers=10, box_size=(20, 20, 20), seed=42)
        mesh = network_to_trimesh(net)
        
        assert mesh is not None
        assert isinstance(mesh, trimesh.Trimesh)
    
    def test_with_parameters(self):
        """Test conversion with custom parameters."""
        net = gen.random_straight_3d(num_fibers=10, box_size=(20, 20, 20), seed=42)
        mesh = network_to_trimesh(net, radial_segments=8)
        
        assert mesh is not None


@pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="Trimesh not available")
class TestAnalyzeMeshProperties:
    """Test mesh property analysis."""
    
    def test_analyze_properties(self):
        """Test analyzing mesh properties."""
        net = gen.random_straight_3d(num_fibers=10, box_size=(20, 20, 20), seed=42)
        mesh = network_to_trimesh(net)
        
        props = analyze_mesh_properties(mesh)
        
        assert 'volume' in props
        assert 'surface_area' in props
        assert 'bounds' in props
        assert 'centroid' in props
        assert 'num_vertices' in props
        assert 'num_faces' in props
        
        assert props['volume'] > 0
        assert props['surface_area'] > 0
        assert props['num_vertices'] > 0
        assert props['num_faces'] > 0


@pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="Trimesh not available")
class TestBooleanOperation:
    """Test boolean operations."""
    
    @pytest.mark.skipif(not MANIFOLD_AVAILABLE, reason="manifold3d not available")
    def test_union(self):
        """Test union operation."""
        net1 = gen.random_straight_3d(num_fibers=5, box_size=(20, 20, 20), seed=42)
        net2 = gen.random_straight_3d(num_fibers=5, box_size=(20, 20, 20), seed=43)
        
        mesh1 = network_to_trimesh(net1)
        mesh2 = network_to_trimesh(net2)
        
        union = boolean_operation(mesh1, mesh2, 'union')
        
        assert union is not None
        assert isinstance(union, trimesh.Trimesh)
    
    def test_invalid_operation(self):
        """Test invalid operation."""
        net = gen.random_straight_3d(num_fibers=5, box_size=(20, 20, 20), seed=42)
        mesh = network_to_trimesh(net)
        
        with pytest.raises(ValueError):
            boolean_operation(mesh, mesh, 'invalid')


@pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="Trimesh not available")
class TestRepairMesh:
    """Test mesh repair."""
    
    def test_repair(self):
        """Test mesh repair."""
        net = gen.random_straight_3d(num_fibers=10, box_size=(20, 20, 20), seed=42)
        mesh = network_to_trimesh(net)
        
        repaired = repair_mesh(mesh)
        
        assert repaired is not None
        assert isinstance(repaired, trimesh.Trimesh)


@pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="Trimesh not available")
class TestSimplifyMesh:
    """Test mesh simplification."""
    
    @pytest.mark.skipif(not FAST_SIMPLIFICATION_AVAILABLE, reason="fast_simplification not available")
    def test_simplify(self):
        """Test mesh simplification."""
        net = gen.random_straight_3d(num_fibers=10, box_size=(20, 20, 20), seed=42)
        mesh = network_to_trimesh(net, radial_segments=12)
        
        original_faces = len(mesh.faces)
        target_faces = min(100, original_faces // 2)
        
        simplified = simplify_mesh(mesh, target_faces)
        
        assert simplified is not None
        assert len(simplified.faces) <= original_faces


