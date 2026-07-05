"""Tests for PyVista integration."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

# Set environment variables for headless rendering
os.environ['PYVISTA_OFF_SCREEN'] = 'true'

from fibernet import gen
from fibernet.pyvista_viz import PyVistaVisualizer, visualize_network_3d, PYVISTA_AVAILABLE


@pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista not available")
class TestPyVistaVisualizer:
    """Test PyVista visualizer."""
    
    def test_initialization(self):
        """Test visualizer initialization."""
        net = gen.random_straight_3d(num_fibers=10, box_size=(20, 20, 20), seed=42)
        viz = PyVistaVisualizer(net)
        
        assert viz.network == net
        assert viz.mesh is not None
    
    def test_build_mesh(self):
        """Test mesh building."""
        net = gen.random_straight_3d(num_fibers=10, box_size=(20, 20, 20), seed=42)
        viz = PyVistaVisualizer(net)
        
        # Check that mesh was created
        assert viz.mesh is not None
        assert len(viz.mesh.cell_data['length']) == 10
    
    @pytest.mark.skipif(os.environ.get("CI") == "true", reason="VTK segfault in CI")
    def test_save_screenshot(self):
        """Test saving screenshot."""
        net = gen.random_straight_3d(num_fibers=10, box_size=(20, 20, 20), seed=42)
        viz = PyVistaVisualizer(net)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            filename = f.name
        
        try:
            viz.save_screenshot(filename, window_size=(400, 300))
            assert Path(filename).exists()
            assert Path(filename).stat().st_size > 0
        finally:
            Path(filename).unlink(missing_ok=True)
    
    def test_export_vtk(self):
        """Test VTK export."""
        net = gen.random_straight_3d(num_fibers=10, box_size=(20, 20, 20), seed=42)
        viz = PyVistaVisualizer(net)
        
        with tempfile.NamedTemporaryFile(suffix='.vtk', delete=False) as f:
            filename = f.name
        
        try:
            viz.export_vtk(filename)
            assert Path(filename).exists()
            assert Path(filename).stat().st_size > 0
        finally:
            Path(filename).unlink(missing_ok=True)
    
    def test_color_by_length(self):
        """Test coloring by length."""
        net = gen.random_straight_3d(num_fibers=10, box_size=(20, 20, 20), seed=42)
        viz = PyVistaVisualizer(net)
        
        # Should not raise error
        viz.color_by_property('length', colormap='viridis')
        
        # Check that data was added
        assert 'length' in viz.mesh.cell_data
    
    def test_color_by_radius(self):
        """Test coloring by radius."""
        net = gen.random_straight_3d(num_fibers=10, box_size=(20, 20, 20), seed=42)
        viz = PyVistaVisualizer(net)
        
        # Should not raise error
        viz.color_by_property('radius', colormap='plasma')
        
        # Check that data was added
        assert 'radius' in viz.mesh.cell_data
    
    def test_color_by_orientation(self):
        """Test coloring by orientation."""
        net = gen.random_straight_3d(num_fibers=10, box_size=(20, 20, 20), seed=42)
        viz = PyVistaVisualizer(net)
        
        # Should not raise error
        viz.color_by_property('orientation', colormap='hsv')
        
        # Check that data was added
        assert 'orientation' in viz.mesh.cell_data
    
    def test_color_by_invalid_property(self):
        """Test coloring by invalid property."""
        net = gen.random_straight_3d(num_fibers=10, box_size=(20, 20, 20), seed=42)
        viz = PyVistaVisualizer(net)
        
        with pytest.raises(ValueError):
            viz.color_by_property('invalid_property')


@pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista not available")
class TestPyVistaAvailability:
    """Test PyVista availability check."""
    
    def test_pyvista_available(self):
        """Test that PyVista is available."""
        assert PYVISTA_AVAILABLE
    
    def test_import_pyvista(self):
        """Test importing PyVista."""
        import pyvista as pv
        assert pv is not None


