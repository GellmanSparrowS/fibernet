"""Tests for visualization module."""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

from fibernet import gen, viz


class TestMatplotlibVisualization:
    """Test matplotlib-based visualizations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.network = gen.random_straight_2d(
            num_fibers=20,
            fiber_length=10.0,
            box_size=(30, 30),
            seed=42
        )
    
    def test_visualize_3d_basic(self):
        """Test basic 3D visualization."""
        fig, ax = viz.visualize_3d_matplotlib(self.network)
        assert fig is not None
        assert ax is not None
    
    def test_visualize_3d_with_options(self):
        """Test 3D visualization with custom options."""
        fig, ax = viz.visualize_3d_matplotlib(
            self.network,
            color='red',
            linewidth=2.0,
            alpha=0.8,
            show_crosslinks=True,
            crosslink_color='blue',
            title='Test Network'
        )
        assert fig is not None
    
    def test_visualize_3d_without_crosslinks(self):
        """Test 3D visualization without crosslinks."""
        fig, ax = viz.visualize_3d_matplotlib(
            self.network,
            show_crosslinks=False
        )
        assert fig is not None
    
    def test_visualize_network_stress(self):
        """Test stress visualization."""
        stress_values = np.random.uniform(0, 1, self.network.num_fibers)
        fig, ax = viz.visualize_network_stress(
            self.network,
            stress_values,
            cmap='coolwarm'
        )
        assert fig is not None
    
    def test_visualize_network_stress_with_options(self):
        """Test stress visualization with options."""
        stress_values = np.random.uniform(0, 1, self.network.num_fibers)
        fig, ax = viz.visualize_network_stress(
            self.network,
            stress_values,
            linewidth=2.0,
            colorbar=True,
            title='Stress Test'
        )
        assert fig is not None
    
    def test_visualize_damage_evolution(self):
        """Test damage evolution visualization."""
        damage_result = {
            'strain': np.linspace(0, 0.1, 20),
            'stress': np.random.uniform(0, 100, 20),
            'damage': np.linspace(0, 0.8, 20),
            'broken_elements': np.linspace(0, 50, 20).astype(int),
        }
        fig = viz.visualize_damage_evolution(damage_result)
        assert fig is not None


class TestVisualizationIntegration:
    """Integration tests for visualization."""
    
    def test_visualize_different_networks(self):
        """Test visualization with different network types."""
        networks = [
            gen.random_straight_2d(num_fibers=10, fiber_length=10.0, box_size=(20, 20)),
            gen.random_straight_3d(num_fibers=10, fiber_length=10.0, box_size=(20, 20, 20)),
        ]
        
        for network in networks:
            fig, ax = viz.visualize_3d_matplotlib(network)
            assert fig is not None
