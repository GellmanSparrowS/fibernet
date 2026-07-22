"""
Tests for visualization module.
"""

import pytest
import numpy as np
from fibernet import gen
from fibernet.visualization import (
    NetworkVisualizer, PlotStyle, visualize_network
)

# Skip tests if matplotlib not available
matplotlib = pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt


class TestPlotStyle:
    """Test PlotStyle dataclass."""
    
    def test_default_style(self):
        style = PlotStyle()
        assert style.fiber_color == 'blue'
        assert style.fiber_linewidth == 1.0
        assert style.crosslink_color == 'red'
    
    def test_custom_style(self):
        style = PlotStyle(
            fiber_color='green',
            fiber_linewidth=2.0,
            crosslink_size=10.0,
        )
        assert style.fiber_color == 'green'
        assert style.fiber_linewidth == 2.0
        assert style.crosslink_size == 10.0
    
    def test_to_dict(self):
        style = PlotStyle()
        data = style.to_dict()
        assert isinstance(data, dict)
        assert 'fiber_color' in data
        assert 'grid' in data


class TestNetworkVisualizer:
    """Test NetworkVisualizer."""
    
    def test_initialization(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        viz = NetworkVisualizer(net)
        assert viz.network == net
        assert viz.style is not None
    
    def test_initialization_with_style(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        style = PlotStyle(fiber_color='red')
        viz = NetworkVisualizer(net, style=style)
        assert viz.style.fiber_color == 'red'
    
    def test_plot_2d(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        viz = NetworkVisualizer(net)
        fig = viz.plot_2d(title="Test Network")
        assert fig is not None
        assert viz.fig is not None
        assert viz.ax is not None
        plt.close(fig)
    
    def test_plot_2d_color_by_length(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        viz = NetworkVisualizer(net)
        fig = viz.plot_2d(color_by='length')
        assert fig is not None
        plt.close(fig)
    
    def test_plot_2d_color_by_orientation(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        viz = NetworkVisualizer(net)
        fig = viz.plot_2d(color_by='orientation')
        assert fig is not None
        plt.close(fig)
    
    def test_plot_2d_no_crosslinks(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        viz = NetworkVisualizer(net)
        fig = viz.plot_2d(show_crosslinks=False)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_3d(self):
        net = gen.random_straight_3d(num_fibers=30, seed=42)
        viz = NetworkVisualizer(net)
        fig = viz.plot_3d(title="3D Network")
        assert fig is not None
        plt.close(fig)
    
    def test_plot_3d_color_by_length(self):
        net = gen.random_straight_3d(num_fibers=30, seed=42)
        viz = NetworkVisualizer(net)
        fig = viz.plot_3d(color_by='length')
        assert fig is not None
        plt.close(fig)
    
    def test_plot_stress_field(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        viz = NetworkVisualizer(net)
        stress = np.random.rand(net.num_fibers) * 1e6
        fig = viz.plot_stress_field(stress, title="Stress Field")
        assert fig is not None
        plt.close(fig)
    
    def test_save(self, tmp_path):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        viz = NetworkVisualizer(net)
        viz.plot_2d()
        
        filename = tmp_path / "test_plot.png"
        viz.save(str(filename), dpi=100)
        assert filename.exists()
        plt.close(viz.fig)


class TestVisualizeNetwork:
    """Test convenience function."""
    
    def test_2d_visualization(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        fig = visualize_network(net, dimension=2, title="Test")
        assert fig is not None
        plt.close(fig)
    
    def test_3d_visualization(self):
        net = gen.random_straight_3d(num_fibers=30, seed=42)
        fig = visualize_network(net, dimension=3, title="Test")
        assert fig is not None
        plt.close(fig)
    
    def test_invalid_dimension(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        with pytest.raises(ValueError):
            visualize_network(net, dimension=4)
