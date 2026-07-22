"""
Visualization module for fiber networks.

Provides publication-quality 2D/3D rendering of FiberNetwork objects
with configurable themes, color mapping, and statistical analysis.

Quick Start:
    >>> import fibernet as fn
    >>> net = fn.create("reentrant_honeycomb_2d")
    >>> fig, ax = fn.viz.plot(net, color_by="orientation", theme="dark")
    >>> fn.viz.save_figure(fig, "output.png", dpi=300)
"""

from .visualization import (
    # Core plotting
    plot,
    plot_3d,
    plot_comparison,
    plot_statistics,
    # Aliases for backward compatibility
    visualize_3d_matplotlib,
    visualize_3d_pyvista,
    plot_network_2d,
    plot_network_3d,
    render_network_3d,
    visualize_damage_evolution,
    visualize_deformation,
    save_figure,
)

# Graph plotting (NetworkX)
from .graph_plot import (
    plot_graph,
    plot_graph_comparison,
    plot_structure_stats,
)

__all__ = [
    # Core
    'plot',
    'plot_3d',
    'plot_comparison',
    'plot_statistics',
    'save_figure',
    # Backward compat
    'visualize_3d_matplotlib',
    'visualize_3d_pyvista',
    'visualize_network_stress',
    'animate_deformation',
    'visualize_damage_evolution',
    'plot_network_2d',
    'plot_network_3d',
    'render_network_3d',
    # Graph
    'plot_graph',
    'plot_graph_comparison',
    'plot_structure_stats',
]

# Add backward compat aliases
def visualize_network_stress(*args, **kwargs):
    """Backward compat: use plot(network, color_by='custom')."""
    return plot(*args, color_by="custom", **kwargs)

def animate_deformation(*args, **kwargs):
    """Backward compat: use visualize_deformation()."""
    return visualize_deformation(*args, **kwargs)

# Interactive Plotly visualization (optional)
try:
    from .plotly_viz import (
        visualize_interactive,
        visualize_stress_field,
        visualize_comparison,
        export_html,
    )
    __all__ += [
        'visualize_interactive',
        'visualize_stress_field',
        'visualize_comparison',
        'export_html',
    ]
except ImportError:
    pass

# Publication-quality renderer
from .renderer import (
    render_network_2d,
    render_network_3d,
    render_comparison_grid,
    render_parametric_study,
    BG_COLOR,
    FIBER_COLOR_2D,
    FIBER_COLOR_3D,
)

__all__ += [
    'render_network_2d',
    'render_network_3d',
    'render_comparison_grid',
    'render_parametric_study',
    'BG_COLOR',
    'FIBER_COLOR_2D',
    'FIBER_COLOR_3D',
]
