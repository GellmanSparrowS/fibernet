"""
Visualization module for fiber networks.
"""

from .visualization import (
    visualize_3d_matplotlib,
    visualize_3d_pyvista,
    visualize_network_stress,
    animate_deformation,
    visualize_damage_evolution,
    plot_network_2d,
    plot_network_3d,
    render_network_3d,
)

__all__ = [
    'visualize_3d_matplotlib',
    'visualize_3d_pyvista',
    'visualize_network_stress',
    'animate_deformation',
    'visualize_damage_evolution',
    'plot_network_2d',
    'plot_network_3d',
    'render_network_3d',
]

# Interactive Plotly visualization
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
