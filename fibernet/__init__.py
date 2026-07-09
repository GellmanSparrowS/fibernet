"""
FiberNet - A comprehensive toolkit for fiber network structure research.

Provides tools for:
- Generation: 2D/3D fiber networks (ordered, disordered, chiral, bundled, woven, hierarchical)
- Simulation: Mechanical, dynamics, fracture, thermal, electromagnetic, fluid, acoustic
- Analysis: Topology, morphology, properties, percolation, multi-scale
- Visualization: 3D rendering, animation, 2D plots, interactive
- Machine Learning: Feature extraction, GNN models

Homepage: https://ml-biomat.com
GitHub: https://github.com/GellmanSparrowS/fibernet

Quick Start
-----------
>>> import fibernet as fn
>>> # Create a random 2D network
>>> net = fn.create("random_2d", num_fibers=100, fiber_length=10.0, box_size=(20, 20))
>>> # Analyze structure
>>> stats = fn.analyze(net)
>>> # Run mechanical simulation
>>> result = fn.simulate_mechanics(net, strain=0.01)
>>> # Visualize
>>> fn.plot(net)

For more examples, see the tutorials/ directory.
"""

__version__ = "2.0.0"
__author__ = "ML-BioMat Lab"

# Core data structures
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.core.material import Material
from fibernet.core.copy_utils import copy_fiber, copy_material, copy_network

# High-level convenience API
from .api import (
    create, mirror, rotate, scale, translate, merge, tile,
    simulate_mechanics, simulate_dynamics, simulate_thermal, analyze, export, load, plot,
    plot_dynamics, plot_metamaterial, plot_stress_strain,
    list_generators, list_backends,
    register_generator, register_backend,
    create_metamaterial, print_metamaterial_info,
)

# Version information
from .version import __version__, __author__, __license__

# Simulation modules (lazy imports for optional dependencies)
from .sim.fluid import DarcySolver, PoreNetworkModel
from .sim.acoustic import AcousticSolver

# Visualization module
from . import viz
from . import graph  # Graph I/O and weld operations
from .graph import (
    to_networkx, from_networkx,
    load_graph_json, save_graph_json,
    weld_graph, find_intersections, merge_coincident_nodes,
)
from .gen.regular import RegularNetworkGenerator
from .gen.zigzag import ZigZagGenerator
from .analysis.graph_features import GraphFeatureExtractor

# Graph visualization
from .viz.graph_plot import plot_graph, plot_graph_comparison, plot_structure_stats


def extract_features(network_or_graph, canvas_size=512, thick=5, **kwargs):
    """Convenience function to extract 94-dimensional features.

    Parameters
    ----------
    network_or_graph : nx.Graph or FiberNetwork
        The network to analyze.
    canvas_size : int
        Resolution for image-based analysis.
    thick : int
        Line thickness for rendering.
    **kwargs
        Additional arguments passed to GraphFeatureExtractor.

    Returns
    -------
    dict
        94-dimensional feature dictionary.

    Examples
    --------
    >>> import fibernet as fn
    >>> gen = fn.RegularNetworkGenerator(side_length=10, tiling=3)
    >>> G = gen.generate()
    >>> features = fn.extract_features(G)
    >>> print(f"Nodes: {features['n_node']}, Edges: {features['n_edge']}")
    """
    extractor = GraphFeatureExtractor(canvas_size=canvas_size, thick=thick, **kwargs)
    return extractor.extract(network_or_graph)


# ML module (optional)
try:
    from . import ml
except ImportError:
    ml = None

# Submodules for advanced usage
from . import gen      # Network generators
from . import sim      # Simulators
from . import analysis # Analyzers
from . import io       # I/O utilities
from . import utils    # Utilities

__all__ = [
    # Core classes
    "Fiber",
    "FiberNetwork", 
    "Material",
    "copy_fiber",
    "copy_material",
    "copy_network",
    
    # High-level API
    "create",
    "mirror",
    "rotate",
    "scale",
    "translate",
    "merge",
    "tile",
    "simulate_mechanics",
    "simulate_dynamics",
    "simulate_thermal",
    "analyze",
    "export",
    "load",
    "plot",
    "plot_dynamics",
    "plot_metamaterial",
    "plot_stress_strain",
    "list_generators",
    "list_backends",
    "register_generator",
    "register_backend",
    "create_metamaterial",
    "print_metamaterial_info",
    
    # Simulators
    "DarcySolver",
    "PoreNetworkModel",
    "AcousticSolver",
    
    # Submodules
    "gen",
    "sim",
    "analysis",
    "io",
    "viz",
    "ml",
    "utils",
    
    # Graph operations
    "graph",
    "to_networkx",
    "from_networkx",
    "load_graph_json",
    "save_graph_json",
    "weld_graph",
    "find_intersections",
    "merge_coincident_nodes",
    "plot_graph",
    "plot_graph_comparison",
    "plot_structure_stats",
    "extract_features",
    # Generators
    "RegularNetworkGenerator",
    "ZigZagGenerator",
    "GraphFeatureExtractor",
    # Metadata
    "__version__",
    "__author__",
    "__license__",
]

# Visualization
from .visualization import (
    NetworkVisualizer, PlotStyle, visualize_network
)
__all__.extend([
    "NetworkVisualizer", "PlotStyle", "visualize_network"
])

# Design of Experiments
from .doe import (
    DesignOfExperiments, ExperimentDesign, ExperimentResult, SweepResult,
    run_parameter_sweep
)
__all__.extend([
    "DesignOfExperiments", "ExperimentDesign", "ExperimentResult", "SweepResult",
    "run_parameter_sweep"
])

# Materials database
from .materials import (
    get_material, list_materials, compare_materials, get_material_database, m
)
__all__.extend([
    "get_material", "list_materials", "compare_materials", "get_material_database", "m"
])

# Unit conversions
from .units import (
    convert_length, convert_force, convert_pressure, convert_temperature,
    convert_energy, parse_unit_string, scale_network_properties
)
__all__.extend([
    "convert_length", "convert_force", "convert_pressure", "convert_temperature",
    "convert_energy", "parse_unit_string", "scale_network_properties"
])

# PyVista visualization (lazy import to avoid VTK segfaults in CI)
import os as _os
if _os.environ.get("CI") != "true":
    from .pyvista_viz import PyVistaVisualizer, visualize_network_3d, PYVISTA_AVAILABLE
else:
    PYVISTA_AVAILABLE = False
    PyVistaVisualizer = None
    visualize_network_3d = None
__all__.extend([
    "PyVistaVisualizer", "visualize_network_3d", "PYVISTA_AVAILABLE"
])

# Trimesh integration
from .trimesh_integration import (
    TrimeshConverter, network_to_trimesh, analyze_mesh_properties,
    boolean_operation, repair_mesh, simplify_mesh, TRIMESH_AVAILABLE
)
__all__.extend([
    "TrimeshConverter", "network_to_trimesh", "analyze_mesh_properties",
    "boolean_operation", "repair_mesh", "simplify_mesh", "TRIMESH_AVAILABLE"
])
