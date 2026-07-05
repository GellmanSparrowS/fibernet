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

__version__ = "1.24.0"
__author__ = "ML-BioMat Lab"

# Core data structures
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.core.material import Material
from fibernet.core.copy_utils import copy_fiber, copy_material, copy_network

# High-level convenience API
from .api import (
    create, mirror, rotate, scale, translate, merge, tile,
    simulate_mechanics, simulate_thermal, analyze, export, load, plot,
)

# Version information
from .version import __version__, __author__, __license__

# Simulation modules (lazy imports for optional dependencies)
from .sim.fluid import DarcySolver, PoreNetworkModel
from .sim.acoustic import AcousticSolver

# Visualization module
from . import viz

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
    "simulate_thermal",
    "analyze",
    "export",
    "load",
    "plot",
    
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
