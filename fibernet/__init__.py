"""
FiberNet - A comprehensive toolkit for fiber network structure research.

Provides tools for:
- Generation: 2D/3D fiber networks (ordered, disordered, chiral, bundled, woven, hierarchical)
- Simulation: Mechanical, dynamics, fracture, thermal, electromagnetic
- Analysis: Topology, morphology, properties
- Visualization: 3D rendering, animation, 2D plots

Homepage: https://ml-biomat.com
GitHub: https://github.com/GellmanSparrowS/fibernet
"""

__version__ = "0.9.0"
__author__ = "ML-BioMat Lab"

from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.core.material import Material
from fibernet.core.copy_utils import copy_fiber, copy_material, copy_network

__all__ = [
    "Fiber",
    "FiberNetwork",
    "Material",
    "__version__",
]

# High-level convenience API
from .api import (
    create, mirror, rotate, scale, translate, merge, tile,
    simulate_mechanics, simulate_thermal, analyze, export, load, plot,
)

# Version information
from .version import __version__, __author__, __license__

# ML module (optional)
try:
    from . import ml
except ImportError:
    pass

# Fluid and acoustic simulation
from .sim.fluid import DarcySolver, PoreNetworkModel
from .sim.acoustic import AcousticSolver

# Visualization module
from . import viz
