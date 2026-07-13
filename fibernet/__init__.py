"""
FiberNet — Unified framework for fiber networks, lattices, and metamaterials.

Quick Start
-----------
>>> import fibernet as fn
>>> g = fn.pattern_2d(unit="honeycomb", box=(10, 10), grid=(5, 5))
>>> engine = fn.TaichiEngine()
>>> result = engine.stretch_test(g, target_stretch=2.0)
>>> fig = fn.render_graph(g, theme="dark")
>>> fig.savefig("honeycomb.png", dpi=200)
"""

__version__ = "4.0.0-dev"

# --- Core ---
from fibernet.core.structure_graph import StructureGraph, SNode, SEdge
from fibernet.core.material import Material
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.core.fiber import Fiber

# --- Transforms ---
from fibernet.core.transforms import (
    translate, rotate, mirror, mirror_x, mirror_y, mirror_z,
    scale, compose,
)

# --- Tiling ---
from fibernet.core.tiling import tile_2d, tile_3d, fit_unit_to_box

# --- Pattern Engine (Structure Generation) ---
from fibernet.gen.pattern import (
    pattern_2d, pattern_3d,
    list_units, register_unit,
)

# --- Simulation ---
from fibernet.sim.accelerated import TaichiEngine, SimResult

# --- Visualization ---
from fibernet.viz.render import (
    render_graph, render_graph_3d,
    render_deformation, render_gallery,
    render_with_stats,
    THEMES,
)

# --- Easy API ---
from fibernet.easy import show, simulate, batch_simulate, train_model, train_rl

# --- Analysis ---
from fibernet.analysis.graph_features import GraphFeatureExtractor

__all__ = [
    # Core
    "StructureGraph", "SNode", "SEdge",
    "Material", "Fiber", "FiberNetwork", "Crosslink",
    # Transforms
    "translate", "rotate", "mirror", "mirror_x", "mirror_y", "mirror_z",
    "scale", "compose",
    # Tiling
    "tile_2d", "tile_3d", "fit_unit_to_box",
    # Generation
    "pattern_2d", "pattern_3d", "list_units", "register_unit",
    # Simulation
    "TaichiEngine", "SimResult",
    # Visualization
    "render_graph", "render_graph_3d", "render_deformation",
    "render_gallery", "render_with_stats", "THEMES",
    # Easy API
    "show", "simulate", "batch_simulate", "train_model", "train_rl",
    # Analysis
    "GraphFeatureExtractor",
]
