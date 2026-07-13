"""
FiberNet — Unified framework for fiber networks, lattices, and metamaterials.

Architecture:
    StructureGraph → Pattern Engine → FEM Simulation → Visualization → ML/RL

Quick Start
-----------
>>> import fibernet as fn
>>> # Generate a honeycomb structure
>>> g = fn.pattern_2d(unit="honeycomb", box=(10, 10), grid=(5, 5), n_internal=8)
>>> print(g)
StructureGraph(dim=2, nodes=90, edges=130, box=[50.0, 50.0])

>>> # Run mechanical analysis
>>> engine = fn.TaichiEngine()
>>> result = fem.uniaxial_tension(strain=0.01)
>>> print(f"E* = {result.effective_youngs_modulus:.2e} Pa")

>>> # Visualize
>>> fig = fn.render_graph(g, theme="dark", color_by="orientation")
>>> fig.savefig("honeycomb.png", dpi=200)

>>> # Generate ML dataset
>>> ds = fn.generate_dataset(units=["honeycomb", "square"], save_dir="datasets/")

>>> # RL environment
>>> env = fn.FiberNetworkEnv(target_E=1e5, target_nu=-0.3)
"""

__version__ = "3.0.0"

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

# --- Visualization ---
from fibernet.viz.render import (
    render_graph, render_graph_3d,
    render_deformation, render_gallery,
    render_with_stats,
    THEMES,
)

# --- ML / RL ---
from fibernet.ml.dataset_v2 import generate_dataset, extract_features
from fibernet.sim.rl_env import FiberNetworkEnv

# Easy API (一行代码)
from fibernet.sim.accelerated import TaichiEngine, SimResult
from fibernet.easy import show, simulate, batch_simulate, train_model, train_rl

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
    # Visualization
    "render_graph", "render_graph_3d", "render_deformation",
    "render_gallery", "render_with_stats", "THEMES",
    # ML/RL
    "generate_dataset", "extract_features", "FiberNetworkEnv",
    # Easy API
    "show", "simulate", "batch_simulate", "train_model", "train_rl",
    # Simulation backends
    "TaichiEngine", "SimResult",
]
