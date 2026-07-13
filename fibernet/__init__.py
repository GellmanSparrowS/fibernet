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

ML Pipeline (one-liners)
------------------------
>>> from fibernet.ml import train_predictor, predict_from_csv, cross_validate
>>> model, metrics = train_predictor(X, y, model_type="rf")
>>> result = predict_from_csv("data.csv", target="max_force")

RL Pipeline (one-liners)
------------------------
>>> from fibernet.rl import plot_reward_curve, run_bayesian_optimization
>>> plot_reward_curve(rewards, window=20, save_path="reward.png")
>>> best = run_bayesian_optimization(objective_fn, param_space, n_iter=50)

Parametric Structure Generation (for RL)
------------------------------------------
>>> # Method 1: Displacements at generation time
>>> g = fn.pattern_2d("square", box=(10,10), grid=(3,3),
...                    n_pts_per_side=3, point_displacements=disps)
>>> # Method 2: Post-generation node manipulation
>>> g.displace_node(5, [0.1, 0.2])
>>> g.set_node_positions({1: [2.5, 0.5], 3: [7.5, 1.0]})
>>> internal = g.get_internal_nodes()  # Nodes available for RL actions
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
    render_trajectory,
    render_graph, render_graph_3d,
    render_deformation, render_gallery,
    render_with_stats,
    THEMES,
)

# --- Easy API ---
from fibernet.easy import show, simulate, batch_simulate, train_model, train_rl

# --- Analysis ---
from fibernet.analysis.graph_features import GraphFeatureExtractor

# --- ML Utilities (lazy import to avoid hard sklearn dependency) ---
try:
    from fibernet.ml.utils import (
        train_predictor,
        cross_validate,
        compare_models,
        predict_from_csv,
        plot_predictions,
        plot_feature_importance,
        plot_residuals,
        plot_learning_curve,
    )
    _HAS_ML = True
except ImportError:
    _HAS_ML = False

# --- RL Utilities (lazy import to avoid hard gymnasium dependency) ---
try:
    from fibernet.rl.utils import (
        plot_reward_curve,
        plot_convergence,
        plot_action_distribution,
        evaluate_agent,
        save_agent,
        load_agent,
        run_bayesian_optimization,
    )
    _HAS_RL = True
except ImportError:
    _HAS_RL = False

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
    "render_gallery", "render_with_stats", "render_trajectory", "THEMES",
    # Easy API
    "show", "simulate", "batch_simulate", "train_model", "train_rl",
    # Analysis
    "GraphFeatureExtractor",
    # ML (if available)
    "train_predictor", "cross_validate", "compare_models",
    "predict_from_csv", "plot_predictions", "plot_feature_importance",
    "plot_residuals", "plot_learning_curve",
    # RL (if available)
    "plot_reward_curve", "plot_convergence", "plot_action_distribution",
    "evaluate_agent", "save_agent", "load_agent",
    "run_bayesian_optimization",
]
