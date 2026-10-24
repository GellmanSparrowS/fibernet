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
>>> g = fn.pattern_2d(unit="square", box=(10,10), grid=(3,3),
...                    n_pts_per_side=3, point_displacements=disps)
>>> # Method 2: Post-generation node manipulation
>>> g.displace_node(5, [0.1, 0.2])
>>> g.set_node_positions({1: [2.5, 0.5], 3: [7.5, 1.0]})
>>> internal = g.get_internal_nodes()  # Nodes available for RL actions
"""

from fibernet.version import __version__

# --- Core ---
from fibernet.core.structure_graph import StructureGraph, SNode, SEdge
from fibernet.core.material import Material
from fibernet.core.viscoelastic import LinearViscoelasticMaterial
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.core.fiber import Fiber
from fibernet.core.fiber_network_profile import (
    FiberNetworkProfile, FiberNetworkProjectionLimits, FiberNetworkProjectionError,
)

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
    list_units_3d, register_unit_3d,
)
from fibernet.gen.manufacturing import (PlanarManufacturingConfig,
                                        ManufacturingNetwork, compile_planar,
                                        compile_surface,
                                        manufacturable_graph)
from fibernet.gen.solid_export import (PrintSettings, FiberSolid, build_solid,
                                       build_voxel_solid, export_solid,
                                       export_route)

from fibernet.gen.manufacturing_bridge import manufacturing_from_graph, export_mapped_route
from fibernet.gen.manufacturing_trails import (ManufacturingTrails,
                                              manufacturing_trails_from_graph,
                                              export_mapped_trails)

# --- Deferred public exports ---
# Import the owning implementation only when its public symbol is requested.
# Do not wrap, replace, or initialize any backend here.
_LAZY_EXPORTS = {
    "TaichiEngine": "fibernet.sim.accelerated",
    "SimResult": "fibernet.sim.accelerated",
    "solve_linear_transport": "fibernet.sim.linear_transport",
    "ViscoelasticResult": "fibernet.sim._viscoelastic_fem",
    "JouleHeatingLoad": "fibernet.sim.joule_heating",
    "JouleHeatingRun": "fibernet.sim.joule_heating",
    "assemble_joule_heating": "fibernet.sim.joule_heating",
    "run_joule_heating": "fibernet.sim.joule_heating",
    "ReducedBeamConfig": "fibernet.sim.reduced_beam",
    "ReducedBeamResult": "fibernet.sim.reduced_beam",
    "ReducedBeamSolver": "fibernet.sim.reduced_beam",
    **dict.fromkeys(("show", "simulate", "batch_simulate", "batch_simulate_from_json",
                     "train_model", "train_rl", "predict_named"), "fibernet.easy"),
    "GraphFeatureExtractor": "fibernet.analysis.graph_features",
    "GraphFeatureExtractor3D": "fibernet.analysis.graph_features_3d",
    **dict.fromkeys(("plot_reward_curve", "plot_convergence", "plot_action_distribution",
                     "evaluate_agent", "save_agent", "load_agent", "run_bayesian_optimization"),
                    "fibernet.rl.utils"),
}
_LAZY_MODULES = {name: "fibernet." + name for name in ("sim", "easy", "analysis", "rl")}

# reload() retains the module dictionary. Drop only bindings newly deferred by
# this facade, so their next access resolves the current owning implementation.
for _name in (*_LAZY_EXPORTS, *_LAZY_MODULES, "_HAS_RL"):
    globals().pop(_name, None)
del _name

# --- Visualization (resolved only when explicitly requested) ---
_RENDER_EXPORTS = (
    "render_trajectory",
    "render_graph", "render_graph_3d",
    "render_deformation", "render_gallery",
    "render_deformation_3d", "render_trajectory_3d", "render_gallery_3d",
    "render_stress_3d", "render_comparison_3d", "render_multi_angle_3d",
    "render_with_stats", "render_fem_stress",
    "THEMES",
)

# --- PyVista 3D Rendering (optional) ---
try:
    from fibernet.viz.render3d import render_structure_graph_3d, render_stress_pyvista
    _HAS_PYVISTA = True
except ImportError:
    _HAS_PYVISTA = False

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

__all__ = [
    # Core
    "StructureGraph", "SNode", "SEdge",
    "Material", "Fiber", "FiberNetwork", "Crosslink",
    "FiberNetworkProfile", "FiberNetworkProjectionLimits", "FiberNetworkProjectionError",
    # Transforms
    "translate", "rotate", "mirror", "mirror_x", "mirror_y", "mirror_z",
    "scale", "compose",
    # Tiling
    "tile_2d", "tile_3d", "fit_unit_to_box",
    # Generation
    "pattern_2d", "pattern_3d", "list_units", "register_unit",
    "list_units_3d", "register_unit_3d",
    "PlanarManufacturingConfig", "ManufacturingNetwork",
    "compile_planar", "compile_surface", "manufacturable_graph",
    "manufacturing_from_graph", "export_mapped_route",
    "ManufacturingTrails", "manufacturing_trails_from_graph", "export_mapped_trails",
    "PrintSettings", "FiberSolid", "build_solid", "build_voxel_solid",
    "export_solid", "export_route",
    # Simulation
    "TaichiEngine", "SimResult", "solve_linear_transport",
    "LinearViscoelasticMaterial", "ViscoelasticResult",
    "JouleHeatingLoad", "JouleHeatingRun", "assemble_joule_heating", "run_joule_heating",
    "ReducedBeamConfig", "ReducedBeamResult", "ReducedBeamSolver",
    # Visualization
    "render_graph", "render_graph_3d", "render_deformation",
    "render_gallery", "render_with_stats", "render_trajectory",
    "render_deformation_3d", "render_trajectory_3d", "render_gallery_3d",
    "render_stress_3d", "render_comparison_3d", "render_multi_angle_3d",
    "render_fem_stress",
    "THEMES",
    # Easy API
    "show", "simulate", "batch_simulate", "batch_simulate_from_json", "train_model", "train_rl", "predict_named",
    # Analysis
    "GraphFeatureExtractor",
    "GraphFeatureExtractor3D",
    # ML (if available)
    "train_predictor", "cross_validate", "compare_models",
    "predict_from_csv", "plot_predictions", "plot_feature_importance",
    "plot_residuals", "plot_learning_curve",
    # RL (if available)
    "plot_reward_curve", "plot_convergence", "plot_action_distribution",
    "evaluate_agent", "save_agent", "load_agent",
    "run_bayesian_optimization",
]


def __getattr__(name):
    """Resolve public exports on demand, propagating implementation failures."""
    from importlib import import_module
    if name in _LAZY_EXPORTS:
        value = getattr(import_module(_LAZY_EXPORTS[name]), name)
    elif name in _RENDER_EXPORTS:
        value = getattr(import_module("fibernet.viz.render"), name)
    elif name in _LAZY_MODULES:
        value = import_module(_LAZY_MODULES[name])
    elif name == "_HAS_RL":
        # Historical private flag described facade import success, not whether
        # Gymnasium or any runtime was installed. Evaluate only if requested.
        try:
            import_module("fibernet.rl.utils")
        except ImportError:
            value = False
        else:
            value = True
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_RENDER_EXPORTS) | set(_LAZY_EXPORTS)
                  | set(_LAZY_MODULES) | {"_HAS_RL"})
