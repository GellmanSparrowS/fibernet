<div align="center">

# 🧬 FiberNet v4

**Python Toolkit for Fiber Network Design, Simulation & Intelligent Optimization**

[![PyPI](https://img.shields.io/pypi/v/fibernet?logo=pypi&logoColor=white)](https://pypi.org/project/fibernet/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CI](https://github.com/GellmanSparrowS/fibernet/actions/workflows/ci.yml/badge.svg)](https://github.com/GellmanSparrowS/fibernet/actions)
[![Downloads](https://img.shields.io/pypi/dm/fibernet)](https://pypi.org/project/fibernet/)

[中文文档](README_CN.md) · [PyPI](https://pypi.org/project/fibernet/) · [Tutorial](#-tutorial) · [API Docs](#-api-reference)

*Developed by [ML-BioMat Lab](https://ml-biomat.com/) @ [BMG-FDU](https://github.com/BMG-FDU)*

</div>

---

## Overview

FiberNet is a research-grade Python toolkit for **computational design of fiber network metamaterials**. It provides a complete closed-loop workflow:

```
Generation → Simulation (Mass-Spring / FEM) → Feature Extraction → Machine Learning → Reinforcement Learning
```

| Feature | Description |
|---------|-------------|
| **26 Unit Types** | 12 2D + 14 3D: honeycomb, kagome, reentrant, octet, diamond\_3d, fcc, bcc, gyroid, TPMS… |
| **Parametric Control** | Internal point displacements for RL-ready continuous action spaces |
| **Dual Simulation** | Taichi mass-spring (GPU) + Beam Frame FEM (Euler–Bernoulli) |
| **94-Dim Features** | Structural + pore + contact feature extraction |
| **One-Line ML** | `predict_from_csv()` → train, evaluate, visualize, save |
| **One-Line RL** | `run_bayesian_optimization()` or CEM optimization |

---

## 🖼️ Showcase

<div align="center">
<img src="docs/images/01_2d_gallery.png" width="80%" alt="2D Structure Gallery" />
</div>

*12 2D unit types: square, triangle, hexagon, honeycomb, kagome, voronoi, chiral, reentrant, star, cross, diamond, missing\_rib.*

<div align="center">
<img src="docs/images/voronoi_1.5x_auto.png" width="80%" alt="Voronoi Stretch" />
</div>

*Voronoi structure under 1.5× uniaxial stretch (mass-spring model) — deformation and stress distribution.*

<div align="center">
<img src="docs/images/05_trajectory_dark.png" width="80%" alt="Deformation Trajectory" />
</div>

*8-frame deformation trajectory: honeycomb under stretch, colored by edge stretch ratio.*

<div align="center">
<img src="docs/images/fem_showcase_dark.png" width="80%" alt="FEM Deformation Showcase" />
</div>

*Beam Frame FEM analysis: uniaxial stretch (2×) and compression (0.5×) across multiple topologies and fiber radii. Bright color = high von Mises stress. Structures modeled as welded frames with radius-dependent bending stiffness.*

<div align="center">
<img src="docs/images/09_ml_analysis_dark.png" width="80%" alt="ML Analysis" />
</div>

*ML analysis: confusion matrix, ROC curves, and learning curves.*

<div align="center">
<img src="docs/images/11_rl_reward_dark.png" width="80%" alt="RL Reward" />
</div>

*CEM reinforcement learning: reward per episode and monotonically increasing best reward.*

---

## 🚀 Quick Start

### One-Line API

```python
import fibernet as fn

g = fn.pattern_2d(unit="honeycomb", box=(10, 10), grid=(4, 4))
fn.show(g)  # one-line visualization
```

```python
r = fn.simulate(g, mode="stretch", strain=1.5, backend="spring")
print(f"max_force={r.max_force:.0f} N, max_stretch={r.max_stretch:.3f}")
```

### FEM in 3 Lines

```python
from fibernet.ml import BeamFrameFEM

solver = BeamFrameFEM(E=1e9, nu=0.3)
g = fn.pattern_2d(unit="honeycomb", box=(10, 10), grid=(4, 4), radius=0.05)
result = solver.stretch_test(g, target_stretch=2.0)

print(f"Max stress: {result['sigma_total'].max()/1e6:.1f} MPa")
print(f"Max displacement: {result['max_displacement']:.4f} m")
```

### Complete Pipeline

```python
import fibernet as fn
import numpy as np

# 1. Parametric structure (20 displacement params for RL)
displacements = [(np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3))
                 for _ in range(20)]
g = fn.pattern_2d(unit="square", box=(10, 10), grid=(3, 3),
                  n_pts_per_side=5, point_displacements=displacements)

# 2a. Taichi mass-spring simulation
engine = fn.TaichiEngine()
r = engine.stretch_test(g, target_stretch=1.5, stiffness=1e5,
                        damping=0.3, num_steps=1000, save_interval=200)

# 2b. Or Beam Frame FEM (Euler-Bernoulli, welded joints)
from fibernet.ml import BeamFrameFEM
fem = BeamFrameFEM(E=1e9, nu=0.3)
fem_result = fem.stretch_test(g, target_stretch=1.5)
sim_r = fem.to_sim_result(fem_result, graph=g)

# 3. Visualization
fig = fn.render_trajectory(g, r.positions_trajectory, r.edge_stretches,
                           n_frames=6, title="Stretch Process")
fig.savefig("deformation.png", dpi=150)

# 4. Feature extraction (94-dim vector)
ext = fn.GraphFeatureExtractor()
features = ext.extract(g)

# 5. Node manipulation (for RL action space)
internal = g.get_internal_nodes()
g.displace_node(internal[0], [0.1, 0.2])
```

---

## 📦 Installation

```bash
pip install fibernet          # core
pip install fibernet[full]    # ML + RL + viz + simulation
pip install fibernet[ml]      # ML only
pip install fibernet[rl]      # RL only
```

| Optional Group | Packages |
|---------------|----------|
| `ml` | scikit-learn, pandas, tqdm |
| `rl` | gymnasium, scikit-optimize, stable-baselines3 |
| `accel` | taichi (GPU simulation) |
| `viz` | pyvista (3D visualization) |
| `full` | all of the above |

---

## 🔬 Beam Frame FEM

FiberNet v4.1 introduces a production-grade **Beam Frame Finite Element Method** solver based on Euler–Bernoulli beam theory, providing physically accurate mechanical analysis beyond the mass-spring model.

### Physics Model

Unlike mass-spring models where fiber diameter is cosmetic, the FEM solver treats the structure as a **welded frame** — joints are rigidly connected and fiber radius directly determines bending and axial stiffness:

```
EI = E × πr⁴ / 4          (bending rigidity)
EA = E × πr²              (axial rigidity)
σ_axial = N / A            (axial stress)
σ_bending = M·r / I        (bending stress)
σ_total = σ_axial + σ_bending
```

Doubling the fiber radius increases bending stiffness by **16×** (r⁴ dependence), correctly capturing the physics of fiber network metamaterials.

### Verification Results

Validated across **152 simulations** (8 2D + 6 3D topologies × 4 radii × 4 stretch targets):

| Radius | Max Stress (2× stretch, honeycomb) | Dominant Mode |
|--------|-----------------------------------|---------------|
| r = 0.02 | 2,967 MPa | Bending-dominated |
| r = 0.05 | 7,418 MPa | Bending-dominated |
| r = 0.10 | 14,835 MPa | Bending-dominated |
| r = 0.20 | 20,997 MPa | Bending-dominated |

All structures show **full deformation propagation** — boundary displacement transmits through the entire structure, not just the first few layers.

### API

```python
from fibernet.ml import BeamFrameFEM
import fibernet as fn

solver = BeamFrameFEM(E=1e9, nu=0.3)

# Generate structure (radius matters for FEM!)
g = fn.pattern_2d(unit="honeycomb", box=(10, 10), grid=(4, 4), radius=0.05)

# One-liner stretch test (auto-selects linear/nonlinear solver)
result = solver.stretch_test(g, target_stretch=2.0)

# Access results
u = result['u']                      # nodal displacements
sigma = result['sigma_total']        # per-element total stress
sigma_axial = result['sigma_axial']  # axial component
sigma_bend = result['sigma_bending'] # bending component
reactions = result['reactions']      # boundary reaction forces

# Low-level API for custom analysis
fem_input = solver.graph_to_fem_input(g, dim=2, pct=0.1)
result = solver.solve_2d(**fem_input)              # linear
result = solver.solve_2d_nonlinear(**fem_input)    # geometrically nonlinear
result = solver.solve_3d(**fem_input)              # 3D analysis

# Convert to SimResult for viz/ML compatibility
sim_result = solver.to_sim_result(result, graph=g)
```

### Supported Solvers

| Solver | Use Case |
|--------|----------|
| `solve_2d()` | Linear 2D — small deformations, fast |
| `solve_2d_nonlinear()` | Nonlinear 2D — large deformations (co-rotational) |
| `solve_3d()` | 3D beam frame analysis |
| `stretch_test()` | Convenience wrapper — auto-selects solver |

---

## 📚 API Reference

### Structure Generation

```python
import numpy as np
disps = [(np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3))
         for _ in range(20)]

g = fn.pattern_2d(
    unit="square",              # 12 2D unit types
    box=(10, 10),               # cell size
    grid=(3, 3),                # tiling grid
    n_pts_per_side=5,           # internal points per edge
    point_displacements=disps,  # parametric control
    radius=0.05,                # fiber radius (for FEM)
    seed=42,
)

# 3D structures
g3d = fn.pattern_3d(unit="octet", box=(5, 5, 5), grid=(2, 2, 2))
```

**Available unit types:**

- **2D:** chiral, cross, diamond, hexagon, honeycomb, kagome, missing\_rib, reentrant, square, star, triangle, voronoi
- **3D:** bcc, chiral\_3d, cubic, diamond\_3d, fcc, gyroid, hcp, iwp, lidinoid, neovius, octet, reentrant\_3d, schwarz\_d, schwarz\_p

### Node Manipulation

```python
g.displace_node(node_id, [dx, dy])
g.set_node_position(node_id, [x, y])
g.set_node_positions({1: [2.5, 0.5], 3: [7.5, 1.0]})

internal = g.get_internal_nodes()
boundary = g.get_boundary_nodes()
```

### Simulation — Mass-Spring

Taichi-based GPU-accelerated mass-spring dynamics. Fibers are modeled as point masses connected by linear springs. Suitable for **large-scale dynamic simulation** and **fast prototyping**. Fiber diameter does not affect mechanics (cosmetic only).

```python
engine = fn.TaichiEngine()
r = engine.stretch_test(g,
    target_stretch=1.5,
    stiffness=1e5,
    damping=0.3,
    num_steps=5000,
    ramp_fraction=0.2,
    save_interval=1000)

r.max_force           # max edge force (N)
r.edge_forces         # per-edge forces
r.edge_stretches      # per-edge stretch ratios
r.positions_trajectory # list of (N,3) position arrays

r.save("result.json", detailed=True)
r2 = fn.SimResult.load("result.json")
```

### Simulation — FEM (BeamFrameFEM)

Euler–Bernoulli beam frame FEM. Fibers are modeled as **beam elements with welded joints** — radius directly determines bending (r⁴) and axial (r²) stiffness. Provides physically accurate stress decomposition (axial + bending). Use for **quantitative mechanical analysis** and **design validation**.

Supports both **linear** solver (small deformation, fast) and **geometrically nonlinear** solver (large deformation, co-rotational incremental). The convenience method `stretch_test()` auto-selects based on strain magnitude.

```python
from fibernet.ml import BeamFrameFEM

solver = BeamFrameFEM(E=1e9, nu=0.3)  # Young's modulus, Poisson's ratio

# One-liner (auto-selects solver)
result = solver.stretch_test(g, target_stretch=2.0, dim=2)

# Full access
u = result['u']                    # nodal displacements (N×dim)
sigma = result['sigma_total']      # per-element total stress
sigma_axial = result['sigma_axial']  # axial stress component
sigma_bend = result['sigma_bending'] # bending stress component
reactions = result['reactions']    # boundary reaction forces
edge_forces = result['edge_forces']  # per-element internal forces

# Low-level API
fem_input = solver.graph_to_fem_input(g, dim=2, pct=0.1)
result = solver.solve_2d(**fem_input)              # linear 2D
result = solver.solve_2d_nonlinear(**fem_input)    # nonlinear 2D (large deformation)
result = solver.solve_3d(**fem_input)              # 3D beam frame

# Compatible with viz/ML pipeline
sim_result = solver.to_sim_result(result, graph=g)
```

**Mass-Spring vs FEM comparison:**

| Aspect | Mass-Spring | BeamFrameFEM |
|--------|-------------|--------------|
| Physics | Point masses + linear springs | Euler–Bernoulli beam elements |
| Joints | Pinned (no moment transfer) | Welded (rigid, moment transfer) |
| Radius effect | Cosmetic only | Physical (EI ∝ r⁴, EA ∝ r²) |
| Stress output | Edge stretch ratio | Full decomposition (axial + bending) |
| Speed | GPU-accelerated, fast for dynamics | CPU sparse solver, fast for statics |
| Best for | Large-scale dynamics, RL rewards | Quantitative stress analysis, validation |

### Visualization

```python
fig = fn.render_graph(g, theme="dark")       # dark purple
fig = fn.render_graph(g, theme="light")      # white background
fig = fn.render_graph(g, theme="blueprint")  # blueprint style

fig = fn.render_deformation(g_original, g_deformed, color_by="stress")
fig = fn.render_trajectory(g, r.positions_trajectory, r.edge_stretches,
                           n_frames=6, title="Stretch Process")
```

### Machine Learning

```python
from fibernet.ml import (
    train_predictor, cross_validate, compare_models,
    predict_from_csv, plot_predictions, plot_feature_importance,
)

result = predict_from_csv("sim_results.csv", target="max_force",
                          model_type="rf", output_dir="ml_out/")

model, metrics = train_predictor(X, y, model_type="rf")
print(f"R² = {metrics['r2']:.3f}")

cv = cross_validate(X, y, model_type="ridge", cv=5)
```

### Reinforcement Learning

```python
from fibernet.rl import (
    plot_reward_curve, plot_convergence, plot_action_distribution,
    run_bayesian_optimization, save_agent, load_agent,
)

result = run_bayesian_optimization(
    objective_fn,
    param_space={"grid_x": (2, 5), "stiffness": (1e4, 1e6)},
    n_iter=50)

plot_reward_curve(rewards, window=20, save_path="reward.png")
plot_convergence(objectives, minimize=True, save_path="conv.png")
```

### 🎯 RL Parametric Control

FiberNet exposes **(dx, dy) displacement parameters** for each internal point on every edge — a continuous action space for RL.

```python
# 40-dim action vector → 20 (dx,dy) pairs
action = agent.act(obs)  # shape: (40,), range: [-0.3, 0.3]
displacements = [(action[2*i], action[2*i+1]) for i in range(20)]
g = fn.pattern_2d(unit="square", grid=(3,3), n_pts_per_side=5,
                  point_displacements=displacements)
```

---

## 🎓 Tutorial

A complete end-to-end Jupyter notebook:

```
tutorials/complete_tutorial_v4.ipynb
```

Standalone runner with checkpoint support:

```bash
python3 tutorials/run_pipeline.py                        # full pipeline
python3 tutorials/run_pipeline.py --num-structures 100   # quick test
python3 tutorials/run_pipeline.py --skip-rl              # skip RL section
```

Covers: structure generation → batch simulation → deformation visualization → feature extraction → ML → CEM reinforcement learning.

---

## 🔬 How It Works

### Mass-Spring Model (Taichi)

GPU-accelerated mass-spring dynamics:

1. **Nodes** = point masses with position and velocity
2. **Edges** = linear springs (configurable stiffness, rest length)
3. **Boundary** = fixed nodes (Dirichlet BC) during stretch
4. **Relaxation** = initial energy minimization before loading
5. **Loading** = controlled displacement to target stretch ratio

```
F_spring = k × (L − L₀) / L₀ × direction
F_damping = −c × v_rel · direction × direction × L₀
F_drag = −γ × v
```

### Beam Frame FEM

Euler–Bernoulli beam elements with welded joints:

1. **Nodes** = welded joints (rigid connection, moment transfer)
2. **Elements** = beam elements with axial + bending stiffness
3. **Radius** = fiber radius determines EA and EI (physical cross-section)
4. **Boundary** = 10% each side fixed (Dirichlet BC)
5. **Solver** = linear (small deformation) or co-rotational nonlinear (large deformation)

```
K_global × U = F    →    σ = E × B × U_element
```

### Parametric Structure Control (for RL)

Each edge can have `n_pts_per_side` internal nodes with programmable `(dx, dy)` displacement:

```
Action = [dx₁, dy₁, dx₂, dy₂, ..., dxₙ, dyₙ] ∈ [−0.3, 0.3]^(2n)
```

For square with `n_pts_per_side=5`: **40 continuous parameters** (20 displacement pairs).

---

## 📁 Project Structure

```
fibernet/
├── fibernet/
│   ├── core/         # StructureGraph, Material, transforms
│   ├── gen/          # pattern_2d/3d, unit factories (26 types)
│   ├── sim/          # TaichiEngine (mass-spring), SimResult
│   ├── ml/           # BeamFrameFEM, train_predictor, cross_validate
│   ├── viz/          # render_graph, render_trajectory, themes
│   ├── analysis/     # GraphFeatureExtractor (94-dim)
│   ├── rl/           # CEM env, Bayesian opt, reward curves
│   └── easy.py       # show(), simulate(), batch_simulate()
├── tutorials/        # Jupyter notebook + standalone runner
├── tests/            # 312 tests (pytest)
├── examples/         # 19 example scripts
├── docs/             # Sphinx documentation + images
└── pyproject.toml    # build configuration
```

---

## 📝 Citation

```bibtex
@software{fibernet2026,
  title = {FiberNet: Python Toolkit for Fiber Network Design and Optimization},
  author = {ML-BioMat Lab, BMG-FDU},
  year = {2026},
  url = {https://github.com/GellmanSparrowS/fibernet},
  version = {4.1.5},
}
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE).

---

<div align="center">

**[中文文档](README_CN.md)** · [PyPI](https://pypi.org/project/fibernet/4.1.5/) · [GitHub](https://github.com/GellmanSparrowS/fibernet)

</div>
