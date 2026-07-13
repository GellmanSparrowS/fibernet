<div align="center">

# 🧬 FiberNet

### Research-Grade Python Toolkit for Fiber Network Design & Optimization
### 面向材料科学的纤维网络结构生成、模拟与智能优化工具包

---

[![PyPI version](https://img.shields.io/pypi/v/fibernet.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/fibernet/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/GellmanSparrowS/fibernet/actions/workflows/ci.yml/badge.svg)](https://github.com/GellmanSparrowS/fibernet/actions/workflows/ci.yml)
[![Downloads](https://img.shields.io/pypi/dm/fibernet.svg?label=Downloads&color=brightgreen)](https://pypi.org/project/fibernet/)
[![Docs](https://readthedocs.org/projects/fibernet/badge/?version=latest)](https://fibernet.readthedocs.io/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

**[Installation](#-installation)** · **[Quick Start](#-quick-start)** · **[Features](#-key-features)** · **[Tutorials](#-tutorials)** · **[Documentation](https://fibernet.readthedocs.io/)** · **[Examples](examples/)**

*Developed by [ML-BioMat Lab](https://ml-biomat.com/) @ [BMG-FDU](https://github.com/BMG-FDU)*

---

</div>

## 📖 Overview

FiberNet is a comprehensive Python toolkit for **computational design of fiber network structures** — from simple random deposition to architectured metamaterials. It provides a complete workflow from structure generation to mechanical simulation and intelligent optimization.

### ✨ Core Capabilities

- **80+ Network Generators** spanning 15 architecture families, including **field-guided synthesis** for biomimetic alignment patterns
- **Custom FEM Solver** (Euler-Bernoulli beam theory, built on NumPy + SciPy, no external FEM dependencies)
- **Reinforcement Learning** environments for inverse design and multi-objective optimization
- **94-D Feature Extraction** for structure-property analysis
- **22+ Physics Modules** — mechanics, dynamics, fracture, thermal, electromagnetic, fluid, acoustic
- **Gibson-Ashby Validation** — analytical benchmarks for cellular solid models
- **TPMS Support** — Triply Periodic Minimal Surface lattice generation
- **Optional GPU Acceleration** via [Taichi](https://github.com/taichi-dev/taichi)

---

## 🚀 Installation

### Option 1: Install from PyPI (Recommended)

```bash
# Standard installation (core functionality)
pip install fibernet

# Full installation with all optional dependencies
pip install fibernet[full]
```

### Option 2: Install from GitHub

```bash
pip install git+https://github.com/GellmanSparrowS/fibernet.git

# With full dependencies
pip install "fibernet[full] @ git+https://github.com/GellmanSparrowS/fibernet.git"
```

### Option 3: Development Installation

```bash
git clone https://github.com/GellmanSparrowS/fibernet.git
cd fibernet
pip install -e ".[dev,full]"
```

### Optional Dependency Groups

| Group | Description | Install Command |
|:------|:------------|:----------------|
| `viz` | PyVista 3D + matplotlib 2D visualization | `pip install fibernet[viz]` |
| `mesh` | Trimesh mesh operations | `pip install fibernet[mesh]` |
| `io` | HDF5 support via h5py | `pip install fibernet[io]` |
| `accel` | Taichi GPU acceleration | `pip install fibernet[accel]` |
| `ml` | scikit-learn ML integration | `pip install fibernet[ml]` |
| `graph` | NetworkX graph analysis | `pip install fibernet[graph]` |
| `rl` | Gymnasium + stable-baselines3 | `pip install fibernet[rl]` |
| `full` | All optional dependencies | `pip install fibernet[full]` |

---

## ⚡ Quick Start

### Generate → Analyze → Simulate

```python
import fibernet as fn
import numpy as np

# 1. Generate a random 2D fiber network
net = fn.create("random_2d", num_fibers=100, fiber_length=10.0,
                box_size=(30, 30), seed=42)

# 2. Analyze structural properties
stats = fn.analyze(net)
print(f"Fibers: {stats['num_fibers']}, Nematic order: {stats['nematic_order']:.3f}")

# 3. Run mechanical simulation (CPU, custom FEM)
from fibernet.sim.mechanical import FiberFEM
fem = FiberFEM(net, segments_per_fiber=5)
E_eff = fem.effective_modulus(strain=0.001)
print(f"Effective modulus: {E_eff:.2e} Pa")
```

### 🎯 Field-Guided Network Generation (NEW in v1.25)

Generate biomimetic fiber networks guided by orientation fields:

```python
from fibernet.gen.field_guided import (
    OrientationField, FieldGuidedConfig, field_guided_network
)

# Create radial orientation field
field = OrientationField(canvas_size=512, field_type="radial")

# Generate network following the field
config = FieldGuidedConfig(
    fiber_count=2000,
    field_strength=0.7,
    fiber_length_mean=100.0,
    seed=42
)
net = field_guided_network(config=config, field=field, box_size=(50, 50))

print(f"Generated {len(net.fibers)} fibers with field-guided alignment")
```

### 🔄 Reinforcement Learning for Inverse Design (NEW in v1.25)

Use RL to find optimal structures with target properties:

```python
from fibernet.sim.rl_environment import FiberNetworkEnv, RLEnvConfig

# Create RL environment
env = FiberNetworkEnv(RLEnvConfig(
    target_modulus=1e7,      # Target: 10 MPa
    target_poisson=-0.3,     # Target: auxetic
    generator_type="reentrant"
))

# Run optimization
obs, info = env.reset()
best_reward = -np.inf
best_params = None

for step in range(50):
    action = env.action_space.sample()  # or use your RL algorithm
    obs, reward, done, info = env.step(action)
    
    if reward > best_reward:
        best_reward = reward
        best_params = info['params']
    
    if done:
        break

print(f"Best: E={info['modulus']:.2e} Pa, ν={info['poisson']:.3f}")
```

### 📊 94-D Feature Extraction

Extract comprehensive structural features for ML:

```python
from fibernet.analysis.graph_features import GraphFeatureExtractor

extractor = GraphFeatureExtractor(canvas_size=512)
features = extractor.extract(net)

print(f"Extracted {len(features)} features:")
print(f"  Nodes: {features['n_node']}, Edges: {features['n_edge']}")
print(f"  Anisotropy: {features['anisotropy']:.3f}")
print(f"  Nematic order: {features.get('nematic_order', 0):.3f}")
```

---

## 🏗️ Complete Closed-Loop Pipeline

FiberNet enables a complete **generative → predictive → optimization** workflow:

```
┌─────────────────────────────────────────────────────────────┐
│  1. STRUCTURE GENERATION                                     │
│  • 80+ generators (random, ordered, chiral, woven, TPMS)    │
│  • Field-guided synthesis for biomimetic patterns           │
│  • Hierarchical and metamaterial architectures              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  2. MECHANICAL SIMULATION (Custom FEM)                       │
│  • Euler-Bernoulli beam elements (12 DOF/element)           │
│  • scipy.sparse + SuperLU direct solver                     │
│  • Linear, nonlinear, hyperelastic, plasticity models       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  3. FEATURE EXTRACTION & ML                                  │
│  • 94-dimensional structural features                       │
│  • Train surrogate models (RF, GBM, MLP, GNN)              │
│  • Rapid property prediction (microseconds vs seconds)      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  4. REINFORCEMENT LEARNING OPTIMIZATION                      │
│  • RL agent proposes new structural parameters              │
│  • ML surrogate provides instant reward feedback            │
│  • Multi-objective optimization (stiffness + auxetic)       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  5. VALIDATION                                               │
│  • Gibson-Ashby scaling laws                                │
│  • Cantilever beam analytical solutions                     │
│  • FEM verification of RL-optimized designs                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎨 Key Features

| Category | Capabilities |
|:---------|:-------------|
| **Generators** | 80+ across 15 families: random, ordered, chiral, woven, hierarchical, **TPMS** (gyroid, diamond, primitive), bundles, curved, laminates, fractal, gradient, biomimetic, CNT, **field-guided** |
| **FEM Solver** | Custom Euler-Bernoulli 3D beam elements (12 DOF/element), sparse direct solver (SuperLU), Tikhonov regularization, h-refinement convergence |
| **Mechanics** | Linear elastic, nonlinear (Newton-Raphson), hyperelastic (Neo-Hookean, Mooney-Rivlin), plasticity, viscoelasticity, damage/fatigue, fracture (LEFM, cohesive zone) |
| **Reinforcement Learning** | Gymnasium-compatible environments, multi-objective optimization, stable-baselines3 support (PPO/SAC/TD3) |
| **Feature Extraction** | 94-dimensional vector: 34 structural, 18 pore, 42 contact features |
| **Validation** | Gibson-Ashby benchmarks, cantilever analytical solutions, patch tests, convergence studies |
| **Multi-Physics** | Thermal, electromagnetic, acoustic, fluid (Darcy flow), rheology, DMA, diffusion, coupled fields, buckling |
| **Analysis** | Morphology, topology, spectral, percolation, pore structure, homogenization, anisotropy, effective properties |
| **ML** | Feature extraction (30+), GNN models, property prediction, dataset generation |
| **I/O** | JSON, YAML, LAMMPS, VTK, GMSH, PDB, XYZ, HDF5, pandas DataFrames |
| **Visualization** | PyVista 3D interactive, matplotlib 2D, Plotly web, animations |
| **Acceleration** | Taichi CPU/GPU parallel FEM (optional) |

---

## 📚 Tutorials

Complete Jupyter notebooks demonstrating the full workflow:

| Tutorial | Description | Key Topics |
|:---------|:------------|:-----------|
| [`01_getting_started.ipynb`](tutorials/01_getting_started.ipynb) | Installation, core concepts, basic workflow | FiberNetwork, Material, basic analysis |
| [`02_mechanical_simulation.ipynb`](tutorials/02_mechanical_simulation.ipynb) | FEM mechanics, stress-strain, effective properties | FiberFEM, constitutive models, validation |
| [`03_machine_learning.ipynb`](tutorials/03_machine_learning.ipynb) | Feature extraction, GNN, property prediction | Graph features, RF/GBM/MLP, GNN |
| [`metamaterial_design.ipynb`](tutorials/metamaterial_design.ipynb) | **Complete pipeline with RL** | Structure generation → FEM → ML → **RL optimization** → **Validation** |
| [`04_reinforcement_validation.ipynb`](tutorials/04_reinforcement_validation.ipynb) | TPMS, FEM validation, Gibson-Ashby | TPMS lattices, convergence, benchmarks |
| [`05_complete_pipeline_rl_validation.ipynb`](tutorials/05_complete_pipeline_rl_validation.ipynb) | Diverse structures + RL + validation | Field-guided, multi-scale, closed-loop |

### Tutorial Highlights

#### `metamaterial_design.ipynb` — The Complete Story

This tutorial walks through the **entire closed-loop pipeline**:

1. ✅ **Generate** 9+ metamaterial structures (auxetic, chiral, star, arrowhead, octet, diamond, hierarchical, woven, braided)
2. ✅ **Visualize** with publication-quality matplotlib plots
3. ✅ **Parametric study**: Sweep re-entrant angle, compute E*, ν*, ρ*
4. ✅ **FEM simulation**: Stress-strain curves, deformation visualization
5. ✅ **ML surrogate**: Train RF/GBM/MLP models, screen 500 candidates
6. ✅ **Reinforcement Learning**: Closed-loop optimization using ML as reward
7. ✅ **Validation**: Cantilever beam, Gibson-Ashby scaling, FEM verification

**All computations on CPU, no GPU required.**

---

## 🔬 How the Mechanical Simulation Works

FiberNet implements a **custom finite element solver** — it does **not** wrap or depend on any existing open-source FEM library (FEniCS, SfePy, PyFEM, etc.).

### Technical Highlights

| Aspect | Implementation |
|:-------|:---------------|
| **Element type** | 3D Euler-Bernoulli beam (12 DOF/element: 3 translations + 3 rotations per node) |
| **Stiffness matrix** | Analytical 12×12 local stiffness with coordinate transformation |
| **Sparse solver** | `scipy.sparse.linalg.spsolve` (SuperLU direct) |
| **Regularization** | Tikhonov regularization for near-singular systems |
| **Nonlinear** | Newton-Raphson iteration with arc-length control |
| **Constitutive models** | Linear elastic, bilinear plasticity, Neo-Hookean, Mooney-Rivlin, Arruda-Boyce, Maxwell, Kelvin-Voigt |
| **Dependencies** | Only NumPy + SciPy (no heavy FEM stack) |

### Mathematical Foundation

**Euler-Bernoulli Beam Theory:**
```
δ_tip = PL³ / (3EI)              # Cantilever deflection
E_eff = 2U / (V · ε²)            # Effective modulus from strain energy
```

**Gibson-Ashby Cellular Solids:**
```
E*/E_s ∝ (ρ*/ρ_s)³              # 2D honeycomb (bending-dominated)
E*/E_s ∝ (ρ*/ρ_s)²              # 3D open-cell foam (stretching-dominated)
```

> 📖 See [docs/fem_implementation.md](docs/fem_implementation.md) for full mathematical details.

### Validation Results

```python
from fibernet.sim.validation import run_all_validations, print_validation_report

results = run_all_validations(E_solid=1e9)
print(print_validation_report(results))
```

**Benchmark tests:**
- ✅ Cantilever beam vs. Euler-Bernoulli analytical solution (error: 0.00%)
- ✅ Gibson-Ashby honeycomb scaling (E* ∝ ρ³)
- ✅ Patch test (uniform strain, machine precision)
- ✅ h-refinement convergence study

---

## 🎯 Field-Guided Network Generation (NEW in v1.25)

Generate biomimetic fiber networks guided by **orientation fields**, inspired by natural fiber alignment patterns (collagen, cellulose, muscle tissue).

### Supported Field Types

| Field Type | Description | Use Case |
|:-----------|:------------|:---------|
| `uniform` | Constant direction everywhere | Aligned composites |
| `radial` | Outward from center | Radial symmetry |
| `vortex` | Circular pattern | Torsional structures |
| `gradient` | Linear variation | Functionally graded materials |
| `random_smooth` | Smoothly varying random field | Biomimetic networks |

### Example: Vortex Pattern

```python
from fibernet.gen.field_guided import (
    OrientationField, FieldGuidedConfig, 
    field_guided_network, multi_scale_orientation_analysis
)
import matplotlib.pyplot as plt

# Create vortex field
field = OrientationField(canvas_size=512, field_type="vortex")

# Visualize field
fig, ax = plt.subplots(figsize=(8, 8))
field.visualize(ax=ax, stride=30, length=20)
plt.show()

# Generate network
config = FieldGuidedConfig(fiber_count=3000, field_strength=0.8, seed=42)
net = field_guided_network(config=config, field=field, box_size=(50, 50))

# Analyze orientation
analysis = multi_scale_orientation_analysis(net)
print(f"Nematic order: {analysis['nematic_order']:.3f}")
print(f"Dominant angle: {np.degrees(analysis['dominant_angle']):.1f}°")
```

---

## 🔄 Reinforcement Learning for Inverse Design (NEW in v1.25)

Use RL to autonomously discover optimal structures with target mechanical properties.

### Closed-Loop Architecture

```
Generate Structure → FEM Simulate → Extract Features → Train ML Model
      ↑                                                          ↓
      └── RL Agent proposes new parameters ← Evaluate with ML ←──┘
```

### Gymnasium-Compatible Environment

```python
from fibernet.sim.rl_environment import FiberNetworkEnv, RLEnvConfig

# Define target properties
config = RLEnvConfig(
    target_modulus=5e7,      # 50 MPa
    target_poisson=-0.3,     # Auxetic
    generator_type="reentrant",
    reward_mode="multi_objective"
)

env = FiberNetworkEnv(config)

# Compatible with stable-baselines3
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Get optimized design
obs, info = env.reset()
for _ in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        break

print(f"Optimized: E={info['modulus']:.2e} Pa, ν={info['poisson']:.3f}")
```

### Supported Generator Types

| Generator | Parameters | Use Case |
|:----------|:-----------|:---------|
| `reentrant` | angle, grid_x, grid_y, radius | Auxetic metamaterials |
| `random` | num_fibers, fiber_length, radius | Random networks |
| `honeycomb` | grid_x, radius | Regular honeycomb |
| `field_guided` | fiber_count, field_strength, radius | Biomimetic patterns |

---

## 🏛️ Architecture

```
fibernet/
├── core/              Fiber, FiberNetwork, Material, Crosslink, PBC, Transform
├── gen/               80+ network generators (15 modules)
│   ├── ordered.py       Square, triangular, honeycomb, cubic, octet, kagome
│   ├── disordered.py    Random deposition, oriented, Poisson disk
│   ├── chiral.py        Helix, double helix, braid, twisted bundle
│   ├── woven.py         Plain, twill, satin, 3D orthogonal
│   ├── hierarchical.py  Bundle, gradient, core-shell, fractal
│   ├── metamaterials.py Re-entrant, chiral, star, arrowhead, diamond, gyroid
│   ├── tpms.py          Gyroid, Diamond, Primitive, I-WP, Neovius
│   ├── bundles.py       Parallel, twisted, braided, tendon
│   ├── curved.py        Sinusoidal, helical, Bezier, crimped
│   ├── laminates.py     Unidirectional, cross-ply, angle-ply, sandwich
│   ├── field_guided.py  Orientation field, field-guided synthesis (NEW)
│   └── ...              (advanced, fractal, gradient, specialized, variants)
├── sim/               22+ physics modules
│   ├── mechanical.py    Custom FEM (Euler-Bernoulli beam, scipy.sparse)
│   ├── nonlinear.py     Newton-Raphson, hyperelastic, plasticity
│   ├── validation.py    Gibson-Ashby benchmarks, analytical tests
│   ├── rl_environment.py RL environments for inverse design (NEW)
│   ├── incremental_fem.py Incremental loading with damage
│   ├── buckling_analysis.py Eigenvalue buckling
│   └── ...              (dynamics, fracture, thermal, EM, fluid, acoustic, ...)
├── analysis/          Morphology, topology, percolation, homogenization
│   └── graph_features.py  94-dimensional feature extraction (NEW)
├── ml/                Feature extraction, GNN, prediction
├── viz/               PyVista 3D, matplotlib 2D, Plotly, animations
├── io/                JSON, YAML, LAMMPS, VTK, GMSH, PDB, XYZ, HDF5
└── utils/             Config, validation, parametric, batch, geometry, units
```

---

## 📊 Performance

### Computational Complexity

| Operation | Complexity | Notes |
|:----------|:-----------|:------|
| **FEM Assembly** | O(N_elem × 144) | Linear in number of elements |
| **FEM Solving** | O(N_dof^1.5) for 2D, O(N_dof^2) for 3D | Sparse direct solver |
| **Feature Extraction** | O(N_elem + N_nodes) | Graph-based features |
| **ML Prediction** | O(1) | Microseconds (trained model) |

### Typical Performance (CPU)

| Network Size | Elements | DOF | FEM Solve Time | ML Predict Time |
|:-------------|:---------|:----|:---------------|:----------------|
| Small (2D)   | ~500     | ~3000 | <0.1 s | <0.001 s |
| Medium (2D)  | ~2000    | ~12000 | ~1 s | <0.001 s |
| Large (3D)   | ~10000   | ~60000 | ~10 s | <0.001 s |

### GPU Acceleration (Optional)

The `sim/accelerated.py` module provides Taichi-based GPU acceleration for large-scale problems. However, **all core FEM functionality works on CPU only** with no GPU dependencies.

---

## 📝 Citation

If you use FiberNet in your research, please cite:

```bibtex
@software{fibernet2025,
  title     = {FiberNet: A Comprehensive Python Toolkit for Fiber Network
               Generation, Simulation, and Analysis},
  author    = {FiberNet Contributors},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/GellmanSparrowS/fibernet},
  version   = {1.25.0},
  doi       = {10.5281/zenodo.XXXXXXX}
}
```

---

## 🙏 Acknowledgments

- Built with [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [NetworkX](https://networkx.org/), [matplotlib](https://matplotlib.org/), [PyVista](https://pyvista.org/), [Taichi](https://taichi-lang.org/)
- Supported by the [**ML-BioMat**](https://ml-biomat.com/) research group ([BMG-FDU](https://github.com/BMG-FDU))
- FEM validated against Gibson-Ashby cellular solid theory
- RL environments compatible with [stable-baselines3](https://stable-baselines3.readthedocs.io/)

---

## 📄 License

[MIT License](LICENSE) — free for academic and commercial use.

---

## 🔗 Links

- **PyPI**: https://pypi.org/project/fibernet/
- **Documentation**: https://fibernet.readthedocs.io/
- **GitHub**: https://github.com/GellmanSparrowS/fibernet
- **Examples**: https://github.com/GellmanSparrowS/fibernet/tree/main/examples
- **Tutorials**: https://github.com/GellmanSparrowS/fibernet/tree/main/tutorials

---

<a id="中文"></a>

## 📖 中文概述

FiberNet 是一个面向材料科学研究的**纤维网络结构**生成、模拟与智能优化 Python 工具包。

### 核心特点

- **80+ 网络生成器**：涵盖 15 个结构家族（随机、有序、手性、编织、层级、TPMS、**场引导**等）
- **自研 FEM 求解器**：基于 Euler-Bernoulli 梁理论，仅依赖 NumPy + SciPy，不依赖任何开源 FEM 库
- **强化学习环境**：Gymnasium 兼容，支持稳定基线3（PPO/SAC/TD3）进行逆设计
- **94 维特征提取**：结构、孔隙、接触特征的完整描述
- **Gibson-Ashby 验证**：提供蜂窝材料和泡沫材料的解析解基准测试
- **TPMS 超材料**：支持 Gyroid、Diamond、Primitive 等三周期极小曲面结构
- **22+ 物理模块**：力学、动力学、断裂、热传导、电磁、流体、声学

### 完整闭环流程

```
结构生成 → 力学模拟 → 特征提取 → 机器学习 → 强化学习优化 → 验证
   ↑                                                        ↓
   └────────────────────────────────────────────────────────┘
```

### 快速开始

```bash
pip install fibernet
```

```python
import fibernet as fn

# 生成随机网络
net = fn.create("random_2d", num_fibers=100, fiber_length=10.0)

# 分析结构
stats = fn.analyze(net)
print(f"纤维数: {stats['num_fibers']}, 向序参数: {stats['nematic_order']:.3f}")

# 力学模拟
from fibernet.sim.mechanical import FiberFEM
fem = FiberFEM(net, segments_per_fiber=5)
E_eff = fem.effective_modulus(strain=0.001)
print(f"有效模量: {E_eff:.2e} Pa")
```

### 场引导网络生成（v1.25 新增）

```python
from fibernet.gen.field_guided import OrientationField, field_guided_network

# 创建径向取向场
field = OrientationField(canvas_size=512, field_type="radial")

# 生成仿生纤维网络
net = field_guided_network(field=field, box_size=(50, 50))
print(f"生成 {len(net.fibers)} 根场引导纤维")
```

### 强化学习逆设计（v1.25 新增）

```python
from fibernet.sim.rl_environment import FiberNetworkEnv, RLEnvConfig

# 创建 RL 环境
env = FiberNetworkEnv(RLEnvConfig(
    target_modulus=1e7,      # 目标模量 10 MPa
    target_poisson=-0.3,     # 目标泊松比（拉胀）
))

# 运行优化
obs, info = env.reset()
for _ in range(50):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

print(f"最优: E={info['modulus']:.2e} Pa, ν={info['poisson']:.3f}")
```

### 力学模拟说明

FiberNet 的力学模拟采用**完全自研的有限元实现**：
- 单元类型：3D Euler-Bernoulli 梁单元（12 DOF/单元）
- 稀疏求解：scipy.sparse + SuperLU
- 非线性：Newton-Raphson 迭代
- 本构模型：线弹性、双线性塑性、超弹性（Neo-Hookean）、粘弹性（Maxwell, Kelvin-Voigt）
- 验证：悬臂梁解析解、Patch Test、Gibson-Ashby 标度律

---

<div align="center">

**[⬆ Back to Top](#-fibernet)** · **[PyPI](https://pypi.org/project/fibernet/)** · **[Docs](https://fibernet.readthedocs.io/)** · **[GitHub](https://github.com/GellmanSparrowS/fibernet)**

</div>
