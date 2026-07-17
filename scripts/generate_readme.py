"""Generate bilingual README.md (EN) and README_CN.md (CN) with language toggle.

Usage:
    python scripts/generate_readme.py

Output:
    README.md      - English version (primary)
    README_CN.md   - Chinese version
"""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)

# ── Shared Content ──────────────────────────────────────────────

BADGES = """\
[![PyPI](https://img.shields.io/pypi/v/fibernet?logo=pypi&logoColor=white)](https://pypi.org/project/fibernet/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CI](https://github.com/GellmanSparrowS/fibernet/actions/workflows/ci.yml/badge.svg)](https://github.com/GellmanSparrowS/fibernet/actions)
[![Downloads](https://img.shields.io/pypi/dm/fibernet)](https://pypi.org/project/fibernet/)"""

CODE_GENERATION = '''\
```python
import fibernet as fn

g = fn.pattern_2d(unit="honeycomb", box=(10, 10), grid=(4, 4))
fn.show(g)  # one-line visualization
```'''

CODE_SIMULATE = '''\
```python
r = fn.simulate(g, mode="stretch", strain=1.5, backend="spring")
print(f"max_force={r.max_force:.0f} N, max_stretch={r.max_stretch:.3f}")
```'''

CODE_PIPELINE = '''\
```python
import fibernet as fn
import numpy as np

# 1. Parametric structure (20 displacement params for RL)
displacements = [(np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3))
                 for _ in range(20)]
g = fn.pattern_2d(unit="square", box=(10, 10), grid=(3, 3),
                  n_pts_per_side=5, point_displacements=displacements)

# 2. Taichi simulation
engine = fn.TaichiEngine()
r = engine.stretch_test(g, target_stretch=1.5, stiffness=1e5,
                        damping=0.3, num_steps=1000, save_interval=200)

# 3. Visualization with stress
fig = fn.render_trajectory(g, r.positions_trajectory, r.edge_stretches,
                           n_frames=6, title="Stretch Process")
fig.savefig("deformation.png", dpi=150)

# 4. Feature extraction (94-dim vector)
ext = fn.GraphFeatureExtractor()
features = ext.extract(g)

# 5. Node manipulation (for RL action space)
internal = g.get_internal_nodes()
g.displace_node(internal[0], [0.1, 0.2])
```'''

CODE_NODE = '''\
```python
g.displace_node(node_id, [dx, dy])          # relative displacement
g.set_node_position(node_id, [x, y])        # absolute position
g.set_node_positions({1: [2.5, 0.5], 3: [7.5, 1.0]})  # batch

internal = g.get_internal_nodes()  # RL action targets
boundary = g.get_boundary_nodes()
```'''

CODE_SIM_DETAIL = '''\
```python
g = fn.pattern_2d(unit="honeycomb", box=(10, 10), grid=(4, 4))
engine = fn.TaichiEngine()
r = engine.stretch_test(g,
    target_stretch=1.5,     # stretch ratio
    stiffness=1e5,          # spring constant
    damping=0.3,            # damping ratio
    num_steps=5000,         # total steps
    ramp_fraction=0.2,      # 20% ramp + 80% hold (relaxation)
    save_interval=1000)

r.max_force           # max edge force (N)
r.edge_forces         # per-edge forces
r.edge_stretches      # per-edge stretch ratios
r.positions_trajectory # list of (N,3) position arrays

r.save("result.json", detailed=True)
r2 = fn.SimResult.load("result.json")
```'''

CODE_VIZ = '''\
```python
fig = fn.render_graph(g, theme="dark")       # dark purple
fig = fn.render_graph(g, theme="light")      # white background
fig = fn.render_graph(g, theme="blueprint")  # blueprint style

fig = fn.render_deformation(g_original, g_deformed, color_by="stress")
fig = fn.render_trajectory(g, r.positions_trajectory, r.edge_stretches,
                           n_frames=6, title="Stretch Process")
```'''

CODE_ML = '''\
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
```'''

CODE_RL = '''\
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
```'''

CODE_RL_PARAM = '''\
```python
# 40-dim action vector → 20 (dx,dy) pairs
action = agent.act(obs)  # shape: (40,), range: [-0.3, 0.3]
displacements = [(action[2*i], action[2*i+1]) for i in range(20)]
g = fn.pattern_2d(unit="square", grid=(3,3), n_pts_per_side=5,
                  point_displacements=displacements)

# Post-generation refinement
for nid in g.get_internal_nodes():
    g.displace_node(nid, agent.refine(nid))
```'''


# ── English README ──────────────────────────────────────────────

README_EN = f"""\
<div align="center">

# 🧬 FiberNet v4

**Python Toolkit for Fiber Network Design, Simulation & Intelligent Optimization**

{BADGES}

[中文文档](README_CN.md) · [PyPI](https://pypi.org/project/fibernet/) · [Tutorial](#-tutorial) · [API Docs](#-api-reference)

*Developed by [ML-BioMat Lab](https://ml-biomat.com/) @ [BMG-FDU](https://github.com/BMG-FDU)*

</div>

---

## Overview

FiberNet is a research-grade Python toolkit for **computational design of fiber network metamaterials**. It provides a complete closed-loop workflow:

```
Generation → Simulation → Feature Extraction → Machine Learning → Reinforcement Learning
```

| Feature | Description |
|---------|-------------|
| **12 Unit Types** | square, triangle, hexagon, honeycomb, kagome, voronoi, chiral, reentrant, star, cross, diamond, missing_rib |
| **Parametric Control** | Internal point displacements for RL-ready continuous action spaces |
| **Taichi Simulation** | GPU-accelerated mass-spring dynamics with auto-relaxation |
| **94-Dim Features** | Structural + pore + contact feature extraction |
| **One-Line ML** | `predict_from_csv()` → train, evaluate, visualize, save |
| **One-Line RL** | `run_bayesian_optimization()` or CEM optimization |

---

## 🖼️ Showcase

<div align="center">
<img src="docs/images/01_2d_gallery.png" width="80%" alt="2D Structure Gallery" />
</div>

*12 unit types: square, triangle, hexagon, honeycomb, kagome, voronoi, chiral, reentrant, star, cross, diamond, missing_rib.*

<div align="center">
<img src="docs/images/voronoi_1.5x_auto.png" width="80%" alt="Voronoi Stretch" />
</div>

*Voronoi structure under 1.5× uniaxial stretch — deformation and stress distribution.*

<div align="center">
<img src="docs/images/05_trajectory_dark.png" width="80%" alt="Deformation Trajectory" />
</div>

*8-frame deformation trajectory: honeycomb under stretch, colored by edge stretch ratio.*

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

{CODE_GENERATION}

{CODE_SIMULATE}

### Complete Pipeline

{CODE_PIPELINE}

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

## 📚 API Reference

### Structure Generation

```python
import numpy as np
disps = [(np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3))
         for _ in range(20)]

g = fn.pattern_2d(
    unit="square",              # 12 unit types available
    box=(10, 10),               # cell size
    grid=(3, 3),                # tiling grid
    n_pts_per_side=5,           # internal points per edge
    point_displacements=disps,  # parametric control
    seed=42,
)

print(fn.list_units())
# ['chiral', 'cross', 'diamond', 'hexagon', 'honeycomb', 'kagome',
#  'missing_rib', 'reentrant', 'square', 'star', 'triangle', 'voronoi']
```

### Node Manipulation (RL Action Space)

{CODE_NODE}

### Simulation

{CODE_SIM_DETAIL}

### Visualization

{CODE_VIZ}

### Machine Learning

{CODE_ML}

### Reinforcement Learning

{CODE_RL}

### 🎯 RL Parametric Control

FiberNet exposes **(dx, dy) displacement parameters** for each internal point on every edge — a continuous action space for RL, equivalent to `move_AB(G, num, dx, dy)` in research code.

{CODE_RL_PARAM}

---

## 🎓 Tutorial

A complete end-to-end Jupyter notebook is available:

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
F_spring = k × (L - L₀) / L₀ × direction
F_damping = -c × v_rel · direction × direction × L₀
F_drag = -γ × v  (air drag)
```

### Parametric Structure Control (for RL)

Each edge can have `n_pts_per_side` internal nodes with programmable `(dx, dy)` displacement:

```
Action = [dx₁, dy₁, dx₂, dy₂, ..., dxₙ, dyₙ] ∈ [-0.3, 0.3]^(2n)
```

For square with `n_pts_per_side=5`: **40 continuous parameters** (20 displacement pairs).

---

## 📁 Project Structure

```
fibernet/
├── fibernet/
│   ├── core/         # StructureGraph, Material, transforms
│   ├── gen/          # pattern_2d/3d, unit factories
│   ├── sim/          # TaichiEngine (mass-spring), SimResult
│   ├── viz/          # render_graph, render_trajectory, themes
│   ├── analysis/     # GraphFeatureExtractor (94-dim)
│   ├── ml/           # train_predictor, cross_validate, plots
│   ├── rl/           # CEM env, Bayesian opt, reward curves
│   └── easy.py       # show(), simulate(), batch_simulate()
├── tutorials/        # Jupyter notebook + standalone runner
├── tests/            # 189 tests (pytest --forked)
├── examples/         # 19 example scripts
└── pyproject.toml    # build configuration
```

---

## 📊 Performance

| Task | Time | Hardware |
|------|------|----------|
| Generation (square 3×3) | ~0.1s | CPU |
| Stretch simulation (5000 steps) | ~6s | Taichi x64 |
| Feature extraction (94-dim) | ~0.5s | CPU |
| ML training (RF, 100 samples) | ~1s | CPU |
| CEM optimization (200 episodes) | ~6 min | CPU |

---

## 📝 Citation

```bibtex
@software{{fibernet2026,
  title = {{FiberNet: Python Toolkit for Fiber Network Design and Optimization}},
  author = {{ML-BioMat Lab, BMG-FDU}},
  year = {{2026}},
  url = {{https://github.com/GellmanSparrowS/fibernet}},
  version = {{4.0.5}},
}}
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE).

---

<div align="center">

**[中文文档](README_CN.md)** · [PyPI](https://pypi.org/project/fibernet/4.0.5/) · [GitHub](https://github.com/GellmanSparrowS/fibernet)

</div>
"""


# ── Chinese README ──────────────────────────────────────────────

README_CN = f"""\
<div align="center">

# 🧬 FiberNet v4

**纤维网络结构生成、力学模拟与智能优化 Python 工具包**

{BADGES}

[English](README.md) · [PyPI](https://pypi.org/project/fibernet/) · [教程](#-教程) · [API 参考](#-api-参考)

*由 [ML-BioMat Lab](https://ml-biomat.com/) @ [BMG-FDU](https://github.com/BMG-FDU) 开发*

</div>

---

## 概述

FiberNet 是一个面向材料科学的 Python 工具包，用于**纤维网络超结构的计算设计**。提供完整闭环工作流：

```
结构生成 → 力学模拟 → 特征提取 → 机器学习 → 强化学习
```

| 功能 | 说明 |
|------|------|
| **12种基元** | square, triangle, hexagon, honeycomb, kagome, voronoi, chiral, reentrant, star, cross, diamond, missing_rib |
| **参数化控制** | 内部点位移，支持RL连续动作空间 |
| **Taichi模拟** | GPU加速质点弹簧动力学，自动弛豫 |
| **94维特征** | 结构 + 孔隙 + 接触特征提取 |
| **一行ML** | `predict_from_csv()` → 训练、评估、可视化、保存 |
| **一行RL** | `run_bayesian_optimization()` 或 CEM 优化 |

---

## 🖼️ 展示

<div align="center">
<img src="docs/images/01_2d_gallery.png" width="80%" alt="2D结构库" />
</div>

*12种基元类型：正方形、三角形、六边形、蜂窝、kagome、Voronoi、手性、凹角、星形、交叉、钻石、缺肋。*

<div align="center">
<img src="docs/images/voronoi_1.5x_auto.png" width="80%" alt="Voronoi拉伸" />
</div>

*Voronoi结构在1.5×单轴拉伸下的变形与应力分布。*

<div align="center">
<img src="docs/images/05_trajectory_dark.png" width="80%" alt="变形轨迹" />
</div>

*8帧变形轨迹：蜂窝结构拉伸过程，按边拉伸比着色。*

<div align="center">
<img src="docs/images/09_ml_analysis_dark.png" width="80%" alt="ML分析" />
</div>

*机器学习分析：混淆矩阵、ROC曲线和学习曲线。*

<div align="center">
<img src="docs/images/11_rl_reward_dark.png" width="80%" alt="RL奖励" />
</div>

*CEM强化学习：每episode奖励和单调递增最佳奖励。*

---

## 🚀 快速开始

### 一行API

```python
import fibernet as fn

g = fn.pattern_2d(unit="honeycomb", box=(10, 10), grid=(4, 4))
fn.show(g)  # 一行出图
```

```python
r = fn.simulate(g, mode="stretch", strain=1.5, backend="spring")
print(f"最大力={{r.max_force:.0f}} N, 最大拉伸={{r.max_stretch:.3f}}")
```

### 完整流水线

{CODE_PIPELINE}

---

## 📦 安装

```bash
pip install fibernet          # 核心
pip install fibernet[full]    # 完整 (ML + RL + 可视化 + 模拟)
pip install fibernet[ml]      # 仅ML
pip install fibernet[rl]      # 仅RL
```

| 可选组 | 包 |
|--------|-----|
| `ml` | scikit-learn, pandas, tqdm |
| `rl` | gymnasium, scikit-optimize, stable-baselines3 |
| `accel` | taichi (GPU模拟) |
| `viz` | pyvista (3D可视化) |
| `full` | 以上全部 |

---

## 📚 API 参考

### 结构生成

```python
import numpy as np
disps = [(np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3))
         for _ in range(20)]

g = fn.pattern_2d(
    unit="square",              # 12种可用基元
    box=(10, 10),               # 单元格尺寸
    grid=(3, 3),                # 铺排网格
    n_pts_per_side=5,           # 每边内部点数
    point_displacements=disps,  # 参数化控制
    seed=42,
)

print(fn.list_units())
# ['chiral', 'cross', 'diamond', 'hexagon', 'honeycomb', 'kagome',
#  'missing_rib', 'reentrant', 'square', 'star', 'triangle', 'voronoi']
```

### 节点操控（RL动作空间）

{CODE_NODE}

### 模拟

{CODE_SIM_DETAIL}

### 可视化

{CODE_VIZ}

### 机器学习

{CODE_ML}

### 强化学习

{CODE_RL}

### 🎯 RL参数化控制

FiberNet 为每条边上的每个内部点暴露 **(dx, dy) 位移参数**，为强化学习提供连续动作空间，等效于研究代码中的 `move_AB(G, num, dx, dy)`。

{CODE_RL_PARAM}

---

## 🎓 教程

完整端到端 Jupyter 教程：

```
tutorials/complete_tutorial_v4.ipynb
```

独立运行脚本（支持断点续跑）：

```bash
python3 tutorials/run_pipeline.py                        # 完整流程
python3 tutorials/run_pipeline.py --num-structures 100   # 快速测试
python3 tutorials/run_pipeline.py --skip-rl              # 跳过RL
```

覆盖：结构生成 → 批量模拟 → 形变可视化 → 特征提取 → 机器学习 → CEM强化学习。

---

## 🔬 工作原理

### 质点弹簧模型 (Taichi)

GPU加速质点弹簧动力学：

1. **节点** = 带位置和速度的质点
2. **边** = 线性弹簧（可配置刚度和静息长度）
3. **边界** = 拉伸时固定节点（Dirichlet边界条件）
4. **弛豫** = 加载前的初始能量最小化
5. **加载** = 受控位移至目标拉伸比

```
F_弹簧 = k × (L - L₀) / L₀ × 方向
F_阻尼 = -c × v_相对 · 方向 × 方向 × L₀
F_阻力 = -γ × v  (空气阻力)
```

### 参数化结构控制（用于RL）

每条边可有 `n_pts_per_side` 个内部节点，每个带可编程 `(dx, dy)` 位移：

```
动作 = [dx₁, dy₁, dx₂, dy₂, ..., dxₙ, dyₙ] ∈ [-0.3, 0.3]^(2n)
```

正方形 `n_pts_per_side=5`：**40个连续参数**（20个位移对）。

---

## 📊 性能

| 任务 | 时间 | 硬件 |
|------|------|------|
| 结构生成 (square 3×3) | ~0.1s | CPU |
| 拉伸模拟 (5000步) | ~6s | Taichi x64 |
| 特征提取 (94维) | ~0.5s | CPU |
| ML训练 (RF, 100样本) | ~1s | CPU |
| CEM优化 (200 episodes) | ~6 min | CPU |

---

## 📝 引用

```bibtex
@software{{fibernet2026,
  title = {{FiberNet: 纤维网络设计与优化Python工具包}},
  author = {{ML-BioMat Lab, BMG-FDU}},
  year = {{2026}},
  url = {{https://github.com/GellmanSparrowS/fibernet}},
  version = {{4.0.5}},
}}
```

---

## 📄 许可证

MIT License. 详见 [LICENSE](LICENSE)。

---

<div align="center">

**[English](README.md)** · [PyPI](https://pypi.org/project/fibernet/4.0.5/) · [GitHub](https://github.com/GellmanSparrowS/fibernet)

</div>
"""


def main():
    en_path = os.path.join(REPO_DIR, "README.md")
    cn_path = os.path.join(REPO_DIR, "README_CN.md")

    with open(en_path, "w") as f:
        f.write(README_EN)
    print(f"Generated {en_path} ({len(README_EN)} chars)")

    with open(cn_path, "w") as f:
        f.write(README_CN)
    print(f"Generated {cn_path} ({len(README_CN)} chars)")


if __name__ == "__main__":
    main()
