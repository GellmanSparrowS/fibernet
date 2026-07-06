<div align="center">

# 🧶 FiberNet

### A Python Toolkit for Fiber Network Generation, Simulation, and Analysis
### 纤维网络结构生成、模拟与分析 Python 工具包

---

[![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI/CD](https://github.com/GellmanSparrowS/fibernet/actions/workflows/ci.yml/badge.svg)](https://github.com/GellmanSparrowS/fibernet/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-925%2B%20passing-brightgreen.svg)]()
[![Documentation](https://readthedocs.org/projects/fibernet/badge/?version=latest)](https://fibernet.readthedocs.io/en/latest/?badge=latest)

**[🇬🇧 English](#english) &nbsp;|&nbsp; [🇨🇳 中文](#中文)**

Homepage: [ml-biomat.com](https://ml-biomat.com/) &nbsp;|&nbsp; Organization: [BMG-FDU](https://github.com/BMG-FDU)

</div>

---

<a id="english"></a>

## Overview

FiberNet is a research-grade Python toolkit for computational study of **fiber network structures**. It provides a unified framework for generating, simulating, analyzing, and visualizing diverse fiber architectures — from simple random networks to complex hierarchical, chiral, woven, and metamaterial structures.

### Key Features

| Category | Capabilities |
|----------|-------------|
| **Generators** (79+) | Random, ordered, chiral, woven, hierarchical, bundles, curved, fractal, gradient, laminates, **metamaterials** (auxetic, octet, diamond, gyroid) |
| **Simulations** (22+) | FEM mechanics, nonlinear, dynamics, fracture, damage, fatigue, creep, thermal, electromagnetic, acoustic, fluid, rheology, DMA, viscoelastic, multiscale |
| **Analysis** | Morphology, topology, statistics, percolation, homogenization, stress-strain, spatial, effective properties, Poisson's ratio |
| **ML** | Feature extraction, dataset creation, GNN, property prediction |
| **I/O** | JSON, YAML, LAMMPS, VTK, GMSH, PDB, XYZ, HDF5, pandas, FEA export |
| **Visualization** | PyVista 3D, matplotlib 2D, Plotly interactive, animations |

### Installation

```bash
# From GitHub (recommended for latest version)
pip install git+https://github.com/GellmanSparrowS/fibernet.git

# Minimal install (core only)
pip install git+https://github.com/GellmanSparrowS/fibernet.git

# Full install (visualization + ML + all optional deps)
pip install "git+https://github.com/GellmanSparrowS/fibernet.git#egg=fibernet[full]"

# Development install
git clone https://github.com/GellmanSparrowS/fibernet.git
cd fibernet
pip install -e ".[dev,full]"
```

> **Requirements**: Python 3.9+, NumPy ≥ 1.21, SciPy ≥ 1.7

### Quick Start

```python
from fibernet import gen
from fibernet.sim.mechanical import FiberFEM, compute_effective_properties

# Generate a re-entrant auxetic honeycomb
net = gen.reentrant_honeycomb_2d(
    reentrant_angle=150,    # degrees (>120° = auxetic)
    grid_size=(8, 8),
    cell_height=10, cell_width=10,
)
print(f"Fibers: {net.num_fibers}, Crosslinks: {net.num_crosslinks}")

# Compute effective mechanical properties
props = compute_effective_properties(net, strain=0.001)
print(props.summary())

# Visualize
from fibernet.viz.plot2d import plot_network_2d
plot_network_2d(net, title="Re-entrant Auxetic Honeycomb", save_path="auxetic.png")
```

### Metamaterial Structures

FiberNet includes specialized metamaterial generators for mechanics design:

```python
from fibernet.gen.metamaterials import (
    reentrant_honeycomb_2d,   # Classic auxetic (negative Poisson's ratio)
    reentrant_honeycomb_3d,   # 3D re-entrant auxetic
    chiral_honeycomb_2d,      # Node-ligament chiral structure
    star_honeycomb_2d,        # Star-shaped auxetic
    arrowhead_auxetic_2d,     # Fishbone auxetic
    hierarchical_lattice_2d,  # Self-similar multi-scale
    proper_octet_truss_3d,    # Stretch-dominated FCC lattice
    diamond_lattice_3d,       # Tetrahedral coordination
    gyroid_lattice_3d,        # TPMS-inspired
    missing_rib_auxetic_2d,   # Missing-rib model
    plate_lattice_3d,         # Near-optimal stiffness
)
```

### Documentation & Tutorials

- [📓 Full Tutorial Notebook](tutorials/metamaterial_design.ipynb) — Install → Generation → Visualization → FEM → ML pipeline
- [API Documentation](https://fibernet.readthedocs.io/)
- [Example Scripts](examples/)

### Platform Support

| Platform | Python | Status |
|----------|--------|--------|
| Ubuntu | 3.9–3.12 | ✅ All tests pass |
| macOS (ARM64) | 3.9–3.12 | ✅ All tests pass |
| Windows | 3.9–3.12 | ✅ All tests pass |

### Citation

If you use FiberNet in your research, please cite:

```bibtex
@software{fibernet2026,
  title  = {FiberNet: A Python Toolkit for Fiber Network Generation, Simulation, and Analysis},
  author = {BMG-FDU Research Group},
  year   = {2026},
  url    = {https://github.com/GellmanSparrowS/fibernet}
}
```

### License

MIT License. See [LICENSE](LICENSE) and [ATTRIBUTIONS.md](ATTRIBUTIONS.md) for dependency license information.

---

<a id="中文"></a>

## 概述

FiberNet 是一个面向**纤维网络结构**计算研究的 Python 工具包。它提供了统一的框架，用于生成、模拟、分析和可视化各种纤维结构——从简单随机网络到复杂的层次结构、手性结构、编织结构和超材料结构。

### 核心功能

| 类别 | 功能 |
|------|------|
| **结构生成**（79+） | 随机、有序、手性、编织、层次、纤维束、弯曲、分形、梯度、层合板、**超材料**（负泊松比、八叉桁架、金刚石、螺旋极小面） |
| **物理模拟**（22+） | FEM 力学、非线性、动力学、断裂、损伤、疲劳、蠕变、热传导、电磁、声学、流体、流变、DMA、粘弹性、多尺度 |
| **分析工具** | 形态学、拓扑、统计、渗流、均匀化、应力应变、空间分析、有效性能、泊松比 |
| **机器学习** | 特征提取、数据集构建、图神经网络、性能预测 |
| **输入输出** | JSON、YAML、LAMMPS、VTK、GMSH、PDB、XYZ、HDF5、pandas、FEA 导出 |
| **可视化** | PyVista 3D、matplotlib 2D、Plotly 交互、动画 |

### 安装

```bash
# 从 GitHub 安装（推荐，获取最新版本）
pip install git+https://github.com/GellmanSparrowS/fibernet.git

# 完整安装（含可视化和 ML 功能）
pip install "git+https://github.com/GellmanSparrowS/fibernet.git#egg=fibernet[full]"

# 开发安装
git clone https://github.com/GellmanSparrowS/fibernet.git
cd fibernet
pip install -e ".[dev,full]"
```

### 快速开始

```python
from fibernet import gen
from fibernet.sim.mechanical import compute_effective_properties

# 生成反入蜂窝（负泊松比超结构）
net = gen.reentrant_honeycomb_2d(
    reentrant_angle=150,    # 度（>120° = 拉胀）
    grid_size=(8, 8),
)

# 计算有效力学性能
props = compute_effective_properties(net, strain=0.001)
print(props.summary())
```

### 超材料结构

专为力学超材料设计提供的生成器：

- `reentrant_honeycomb_2d/3d` — 经典拉胀（负泊松比）结构
- `chiral_honeycomb_2d` — 节点-韧带手性蜂窝
- `arrowhead_auxetic_2d` — 箭头/鱼骨拉胀结构
- `proper_octet_truss_3d` — 拉伸主导 FCC 晶格
- `diamond_lattice_3d` — 四面体配位弯曲主导晶格
- `gyroid_lattice_3d` — TPMS 螺旋极小面结构
- `hierarchical_lattice_2d` — 自相似多尺度层次晶格
- `plate_lattice_3d` — 近最优刚度板格

### 教程

- [📓 完整教程 Notebook](tutorials/metamaterial_design.ipynb) — 安装 → 生成 → 可视化 → FEM 模拟 → ML 预测

### 平台支持

| 平台 | Python | 状态 |
|------|--------|------|
| Ubuntu | 3.9–3.12 | ✅ 全部通过 |
| macOS (ARM64) | 3.9–3.12 | ✅ 全部通过 |
| Windows | 3.9–3.12 | ✅ 全部通过 |

### 引用

```bibtex
@software{fibernet2026,
  title  = {FiberNet: A Python Toolkit for Fiber Network Generation, Simulation, and Analysis},
  author = {BMG-FDU Research Group},
  year   = {2026},
  url    = {https://github.com/GellmanSparrowS/fibernet}
}
```

### 许可证

MIT 许可证。依赖许可证信息见 [ATTRIBUTIONS.md](ATTRIBUTIONS.md)。

---

<div align="center">

**Made with ❤️ by [BMG-FDU](https://github.com/BMG-FDU) Research Group**

[⬆ Back to top](#-fibernet)

</div>
