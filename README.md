<div align="center">

# FiberNet

**A Comprehensive Python Toolkit for Fiber Network Generation, Simulation, and Analysis**

**面向材料科学研究的完整纤维网络结构生成、模拟与分析 Python 工具包**

---

[🇬🇧 English](#english) | [🇨🇳 中文](#中文)

[![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pip%20install-fibernet-blue?logo=pypi&logoColor=white)](https://pypi.org/project/fibernet/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI/CD](https://github.com/GellmanSparrowS/fibernet/actions/workflows/ci.yml/badge.svg)](https://github.com/GellmanSparrowS/fibernet/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/fibernet/badge/?version=latest)](https://fibernet.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://img.shields.io/badge/tests-889%20passing-brightgreen.svg)]()
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.pending-orange.svg)]()
[![GitHub Stars](https://img.shields.io/github/stars/GellmanSparrowS/fibernet?style=social)](https://github.com/GellmanSparrowS/fibernet)

**Homepage / 主页**: [ml-biomat.com](https://ml-biomat.com/)

</div>

---

<a id="english"></a>

## 📖 Overview

FiberNet is a research-grade Python toolkit designed for computational study of fiber network structures at the *Nature Materials* level. It provides a unified framework for generating, simulating, analyzing, and visualizing diverse fiber architectures — from simple random networks to complex hierarchical, chiral, and woven structures.

The toolkit supports multi-physics simulations including mechanics, dynamics, fracture, thermal transport, electromagnetics, fluid flow, and acoustics, with optional GPU acceleration via [Taichi](https://github.com/taichi-dev/taichi).

### Who is it for?

- **Materials scientists** studying nonwoven, woven, and composite fiber architectures
- **Biomechanics researchers** modeling tissue scaffolds, extracellular matrices, and biological networks
- **Polymer physicists** investigating entanglement, percolation, and rheology
- **Computational engineers** performing multi-scale FEM and multi-physics simulations

---

### ⚡ Key Features

| Category | Capabilities |
|:---------|:-------------|
| **Network Generation** | 68 generators — random, ordered, chiral, woven, hierarchical, bundles, curved, biomimetic, CNT, electrospun, textile, paper |
| **Physics Simulation** | FEM, nonlinear mechanics, dynamics, fracture, damage/fatigue, thermal, electromagnetic, acoustic, fluid, rheology, DMA |
| **Crosslink Models** | Rigid, spring, breakable, friction, bonded (covalent, hydrogen, ionic, entanglement) |
| **Constitutive Models** | Linear elastic, bilinear plasticity, power-law, neo-Hookean, Mooney–Rivlin, Arruda–Boyce, Maxwell, Kelvin–Voigt, SLS |
| **Analysis** | Morphology, topology, spectral, pore structure, anisotropy, percolation, multi-scale homogenization |
| **Machine Learning** | Feature extraction (30+), GNN models, property prediction, dataset generation |
| **I/O Formats** | JSON, YAML, LAMMPS, VTK, GMSH, PDB, XYZ, HDF5, pandas |
| **Visualization** | PyVista 3D interactive, matplotlib 2D, Plotly web, animations, screenshots |
| **Acceleration** | Taichi CPU/GPU parallel FEM, parallel contact detection |
| **Optimization** | Energy minimization (L-BFGS-B, CG, BFGS, Powell), parameter sweeps, Monte Carlo, sensitivity analysis |

---

### 📦 Installation

```bash
# Standard installation
pip install fibernet

# Full installation with all optional dependencies
pip install fibernet[full]

# Development installation
git clone https://github.com/GellmanSparrowS/fibernet.git
cd fibernet && pip install -e ".[dev,full]"
```

**Optional dependency groups:**

| Group | Description |
|:------|:------------|
| `viz` | PyVista 3D + matplotlib 2D visualization |
| `mesh` | Trimesh mesh operations (STL/OBJ/PLY export) |
| `io` | HDF5 support via h5py |
| `accel` | Taichi GPU acceleration |
| `ml` | scikit-learn ML integration |
| `graph` | NetworkX graph analysis |
| `full` | All optional dependencies |

---

### 🚀 Quick Start

```python
import fibernet as fn

# 1. Generate a random 2D fiber network
net = fn.create("random_2d", num_fibers=100, fiber_length=10.0,
                box_size=(30, 30), seed=42)

# 2. Analyze structural properties
stats = fn.analyze(net)
print(f"Fibers: {stats['num_fibers']}, "
      f"Nematic order: {stats['nematic_order']:.3f}")

# 3. Run mechanical simulation
result = fn.simulate_mechanics(net, strain=0.01)
print(f"Effective modulus: {result['effective_modulus']:.2e} Pa")

# 4. Visualize and export
fn.plot(net)
fn.export(net, "network.vtk", format="vtk")
```

#### Advanced: Parametric Study

```python
from fibernet.utils.parametric import parametric_sweep

results = parametric_sweep(
    generator="random_2d",
    parameters={"num_fibers": [50, 100, 200], "fiber_length": [5.0, 10.0]},
    analysis=["nematic_order", "connectivity"],
    simulation={"type": "mechanical", "strain": 0.01},
)
```

---

### 🏗️ Architecture

```
fibernet/
├── core/              # Data structures: Fiber, FiberNetwork, Material, Crosslink
├── gen/               # 68 network generators
│   ├── disordered.py  #   Random deposition, crossing, electrospun
│   ├── ordered.py     #   Lattice, periodic, aligned, grid
│   ├── chiral.py      #   Helical, twisted, chiral bundles
│   ├── woven.py       #   Plain, twill, satin weave patterns
│   ├── hierarchical.py#   Multi-scale, fractal, self-similar
│   ├── bundle.py      #   Yarns, tows, fiber bundles
│   ├── curved.py      #   Curved fibers with bending mechanics
│   └── variant.py     #   2D→3D, multi-radius, gyroid, foam
├── sim/               # Simulation engines
│   ├── mechanical.py  #   FEM solver (linear & nonlinear)
│   ├── dynamics.py    #   Molecular dynamics, Brownian
│   ├── fracture.py    #   Crack propagation, LEFM
│   ├── damage.py      #   Damage mechanics, fatigue
│   ├── thermal.py     #   Heat conduction, convection
│   ├── electromagnetic.py # Maxwell equations, permittivity
│   ├── fluid.py       #   Darcy flow, pore network
│   ├── acoustic.py    #   Wave propagation, band structure
│   ├── coupled.py     #   Multi-physics coupling
│   └── viscoelastic.py#   DMA, Maxwell, Kelvin-Voigt
├── analysis/          # Structural analysis tools
├── ml/                # Machine learning integration
├── viz/               # 2D/3D visualization
├── io/                # File I/O (7+ formats)
├── utils/             # Utilities, parametric studies
├── api.py             # High-level convenience API
└── transforms/        # Network transformations
```

---

### 📊 Test Results

889 tests passing across 65+ test files covering all modules:

```bash
# Run all tests
pytest tests/ -v

# Run specific module
pytest tests/test_generators.py tests/test_integration.py -v

# Run with coverage
pytest tests/ --cov=fibernet --cov-report=term-missing
```

---

### 📚 Examples

Runnable examples are provided in the [`examples/`](examples/) directory:

| Example | Description |
|:--------|:------------|
| `basic_usage.py` | Quick start: generation, analysis, visualization |
| `full_workflow.py` | Complete pipeline: generate → analyze → simulate → export |
| `ml_example.py` | Machine learning feature extraction and prediction |
| `comprehensive_demo.py` | Showcase of all major capabilities |
| `advanced_analysis.py` | Structural statistics and comparison |
| `ml_property_prediction.py` | GNN-based property prediction workflow |

```bash
python examples/full_workflow.py
```

---

### 📖 Documentation

Full API reference and user guides: **[fibernet.readthedocs.io](https://fibernet.readthedocs.io/)**

Jupyter tutorials available in [`tutorials/`](tutorials/).

---

### 📝 Citation

If you use FiberNet in your research, please cite:

```bibtex
@software{fibernet2025,
  title     = {FiberNet: A Comprehensive Python Toolkit for Fiber Network
               Generation, Simulation, and Analysis},
  author    = {FiberNet Contributors},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/GellmanSparrowS/fibernet},
  version   = {1.24.0}
}
```

---

### 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

### 🙏 Acknowledgments

- Built with [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [NetworkX](https://networkx.org/), [matplotlib](https://matplotlib.org/), [PyVista](https://pyvista.org/), [Plotly](https://plotly.com/), and [Taichi](https://taichi-lang.org/)
- Supported by the [**ML-BioMat**](https://ml-biomat.com/) research group ([BMG-FDU](https://github.com/BMG-FDU))
- Inspired by research in computational materials science, polymer physics, and biomechanics

---

### 📄 License

This project is licensed under the [MIT License](LICENSE).

---
---

<a id="中文"></a>

## 📖 概述

FiberNet 是一个面向 *Nature Materials* 级别研究的中国产 Python 工具包，专用于纤维网络结构的计算研究。它提供了统一的框架，用于生成、模拟、分析和可视化各种纤维结构——从简单的随机网络到复杂的层次、手性和编织结构。

该工具包支持多物理场模拟，包括力学、动力学、断裂、热传导、电磁学、流体流动和声学，并可通过 [Taichi](https://github.com/taichi-dev/taichi) 实现 GPU 加速。

### 适用对象

- **材料科学家** — 研究非织造、编织和复合纤维结构
- **生物力学研究者** — 建模组织支架、细胞外基质和生物网络
- **高分子物理学家** — 研究缠结、渗流和流变学
- **计算工程师** — 执行多尺度有限元和多物理场模拟

---

### ⚡ 核心功能

| 类别 | 能力 |
|:-----|:-----|
| **网络生成** | 68种生成器 — 随机、有序、手性、编织、层次、束、弯曲、仿生、碳纳米管、电纺、纺织、纸张 |
| **物理模拟** | 有限元、非线性力学、动力学、断裂、损伤/疲劳、热学、电磁、声学、流体、流变、DMA |
| **交联模型** | 刚性、弹簧、可断裂、摩擦、键合（共价、氢键、离子、缠结） |
| **本构模型** | 线弹性、双线性塑性、幂律、neo-Hookean、Mooney–Rivlin、Arruda–Boyce、Maxwell、Kelvin–Voigt、SLS |
| **分析工具** | 形态学、拓扑、谱分析、孔隙结构、各向异性、渗流、多尺度均匀化 |
| **机器学习** | 特征提取（30+）、图神经网络、性能预测、数据集生成 |
| **输入输出** | JSON、YAML、LAMMPS、VTK、GMSH、PDB、XYZ、HDF5、pandas |
| **可视化** | PyVista 3D交互、matplotlib 2D、Plotly网页、动画、截图 |
| **加速** | Taichi CPU/GPU并行有限元、并行接触检测 |
| **优化** | 能量最小化（L-BFGS-B、CG、BFGS、Powell）、参数扫描、蒙特卡洛、灵敏度分析 |

---

### 📦 安装

```bash
# 标准安装
pip install fibernet

# 完整安装（含所有可选依赖）
pip install fibernet[full]

# 开发安装
git clone https://github.com/GellmanSparrowS/fibernet.git
cd fibernet && pip install -e ".[dev,full]"
```

---

### 🚀 快速入门

```python
import fibernet as fn

# 1. 生成随机二维纤维网络
net = fn.create("random_2d", num_fibers=100, fiber_length=10.0,
                box_size=(30, 30), seed=42)

# 2. 分析结构属性
stats = fn.analyze(net)
print(f"纤维数: {stats['num_fibers']}, "
      f"向序参数: {stats['nematic_order']:.3f}")

# 3. 运行力学模拟
result = fn.simulate_mechanics(net, strain=0.01)
print(f"等效模量: {result['effective_modulus']:.2e} Pa")

# 4. 可视化与导出
fn.plot(net)
fn.export(net, "network.vtk", format="vtk")
```

---

### 📊 测试结果

889个测试全部通过，覆盖65+测试文件和所有模块：

```bash
pytest tests/ -v
pytest tests/test_generators.py tests/test_integration.py -v
```

---

### 📖 文档

完整 API 参考和用户指南：**[fibernet.readthedocs.io](https://fibernet.readthedocs.io/)**

Jupyter 教程位于 [`tutorials/`](tutorials/) 目录。

---

### 📝 引用

如果在研究中使用了 FiberNet，请引用：

```bibtex
@software{fibernet2025,
  title     = {FiberNet: A Comprehensive Python Toolkit for Fiber Network
               Generation, Simulation, and Analysis},
  author    = {FiberNet Contributors},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/GellmanSparrowS/fibernet},
  version   = {1.24.0}
}
```

---

### 🤝 贡献

欢迎贡献代码！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

---

### 🙏 致谢

- 基于 [NumPy](https://numpy.org/)、[SciPy](https://scipy.org/)、[NetworkX](https://networkx.org/)、[matplotlib](https://matplotlib.org/)、[PyVista](https://pyvista.org/)、[Plotly](https://plotly.com/) 和 [Taichi](https://taichi-lang.org/) 构建
- 由 [**ML-BioMat**](https://ml-biomat.com/) 研究组（[BMG-FDU](https://github.com/BMG-FDU)）提供支持
- 受计算材料科学、高分子物理和生物力学领域研究启发

---

### 📄 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。

---

<div align="center">

**[⬆ Back to top / 返回顶部](#fibernet)**

*Made with ❤️ by the FiberNet contributors · [ml-biomat.com](https://ml-biomat.com/)*

</div>
