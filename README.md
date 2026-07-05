# FiberNet

**A comprehensive Python toolkit for fiber network structure generation, simulation, and analysis.**
**面向 Nature Materials 级别研究的开源纤维网络结构 Python 工具包。**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-889%20passing-green.svg)]()
[![Version](https://img.shields.io/badge/version-1.24.0-blue.svg)]()
[![CI/CD](https://github.com/GellmanSparrowS/fibernet/actions/workflows/ci.yml/badge.svg)](https://github.com/GellmanSparrowS/fibernet/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/fibernet/badge/?version=latest)](https://fibernet.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)]()
[![GitHub](https://img.shields.io/github/stars/GellmanSparrowS/fibernet?style=social)](https://github.com/GellmanSparrowS/fibernet)

---

## 📖 Overview / 概述

**English**: FiberNet enables researchers to generate, simulate, and analyze fiber network structures for applications in materials science, biomechanics, polymer physics, composites engineering, and more. Designed as a research-grade tool with emphasis on reproducibility and extensibility.

**中文**: FiberNet 使研究人员能够生成、模拟和分析纤维网络结构，适用于材料科学、生物力学、高分子物理、复合材料工程等领域。作为研究级工具，强调可重复性和可扩展性。

**Homepage / 主页**: [https://ml-biomat.com/](https://ml-biomat.com/)

---

## ⚡ Key Features / 核心功能

| Category / 类别 | Capabilities / 能力 |
|----------|-------------|
| **Generation / 生成** | 68 generators: random, ordered, chiral, woven, hierarchical, bundles, curved fibers, biomimetic, CNT, paper, textile, electrospun / 68种生成器：随机、有序、手性、编织、层次、束、弯曲纤维、仿生、碳纳米管、纸张、纺织、电纺 |
| **Simulation / 模拟** | FEM, dynamics, fracture, damage/fatigue, thermal, electromagnetic, acoustic, fluid, rheology, DMA, multi-scale, optimization / 有限元、动力学、断裂、损伤/疲劳、热学、电磁、声学、流体、流变、DMA、多尺度、优化 |
| **Crosslinks / 交联** | Rigid, spring, breakable, friction, bonded, covalent, hydrogen bond, ionic, physical entanglement / 刚性、弹簧、可断裂、摩擦、键合、共价、氢键、离子、物理缠结 |
| **Analysis / 分析** | Morphology, topology, spectral, pore structure, anisotropy, percolation, multi-scale homogenization / 形态学、拓扑、谱分析、孔隙结构、各向异性、渗流、多尺度均匀化 |
| **Mesh Operations / 网格** | Trimesh integration, mesh conversion, boolean operations, mesh analysis, repair, simplification / Trimesh集成、网格转换、布尔运算、网格分析、修复、简化 |
| **Optimization / 优化** | Energy minimization (L-BFGS-B, CG, BFGS, Powell), parameter optimization, global optimization (differential evolution) / 能量最小化、参数优化、全局优化（差分进化） |
| **ML Integration / 机器学习** | Feature extraction, GNN model, property prediction, dataset generation / 特征提取、图神经网络、性能预测、数据集生成 |
| **I/O / 输入输出** | JSON, YAML, LAMMPS, VTK, GMSH, PDB, XYZ, pandas, HDF5 formats / JSON、YAML、LAMMPS、VTK、GMSH、PDB、XYZ、pandas、HDF5格式 |
| **Acceleration / 加速** | Taichi CPU/GPU parallel FEM, parallel contact detection / Taichi CPU/GPU并行有限元、并行接触检测 |
| **Visualization / 可视化** | PyVista 3D interactive, matplotlib, plotly web, screenshots, animations / PyVista 3D交互、matplotlib、plotly网页、截图、动画 |

---

## 📦 Installation / 安装

```bash
# Basic installation / 基础安装
pip install fibernet

# With all optional dependencies / 安装所有可选依赖
pip install fibernet[full]

# Development install / 开发安装
git clone https://github.com/GellmanSparrowS/fibernet.git
cd fibernet && pip install -e ".[dev]"
```

---

## 🚀 Quick Start / 快速入门

### High-Level API / 高级API

```python
import fibernet as fn

# Create a random 2D network / 创建随机2D网络
net = fn.create("random_2d", num_fibers=100, fiber_length=10.0, box_size=(30, 30), seed=42)

# Analyze structure / 分析结构
stats = fn.analyze(net)
print(f"Fibers: {stats['num_fibers']}, Order: {stats['nematic_order']:.3f}")

# Run mechanical simulation / 运行力学模拟
result = fn.simulate_mechanics(net, strain=0.01)

# Transform / 变换
net_scaled = fn.scale(net, factor=2.0)
net_rotated = fn.rotate(net, angle=0.785, axis=[0, 0, 1])

# Visualize / 可视化
fn.plot(net)

# Export / 导出
fn.export(net, "network.json", format="json")
fn.export(net, "network.vtk", format="vtk")
```

### Network Generation / 网络生成

```python
from fibernet import gen

# Random 2D fiber network / 随机2D纤维网络
net = gen.random_straight_2d(
    num_fibers=100, fiber_length=15.0, box_size=(50, 50),
    radius=0.1, seed=42
)

# 3D random network / 3D随机网络
net_3d = gen.random_straight_3d(
    num_fibers=200, fiber_length=20.0, box_size=(50, 50, 50), seed=42
)

# Ordered lattices / 有序晶格
square = gen.square_lattice_2d(spacing=5.0, grid_size=(10, 10))
honeycomb = gen.honeycomb_lattice_2d(cell_size=5.0, grid_size=(10, 10))
triangular = gen.triangular_lattice_2d(spacing=5.0, grid_size=(10, 10))
cubic = gen.cubic_lattice_3d(spacing=5.0, grid_size=(5, 5, 5))
octet = gen.octet_truss_3d(spacing=5.0, grid_size=(3, 3, 3))

# Specialized structures / 专用结构
chiral = gen.chiral_network_2d(
    num_fibers=50, fiber_length=15.0, box_size=(50, 50),
    chirality=0.5, seed=42
)
woven = gen.woven_2d(warp_count=10, weft_count=10, spacing=5.0)
helix = gen.single_helix(radius=5.0, pitch=2.0, turns=3, num_points=100)
dna = gen.double_helix(radius=5.0, pitch=2.0, turns=3)

# Biomimetic / 仿生结构
collagen = gen.biomimetic_collagen(num_fibers=100, box_size=(50, 50), seed=42)
fibrin = gen.biomimetic_fibrin(num_fibers=100, box_size=(50, 50), seed=42)
electrospun = gen.electrospun(num_fibers=200, box_size=(50, 50), seed=42)
```

### Fiber Bundles / 纤维束

```python
from fibernet.gen.bundles import (
    parallel_bundle, twisted_bundle, random_bundle,
    rope_bundle, braided_bundle,
)

# Parallel fiber bundle / 平行纤维束
parallel = parallel_bundle(
    num_fibers=20, fiber_length=50.0, bundle_radius=2.0,
    centerline=np.array([[0, 0, 0], [50, 0, 0]])
)

# Twisted bundle / 扭转纤维束
twisted = twisted_bundle(
    num_fibers=20, fiber_length=50.0, bundle_radius=2.0,
    twist_rate=0.1, centerline=np.array([[0, 0, 0], [50, 0, 0]])
)
```

### Mechanical Simulation / 力学模拟

```python
from fibernet.sim import mechanical

# Linear FEM / 线性有限元
result = mechanical.MechanicalSimulator().simulate(
    net, strain=0.01, method='fem'
)
print(f"Stress: {result.stress:.2e} Pa")
print(f"Young's modulus: {result.youngs_modulus:.2e} Pa")

# Nonlinear FEM with large deformation / 大变形的非线性有限元
result_nl = mechanical.MechanicalSimulator().simulate(
    net, strain=0.1, method='nonlinear_fem',
    geometric_nonlinearity=True
)
```

### Crosslinks / 交联

```python
from fibernet.core.crosslinks import (
    RigidCrosslink, SpringCrosslink, BreakableCrosslink,
)

# Add rigid crosslinks / 添加刚性交联
net.add_crosslinks(RigidCrosslink(stiffness=1e9))

# Add breakable crosslinks / 添加可断裂交联
net.add_crosslinks(BreakableCrosslink(
    stiffness=1e8, breaking_force=1e-6
))
```

### Analysis / 分析

```python
from fibernet.analysis import MorphologyAnalyzer, TopologyAnalyzer

# Morphological analysis / 形态分析
morph = MorphologyAnalyzer(net)
print(f"Nematic order: {morph.nematic_order_parameter():.3f}")
print(f"Porosity: {morph.porosity():.3f}")
print(f"Mean fiber length: {morph.mean_fiber_length():.2f}")

# Topological analysis (requires networkx) / 拓扑分析（需要networkx）
topo = TopologyAnalyzer(net)
result = topo.analyze()
print(f"Components: {result.num_components}")
print(f"Clustering: {result.clustering_coefficient:.3f}")
```

### Ensemble Analysis / 集合分析

```python
from fibernet.utils.ensemble import ensemble_analysis
from fibernet.analysis import MorphologyAnalyzer

# Generate ensemble / 生成集合
ensemble = fn.create_ensemble(
    gen.random_straight_2d, num_networks=50, base_seed=42,
    num_fibers=100, fiber_length=10.0, box_size=(50, 50)
)

def analyze(net):
    morph = MorphologyAnalyzer(net)
    return {
        'nematic_order': morph.nematic_order_parameter(),
        'porosity': morph.porosity(),
    }

stats = ensemble_analysis(ensemble, analyze)
print(f"Nematic order: {stats['nematic_order']['mean']:.3f} ± {stats['nematic_order']['std']:.3f}")
```

### Visualization / 可视化

```python
# Matplotlib (static) / 静态图
from fibernet.viz import visualize_3d_matplotlib
fig, ax = visualize_3d_matplotlib(net, show_crosslinks=True)

# Plotly (interactive web) / 交互式网页
from fibernet.viz import visualize_interactive, visualize_stress_field, export_html
fig = visualize_interactive(net, color_by='orientation', title="My Network")
export_html(fig, "network.html", auto_open=True)

# PyVista (interactive 3D) / 交互式3D
from fibernet.viz import visualize_3d_pyvista
plotter = visualize_3d_pyvista(net, fiber_radius=0.1)
```

### Export to Simulation Software / 导出到仿真软件

```python
from fibernet.io import to_vtk, to_lammps, to_gmsh, to_pdb

to_vtk(net, 'network.vtk')         # ParaView
to_lammps(net, 'network.lammps')   # LAMMPS MD
to_gmsh(net, 'network.geo')        # GMSH meshing
to_pdb(net, 'network.pdb')         # Protein Data Bank
```

### Unit Systems / 单位制

```python
from fibernet.utils.units import convert_network

# Convert to micrometers / 转换为微米
net_micro = convert_network(net, from_unit='si', to_unit='micro')
```

---

## 📁 Project Structure / 项目结构

```
fibernet/
├── core/              # Core data structures / 核心数据结构
│                      # (Fiber, FiberNetwork, Material, Crosslinks)
├── gen/               # 68 network generators / 68种网络生成器
├── sim/               # Simulation engines / 模拟引擎
│   ├── mechanical.py  # FEM (linear/nonlinear) / 有限元（线性/非线性）
│   ├── accelerated.py # Taichi GPU-accelerated FEM / GPU加速有限元
│   ├── thermal.py     # Heat conduction / 热传导
│   ├── electromagnetic.py # Electrical conductivity / 电导率
│   ├── acoustic.py    # Acoustic wave propagation / 声波传播
│   ├── fracture.py    # Crack propagation / 裂纹扩展
│   ├── damage.py      # Damage mechanics, fatigue / 损伤力学、疲劳
│   ├── rheology.py    # Fiber suspension rheology / 纤维悬浮流变
│   ├── fluid.py       # Darcy flow, pore network / 达西流、孔隙网络
│   ├── multiscale.py  # Homogenization, RVE / 均匀化、代表性体积元
│   └── viscoelastic.py # DMA, Maxwell, Kelvin-Voigt / 粘弹性
├── analysis/          # Analysis tools / 分析工具
├── ml/                # Machine learning / 机器学习
├── io/                # I/O formats / 输入输出格式
├── viz/               # Visualization / 可视化
├── utils/             # Utilities / 工具函数
├── api.py             # High-level convenience API / 高级便捷API
└── transforms/        # Network transformations / 网络变换
```

---

## 📊 Test Results / 测试结果

**English**: 889 tests passing across 65+ test files covering all modules.

**中文**: 889个测试全部通过，覆盖65+测试文件和所有模块。

```bash
# Run all tests / 运行所有测试
pytest tests/ -v

# Run specific module tests / 运行特定模块测试
pytest tests/test_core.py tests/test_generators.py
pytest tests/test_integration.py -v
```

---

## 📚 Examples / 示例

FiberNet includes example scripts in the `examples/` directory:
FiberNet 在 `examples/` 目录中包含示例脚本：

- **`basic_usage.py`** — Quick start with network generation and analysis / 网络生成与分析快速入门
- **`full_workflow.py`** — Complete pipeline: generate → analyze → simulate → export / 完整流程
- **`ml_example.py`** — Machine learning integration / 机器学习集成
- **`comprehensive_demo.py`** — Showcase of all major features / 所有主要功能展示

Run any example / 运行示例:
```bash
python examples/full_workflow.py
```

---

## 📖 Documentation / 文档

Full documentation with API reference / 包含API参考的完整文档:
[https://fibernet.readthedocs.io/](https://fibernet.readthedocs.io/)

Jupyter notebook tutorials are available in the `tutorials/` directory.
Jupyter 笔记本教程可在 `tutorials/` 目录中找到。

---

## 🏗️ Architecture / 架构

```
┌─────────────────────────────────────────────────┐
│                  High-Level API                  │
│               高级API (api.py)                    │
├──────┬──────┬──────┬──────┬──────┬──────┬───────┤
│ gen  │ sim  │analysis│ viz │  ml  │  io  │transforms│
│生成器 │模拟  │ 分析  │可视化│机器学习│输入输出│  变换  │
├──────┴──────┴──────┴──────┴──────┴──────┴───────┤
│              Core Data Structures                │
│           核心数据结构 (core/)                     │
│  Fiber │ FiberNetwork │ Material │ Crosslinks    │
├──────────────────────────────────────────────────┤
│              Utils / Integrations                │
│           工具函数 / 集成                          │
│  config │ units │ validation │ networkx │ lammps │
└──────────────────────────────────────────────────┘
```

---

## 📝 Citation / 引用

If you use FiberNet in your research, please cite:
如果在研究中使用了 FiberNet，请引用：

```bibtex
@software{fibernet2025,
  title = {FiberNet: A Comprehensive Python Toolkit for Fiber Network Generation, Simulation, and Analysis},
  author = {FiberNet Contributors},
  year = {2025},
  url = {https://github.com/GellmanSparrowS/fibernet},
  version = {1.24.0}
}
```

---

## 🤝 Contributing / 贡献

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

---

## 📄 License / 许可证

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
本项目基于 MIT 许可证 - 详情请查看 [LICENSE](LICENSE) 文件。

---

## 🙏 Acknowledgments / 致谢

- Built with NumPy, SciPy, NetworkX, matplotlib, pyvista, plotly, and Taichi
- 基于 NumPy、SciPy、NetworkX、matplotlib、pyvista、plotly 和 Taichi 构建
- Inspired by research in computational materials science and biomechanics
- 受计算材料科学和生物力学研究的启发
- Supported by the [ML-BioMat](https://ml-biomat.com/) research group
- 由 [ML-BioMat](https://ml-biomat.com/) 研究组支持
