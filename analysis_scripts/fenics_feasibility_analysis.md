# FEniCS 模拟子 API 可行性分析

## 当前模拟系统

### 已有后端

| 后端 | 文件 | 能力 | Windows 兼容性 |
|------|------|------|----------------|
| **BeamFEM** (scipy) | `sim/fem.py` | Euler-Bernoulli 梁 FEM, 静力/动力 | ✅ 纯 Python |
| **Taichi 加速** | `sim/accelerated.py` | GPU/CPU 并行 FEM 组装 | ✅ 原生支持 |
| **力学测试** | `sim/mechanical.py` | 拉伸/压缩/剪切 | ✅ 纯 Python |
| **多物理场** | `sim/coupled.py` 等 | 热-力耦合、声学、电磁 | ✅ 纯 Python |

### 当前架构特点
- `HAS_TAICHI` flag 实现优雅降级（Taichi 不可用时回退到 scipy）
- FEM 求解器使用 `scipy.sparse.linalg.spsolve`（直接求解器）
- 支持增量加载、应力-应变曲线、屈曲分析
- 所有模拟接受 `StructureGraph` 输入 → 与 pattern engine 无缝集成

---

## FEniCS 可行性评估

### FEniCS vs FEniCSx

| | FEniCS (legacy) | FEniCSx (dolfinx) |
|---|---|---|
| Python 包 | `dolfin` | `fenics-dolfinx` |
| 活跃开发 | ❌ 已停止 | ✅ 活跃 |
| Windows 原生 | ❌ 不支持 | ❌ 不支持 |
| WSL2 支持 | ⚠️ 困难 | ✅ 可行 |
| pip 安装 | ❌ 需要 conda | ⚠️ Linux 上 `pip install fenics-dolfinx` |
| 依赖复杂度 | 高（PETSc, MPI） | 高（PETSc, MPI, UFCx） |
| 适合场景 | 传统 FEM | 高阶/变分 FEM |

### Windows 兼容性详细分析

#### ❌ 原生 Windows 不支持
FEniCS/FEniCSx **没有原生 Windows 支持**。原因：
1. 依赖 PETSc（并行线性代数库），编译需要 POSIX 环境
2. 依赖 MPI 实现（OpenMPI/MPICH），Windows 上常用 MS-MPI 但兼容性差
3. C++ 编译器差异（MSVC vs GCC/Clang）
4. 构建系统基于 CMake + Linux 工具链

#### ⚠️ WSL2 方案可行但增加门槛
- 用户需安装 WSL2 + Ubuntu
- 在 WSL2 内 `pip install fenics-dolfinx`
- 代码可直接运行（文件系统互通）
- **缺点**: 普通 Windows 用户可能不熟悉 WSL2

#### ⚠️ Docker 方案
- `docker pull dolfinx/dolfinx` 镜像
- 可在 Windows 上通过 Docker Desktop 运行
- **缺点**: Docker Desktop 占用资源大，学习曲线

### 功能增益评估

FEniCS 相比现有 BeamFEM 能提供什么？

| 能力 | 现有 BeamFEM | FEniCS/FEniCSx |
|------|-------------|----------------|
| 单元类型 | Euler-Bernoulli 梁 | 梁/壳/实体/高阶 |
| 网格 | 图结构（节点+边） | 三角形/四边形/四面体 |
| 材料模型 | 线弹性 | 非线性/超弹/塑性 |
| 多物理场 | 基础耦合 | 完整多物理场 |
| 自适应网格 | ❌ | ✅ |
| 周期边界 | ⚠️ 手动 | ✅ 原生支持 |
| 安装难度 | 零（pip） | 高（PETSc/MPI） |
| Windows 支持 | ✅ | ❌（需 WSL2） |

### 关键问题：FEniCS 对本项目的实际价值

**现有 BeamFEM 已经足够好的场景**：
- 梁/杆系结构（这正是 FiberNet 的核心对象）
- 线弹性分析
- 有效属性计算（E*, ν*, G*）
- 变形可视化

**FEniCS 真正有优势的场景**：
1. **实体/壳单元**：如果结构需要 2D/3D 实体网格而非梁模型
2. **高阶单元**：二次/三次单元提高精度
3. **非线性材料**：超弹性、塑性、蠕变
4. **自适应网格细化**：应力集中区域自动细化
5. **周期边界条件的变分形式**：更严格的均质化理论

**但对于 FiberNet 的核心用例（纤维网络/格子结构），梁 FEM 是正确选择**：
- 纤维网络本质是 1D 梁的集合
- Euler-Bernoulli/Timoshenko 梁理论精确描述细梁行为
- FEniCS 的实体单元反而不合适（需要大量单元来离散一根细梁）

---

## 建议方案

### 方案 A：不添加 FEniCS（推荐）

**理由**：
1. FiberNet 的对象是梁系结构，现有 BeamFEM 精确且高效
2. FEniCS 在 Windows 上不可用（需 WSL2），会阻碍用户
3. 引入高复杂度依赖（PETSc、MPI）与项目轻量级理念冲突
4. Taichi 加速已提供足够的性能

### 方案 B：添加 FEniCS 作为可选子模块（折中）

```python
# fibernet/sim/fenics_backend.py
try:
    import dolfinx
    HAS_FENICS = True
except ImportError:
    HAS_FENICS = False

class FEniCSFEM:
    """FEniCS-based FEM solver for StructureGraph.
    
    Requires: pip install fenics-dolfinx (Linux/macOS only)
    Windows users: Use WSL2 or Docker.
    
    Provides:
    - Higher-order beam elements
    - Nonlinear material models
    - Periodic boundary conditions via variational form
    - Adaptive mesh refinement
    """
    
    def __init__(self, graph: StructureGraph):
        if not HAS_FENICS:
            raise ImportError(
                "FEniCS required. Install with: pip install fenics-dolfinx\n"
                "Windows users: Requires WSL2. See docs for details."
            )
        self.graph = graph
    
    def solve(self, ...):
        """Solve using FEniCS variational form."""
        ...
```

**架构**：
```
sim/
├── fem.py              # BeamFEM (默认, 纯 Python, 全平台)
├── fenics_backend.py   # FEniCSFEM (可选, Linux/macOS, 高级功能)
├── accelerated.py      # Taichi 加速 (可选, 全平台)
└── __init__.py         # 智能路由
```

**使用方式**：
```python
# 默认：纯 Python BeamFEM（全平台）
fem = fn.sim.solve(graph, method="beam")

# 高级：FEniCS（需要安装）
fem = fn.sim.solve(graph, method="fenics")

# 加速：Taichi（需要安装）
fem = fn.sim.solve(graph, method="taichi")
```

### 方案 C：添加轻量替代方案（务实）

如果确实需要 FEM 增强但要保持 Windows 兼容，可以考虑：

| 替代方案 | 特点 | Windows |
|---------|------|---------|
| **scikit-fem (skfem)** | 轻量纯 Python FEM | ✅ pip install |
| **SfePy** | 完整 FEM, 纯 Python | ✅ pip install |
| **PyAEDT/Ansys** | 商业 FEM | ✅ 但需许可证 |

推荐 **scikit-fem** 作为轻量增强：
- 纯 Python, pip 可安装, Windows 友好
- 支持 2D/3D 实体单元
- 比 FEniCS 简单得多
- 可作为 BeamFEM 的补充（处理需要实体网格的场景）

---

## 结论

| 方案 | 推荐度 | 理由 |
|------|--------|------|
| A: 不加 FEniCS | ⭐⭐⭐⭐⭐ | 现有系统已足够，保持简洁 |
| B: 可选 FEniCS | ⭐⭐⭐ | 高级用户有需求时可扩展 |
| C: scikit-fem | ⭐⭐⭐⭐ | 保持 Windows 兼容的轻量增强 |

**我的建议**: 先维持方案 A。如果后续有明确需求（如非线性材料、自适应网格），再实施方案 B 作为可选模块。Windows 用户通过 `HAS_FENICS` flag 优雅降级，文档中明确说明需要 WSL2。
