# FiberNet v3 重构进度

**最后更新**: 2026-07-09  
**当前状态**: Phase 1-6 完成，待优化中间点可编程性

---

## 架构概览

```
fibernet/
├── core/
│   ├── structure_graph.py    ← StructureGraph: 统一数据结构
│   ├── transforms.py         ← 几何变换: translate, rotate, mirror, scale
│   ├── tiling.py             ← 平铺焊接: tile_2d, tile_3d
│   └── material.py           ← 材料属性
├── gen/
│   ├── pattern.py            ← 图案引擎: pattern_2d, pattern_3d (主入口)
│   └── (其他生成器保留)
├── sim/
│   ├── fem.py                ← 梁单元 FEM: BeamFEM
│   └── rl_env.py             ← RL 环境: FiberNetworkEnv
├── viz/
│   └── render.py             ← 可视化: render_graph, render_gallery
├── ml/
│   └── dataset_v2.py         ← ML 数据集: generate_dataset
└── __init__.py               ← 统一 API 导出
```

---

## 已完成阶段

### Phase 1: 核心基础 ✅
- **StructureGraph**: NumPy 原生，空间哈希节点合并，边去重，边界标志，内部点，JSON/networkx/numpy 转换
- **Transforms**: translate, rotate (2D/3D), mirror (x/y/z), scale, compose
- **Tiling**: tile_2d/tile_3d 自动节点焊接，tile_with_transforms, fit_unit_to_box

### Phase 2: 图案引擎 ✅
- **Pattern Engine**: `pattern_2d()` / `pattern_3d()` 统一 API
- **11 个 2D 基元**: square, triangle, hexagon, honeycomb, kagome, reentrant (拉胀), chiral, star, cross, missing_rib, diamond
- **3 个 3D 基元**: cubic, octet, diamond_3d
- 所有 11 个 2D 基元平铺后连通
- 确定性生成（除非显式设置 seed）
- 边离散化：每条边 N 个内部点用于变形可视化

### Phase 3: 仿真 ✅
- **BeamFEM**: Euler-Bernoulli 梁 FEM，scipy.sparse 组装
- 单轴拉伸、剪切测试、应力-应变曲线
- 有效 E*, ν*, G* 提取
- 变形图输出用于可视化
- Reentrant 验证拉胀（负泊松比）

### Phase 4: 可视化 ✅
- **render.py**: render_graph, render_graph_3d, render_deformation, render_gallery
- 4 个主题（dark, light, blueprint, publication），发光效果
- 颜色模式：uniform, orientation, length, stress, strain, custom
- 9 张展示图在 `output_viz/`

### Phase 5: ML/RL ✅
- **dataset_v2.py**: 参数扫描 → FEM 标注 → numpy/JSON 导出（断点续跑）
- 特征提取（18 个拓扑+几何特征）
- **rl_env.py**: Gymnasium 兼容 FiberNetworkEnv
  - Action: 选择基元、网格、半径
  - Reward: 到目标 E*, ν* 的距离

### Phase 6: 集成 ✅
- 统一顶层 API: `import fibernet as fn`
- 9/9 集成测试通过（~2秒）
- 干净的 git 历史，每个阶段可回滚

---

## 当前 API 文档

### 主入口: pattern_2d()

```python
from fibernet import pattern_2d

g = pattern_2d(
    unit="honeycomb",           # 内置基元名称 (见 list_units())
    box=(10, 10),               # 单元尺寸 (w, h)
    grid=(5, 5),                # 平铺网格 (nx, ny)
    n_internal=8,               # 每条边的内部点数（用于变形可视化）
    radius=0.1,                 # 梁半径
    
    # 变换（可选）
    mirror_x=False,
    mirror_y=False,
    rotation=0.0,               # 旋转角度（度）
    
    # 自定义形状（与 unit 互斥）
    points=None,                # [(x,y), ...] 自定义多边形顶点
    closed=True,                # 是否闭合
    fit_to_box=False,           # 自动缩放到 box
    
    # 边界处理
    boundary_mode="none",       # "none" | "error" | "extend"
    
    # 扰动（仅当 seed 设置时生效）
    perturbation=0.0,           # 扰动幅度（边长的比例）
    seed=None,                  # 随机种子（确定性）
    
    # 单位工厂参数（传递给特定基元）
    unit_kwargs={},             # 如 reentrant: {"angle": 20}
)
```

### 主入口: pattern_3d()

```python
from fibernet import pattern_3d

g = pattern_3d(
    unit="cubic",               # "cubic" | "octet" | "diamond_3d"
    box=(10, 10, 10),           # 单元尺寸 (w, h, d)
    grid=(3, 3, 3),             # 平铺网格 (nx, ny, nz)
    n_internal=4,               # 每条边的内部点数
    radius=0.1,                 # 梁半径
)
```

### FEM 仿真: BeamFEM

```python
from fibernet import BeamFEM

fem = BeamFEM(graph, default_E=1e9, default_nu=0.3)

# 单轴拉伸
result = fem.uniaxial_tension(strain=0.01)
print(f"E* = {result.effective_youngs_modulus:.2e} Pa")
print(f"ν* = {result.effective_poissons_ratio:.3f}")

# 变形图
deformed = result.deformed_graph

# 应力-应变曲线
strains, stresses = fem.stress_strain_curve(max_strain=0.05, n_steps=10)
```

### 可视化: render_graph()

```python
from fibernet import render_graph, render_gallery

fig = render_graph(
    g,
    theme="dark",               # "dark" | "light" | "blueprint" | "publication"
    color_by="orientation",     # "uniform" | "orientation" | "length" | "stress" | "strain" | "custom"
    line_width=1.5,
    show_nodes=False,           # 是否显示节点锚点
    title="Honeycomb",
    save_path="output.png",
)
```

### ML 数据集: generate_dataset()

```python
from fibernet import generate_dataset

ds = generate_dataset(
    units=["honeycomb", "square"],
    grid_range=[(3,3), (5,5)],
    radius_range=[0.05, 0.1, 0.2],
    save_dir="datasets/",
    checkpoint_file="ckpt.json",  # 断点续跑
)
```

---

## 待优化：中间点可编程性

### 问题
当前 `pattern_2d()` 的内置基元（如 honeycomb）只在多边形顶点处有节点，每条边是直线。缺少：
1. **n_pts_per_side**: 每条多边形边上的中间点数量（可调）
2. **per-point displacement**: 每个中间点的 (dx, dy) 独立可调
3. **默认行为**: 每个中间点应有非零随机位移

### 目标
让用户能精确控制每条边的形状，生成复杂的超材料结构。

### 设计方案

```python
pattern_2d(
    unit="honeycomb",
    box=(10, 10),
    grid=(5, 5),
    n_pts_per_side=5,           # 每条多边形边上 5 个中间点
    
    # 方式1: 显式逐点位移（完全控制）
    point_displacements={
        "side_0": [(0.1, 0.2), (0.05, -0.1), ...],  # 5 个位移对应 5 个中间点
        "side_1": [...],
        ...
    },
    
    # 方式2: Cn 对称扰动（保持旋转对称）
    perturbation=0.1,           # 扰动幅度
    seed=42,                    # 确定性随机
    
    # 方式3: 默认 = 自动生成（seed=0, perturbation=0.05）
    # 如果都没设置，使用默认随机位移
    
    n_internal=8,               # FEM 变形用的内部点（独立于 n_pts_per_side）
)
```

### 实现步骤
1. 在 `_unit_honeycomb()` 等工厂函数中添加 `n_pts_per_side` 参数
2. 生成多边形时插入中间点（参考 v7 的 `_generate_polygon()`）
3. 添加 `point_displacements` 参数支持显式位移
4. 默认行为：seed=0 时生成确定性随机位移
5. 更新所有 11 个 2D 基元支持此特性
6. 更新 3D 基元支持（如果适用）
7. 重新生成 showcase 图（确保可见中间点）

---

## Git 历史

```
9f62146 Pre-refactor snapshot
687688e Phase 1a: StructureGraph
102ee5f Phase 1b: transforms
2fd2daf Phase 1c: tiling
49d976d Phase 2a: Pattern Engine
efc6b0d Phase 2b: unit connectivity fixes
14f1be5 Phase 3a: BeamFEM
6b4d800 Phase 4: Visualization
1a5670c Phase 5: ML + RL
1cc040e Phase 6: Integration
```

---

## 下一步行动

1. **立即**: 实现 `n_pts_per_side` + `point_displacements` API
2. **测试**: 生成带中间点的 honeycomb，验证连通性
3. **可视化**: 重新生成 showcase 图，确保可见边上的变化
4. **提交**: git commit "Add intermediate point programmability"

---

**上下文恢复**: 如果断联，读取此文件继续工作，无需依赖聊天历史。
