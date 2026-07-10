# FiberNet v3 重构进度

**最后更新**: 2026-07-10  
**当前状态**: 所有核心功能完成，中间点可编程性已实现

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
│   ├── pattern.py            ← 图案引擎: pattern_2d, pattern_3d (1512行，核心)
│   └── (其他生成器保留)
├── sim/
│   ├── fem.py                ← 梁单元 FEM: BeamFEM
│   └── rl_env.py             ← RL 环境: FiberNetworkEnv
├── viz/
│   └── render.py             ← 可视化: render_graph, render_gallery, render_graph_3d
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
- **Pattern Engine**: `pattern_2d()` / `pattern_3d()` 统一 API (1512行)
- **11 个 2D 基元**: square, triangle, hexagon, honeycomb, kagome, reentrant (拉胀), chiral, star, cross, missing_rib, diamond
- **3 个 3D 基元**: cubic, octet, diamond_3d
- 所有 11 个 2D 基元平铺后连通 ✅
- 确定性生成（除非显式设置 seed）
- 边离散化：每条边 N 个内部点用于变形可视化

### Phase 3: 仿真 ✅
- **BeamFEM**: Euler-Bernoulli 梁 FEM，scipy.sparse 组装
- 单轴拉伸、剪切测试、应力-应变曲线
- 有效 E*, ν*, G* 提取
- 变形图输出用于可视化
- Reentrant 验证拉胀（负泊松比）

### Phase 4: 可视化 ✅
- **render.py**: render_graph, render_graph_3d, render_deformation, render_gallery, render_with_stats
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

## 中间点可编程性（新增核心功能）✅

### 功能描述
每条多边形边可以插入 `n_pts_per_side` 个中间图节点，每个节点可独立编程控制位移。

### API 参数

```python
pattern_2d(
    unit="honeycomb",
    box=(10, 10),
    grid=(5, 5),
    n_pts_per_side=5,              # 每条边上 5 个中间节点（影响结构拓扑）
    point_displacements=[...],     # 显式位移列表 [(dx,dy), ...]
    perturbation=0.1,              # Cn 对称扰动幅度
    seed=42,                       # 确定性随机种子
    n_internal=10,                 # FEM 变形用的内部点（独立于 n_pts_per_side）
)
```

### 实现细节

1. **`_generate_polygon_perimeter()`**: 生成多边形周边点（角点 + 中间点）
2. **`_cn_symmetric_displacements()`**: Cn 对称位移（保持旋转对称）
3. **`_add_edge_with_intermediates()`**: 添加带中间节点的边
4. **`_auto_displacements()`**: 确定性非零位移生成

### 测试结果
- 8/8 单元测试通过 ✅
- 所有 11 个 2D 基元支持中间点 ✅
- 所有 3 个 3D 基元支持中间点 ✅
- 确定性种子验证通过 ✅

### 示例

```python
# 蜂巢细节（每条边 5 个中间点）
g = pattern_2d(unit="honeycomb", box=(10,10), grid=(5,5), n_pts_per_side=5, seed=42)
# 结果: 840 nodes, 900 edges

# Kagome 蓝图（每条边 4 个中间点）
g = pattern_2d(unit="kagome", box=(10,10), grid=(4,4), n_pts_per_side=4, seed=42)
# 结果: 849 nodes, 960 edges

# 显式位移控制
displacements = [(0.5, 0.3), (0.2, -0.4), ...]  # 每个中间点的位移
g = pattern_2d(unit="square", box=(10,10), grid=(2,2), 
               n_pts_per_side=2, point_displacements=displacements)
```

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
    n_pts_per_side=3,           # 每条边的中间图节点数（影响结构拓扑）
    point_displacements=None,   # 显式位移列表
    perturbation=0.0,           # Cn 对称扰动幅度
    seed=42,                    # 确定性随机种子
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
    n_pts_per_side=3,           # 每条边的中间图节点数
    point_displacements=None,   # 显式位移列表
    seed=42,                    # 确定性随机种子
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
from fibernet import render_graph, render_gallery, render_graph_3d

fig = render_graph(
    g,
    theme="dark",               # "dark" | "light" | "blueprint" | "publication"
    color_by="orientation",     # "uniform" | "orientation" | "length" | "stress" | "strain" | "custom"
    line_width=1.5,
    show_nodes=False,           # 是否显示节点锚点（默认关闭）
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
abbe1b3 Checkpoint: Save comprehensive API documentation to PROGRESS.md
16b6583 Add intermediate point programmability to pattern engine
```

---

## 展示图说明

所有展示图使用 `n_pts_per_side > 0` 生成，展示中间点可编程性：

1. **01_2d_gallery.png**: 11 个 2D 基元，n_pts_per_side=3
2. **02_honeycomb_detail.png**: 蜂巢，n_pts_per_side=5，840 节点
3. **03_kagome_blueprint.png**: Kagome，n_pts_per_side=4，849 节点
4. **04_auxetic_comparison.png**: 蜂巢 vs 拉胀，n_pts_per_side=4
5. **05_3d_cubic.png**: 3D 立方，n_pts_per_side=3，1036 节点
6. **06_3d_octet.png**: 3D 八面体，n_pts_per_side=3，515 节点
7. **07_chiral_stats.png**: 手性蜂巢带统计，n_pts_per_side=4
8. **08_star_pattern.png**: 星形，n_pts_per_side=5，420 节点
9. **09_cross_pattern.png**: 十字形，n_pts_per_side=4，516 节点

---

**上下文恢复**: 如果断联，读取此文件继续工作，无需依赖聊天历史。
