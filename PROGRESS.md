# FiberNet v3 重构进度

**最后更新**: 2026-07-10  
**当前状态**: 所有核心功能完成，可视化优化完成

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
│   ├── pattern.py            ← 图案引擎: pattern_2d, pattern_3d (1600+行，核心)
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

### Phase 1-6: 核心功能 ✅
- StructureGraph, transforms, tiling, pattern engine, FEM, visualization, ML/RL
- 详见 git log

### Phase 7: 中间点可编程性 ✅
- `n_pts_per_side`: 每条边上 N 个中间图节点
- `point_displacements`: 显式位移列表
- Cn 对称扰动
- 默认位移幅度: 0.3 * edge_length
- 11 个 2D 基元 + 3 个 3D 基元全部支持

### Phase 8: 可视化优化 ✅
- 位移幅度从 0.05 → 0.3 (更明显的变形)
- 统一颜色方案 (coolwarm colormap)
- 新增 "fiber" 颜色模式 (每条纤维统一颜色)
- 清晰的纤维渲染 (无散射)
- 新增 Voronoi 生成器 (支持 n_pts_per_side)
- 新增 FEM 变形和应力场可视化
- 新增 ML 数据集可视化
- 新增 RL 环境可视化

---

## 展示图说明 (output_viz/)

1. **01_2d_gallery.png** — 12 个 2D 基元画廊 (n_pts_per_side=3)
2. **02_honeycomb_detail.png** — 蜂巢细节 (n_pts_per_side=5, 840 nodes)
3. **03_kagome_blueprint.png** — Kagome 蓝图 (n_pts_per_side=4, 849 nodes)
4. **04_voronoi.png** — Voronoi 镶嵌 (n_pts_per_side=3, 25 seeds)
5. **05_auxetic_comparison.png** — 拉胀对比 (honeycomb vs reentrant)
6. **06_3d_cubic.png** — 3D 立方 (n_pts_per_side=3, 1036 nodes)
7. **07_3d_octet.png** — 3D 八面体 (n_pts_per_side=3, 515 nodes)
8. **08_fem_deformation.png** — FEM 变形可视化 (应力场着色)
9. **09_fem_stress.png** — FEM 应力场 (reentrant auxetic)
10. **10_ml_dataset.png** — ML 训练数据集 (6 个结构 + FEM 属性)
11. **11_rl_environment.png** — RL 环境动作空间探索
12. **12_chiral_stats.png** — 手性蜂巢统计信息

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
    mirror_x=False,
    mirror_y=False,
    rotation=0.0,
    boundary_mode="none",
    fit_to_box=False,
    unit_kwargs={},
)
```

### 主入口: pattern_3d()

```python
from fibernet import pattern_3d

g = pattern_3d(
    unit="cubic",               # "cubic" | "octet" | "diamond_3d"
    box=(10, 10, 10),           # 单元尺寸 (w, h, d)
    grid=(3, 3, 3),             # 平铺网格 (nx, ny, nz)
    n_internal=4,
    n_pts_per_side=3,
    point_displacements=None,
    seed=42,
    radius=0.1,
)
```

### FEM 仿真: BeamFEM

```python
from fibernet import BeamFEM

fem = BeamFEM(g, default_E=1e9, default_nu=0.3)
result = fem.uniaxial_tension(strain=0.02, deformation_scale=20)

print(f"E* = {result.effective_youngs_modulus:.2e} Pa")
print(f"ν* = {result.effective_poissons_ratio:.3f}")

deformed = result.deformed_graph
stresses = result.stresses
```

### 可视化: render_graph()

```python
from fibernet import render_graph, render_gallery, render_graph_3d

fig = render_graph(
    g,
    theme="dark",               # "dark" | "light" | "blueprint" | "publication"
    color_by="orientation",     # "uniform" | "orientation" | "length" | "stress" | "strain" | "fiber" | "custom"
    colormap="coolwarm",        # 统一颜色方案
    line_width=1.5,
    show_nodes=False,
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
    checkpoint_file="ckpt.json",
)
```

### RL 环境: FiberNetworkEnv

```python
from fibernet import FiberNetworkEnv

env = FiberNetworkEnv(target_E=1e6, target_nu=-0.3)
obs, info = env.reset()

action = {
    "unit_idx": 0,  # honeycomb
    "grid_x": 3,    # grid size - 2
    "grid_y": 3,
    "radius": np.array([0.1]),
}
obs, reward, terminated, truncated, info = env.step(action)
# info["graph"], info["E_star"], info["nu_star"]
env.close()
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
abbe1b3 Checkpoint: Save comprehensive API documentation
16b6583 Add intermediate point programmability
22544ad Phase 7: Regenerate showcase images
```

---

**上下文恢复**: 如果断联，读取此文件继续工作，无需依赖聊天历史。
