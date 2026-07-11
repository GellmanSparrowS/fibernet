# FiberNet 生成范式分析 (v2)

**日期**: 2026-07-11
**状态**: 所有连通性问题已修复 (49/49 测试通过)

---

## 核心范式：折线基元 → 形状 → 网格 → 多层级组合

整个生成系统可以理解为四个层级的组装，就像 PyTorch 中的神经网络层一样——
每一层接受上一层的输出作为输入，层层叠加产生越来越复杂的结构。

```
┌──────────────────────────────────────────────────────┐
│  Level 4: 多层级组合 (Multi-level Composition)        │
│  → 把小基元当成"新折线"，再拼成更大的形状/网格        │
│  → 就像 nn.Sequential 里嵌套 nn.Module                │
├──────────────────────────────────────────────────────┤
│  Level 3: 变换 (Transform)                           │
│  → 镜像、旋转 → 在平铺前/后施加                       │
├──────────────────────────────────────────────────────┤
│  Level 2: 形状组装 + 平铺焊接                         │
│  → 折线拼成方形/蜂巢/Kagome等 → 在网格中平铺 → 焊接   │
├──────────────────────────────────────────────────────┤
│  Level 1: 折线基元 (Polyline Primitive)               │
│  → 一条直线段或折线段，两端定义"虚拟盒子"的端点        │
└──────────────────────────────────────────────────────┘
```

---

## Level 1: 折线基元 (Polyline Primitive)

### 概念

一条折线是最小的几何基元。它由一组有序的 `(x, y)` 坐标点定义。
折线的**两端**（首点和尾点）定义了这个基元的**空间边界**——
就像一个虚拟的盒子的左右端点。

```
(0,5) ─────── (10,5)    ← 直线段，两端 = 虚拟盒子的左右端点

(0,0) ── (5,10) ── (10,0)   ← V形折线

(0,0) ── (3,3) ── (7,7) ── (10,0)   ← 任意折线
```

### 关键特性：每条边可加中间节点

一条直线段 `p1 → p2`，加上 `n_pts_per_side` 个中间节点后，
可以变成曲线/折线段。每个中间节点有 (dx, dy) 位移控制变形。

```
直线: (0,5) ─────────────── (10,5)

加3个中间节点+位移后:
(0,5) ── (2.5+dx₁, 5+dy₁) ── (5+dx₂, 5+dy₂) ── (7.5+dx₃, 5+dy₃) ── (10,5)
变成了一条曲线/变形折线段!
```

### API 示例

```python
import fibernet as fn

# 1a. 最简单的直线段
g = fn.pattern_2d(
    points=[(0, 5), (10, 5)],  # 两个端点
    closed=False,               # 不闭合
    box=(10, 10),               # 虚拟盒子尺寸
    grid=(1, 1),                # 不平铺
)
# → StructureGraph(dim=2, nodes=2, edges=1)

# 1b. 直线 + 中间节点（变形为曲线）
g = fn.pattern_2d(
    points=[(0, 5), (10, 5)],
    closed=False,
    box=(10, 10),
    grid=(1, 1),
    n_pts_per_side=3,           # 每条边3个中间节点
    perturbation=0.3,            # 位移幅度 (30% of edge length)
    seed=42,                     # 确定性随机种子
)
# → StructureGraph(dim=2, nodes=5, edges=4)
# 直线变成了4段折线

# 1c. V形折线（3个点定义2段折线）
g = fn.pattern_2d(
    points=[(0, 0), (5, 10), (10, 0)],
    closed=False,
    box=(10, 10),
    grid=(1, 1),
)
# → StructureGraph(dim=2, nodes=3, edges=2)

# 1d. 闭合多边形
g = fn.pattern_2d(
    points=[(2, 2), (8, 2), (8, 8), (2, 8)],
    closed=True,                # 首尾自动连接
    box=(10, 10),
    grid=(1, 1),
    n_pts_per_side=2,           # 每条边2个中间节点
)
# → StructureGraph(dim=2, nodes=12, edges=12)
# 4个角点 + 4条边×2个中间点 = 12个节点
```

---

## Level 2: 形状组装 + 平铺焊接

### 概念

将折线基元排列成特定的**形状**（正方形、蜂巢、Kagome 等），
定义在一个 `[0, w] × [0, h]` 的单元盒子内。
然后在网格中**平铺**这个单元，相邻单元在边界处自动**焊接**。

```
    ┌────────┐     ┌────────┐     ┌────────┐
    │  Unit  │     │  Unit  │     │  Unit  │
    │  Cell  │ ──→ │  Cell  │ ──→ │  Cell  │     3×3 网格
    │        │     │        │     │        │
    └────────┘     └────────┘     └────────┘
      边界节点 ══════ 焊接 ══════ 边界节点
```

### 内置的12种形状 (2D) + 3种 (3D)

| 形状 | 描述 | 边界接触点 | 典型用途 |
|------|------|-----------|---------|
| `square` | 正方形框 | 四边 | 基础网格 |
| `triangle` | 三角菱形 | 四角+对角线 | 三角网格 |
| `hexagon` | 正六边形 | 上下左右 | 蜂巢变体 |
| `honeycomb` | 蜂窝形 | 上下左右 | 拉胀材料 |
| `kagome` | 角点+边中点+中心 | 四角四边 | 光子晶体 |
| `reentrant` | 内凹箭头 | 上下左右 | 负泊松比 |
| `chiral` | 圆环+切线 | 四角 | 手性超材料 |
| `star` | N角星 | 封闭内部 | 装饰/隔音 |
| `cross` | 十字形 | 上下左右 | 力学超材料 |
| `missing_rib` | 缺肋蜂窝 | 上下右 | 各向异性 |
| `diamond` | 菱形 | 上下左右 | 金刚石晶格 |
| `voronoi` | Voronoi镶嵌 | 随机 | 仿生/随机泡沫 |
| `cubic` (3D) | 立方体 | 六面 | 基础3D |
| `octet` (3D) | 八面体桁架 | 六面+中心 | 轻质高强 |
| `diamond_3d` (3D) | 金刚石3D | 六面+面心 | 光子带隙 |

### API 示例

```python
# 2a. 标准蜂窝 + 3×3 网格
g = fn.pattern_2d(
    unit="honeycomb",
    box=(10, 10),           # 每个单元格 10×10
    grid=(5, 5),            # 5×5 网格 → 共 50×50 的结构
    n_internal=4,           # FEM变形用的内部点
    n_pts_per_side=2,       # 每条边2个中间节点
    radius=0.1,             # 梁半径
)
# → StructureGraph(dim=2, nodes=..., edges=..., box=[50.0, 50.0])

# 2b. Kagome + 变换
g = fn.pattern_2d(
    unit="kagome",
    box=(10, 10),
    grid=(3, 3),
    mirror_x=True,          # 水平镜像
    rotation=30.0,           # 旋转30度 (平铺后旋转，保持连通!)
)

# 2c. 自定义形状 → 自动适配盒子
g = fn.pattern_2d(
    points=[(50, 0), (65, 35), (100, 50), (65, 65), (50, 100), (35, 65), (0, 50), (35, 35)],
    closed=True,
    fit_to_box=True,         # 自动缩放到 box 尺寸
    box=(10, 10),
    grid=(3, 3),
)

# 2d. 注册自定义形状为可复用的 unit
def my_unit_factory(box, radius=0.1, material=None, **kwargs):
    w, h = box
    g = fn.StructureGraph(dimension=2, box_size=[w, h])
    # 定义 Z 字形
    n0 = g.add_node([0, h/2])
    n1 = g.add_node([w/3, h])
    n2 = g.add_node([2*w/3, 0])
    n3 = g.add_node([w, h/2])
    g.add_edge(n0, n1, radius=radius, material=material)
    g.add_edge(n1, n2, radius=radius, material=material)
    g.add_edge(n2, n3, radius=radius, material=material)
    return g

fn.register_unit("zigzag", my_unit_factory)
g = fn.pattern_2d(unit="zigzag", box=(10, 10), grid=(4, 4))

# 2e. 3D 结构
g = fn.pattern_3d(
    unit="octet",
    box=(5, 5, 5),
    grid=(3, 3, 3),
    n_pts_per_side=2,
    radius=0.05,
)
# → StructureGraph(dim=3, nodes=..., edges=..., box=[15, 15, 15])
```

---

## Level 3: 变换 (Transform)

### 概念

在 Level 2 的形状上施加几何变换：
- **镜像 (mirror_x, mirror_y)**: 在平铺前施加，保持边界对齐
- **旋转 (rotation)**: 在平铺**后**施加，保持连通性
- **手动变换**: translate, rotate, mirror, scale (来自 transforms 模块)

```python
# 3a. 内置变换参数
g = fn.pattern_2d(
    unit="honeycomb",
    box=(10, 10),
    grid=(3, 3),
    mirror_x=True,      # 水平翻转
    mirror_y=False,
    rotation=45.0,       # 整个网格旋转45度
)

# 3b. 手动变换已有结构
g = fn.pattern_2d(unit="square", box=(10, 10), grid=(3, 3))
g_rotated = fn.rotate(g, 30.0, center=[15, 15, 0])
g_mirrored = fn.mirror_x(g, origin=15)
g_scaled = fn.scale(g, 2.0)
g_translated = fn.translate(g, [50, 0, 0])

# 3c. 组合变换
g_composed = fn.compose(g, fn.translate([10, 0, 0]), fn.rotate(45.0))
```

---

## Level 4: 多层级组合 (Multi-level Composition)

### 概念

这是最强大的能力：**把一个完整的结构当成"新折线基元"，
再次走 Level 1-3 的流程**。就像 PyTorch 中的 `nn.Sequential` 嵌套：

```python
# PyTorch 类比
model = nn.Sequential(
    nn.Sequential(nn.Linear(10, 20), nn.ReLU()),  # "子网络"
    nn.Sequential(nn.Linear(20, 30), nn.ReLU()),  # "子网络"
)

# FiberNet 类比
big_structure = pattern_2d(
    unit=pattern_2d(unit="honeycomb", grid=(2,2)),  # "子结构作为unit"
    grid=(3, 3),
)
```

### 实现方式

通过 `register_unit` 把任何生成函数注册为新的基元工厂：

```python
# ===== 层级 1: 基础蜂窝 =====
base = fn.pattern_2d(unit="honeycomb", box=(5, 5), grid=(2, 2))
# → 一个小的蜂窝结构 (约 18 nodes)

# ===== 层级 2: 把蜂窝当成 unit，再拼大网格 =====
from fibernet.core.tiling import fit_unit_to_box

def nested_honeycomb_factory(box, **kwargs):
    """内部生成 2×2 蜂窝，然后缩放适配到新 box。"""
    inner = fn.pattern_2d(
        unit="honeycomb",
        box=(box[0] / 2, box[1] / 2),
        grid=(2, 2),
    )
    return fit_unit_to_box(inner, target_box=list(box) + [0.0])

fn.register_unit("nested_honeycomb", nested_honeycomb_factory)

g = fn.pattern_2d(
    unit="nested_honeycomb",
    box=(10, 10),
    grid=(3, 3),
)
# → 3×3 网格，每个格子里是一个 2×2 的小蜂窝
# → 约 126 nodes, 186 edges

# ===== 层级 3: 再嵌套一层 =====
def double_nested_factory(box, **kwargs):
    inner = fn.pattern_2d(
        unit="nested_honeycomb",
        box=(box[0] / 2, box[1] / 2),
        grid=(2, 2),
    )
    return fit_unit_to_box(inner, target_box=list(box) + [0.0])

fn.register_unit("double_nested", double_nested_factory)
g = fn.pattern_2d(unit="double_nested", box=(10, 10), grid=(2, 2))
# → 2×2 网格，每个格子是嵌套蜂窝，总共 4 层嵌套!
```

### 混合结构：不同形状并排

```python
# 左边 square 网格 + 右边 honeycomb 网格
g1 = fn.pattern_2d(unit="square", box=(10, 10), grid=(3, 3))
g2 = fn.pattern_2d(unit="honeycomb", box=(10, 10), grid=(3, 3))
g2_shifted = fn.translate(g2, [30, 0, 0])  # 平移到右侧
combined = g1.merge(g2_shifted)
# → 两个不同结构并排，后续可通过锚点焊接连接
```

### 锚点焊接 (Anchor Point Welding)

最终把多个独立的结构通过共同的锚点连接在一起：

```python
# 结构 A 和结构 B 通过共同位置的节点焊接
from fibernet.core.tiling import tile_2d

# 生成两个结构
g_a = fn.pattern_2d(unit="square", box=(10, 10), grid=(3, 3))
g_b = fn.pattern_2d(unit="kagome", box=(10, 10), grid=(3, 3))

# 平移 B 到 A 的右侧
g_b = fn.translate(g_b, [30, 0, 0])

# 合并 — 空间哈希自动焊接重叠的节点
combined = g_a.merge(g_b)
# 如果两个结构在 x=30 处有共同节点，它们会被自动焊接!
```

---

## 与 PyTorch 的类比

| PyTorch | FiberNet | 说明 |
|---------|----------|------|
| `nn.Linear` | 折线基元 (polyline) | 最基础的操作 |
| `nn.Sequential` | `register_unit` + `pattern_2d` | 组合多个层 |
| `nn.Module` 嵌套 | 嵌套 `register_unit` | 层级组合 |
| `torch.cat` / `torch.stack` | `merge` / `tile_2d` | 拼接多个结构 |
| `F.interpolate` | `fit_unit_to_box` + `scale` | 缩放适配 |
| `nn.Dropout` | `perturbation` + `seed` | 引入随机性 |
| `model.eval()` | `BeamFEM` simulation | 评估结果 |
| `torch.cuda` | Taichi 加速 | 硬件加速 |

---

## 关键 API 速查

```python
# 查看所有可用基元
fn.list_units()
# → ['chiral', 'cross', 'diamond', 'hexagon', 'honeycomb', 'kagome',
#    'missing_rib', 'reentrant', 'square', 'star', 'triangle', 'voronoi']

# 生成 2D 结构
g = fn.pattern_2d(unit=..., box=..., grid=..., n_pts_per_side=..., ...)

# 生成 3D 结构
g = fn.pattern_3d(unit=..., box=..., grid=..., n_pts_per_side=..., ...)

# 注册自定义基元
fn.register_unit("my_shape", my_factory_function)

# 变换
g = fn.translate(g, [dx, dy, dz])
g = fn.rotate(g, angle_deg, center=[cx, cy, cz])
g = fn.mirror_x(g, origin=x)
g = fn.scale(g, factor)

# 平铺
g_tiled = fn.tile_2d(unit_cell, grid=(nx, ny), box_size=(w, h))

# 合并结构
combined = g1.merge(g2)

# 运行 FEM 仿真
fem = fn.BeamFEM(g)
result = fem.uniaxial_tension(strain=0.01)
print(f"E* = {result.effective_youngs_modulus:.2e}")
```
