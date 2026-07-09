# FiberNet Pattern Engine — API Design Proposal

## 核心思路：四大生成范式

基于深入分析，所有结构可以归为四大范式：

### 范式1: Pattern Engine (基元拼接引擎) — `pattern_2d` / `pattern_3d`

**核心理念**: 定义基元 → 变换 → 平铺 → 焊接

这是覆盖最广的API，覆盖:
- 2D: square, triangular, honeycomb, kagome, reentrant, star, arrowhead, chiral, missing_rib, zigzag
- 3D: cubic, octet, diamond, truncated_octahedron, rhombic_dodecahedron

### 范式2: Hierarchical Engine (真层级引擎) — `hierarchical`

**核心理念**: 边替换自相似

Level 0: 基础lattice
Level 1: 每条边替换为Level 0的缩小版(旋转对齐边方向)
Level 2: Level 1的每条边再替换为Level 0的缩小版

### 范式3: Fractal Engine (分形引擎) — `fractal`

**核心理念**: 递归几何极限

sierpinski, koch, hilbert, fractal_tree, dragon_curve

### 范式4: Stochastic Engine (随机过程引擎) — `stochastic`

**核心理念**: 概率分布驱动

random, voronoi, biomimetic, entangled, field_guided, electrospun

---

## 范式1: Pattern Engine 详细设计

### 基元(Base Unit)的定义

基元是定义在一个单元格(cell)内的几何结构：

```python
class BaseUnit:
    """基元 = 单元格内的几何定义"""
    
    # 顶点坐标 (2D或3D)
    vertices: np.ndarray  # shape (N, 2) or (N, 3)
    
    # 边连接 (顶点对索引)
    edges: np.ndarray  # shape (M, 2), each row is (vertex_i, vertex_j)
    
    # 单元格尺寸
    cell_width: float
    cell_height: float
    cell_depth: float  # for 3D
    
    # 边界标记: 哪些顶点在cell边界上 (用于tiling时的welding)
    boundary_vertices: set  # indices of vertices on cell boundary
    
    # 可选: 内部参数 (如角度、比例等)
    params: dict
```

### 内置基元库

#### 2D基元

```
┌─────────────────────────────────────────────────────────────────────┐
│ square          │ triangular      │ honeycomb        │ kagome      │
│ ┌───┐           │   △             │    ⬡             │  ⊗          │
│ │   │           │  △△             │   ⬡⬡            │ ⊗⊗          │
│ └───┘           │                 │                  │             │
│ cell: [a, a]    │ cell: [a, a√3/2]│ cell: [3a, 2a√3] │ cell: [...] │
│ vertices: 4     │ vertices: 3+    │ vertices: 6+     │ vertices:.. │
│ edges: 4        │ edges: 6+       │ edges: 6+        │ edges: ...  │
├─────────────────────────────────────────────────────────────────────┤
│ reentrant       │ star            │ arrowhead        │ chiral      │
│   ><            │    ✦            │    >-<           │    ⊕        │
│  /  \           │   /|\           │   /   \          │   ╱│╲       │
│ └────┘          │  / | \          │  /     \         │  ╱ │ ╲      │
│ cell: [w, h]    │ cell: [a, a]    │ cell: [a, a]     │ cell: [...] │
│ param: angle    │ param: n_arms   │ param: angle     │ param: r    │
├─────────────────────────────────────────────────────────────────────┤
│ missing_rib     │ zigzag          │ custom           │             │
│ ┌─ ─┐           │  ∧  ∧  ∧       │ user-defined     │             │
│ │   │           │ / \/ \/ \       │ vertices+edges   │             │
│ └───┘           │                 │                  │             │
│ param: which    │ param: pts      │                  │             │
└─────────────────────────────────────────────────────────────────────┘
```

#### 3D基元

```
┌──────────────────────────────────────────────────────┐
│ cubic        │ octet       │ diamond      │ tetrakaidecahedron │
│ ┌──┐         │ △△△         │ ◇◇          │                    │
│ │  │         │ △△△△        │ ◇◇◇         │                    │
│ └──┘         │             │             │                    │
│ param: a     │ param: a    │ param: a    │                    │
└──────────────────────────────────────────────────────────────┘
```

### 变换(Transform)系统

```python
class TransformRules:
    """定义每个tile如何变换基元"""
    
    mirror_x: bool          # 奇数列水平镜像
    mirror_y: bool          # 奇数行垂直镜像
    mirror_z: bool          # 奇数层深度镜像 (3D)
    rotation: float         # 每个tile额外旋转(弧度)
    rotation_alternate: float  # 交替tile的旋转
    
    # 高级: 自定义变换函数
    custom_transform: Callable  # (x, y, col, row) -> (x', y')
```

### Tiling系统

```python
class TilingConfig:
    """定义如何平铺"""
    
    grid: tuple             # (nx, ny) for 2D, (nx, ny, nz) for 3D
    cell_spacing: tuple     # cell间距 (默认=cell_size, 可设gap/overlap)
    periodic: bool          # 是否周期边界
    stagger: str            # 'none', 'hex', 'custom'
    stagger_offset: float   # stagger偏移量
```

### Welding系统

```python
class WeldingConfig:
    """定义如何合并边界节点"""
    
    method: str             # 'position' (默认), 'intersection' (Shapely), 'both'
    tolerance: float        # 合并容差
    detect_intersections: bool  # 是否检测边交叉点
    intersection_tolerance: float  # 交叉检测容差
```

### 完整API

```python
import fibernet as fn

# === 基本用法 ===
net = fn.pattern_2d(
    base='honeycomb',          # 内置基元名称
    cell_size=10.0,            # 单元格尺寸
    grid=(6, 6),               # 平铺网格
    mirror_x=True,             # 奇数列水平镜像
    mirror_y=True,             # 奇数行垂直镜像
    radius=0.2,                # 纤维半径
    perturbation=0.0,          # 节点扰动 (0=完美, >0=随机)
)

# === 参数化基元 ===
net = fn.pattern_2d(
    base='reentrant',
    cell_size=10.0,
    grid=(4, 4),
    angle=150.0,               # reentrant角度
    mirror_x=True,
    mirror_y=True,
)

# === 自定义基元 ===
net = fn.pattern_2d(
    base='custom',
    vertices=[(0, 0), (5, 0), (5, 5), (0, 5)],  # 自定义顶点
    edges=[(0, 1), (1, 2), (2, 3), (3, 0)],     # 自定义边
    cell_size=5.0,
    grid=(6, 6),
    mirror_x=True,
)

# === 函数式基元 ===
def my_unit(cell_size, **params):
    """返回 (vertices, edges, cell_width, cell_height)"""
    a = cell_size
    angle = params.get('angle', 60.0)
    # ... 构建自定义几何 ...
    return vertices, edges, a, a

net = fn.pattern_2d(
    base=my_unit,
    cell_size=10.0,
    grid=(5, 5),
    angle=45.0,
)

# === ZigZag (polyline基元) ===
net = fn.pattern_2d(
    base='zigzag',
    points=[(0, 31.7), (75, 75), (31.7, 0), (161.6, 75), (118.3, 0), (193, 43.3)],
    grid=(4, 10),
    mirror_x=True,
    mirror_y=True,
)

# === 带边交叉检测的P1风格 ===
net = fn.pattern_2d(
    base='square',
    cell_size=10.0,
    grid=(6, 6),
    num_points_per_side=3,     # 每条边3个中点
    perturbation=0.3,          # 随机扰动
    detect_intersections=True, # Shapely边交叉检测
    weld_intersections=True,   # 在交叉点添加节点
)

# === 3D Pattern ===
net = fn.pattern_3d(
    base='octet',
    cell_size=10.0,
    grid=(3, 3, 3),
    mirror_x=True,
    mirror_y=True,
    mirror_z=True,
)
```

### 基元实现细节

每个内置基元通过一个工厂函数生成:

```python
def _unit_honeycomb(cell_size: float, **params) -> BaseUnit:
    """生成蜂窝基元"""
    a = cell_size
    # 六边形顶点
    s = a / 3
    h = s * np.sqrt(3) / 2
    
    vertices = np.array([
        [0, 0], [s/2, 0], [s, h], [s/2, 2*h], [0, 2*h], [-s/2, h],
        # 第二个六边形 (stagger)
        [3*s/2, h], [2*s, h], [5*s/2, 2*h], [2*s, 3*h], [3*s/2, 3*h], [s, 2*h],
    ])
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0],  # hex 1
        [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 6],  # hex 2
    ])
    
    cell_w = 3 * s
    cell_h = 2 * h * 2  # = 2 * a * sqrt(3)/3 * 2
    
    return BaseUnit(vertices, edges, cell_w, cell_h)
```

---

## 范式2: Hierarchical Engine 详细设计

### 真层级算法

```
Level 0: 基础 lattice (例如 triangular)
         定义: 3个顶点, 3条边

Level 1: 每条边 → Level 0 的缩小版
         对于三角形每条边:
         1. 计算边向量 v = p2 - p1
         2. 计算边长度 L = |v|
         3. 生成 Level 0 lattice, 缩放到 L * scale_ratio
         4. 旋转使 lattice 的"主轴"对齐 v 方向
         5. 平移到 p1 位置
         6. 端点 p1, p2 与 Level 0 的顶点合并(weld)
         
Level 2: Level 1 的每条边 → Level 0 的更小缩小版
         递归应用上述过程

结果: 自相似结构, 每个尺度上都有相同的基础拓扑
```

### API设计

```python
net = fn.hierarchical(
    base='honeycomb',          # 基础topology
    levels=3,                  # 层级深度
    scale_ratio=0.3,           # 每级缩小比例
    cell_size=100.0,           # 总体尺寸
    edge_replacement=True,     # True=真层级, False=细分+支撑(旧模式)
    rotation_align=True,       # 自动对齐边方向
    radius_decay=0.7,          # 每级半径衰减
    # 高级: 每级可以用不同topology
    level_topologies=['honeycomb', 'triangular', 'square'],  # 各级不同
)
```

### 与Buehler层级蜂窝的对比

Buehler的层级蜂窝:
- H1: 普通六边形蜂窝
- H2: 大六边形, 壁由H1组成
- H3: 更大六边形, 壁由H2组成
- H4: 最大六边形, 壁由H3组成

我们的实现:
- Level 0: 基础lattice (如honeycomb)
- Level 1: 每条边替换为缩小版honeycomb
- Level 2: Level 1的每条边替换为更小版honeycomb
- Level N: 递归替换

**关键差异**: Buehler的层级是在cell level操作的(大cell内包含小cell),
而我们的边替换是在edge level操作的.
两种方法都能产生自相似结构, 但视觉和力学特性不同.

建议同时支持两种模式:
- `mode='edge_replacement'`: 边替换 (更通用)
- `mode='cell_nesting'`: cell嵌套 (Buehler风格)

---

## 范式3: Fractal Engine

已有良好基础, 只需统一API:

```python
net = fn.fractal(
    type='sierpinski',        # sierpinski/koch/hilbert/tree/dragon
    iterations=4,
    size=50.0,
    # type-specific params
    branch_angle=30.0,        # for tree
    branch_ratio=0.7,         # for tree
)
```

---

## 范式4: Stochastic Engine

```python
net = fn.stochastic(
    type='random_2d',         # random_2d/3d, voronoi_2d/3d, biomimetic, entangled
    num_fibers=200,
    box_size=(50, 50, 50),
    # type-specific params
    fiber_length=10.0,
    curvature=0.3,            # for curved/entangled
    persistence_length=5.0,   # for biomimetic
)
```

---

## 完整API索引

```python
import fibernet as fn

# List all available generators
fn.list_generators()

# Pattern engine (periodic structures)
fn.pattern_2d(base, grid, ...)
fn.pattern_3d(base, grid, ...)

# Hierarchical engine (self-similar)
fn.hierarchical(base, levels, ...)

# Fractal engine (recursive limit)
fn.fractal(type, iterations, ...)

# Stochastic engine (random processes)
fn.stochastic(type, ...)

# Legacy API (still supported)
fn.create('square_lattice_2d', ...)
fn.create('reentrant_honeycomb_2d', ...)
```
