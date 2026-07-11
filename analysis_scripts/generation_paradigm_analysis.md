# FiberNet 生成范式深度分析

## 核心架构：四级流水线 + 多层级组合

```
折线基元 → Unit Cell → Transform → Tiling+Welding → (可作为新基元递归)
```

---

## 一、四级流水线详解

### Level 1: 折线基元 (Polyline Primitive)

**概念**：一个折线（或闭合多边形）是最小的几何基元。
它由一组有序的 `(x, y)` 点定义，可以是：
- **闭合的** (`closed=True`)：首尾自动连接，形成多边形
- **开放的** (`closed=False`)：纯折线段

**关键**：折线的两个端点（或闭合多边形的顶点）定义了这个基元的"虚拟盒子"的左右端点。
比如一个 L-shape 折线 `[(0,0),(10,0),(10,3),(3,3),(3,10),(0,10)]`：
- 它的包围盒是 `[0,10] × [0,10]`
- 这些端点也是 cell boundary 的接触点
- 平铺时，这些接触点会被焊接(weld)到相邻 cell 的对应点

**每条边可添加中间节点**（`n_pts_per_side`）：
```python
# 一条直线边 p1→p2 上加 3 个中间节点
p1 = (0, 0)
p2 = (10, 0)
# 等分插值：(2.5, 0), (5.0, 0), (7.5, 0)
# 加位移后：(2.5+dx1, dy1), (5.0+dx2, dy2), (7.5+dx3, dy3)
# 直线变成了折线段！
```

### Level 2: Unit Cell 组装

**概念**：一个 unit cell 是在一个 `[0,w] × [0,h]` 的盒子里放置一组折线基元。

内置的 12 种 unit cell 类型：

| Unit | 基元描述 | 边界接触 |
|------|----------|----------|
| `square` | 4边正方形框 | 四边全接触 |
| `triangle` | 菱形(两个三角) | 四角+对角线 |
| `hexagon` | 6边正六边形 | 上下左右 |
| `honeycomb` | 蜂窝形6段 | 上下左右 |
| `kagome` | 角点+边中点+中心 | 四角四边中 |
| `reentrant` | 内凹箭头形 | 上下左右 |
| `chiral` | 圆环+切线韧带 | 四角 |
| `star` | N角星 | 封闭内部 |
| `cross` | 十字形 | 上下左右 |
| `missing_rib` | 缺肋蜂窝 | 上下右 |
| `diamond` | 菱形 | 上下左右 |
| `voronoi` | Voronoi镶嵌 | 随机 |

**自定义 unit cell**：
```python
# 方式1：直接给点序列（坐标在 box 空间内）
g = fn.pattern_2d(points=[(0,0),(5,5),(10,0)], closed=False, box=(10,10))

# 方式2：用 fit_to_box 自动缩放（任意坐标系→box）
g = fn.pattern_2d(
    points=[(50,0),(65,35),(100,50),(65,65),(50,100)],
    closed=True, fit_to_box=True, box=(10,10)
)

# 方式3：用 register_unit 注册为可复用的工厂
fn.register_unit('my_shape', my_factory_function)
g = fn.pattern_2d(unit='my_shape', box=(10,10), grid=(3,3))
```

### Level 3: Transform (变换)

Unit cell 创建后，可以施加三种变换：
- `mirror_x=True`：沿 y 轴镜像（水平翻转）
- `mirror_y=True`：沿 x 轴镜像（垂直翻转）
- `rotation=45.0`：旋转 45 度

这些变换在 tiling 之前执行，所以平铺出来的结构会反映变换后的形状。

### Level 4: Tiling + Welding (平铺+焊接)

```python
g = fn.pattern_2d(unit='square', box=(10,10), grid=(5,5))
# 结果：5×5=25 个 cell 平铺
# 每个 cell 的边界节点被自动焊接（merge=True）
# 总 box = (50, 50)
```

**焊接算法**：
1. 对每个 grid cell `(i,j)`，复制 unit cell 并平移 `(i*w, j*h)`
2. 所有节点加入新 graph，`merge=True` 触发空间哈希合并
3. 距离 < tolerance 的节点被合并为一个
4. 外边界节点被标记 `boundary=True`

---

## 二、多层级组合（类 PyTorch 的层级叠加）

### 与 PyTorch 的类比

```python
# PyTorch 类比
model = nn.Sequential(
    nn.Conv2d(...),      # Level 1: 基元变换
    nn.BatchNorm2d(),     # Level 2: 规范化
    nn.ReLU(),            # Level 3: 激活
    nn.Conv2d(...),      # Level 4: 再次卷积
)

# FiberNet 类比
structure = compose_layers(
    polyline_primitive,     # Level 1: 定义折线
    assemble_unit_cell,     # Level 2: 组装成 unit
    apply_transforms,       # Level 3: 镜像+旋转
    tile_and_weld,          # Level 4: 平铺成网格
)
```

### 多层级组合示意

```
Level 0: 单条直线段
  ↓ (加中间节点 + 位移)
Level 1: 变形线段 → 折线/曲线
  ↓ (组合成闭合形状)
Level 2: Unit Cell (方形/蜂窝/自定义)
  ↓ (mirror/rotate + tile + weld)
Level 3: 大网格 (正方形排列)
  ↓ (再次作为基元？)
Level 4: 超大结构 (组合不同区域的 pattern)
```

### 实现多层级组合的方式

#### 方式 1：register_unit + pattern_2d 嵌套

```python
import fibernet as fn
from fibernet.core.structure_graph import StructureGraph

def my_complex_unit(box, **kwargs):
    """一个复杂 unit cell，内部调用 pattern_2d 生成子结构。"""
    # 内部生成一个小网格作为 unit cell
    inner = fn.pattern_2d(
        unit='honeycomb', box=(box[0]/2, box[1]/2),
        grid=(2,2),
    )
    # 把小网格缩放适配到当前 box
    from fibernet.core.tiling import fit_unit_to_box
    return fit_unit_to_box(inner, target_box=list(box) + [0.0])

fn.register_unit('complex', my_complex_unit)

# 现在可以用 'complex' 基元做大网格
mega = fn.pattern_2d(unit='complex', box=(10,10), grid=(4,4))
```

#### 方式 2：StructureGraph.merge() 手动组合

```python
g1 = fn.pattern_2d(unit='square', box=(10,10), grid=(3,3))
g2 = fn.pattern_2d(unit='honeycomb', box=(10,10), grid=(3,3))

# 平移 g2 使其接在 g1 右侧
from fibernet.core.transforms import translate
g2_shifted = translate(g2, [30, 0, 0])

# 合并（边界节点自动焊接）
combined = g1.merge(g2_shifted)
```

#### 方式 3：compose() 链式变换

```python
from fibernet.core.transforms import compose, make_translate, make_rotate, make_mirror

pipeline = compose(
    base_graph,
    make_rotate(45),
    make_mirror(axis='x'),
    make_translate([10, 0, 0]),
)
```

---

## 三、API 使用示例

### 示例 1：从单条直线到复杂网格

```python
import fibernet as fn
import numpy as np

# ── Step 1: 单条直线段作为基元 ──
line = fn.pattern_2d(
    points=[(0, 5), (10, 5)],
    closed=False,
    box=(10, 10),
    grid=(3, 3),
    boundary_mode='extend',
)
# 结果: 3×3 网格，每个 cell 里一条水平线

# ── Step 2: 折线基元 ──
v_shape = fn.pattern_2d(
    points=[(0, 5), (5, 10), (5, 0), (10, 5)],
    closed=False,
    box=(10, 10),
    grid=(3, 3),
    mirror_x=True,
)

# ── Step 3: 闭合多边形基元 → 方形排列 ──
square_lattice = fn.pattern_2d(
    unit='square',
    box=(10, 10),
    grid=(5, 5),
    n_pts_per_side=2,
    perturbation=0.15,
    seed=42,
)

# ── Step 4: 自定义形状 + 变换 ──
star_grid = fn.pattern_2d(
    points=[(50,0),(65,35),(100,50),(65,65),(50,100),(35,65),(0,50),(35,35)],
    closed=True,
    fit_to_box=True,
    box=(10, 10),
    grid=(3, 3),
    mirror_x=True,
    mirror_y=True,
)
```

### 示例 2：注册自定义基元

```python
def _unit_zigzag(box, **kwargs):
    """Z字形 unit cell"""
    from fibernet.core.structure_graph import StructureGraph
    from fibernet.core.material import Material
    w, h = box
    r = kwargs.get('radius', 0.1)
    mat = kwargs.get('material', Material.generic())
    
    g = StructureGraph(dimension=2, box_size=[w, h, 0])
    # Z 形: 左下→右上→右下→右上边界
    n0 = g.add_node([0, h/2])       # 左边界
    n1 = g.add_node([w/2, h])       # 上边界
    n2 = g.add_node([w/2, 0])       # 下边界
    n3 = g.add_node([w, h/2])       # 右边界
    g.add_edge(n0, n1, radius=r, material=mat)
    g.add_edge(n1, n2, radius=r, material=mat)
    g.add_edge(n2, n3, radius=r, material=mat)
    return g

fn.register_unit('zigzag', _unit_zigzag)
grid = fn.pattern_2d(unit='zigzag', box=(10,10), grid=(4,4))
```

### 示例 3：内部形状 + boundary_mode='extend'

```python
# 一个不接触边界的正方形，自动加 bridge 连接到 cell 边界
net = fn.pattern_2d(
    points=[(3,3),(7,3),(7,7),(3,7)],
    closed=True,
    box=(10, 10),
    grid=(3, 3),
    boundary_mode='extend',
)
```

---

## 四、范式验证：当前系统 vs 用户描述

| 用户描述 | 当前实现 | 状态 |
|----------|----------|------|
| 单条直线段作为基元 | `points=[(0,5),(10,5)]`, `closed=False` | ✅ |
| 两头定义为盒子左右端点 | 边界检测 `_detect_boundary_nodes` | ✅ |
| 按方形/其他形状排列 | 12种内置 unit + 自定义 `points` | ✅ |
| 长度适配 | `fit_to_box=True` 自动缩放 | ✅ |
| 正方形拼成大网格 | `grid=(N,M)` + `tile_2d` | ✅ |
| 小基元做拼接旋转 | `mirror_x/y`, `rotation`, `merge()` | ✅ |
| 多层级组合 | `register_unit` 嵌套 + `compose` | ✅ |
| 锚点焊接 | `tile_2d` 中的 `merge=True` 空间哈希 | ✅ |
| 像 PyTorch 加层 | `compose()` 链式变换 | ✅ |

### 已知限制

1. **非正交平铺**：当前 `tile_2d` 只支持正交网格排列，不支持六角形平铺（hexagonal tiling）
2. **3D 平铺**：`tile_3d` 已实现但基元较少（cubic, octet）
3. **跨区域组合**：不同区域用不同 pattern 需要手动 `merge()`，没有直接的"分区平铺"API
4. **递归层级深度**：理论上无限，但实际嵌套 3-4 层后焊接精度可能累积误差
