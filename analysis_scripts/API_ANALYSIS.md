# FiberNet 生成范式深度分析 + API 使用示例

## 一、当前生成范式解析

### 核心架构：四级流水线

```
折线基元 → Unit Cell → Transform → Tiling+Welding
```

#### Level 1: 折线基元 (Polyline Primitive)

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

#### Level 2: Unit Cell 组装

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
# 方式1：直接给点序列
g = pattern_2d(points=[(0,0),(5,5),(10,0)], closed=False, box=(10,10))

# 方式2：用 fit_to_box 自动缩放
g = pattern_2d(
    points=[(50,0),(65,35),(100,50),(65,65),(50,100)],
    closed=True, fit_to_box=True, box=(10,10)
)
```

#### Level 3: Transform (变换)

Unit cell 创建后，可以施加三种变换：
- `mirror_x=True`：沿 y 轴镜像（水平翻转）
- `mirror_y=True`：沿 x 轴镜像（垂直翻转）
- `rotation=45.0`：旋转 45 度

这些变换在 tiling 之前执行，所以平铺出来的结构会反映变换后的形状。

#### Level 4: Tiling + Welding (平铺+焊接)

```python
g = pattern_2d(unit='square', box=(10,10), grid=(5,5))
# 结果：5×5=25 个 cell 平铺
# 每个 cell 的边界节点被自动焊接
# 总 box = (50, 50)
```

**焊接算法**：
1. 对每个 grid cell `(i,j)`，复制 unit cell 并平移 `(i*w, j*h)`
2. 所有节点加入新 graph，`merge=True` 触发空间哈希合并
3. 距离 < tolerance 的节点被合并为一个
4. 外边界节点被标记 `boundary=True`

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

这就像 PyTorch 的 `nn.Sequential`：
```python
# 类比 PyTorch
model = nn.Sequential(
    nn.Conv2d(...),   # Level 1: 基元变换
    nn.BatchNorm2d(),  # Level 2: 规范化
    nn.ReLU(),         # Level 3: 激活
    nn.Conv2d(...),   # Level 4: 再次卷积
)

# FiberNet 类比
structure = compose_layers(
    polyline_primitive,     # Level 1: 定义折线
    assemble_unit_cell,     # Level 2: 组装成 unit
    apply_transforms,       # Level 3: 镜像+旋转
    tile_and_weld,          # Level 4: 平铺成网格
)
```

---

## 二、API 使用示例

### 示例 1：从单条直线到复杂网格

```python
import fibernet as fn

# ── Step 1: 单条直线段作为基元 ──
# 直线从 (0,5) 到 (10,5)，这是一条水平线
# 它的"虚拟盒子"是 [0,10] × [0,10]，左右端点就是直线两头
straight_line = fn.pattern_2d(
    points=[(0, 5), (10, 5)],
    closed=False,
    box=(10, 10),
    grid=(3, 3),
    boundary_mode='extend',  # 自动连接到 cell 边界
)
# 结果: 3×3 网格，每个 cell 里一条水平线

# ── Step 2: 折线基元 ──
# V 形折线：从左边界到中心到底部到中心到右边界
v_shape = fn.pattern_2d(
    points=[(0, 5), (5, 10), (5, 0), (10, 5)],
    closed=False,
    box=(10, 10),
    grid=(3, 3),
    mirror_x=True,  # 镜像后变成 X 形
)

# ── Step 3: 闭合多边形基元 → 方形排列 ──
square_lattice = fn.pattern_2d(
    unit='square',
    box=(10, 10),
    grid=(5, 5),
    n_pts_per_side=2,   # 每条边加2个中间节点
    perturbation=0.15,   # 给中间节点随机位移
    seed=42,             # 确定性结果
)

# ── Step 4: 蜂窝形 → 六角排列 ──
honeycomb = fn.pattern_2d(
    unit='honeycomb',
    box=(10, 10),
    grid=(4, 4),
    n_pts_per_side=3,
    seed=0,
)
```

### 示例 2：自定义折线 → 方形盒子 → 大网格

```python
import numpy as np

# 定义一个特殊的折线基元：Z形
z_shape_pts = [
    (0, 10),   # 左上
    (10, 10),  # 右上
    (0, 0),    # 左下（对角线）
    (10, 0),   # 右下
]

# 平铺成 4×4 网格
z_grid = fn.pattern_2d(
    points=z_shape_pts,
    closed=False,
    box=(10, 10),
    grid=(4, 4),
    mirror_x=True,
    mirror_y=True,
)
print(z_grid)
# StructureGraph(dim=2, nodes=..., edges=..., box=[40.0, 40.0])
```

### 示例 3：多层级组合 (小基元 → 拼接 → 旋转)

```python
# 先做一个小基元
base_unit = fn.pattern_2d(
    unit='square',
    box=(5, 5),
    grid=(2, 2),          # 2×2 小网格作为基元
    n_pts_per_side=1,
    perturbation=0.1,
    seed=7,
)

# 把小基元旋转 45 度后平铺
rotated_grid = fn.pattern_2d(
    unit='square',
    box=(10, 10),
    grid=(3, 3),
    rotation=45.0,
    mirror_x=True,
    mirror_y=True,
)
```

### 示例 4：注册自定义 unit 工厂

```python
def my_unit(box, **kwargs):
    """自定义 unit：一个 X 形"""
    from fibernet.core.structure_graph import StructureGraph
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])
    n00 = g.add_node([0, 0])
    n10 = g.add_node([w, 0])
    n01 = g.add_node([0, h])
    n11 = g.add_node([w, h])
    g.add_edge(n00, n11, radius=0.1)  # 对角线1
    g.add_edge(n10, n01, radius=0.1)  # 对角线2
    return g

fn.register_unit('x_shape', my_unit)

# 现在可以像内置 unit 一样使用
x_grid = fn.pattern_2d(unit='x_shape', box=(10, 10), grid=(4, 4))
```

### 示例 5：3D 结构

```python
# 立方体格子
cubic = fn.pattern_3d(unit='cubic', box=(10, 10, 10), grid=(3, 3, 3))

# 八面体桁架
octet = fn.pattern_3d(unit='octet', box=(10, 10, 10), grid=(2, 2, 2))
```

### 示例 6：带模拟的完整流程

```python
# 生成结构
g = fn.pattern_2d(unit='honeycomb', box=(10, 10), grid=(5, 5), n_internal=4)

# FEM 分析
fem = fn.BeamFEM(g)
result = fem.uniaxial_tension(strain=0.01)
print(f"E* = {result.effective_youngs_modulus:.2e} Pa")
print(f"ν* = {result.effective_poissons_ratio:.3f}")

# 可视化变形
fig = fn.render_deformation(g, result, theme='dark', save_path='deformed.png')
```

---

## 三、可视化方法说明

### 当前可视化系统

**核心模块**：`fibernet/viz/render.py` (581 行)

#### 4 种主题预设

| Theme | 背景 | 线条色 | 适用场景 |
|-------|------|--------|----------|
| `dark` | 深黑 `#0a0a0f` | 青绿 `#00e5a0` + 发光 | 展示/演示 |
| `light` | 白色 `#fafafa` | 深蓝 `#2c3e50` | 日常查看 |
| `blueprint` | 深蓝 `#0a1628` | 蓝色 `#4a9eff` + 发光 | 工程风 |
| `publication` | 纯白 `#ffffff` | 黑色 `#333333` | 论文出版 |

#### 5 种渲染函数

1. **`render_graph(g, theme, color_by, ...)`**
   - 基础 2D 渲染
   - `color_by` 支持: `"orientation"`, `"length"`, `"stress"`, `"strain"`, `"uniform"`
   - 支持发光(glow)效果
   - 可选统计信息叠加

2. **`render_graph_3d(g, theme, ...)`**
   - 3D 结构渲染
   - 使用 matplotlib 的 `mpl_toolkits.mplot3d`

3. **`render_deformation(g, fem_result, ...)`**
   - FEM 变形可视化
   - 用颜色映射显示位移/应力分布
   - 变形前后对比

4. **`render_gallery(graphs, titles, ...)`**
   - 多结构并排展示
   - 自动排列成网格

5. **`render_with_stats(g, ...)`**
   - 渲染 + 统计信息面板
   - 节点数、边数、密度、连接度等

#### output_viz 目录内容

当前 `output_viz/` 包含 12 张图（约 20MB），覆盖：
- `01_2d_gallery.png` — 2D 结构总览 (4MB, 最大)
- `02-05` — 各类结构细节 (蜂窝、kagome、voronoi、auxetic)
- `06-07` — 3D 结构 (cubic, octet)
- `08-09` — FEM 变形和应力图
- `10-12` — ML 数据集、RL 环境、chiral 统计

**问题**："感觉很花" 的原因：
1. 暗色主题 + 发光效果在密集结构中产生视觉噪声
2. `n_pts_per_side` 中间节点让边变成曲线，增加视觉复杂度
3. 多图并排时颜色变化过多（`color_by="orientation"` 按角度着色）
4. 缺少统一的图例和比例尺

**建议改进**：
- 对密集结构使用 `publication` 或 `light` 主题
- 关闭 glow (`THEMES["dark"]["glow"] = False`)
- 使用 `color_by="uniform"` 统一着色
- 减少 `n_pts_per_side` 的显示值

---

## 四、FEniCS 集成可行性分析

### 当前状态

**已有 Taichi 集成**：
- `fibernet/sim/fem.py` 中使用 Taichi 加速单元刚度矩阵计算
- `fibernet/sim/accelerated.py` 提供额外的 Taichi 加速模块
- Taichi 在 import 时可选加载 (`HAS_TAICHI` flag)

**当前 FEM 实现**：
- 基于 Euler-Bernoulli 梁单元
- scipy.sparse 稀疏矩阵组装
- scipy.sparse.linalg.spsolve 直接求解器
- 3 DOF/节点 (ux, uy, θ)

### FEniCS 可行性分析

#### FEniCS 在 Windows 上的兼容性

| 方面 | 现状 | 评估 |
|------|------|------|
| 安装方式 | 主要 Linux/macOS，通过 Docker/WSL | ⚠️ 较复杂 |
| pip install | `pip install fenics-dolfinx` 需要 PETSc 等 C 依赖 | ❌ 不直接 |
| Conda | `conda install -c conda-forge fenics-dolfinx` | ✅ 可行 |
| Docker | `docker pull dolfinx/dolfinx` | ✅ 跨平台 |
| WSL2 | 完全支持 | ✅ Win10+ 可行 |
| 原生 Win | 不支持原生 Windows 编译 | ❌ 不行 |

**结论**：FEniCS 在 Windows 上 **不能直接使用**，但可以通过以下方式间接使用：
1. **WSL2** — 最推荐，原生 Linux 环境
2. **Docker** — 跨平台但需要 Docker Desktop
3. **Conda** — `fenics-dolfinx` 可以在 Windows Conda 环境中安装

#### 集成方案设计

```python
# 方案 A：作为可选后端（类似 Taichi 的 HAS_TAICHI 模式）

try:
    import dolfinx
    from dolfinx import mesh, fem, io
    HAS_FENICS = True
except ImportError:
    HAS_FENICS = False

class FEniCSFEM:
    """FEniCS-based FEM solver for StructureGraph.
    
    优势：
    - 支持高阶单元 (Lagrange P2, P3)
    - 自动网格生成 (从 StructureGraph 边到 FEniCS mesh)
    - 支持非线性材料 (hyperelastic, plasticity)
    - 支持多物理场耦合 (热-力, 流-固)
    - 支持周期性边界条件 (built-in)
    - MPI 并行 (自动)
    
    劣势：
    - 重量级依赖 (PETSc, MPI, etc.)
    - Windows 原生不支持
    - 安装复杂
    - 对于简单梁问题可能 overkill
    """
    
    def __init__(self, graph: StructureGraph):
        if not HAS_FENICS:
            raise ImportError("FEniCS required: pip install fenics-dolfinx")
        self.graph = graph
        self._build_mesh()
    
    def _build_mesh(self):
        """Convert StructureGraph edges to FEniCS mesh."""
        # 每条边 → 1D 单元（梁）
        # 或：2D/3D 实体网格
        pass
    
    def linear_elasticity(self, boundary_conditions):
        """Solve linear elasticity problem."""
        pass
    
    def homogenization(self):
        """Compute effective properties via numerical homogenization."""
        pass
```

#### 与现有 Taichi 的对比

| 特性 | Taichi (现有) | FEniCS (拟加) |
|------|--------------|---------------|
| 安装难度 | 简单 (`pip install taichi`) | 复杂 (需要 PETSc/MPI) |
| Windows 支持 | ✅ 原生 | ❌ 需 WSL/Docker |
| GPU 加速 | ✅ CUDA/Metal | ⚠️ 需额外配置 |
| 单元类型 | 梁单元(固定) | 任意(梁/壳/实体) |
| 非线性 | 手动实现 | 内置 UFL 形式 |
| 并行 | GPU 并行 | MPI 分布式 |
| 适用场景 | 快速原型/大规模梁网络 | 精细分析/多物理场 |
| 学习曲线 | 低 | 高 |

#### 建议

**短期（推荐）**：不加 FEniCS，保持 Taichi 加速 + scipy 求解器的轻量方案。
- 用户已有 BeamFEM 能处理大部分超材料分析需求
- Taichi 在 Windows 上原生支持，用户体验好

**中长期（可选）**：作为可选子模块添加
- `fibernet.sim.fenics_backend` — 可选安装
- 提供高阶单元、非线性材料、多物理场能力
- 通过 `HAS_FENICS` flag 优雅降级
- 文档中明确标注 Windows 用户需 WSL2

---

## 五、卍 形结构生成能力分析

### 卍 形结构特征分析

卍 (swastika/manji) 形结构的关键特征：
- **4 重旋转对称** (C4)
- **4 个 L 形臂**，每个臂从中心向外延伸，末端 90° 弯曲
- **中心节点**连接 4 个臂
- 每个臂的结构：从中心 → 水平/垂直延伸 → 直角转弯 → 再延伸

### 能否用现有 API 生成？

**答案：可以，但有多种方法**

#### 方法 1：自定义折线基元 + mirror

```python
import fibernet as fn
import numpy as np

# 卍 的一个臂（1/4 结构）
# 从中心开始：向右延伸，然后向上弯
arm_pts = [
    (5, 5),    # 中心
    (8, 5),    # 向右延伸
    (8, 8),    # 向上弯
]

# 这个臂的端点在 (5,5) 和 (8,8)
# 需要让它接触 cell 边界

# 更合适的方式：定义完整的 卍 形
swastika_pts = [
    # 臂1：上→右
    (5, 10), (5, 7), (8, 7), (8, 10),
    # 臂2：右→下
    (10, 5), (7, 5), (7, 2), (10, 2),
    # 臂3：下→左
    (5, 0), (5, 3), (2, 3), (2, 0),
    # 臂4：左→上
    (0, 5), (3, 5), (3, 8), (0, 8),
]

# 但这个折线不连续，需要分成多条
```

#### 方法 2：自定义 unit 工厂（最推荐）

```python
def _unit_swastika(box, n_internal=0, radius=0.1, material=None,
                   arm_ratio=0.6, bend_ratio=0.3, **kwargs):
    """卍 形 unit cell。
    
    参数:
    arm_ratio: 臂长占 cell 的比例
    bend_ratio: 弯曲段占臂长的比例
    """
    w, h = box
    cx, cy = w/2, h/2
    
    # 臂的几何参数
    arm_len = min(w, h) * arm_ratio / 2  # 从中心到臂端的距离
    bend = arm_len * bend_ratio          # 弯曲段长度
    
    g = StructureGraph(dimension=2, box_size=[w, h])
    
    # 中心节点
    nc = g.add_node([cx, cy])
    
    # 4 个臂端节点（在 cell 边界上）
    n_top = g.add_node([cx, h])        # 上边界
    n_right = g.add_node([w, cy])      # 右边界
    n_bottom = g.add_node([cx, 0])     # 下边界
    n_left = g.add_node([0, cy])       # 左边界
    
    # 4 个弯曲节点
    n_tr = g.add_node([cx + arm_len - bend, cy + arm_len])  # 上臂向右弯
    n_br = g.add_node([cx + arm_len, cy - arm_len + bend])  # 右臂向下弯
    n_bl = g.add_node([cx - arm_len + bend, cy - arm_len])  # 下臂向左弯
    n_tl = g.add_node([cx - arm_len, cy + arm_len - bend])  # 左臂向上弯
    
    # 连接：中心 → 臂中点 → 弯曲点 → 臂端
    # 上臂：中心→上→右弯→顶边
    n_up = g.add_node([cx, cy + arm_len - bend])
    g.add_edge(nc, n_up, radius=radius, material=material)
    g.add_edge(n_up, n_tr, radius=radius, material=material)
    g.add_edge(n_tr, n_top, radius=radius, material=material)
    
    # ... 类似添加其他 3 个臂
    return g
```

#### 方法 3：使用多条折线组合

```python
# 每个臂是一条独立的开放折线
# 然后用 register_unit 组合

def _unit_swastika_v2(box, **kwargs):
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])
    cx, cy = w/2, h/2
    r = min(w, h) * 0.3
    
    # 4 条折线，每条构成一个臂
    arms = [
        [(cx, h), (cx, cy+r), (cx+r, cy+r), (cx+r, cy)],        # 上臂右弯
        [(w, cy), (cx+r, cy), (cx+r, cy-r), (cx, cy-r)],        # 右臂下弯  
        [(cx, 0), (cx, cy-r), (cx-r, cy-r), (cx-r, cy)],        # 下臂左弯
        [(0, cy), (cx-r, cy), (cx-r, cy+r), (cx, cy+r)],        # 左臂上弯
    ]
    
    for arm_pts in arms:
        g.add_polyline(arm_pts, closed=False, radius=kwargs.get('radius', 0.1))
    
    return g

fn.register_unit('swastika', _unit_swastika_v2)
swastika_grid = fn.pattern_2d(unit='swastika', box=(10, 10), grid=(4, 4))
```

### 关键限制

1. **卍 形的臂端需要接触 cell 边界**才能正确焊接
   - 需要确保臂端点位于 `x=0`, `x=w`, `y=0`, `y=h` 上
   - 或者使用 `boundary_mode='extend'` 自动连接

2. **旋转对称性**：
   - 平铺后的焊接需要臂端点精确对齐
   - 如果臂弯曲后不到边界，相邻 cell 的臂不会连接

3. **mirror 的影响**：
   - `mirror_x=True` 会产生 卐 和 卍 交替的图案
   - `mirror_y=True` 同理
   - 不 mirror 则全是同一方向

### 实际可行性评估

**结论**：完全可以生成，但需要注意边界接触。

最简洁的方法是用自定义 unit 工厂，确保 4 个臂端分别在 4 条 cell 边界上。
这样 `pattern_2d(unit='swastika', grid=(5,5))` 就能产生连续的 卍 网格。

如果需要更复杂的变体（如臂上有波纹），可以结合 `n_pts_per_side` 参数。

