# FiberNet 可视化方法分析

## 当前可视化系统架构

FiberNet 有 **3 层可视化系统**，从底层到高级：

### 第 1 层：核心渲染器 (`fibernet/viz/visualization.py`)

- **函数**: `plot()`, `plot_3d()`, `plot_comparison()`, `plot_statistics()`
- **后端**: matplotlib
- **支持对象**: `FiberNetwork`（老 API）和 `StructureGraph`（新 API）
- **4 种主题**:
  - `light`: 白底 + 深色线条
  - `dark`: 深蓝底 (#1A1A2E) + 浅灰线条
  - `publication`: 白底 + 简洁线条（论文用）
  - `blueprint`: 深蓝底 (#0A1628) + 蓝色线条 + 网格
- **着色方式**:
  - `uniform`: 单色
  - `orientation`: 按纤维角度着色（viridis/coolwarm 等 colormap）
  - `length`: 按纤维长度着色
  - `radius`: 按半径着色
  - `material`: 按材料类型着色
  - `custom`: 自定义标量场

### 第 2 层：Showcase 渲染器 (`fibernet/viz/showcase_renderer.py`)

- **函数**: `render_2d_grid()`, `render_2d_single()`, `render_3d_grid()`, `render_3d_single()`
- **风格**: 深色赛博朋克风
  - 背景: `#0d0d0d`（纯黑）
  - 主色: `#00e87b`（霓虹绿）
  - 辅色: `#00b4ff`（霓虹蓝）
  - 暖色: `#ff7744`（霓虹橙）
- **特殊效果**: 3-pass glow（外发光 → 中发光 → 核心线条）
  - 外层: linewidth×3, alpha=0.07
  - 中层: linewidth×1.8, alpha=0.15
  - 核心: linewidth×1, alpha=0.92
- **坐标归一化**: 所有结构归一化到 [0,1]×[0,1] 展示

### 第 3 层：Showcase 生成脚本

| 脚本 | 输出 | 内容 |
|------|------|------|
| `generate_showcase.py` | 12 张图 | 完整 showcase（2D gallery、3D、FEM、ML、RL） |
| `viz_comprehensive.py` | 1 张图（6×5 grid） | 30 个子图，涵盖所有 pattern 主题 |
| `generate_core_gallery.py` | 单独 gallery | 核心基元展示 |

---

## 当前 12 张 showcase 图的内容

| 文件 | 大小 | 内容 | 视觉风格 |
|------|------|------|----------|
| `01_2d_gallery.png` | 3.8 MB | 3×4 grid，12 种 unit × n_pts=3 + dark 主题 + orientation 着色 | 密集、多彩 |
| `02_honeycomb_detail.png` | 1.8 MB | 单图 honeycomb 5×5 + n_pts=5 + coolwarm colormap | 大、详细 |
| `03_kagome_blueprint.png` | 1.1 MB | kagome + blueprint 主题 + 网格背景 | 蓝图风 |
| `04_voronoi.png` | 1.7 MB | voronoi 镶嵌 + orientation 着色 | 随机纹理 |
| `05_auxetic_comparison.png` | 1.6 MB | reentrant + chiral 对比 + 统计面板 | 2×2 对比 |
| `06_3d_cubic.png` | 0.7 MB | 3D 立方体结构 | pyvista 渲染 |
| `07_3d_octet.png` | 0.6 MB | 3D octet truss | pyvista 渲染 |
| `08_fem_deformation.png` | 1.4 MB | FEM 变形结果 + 位移场 | 变形+颜色叠加 |
| `09_fem_stress.png` | 0.9 MB | FEM 应力场 | 热力图 |
| `10_ml_dataset.png` | 3.8 MB | ML 数据集样本 grid | 大量小图 |
| `11_rl_environment.png` | 1.4 MB | RL 环境可视化 | agent 交互 |
| `12_chiral_stats.png` | 1.7 MB | chiral 统计分析 | 直方图+结构 |

---

## 为什么"感觉很花"？

### 1. 高对比度霓虹风格
Showcase 渲染器用纯黑底 (#0d0d0d) + 霓虹绿 (#00e87b)，对比度极高，视觉冲击强。

### 2. 3-pass Glow 效果
每条线画 3 遍（外发光 → 中发光 → 核心），增加了视觉复杂度和"光晕"感。
对于线条密集的结构（如 voronoi 有 250+ 节点），光晕叠加后非常亮。

### 3. Orientation 着色
`color_by="orientation"` 让每条线颜色不同（按角度映射到 colormap），
当结构有很多边时（honeycomb 5×5 = 130 edges），视觉上非常杂乱。

### 4. 多子图密集排列
`viz_comprehensive.py` 把 30 个子图塞到一个 6×5 的 grid 里，
每个子图都是独立的深色主题 + 发光效果，整体看起来非常拥挤。

### 5. 高 DPI + 大尺寸
图片 180-200 DPI，最大 3.8 MB，像素密度高，细节多，看起来更"花"。

---

## 改进建议

### 快速改善（不改代码）
1. 改用 `theme="publication"` 替代 dark 主题 → 白底 + 简洁线条
2. 用 `color_by="uniform"` 替代 orientation → 单色更清爽
3. 减少 grid 尺寸 → 3×3 改为 2×2，减少子图数量
4. 关闭 glow → `glow=False`

### 中期改善（改代码）
1. **添加一个 "clean" 主题**:
   ```python
   _THEMES["clean"] = {
       "bg_color": "#FFFFFF",
       "fiber_color": "#333333",
       "crosslink_color": "#666666",
       "axis_color": "#DDDDDD",
       "text_color": "#333333",
       "grid_alpha": 0.0,
   }
   ```
2. **添加 `showcase_renderer` 的 clean 模式**:
   - 去掉 glow pass
   - 用浅灰色底 + 深灰色线条
   - 加细网格线而不是全黑
3. **分级渲染**:
   - 单结构大图：详细 + 少量着色
   - Gallery grid：简洁 + 单色
   - 对比图：publication 主题

### 推荐的最终方案
提供一个 `render_mode` 参数：
- `render_mode="showcase"`: 当前霓虹风（适合演示/展示）
- `render_mode="paper"`: 白底 + 单色 + 无 glow（适合论文/报告）
- `render_mode="blueprint"`: 蓝图风（适合技术文档）
