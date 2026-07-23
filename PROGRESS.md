# FiberNet v4.1.2 — Post-Release Progress

## 最新状态 (2026-07-23)

### ✅ v4.1.2 Released

**PyPI**: https://pypi.org/project/fibernet/4.1.2/
**GitHub**: https://github.com/GellmanSparrowS/fibernet/releases/tag/v4.1.2

**1. BUG 修复: 纤维半径不影响力学结果 (Critical)**
- 根因: `graph_to_fem_input()` 中 `[e.radius for e in graph.edges]` 遍历 Dict 的 keys (整数), 而非 values (SEdge 对象)
- 修复: 改为 `[e.radius for e in graph.edges.values()]`
- 同时修复 `to_sim_result()` 中能量计算的半径访问
- 验证结果:
  | 半径 | 最大应力 |
  |------|---------|
  | 0.02 | 2967 MPa |
  | 0.05 | 6309 MPa |
  | 0.10 | 11530 MPa |
  | 0.20 | 20997 MPa |

**2. Showcase 布局修复**
- 问题: 原 3×4 网格放不下 19 个面板 → 中间行拉伸和压缩堆叠
- 修复: 改为 5×4 GridSpec 布局
  - Row 1-2: 8 个拉伸结构 (2×4)
  - Row 3: 4 个压缩结构
  - Row 4: 4 个半径变化 (r=0.02, 0.05, 0.10, 0.20)
  - Row 5: 4 个分析图表

**3. 暗色主题可见性修复**
- 线宽: 0.6 → 2.0 (暗色) / 1.5 (亮色)
- 色图: inferno → magma (暗色拉伸), coolwarm → RdYlBu (暗色压缩)
- 添加圆角 capstyle 改善视觉质量

**4. 测试结果**
- 312/312 测试全部通过

### 文件变更
- `fibernet/ml/beam_frame_fem_v6.py`: graph_to_fem_input + to_sim_result 半径修复
- `scripts/fem_showcase.py`: 5×4 GridSpec 布局 + 增亮暗色主题
- `fibernet/version.py`, `fibernet/__init__.py`, `pyproject.toml`: 版本 → 4.1.2
- `CHANGELOG.md`, `PROGRESS.md`: 更新日志
- `output_data/deformation_test/viz/`: 重新生成的展示图

### 提交记录
```
cf891cd fix(fem): radius propagation + showcase layout + dark theme visibility
41c65bc release: v4.1.1 — FEM convenience API + showcase
```

## 2026-07-23 - Fix Showcase Visualization (v4.1.2)

**Issue**: Dark theme showcase images had invisible/dark edges despite bright colormap values

**Root Cause**: `LineCollection(colors=...)` (plural) only sets edge color, leaving face color as black. This made lines appear dark even when the colormap specified bright colors.

**Fix**: Changed `colors=` to `color=` (singular) in `scripts/fem_showcase.py:draw_fem_panel()`
- `colors=` → only sets edgecolor (face stays black)
- `color=` → sets both facecolor and edgecolor (lines become visible)

**Verification**: 
- Before: 0 bright pixels detected
- After: 998,013 bright pixels, mean color RGB(111, 217, 146)
- All 16 panels now show vibrant, visible edges

**Changes**:
- `scripts/fem_showcase.py`: Line 94, changed `colors=` to `color=`
- Regenerated showcase images (dark + light themes)
- Version remains 4.1.2 (no functional API changes)
