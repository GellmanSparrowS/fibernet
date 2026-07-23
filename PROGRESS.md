# FiberNet v4.1.2 — Post-Release Progress

## 最新状态 (2026-07-23)

### ✅ v4.1.2 Released

**1. BUG 修复: 纤维半径不影响力学结果**
- 根因: `graph_to_fem_input()` 中 `[e.radius for e in graph.edges]` 遍历 Dict 的 keys (整数), 而非 values (SEdge 对象)
- 修复: 改为 `[e.radius for e in graph.edges.values()]`
- 验证: r=0.02→σ=2967MPa, r=0.05→σ=6309MPa, r=0.10→σ=11530MPa, r=0.20→σ=20997MPa
- `to_sim_result()` 中能量计算的半径访问也已修复

**2. Showcase 布局修复**
- 问题: 原 3×4 网格放不下 8拉伸+4压缩+3半径+4分析 = 19 个面板
- 修复: 改为 5×4 GridSpec 布局
  - Row 1-2: 8 个拉伸结构 (2×4)
  - Row 3: 4 个压缩结构
  - Row 4: 4 个半径变化 (r=0.02, 0.05, 0.10, 0.20)
  - Row 5: 4 个分析图表

**3. 暗色主题可见性修复**
- 线宽: 0.6 → 2.0 (暗色) / 1.5 (亮色)
- 色图: inferno → magma (暗色拉伸), coolwarm → RdYlBu (暗色压缩)
- 添加了圆角 capstyle 改善视觉质量

**4. 测试结果**
- 312/312 测试全部通过

### 提交记录
```
pending: fix(fem): radius propagation + showcase layout + dark theme visibility
41c65bc release: v4.1.1 — FEM convenience API + showcase
```

---

## 历史记录

### v4.1.1 Released (2026-07-23)
- FEM convenience API (graph_to_fem_input, stretch_test, to_sim_result, render_fem_stress)
- 312/312 tests passing

### v4.1.0 Released (2026-07-23)
- PyPI: https://pypi.org/project/fibernet/4.1.0/
- GitHub Release: https://github.com/GellmanSparrowS/fibernet/releases/tag/v4.1.0
- BeamFrameFEM_v6 + 3D structures + API fixes
- 312/312 tests passing
