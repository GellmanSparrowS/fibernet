# FiberNet v4.1.0+ — Post-Release Progress

## 最新状态 (2026-07-23)

### ✅ 全部完成

**1. BUG 修复**
- BUG 5: 无效节点索引 → 清晰的 ValueError (替代 cryptic IndexError)
- BUG 8: solve_3d() 添加 reactions/edge_forces/K (与 solve_2d 一致)
- 回归: 312/312 测试通过

**2. API 易用性增强 (4 个新功能)**
- `graph_to_fem_input(graph)`: StructureGraph → FEM 输入一步转换
- `stretch_test(graph, target_stretch)`: 一行代码完成单轴拉伸 FEM
- `to_sim_result(fem_result)`: FEM dict → SimResult (后端可替换)
- `render_fem_stress(graph, result)`: 边着色应力可视化

**API 提升**: 基本 FEM 分析从 ~10 行 → 3 行:
```python
solver = BeamFrameFEM_v6(E=1e9, nu=0.3)
g = fn.pattern_2d(unit='honeycomb', box=(10,10), grid=(4,4), radius=0.05)
res = solver.stretch_test(g, target_stretch=2.0)
```

**3. API 分析结果**
- 易用性 (简单场景): ✅ 已改善 (新增 4 个便捷方法)
- 可编程性 (复杂场景): ✅ 良好 (K矩阵、torch支持、非线性历史)
- 两套 API (弹簧/FEM) 不冲突, 可通过 SimResult 无缝切换

**4. FEM 展示图**
- Dark: `fem_showcase_dark.png` (2.0 MB, 3979×2658 px)
- Light: `fem_showcase_light.png` (2.2 MB, 3979×2658 px)
- 8种2D单元 × 拉伸/压缩 + 分析图表
- 脚本: `scripts/fem_showcase.py`

### 提交记录
```
72c9ef9 feat: Add FEM showcase images (dark + light themes)
b487e8f feat: FEM showcase visualization (dark + light themes)
e69fb56 feat(fem): add convenience API for ease-of-use
d0a946d fix(beam_fem_v6): validate node indices + add 3D reactions/edge_forces
31171a4 release: v4.1.0 — BeamFrameFEM + 3D structures + API fixes
```

---

## 历史记录

### v4.1.0 Released (2026-07-23)
- PyPI: https://pypi.org/project/fibernet/4.1.0/
- GitHub Release: https://github.com/GellmanSparrowS/fibernet/releases/tag/v4.1.0
- 312/312 tests passing
