# FiberNet v4.1.0+ — Post-Release Progress

## 最新状态 (2026-07-23)

### ✅ BUG 修复 — 已完成

**BUG 5: 无效节点索引崩溃**
- 现象: `fixed_nodes=[0, 99]` 传入 3 节点图时 → IndexError
- 修复: 在 `solve_2d`, `solve_2d_nonlinear`, `solve_3d` 入口添加 `_validate_nodes()`
- 效果: 现在抛出清晰的 `ValueError` 消息

**BUG 8: 3D 求解器缺少 reactions**
- 现象: `solve_3d()` 返回 dict 中没有 `reactions` 字段
- 修复: 添加 `reactions = (K_damped @ u_full - f_global).reshape(n_nodes, 6)`
- 同时添加: `edge_forces` (n_edges, 3) 和 `K` 矩阵
- 平衡验证: Sum(reactions) = -Sum(applied_forces) ✓

**回归测试**: 312/312 测试全部通过

### 🔄 下一步
- API 易用性分析 (简单场景) + 可编程性分析 (复杂场景)
- FEM 展示图设计 + 生成 (dark+light, 用自带 viz API)
- git push + PROGRESS.md 更新

### 已提交
```
d0a946d fix(beam_fem_v6): validate node indices + add 3D reactions/edge_forces
31171a4 release: v4.1.0 — BeamFrameFEM + 3D structures + API fixes
```

---

## 历史记录

### v4.1.0 Released (2026-07-23)
- PyPI: https://pypi.org/project/fibernet/4.1.0/
- GitHub Release: https://github.com/GellmanSparrowS/fibernet/releases/tag/v4.1.0
- 312/312 tests passing
