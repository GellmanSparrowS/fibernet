# FiberNet v4 — Progress Log

## 最后更新 (2026-07-23)

### 当前版本: v4.1.7

### 已完成
- ✅ v4.1.0: BeamFrameFEM 模块 + 大变形测试套件
- ✅ v4.1.1: FEM 便捷API (stretch_test, graph_to_fem_input, to_sim_result)
- ✅ v4.1.2: 修复 radius 传播 bug + FEM showcase 可视化
- ✅ v4.1.3: 双语 README 大改版 + FEM 板块 + 文件整理
- ✅ v4.1.4: BeamFrameFEM 别名 + 模拟分节 + 图片排序
- ✅ v4.1.5: 重命名 BeamFrameFEM_v6 → BeamFrameFEM + 模块重命名
- ✅ v4.1.6: 完整API清理 - 消除所有不专业命名 + 删除shim
- ✅ v4.1.7: **kwargs修复 + README低级API示例修正

### v4.1.7 变更
- **API修复**: `solve_2d`/`solve_2d_nonlinear`/`solve_3d` 添加 `**kwargs`
  - `graph_to_fem_input()` 返回的 dict 现在可以直接 `**` 展开给求解器
  - 额外的 `boundaries`/`x_range` 键被安全忽略
  - 用法: `solver.solve_2d(**fem_input, fixed_nodes=left)`
- **README修正**: 低级 API 示例补充了正确的边界提取和位移设置
- **pytest**: 235 passed, 1 skipped, 0 failed

### CI 状态
- 12/12 测试 job 通过 (ubuntu/windows/macos × Python 3.9-3.12)
- publish job 失败（与手动 PyPI 上传冲突，非代码问题）

### API 验证 (27项)
- ✓ 核心导入 (StructureGraph, Material)
- ✓ 结构生成 (12种2D + 14种3D单元)
- ✓ 参数化控制 (n_pts_per_side, point_displacements)
- ✓ 节点操作 (displace_node, get_internal_nodes)
- ✓ FEM双路径导入 (fibernet.ml + fibernet.ml.beam_frame_fem)
- ✓ 旧 _v6 导入正确失败
- ✓ stretch_test (2×拉伸 + 0.5×压缩)
- ✓ **kwargs 修复 (solve_2d/solve_2d_nonlinear/solve_3d)
- ✓ 3D分析 (solve_3d, 6 DOF/node)
- ✓ radius物理效应 (EI∝r⁴)
- ✓ to_sim_result 转换
- ✓ 特征提取 (22个特征)
- ✓ 可视化 (3种主题)
- ✓ 一行API (fn.pattern_2d, fn.simulate)

### 版本历史
| 版本 | 关键变更 |
|------|----------|
| 4.1.7 | **kwargs修复 + README低级API示例修正 |
| 4.1.6 | 完整API清理 + 删除shim + 脚本重命名 |
| 4.1.5 | BeamFrameFEM重命名 + 模块重命名 |
| 4.1.4 | BeamFrameFEM别名 + 模拟文档重构 |
| 4.1.3 | 双语README大改版 + FEM板块 |
| 4.1.2 | 修复 radius 传播 bug |
| 4.1.1 | FEM 便捷API + showcase |
| 4.1.0 | BeamFrameFEM + 3D结构 + API修复 |
