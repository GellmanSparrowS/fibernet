# FiberNet v4 — Progress Log

## 最后更新 (2026-07-23 11:40 UTC)

### 当前版本: v4.1.7 ✓ RELEASED

### 已完成
- ✅ v4.1.0: BeamFrameFEM 模块 + 大变形测试套件
- ✅ v4.1.1: FEM 便捷API (stretch_test, graph_to_fem_input, to_sim_result)
- ✅ v4.1.2: 修复 radius 传播 bug + FEM showcase 可视化
- ✅ v4.1.3: 双语 README 大改版 + FEM 板块 + 文件整理
- ✅ v4.1.4: BeamFrameFEM 别名 + 模拟分节 + 图片排序
- ✅ v4.1.5: 重命名 BeamFrameFEM_v6 → BeamFrameFEM + 模块重命名
- ✅ v4.1.6: 完整API清理 - 消除所有不专业命名 + 删除shim
- ✅ **v4.1.7: **kwargs修复 + README低级API示例修正**

### v4.1.7 变更
- **API修复**: `solve_2d`/`solve_2d_nonlinear`/`solve_3d` 添加 `**kwargs`
  - `graph_to_fem_input()` 返回的 dict 现在可以直接 `**` 展开给求解器
  - 额外的 `boundaries`/`x_range` 键被安全忽略
  - 用法: `solver.solve_2d(**fem_input, fixed_nodes=left)`
- **README修正**: 低级 API 示例补充了正确的边界提取和位移设置
- **pytest**: 235 passed, 1 skipped, 0 failed
- **自定义API测试**: 27/27 通过

### 发布状态
- ✅ Git push: `3d9266b` → GitHub master
- ✅ GitHub Release: https://github.com/GellmanSparrowS/fibernet/releases/tag/v4.1.7
- ✅ PyPI: https://pypi.org/project/fibernet/4.1.7/
  - fibernet-4.1.7-py3-none-any.whl (632 KB)
  - fibernet-4.1.7.tar.gz (15.7 MB)
- ◐ GitHub Actions CI: in_progress (最新 commit)

### CI 状态
- 上次运行 (v4.1.6): 12/12 测试 job 通过 (ubuntu/windows/macos × Python 3.9-3.12)
- publish job 失败（与手动 PyPI 上传冲突，非代码问题）

### API 验证 (27项全部通过)
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
| **4.1.7** | **kwargs修复 + README低级API示例修正 |
| 4.1.6 | 完整API清理 + 删除shim + 脚本重命名 |
| 4.1.5 | BeamFrameFEM重命名 + 模块重命名 |
| 4.1.4 | BeamFrameFEM别名 + 模拟文档重构 |
| 4.1.3 | 双语README大改版 + FEM板块 |
| 4.1.2 | 修复 radius 传播 bug |
| 4.1.1 | FEM 便捷API + showcase |
| 4.1.0 | BeamFrameFEM + 3D结构 + API修复 |

### 使用示例

```python
from fibernet.ml import BeamFrameFEM
from fibernet.gen.pattern import pattern_2d

solver = BeamFrameFEM(E=1e9, nu=0.3)
g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(4, 4), radius=0.05)

# 一行API
result = solver.stretch_test(g, target_stretch=2.0)

# 低级API（完整控制）
fem_input = solver.graph_to_fem_input(g, dim=2, pct=0.1)
left = fem_input['boundaries']['left']    # 固定边界
right = fem_input['boundaries']['right']  # 加载边界

# 线性求解
result = solver.solve_2d(**fem_input, fixed_nodes=left)

# 非线性求解（施加强制位移）
target_disp = fem_input['x_range'] * (2.0 - 1.0)  # 2×拉伸
prescribed = {ni: (target_disp, 0.0) for ni in right}
result = solver.solve_2d_nonlinear(**fem_input, prescribed_disp=prescribed,
                                    fixed_nodes=left, n_steps=10)

# 3D分析
result = solver.solve_3d(**fem_input, fixed_nodes=left)
```

### 下一步
- 等待 CI 完成
- 更新复旦开源资源页面 (swdfz.fudan.edu.cn)
