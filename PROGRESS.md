# FiberNet v4 — Progress Log

## 最后更新 (2026-07-23)

### 当前版本: v4.1.6

### 已完成
- ✅ v4.1.0: BeamFrameFEM 模块 + 大变形测试套件
- ✅ v4.1.1: FEM 便捷API (stretch_test, graph_to_fem_input, to_sim_result)
- ✅ v4.1.2: 修复 radius 传播 bug + FEM showcase 可视化
- ✅ v4.1.3: 双语 README 大改版 + FEM 板块 + 文件整理
- ✅ v4.1.4: BeamFrameFEM 别名 + 模拟分节 + 图片排序
- ✅ v4.1.5: 重命名 BeamFrameFEM_v6 → BeamFrameFEM + 模块重命名
- ✅ v4.1.6: 完整API清理 - 消除所有不专业命名

### v4.1.6 变更 (完整API清理)
- 移除 `beam_frame_fem_v6.py` shim 文件（不再保留旧的 `_v6` 导入路径）
- 修复 `fibernet/sim/rl_env.py` 中的破损导入（`dataset_v2` → `dataset._extract_graph_features`）
- 重命名脚本文件：
  - `benchmarks/phase6_v6_validation.py` → `phase6_validation.py`
  - `scripts/run_tutorial_viz_v6.py` → `run_tutorial_viz.py`
- 更新所有内部引用，消除 `_v6` 后缀
- API验证：7/7 测试全部通过

### 审计结果
- ✓ 核心导入正常 (StructureGraph, Material)
- ✓ 生成模块正常 (pattern_2d, pattern_3d, list_units - 12种单元)
- ✓ FEM模块正常 (BeamFrameFEM 双路径导入)
- ✓ 旧的 `_v6` 导入正确失败
- ✓ FEM功能测试通过 (stretch_test)
- ✓ 特征提取正常 (22个特征)
- ✓ RL环境导入正常 (FiberNetworkEnv)

### 版本历史
| 版本 | 日期 | 关键变更 |
|------|------|----------|
| 4.1.6 | 2026-07-23 | 完整API清理 - 消除所有不专业命名 |
| 4.1.5 | 2026-07-23 | BeamFrameFEM重命名 + 模块重命名 |
| 4.1.4 | 2026-07-23 | BeamFrameFEM别名 + 模拟文档重构 |
| 4.1.3 | 2026-07-23 | 双语README大改版 + FEM板块 |
| 4.1.2 | 2026-07-23 | 修复 radius 传播 bug |
| 4.1.1 | 2026-07-23 | FEM 便捷API + showcase |
| 4.1.0 | 2026-07-23 | BeamFrameFEM + 3D结构 + API修复 |

### 下一步
- 更新复旦开源资源页面 (swdfz.fudan.edu.cn)
