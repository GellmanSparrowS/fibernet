# FiberNet v4 — Progress Log

## 最后更新 (2026-07-23 10:15 UTC)

### 已完成
- ✅ v4.1.0: BeamFrameFEM_v6 模块 + 大变形测试套件
- ✅ v4.1.1: FEM 便捷API (stretch_test, graph_to_fem_input, to_sim_result)
- ✅ v4.1.2: 修复 radius 传播 bug + FEM showcase 可视化
- ✅ v4.1.3: 双语 README 大改版 + FEM 板块 + 文件整理
- ✅ v4.1.4: BeamFrameFEM 别名 + 模拟分节 + 图片排序

### v4.1.4 变更
- `BeamFrameFEM_v6` → `BeamFrameFEM`（公共API别名，`_v6` 仍可用向后兼容）
- Simulation 拆分为 **Mass-Spring** 和 **FEM** 两个子节，含对比表（物理/接头/半径/应力/速度/场景）
- Showcase 图片重排：生成 → 质点弹簧 → 轨迹 → FEM → ML → RL
- 删除 Performance 节
- 源码: `fibernet/ml/beam_frame_fem_v6.py` 添加 `BeamFrameFEM = BeamFrameFEM_v6`
- 源码: `fibernet/ml/__init__.py` 懒加载注册 `BeamFrameFEM`
- PyPI: https://pypi.org/project/fibernet/4.1.4/
- GitHub Release: https://github.com/GellmanSparrowS/fibernet/releases/tag/v4.1.4

### 版本历史
| 版本 | 日期 | 关键变更 |
|------|------|----------|
| 4.1.4 | 2026-07-23 | BeamFrameFEM别名 + 模拟文档重构 + 图片排序 |
| 4.1.3 | 2026-07-23 | 双语README大改版 + FEM板块 + 文件整理 |
| 4.1.2 | 2026-07-23 | 修复 radius 传播 bug |
| 4.1.1 | 2026-07-23 | FEM 便捷API + showcase |
| 4.1.0 | 2026-07-23 | BeamFrameFEM_v6 + 3D结构 + API修复 |

### 下一步
- 更新复旦开源资源页面 (swdfz.fudan.edu.cn)
