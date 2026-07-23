# FiberNet v4 — Progress Log

## 最后更新 (2026-07-23 10:05 UTC)

### 已完成
- ✅ v4.1.0: BeamFrameFEM_v6 模块 + 大变形测试套件
- ✅ v4.1.1: FEM 便捷API (stretch_test, graph_to_fem_input, to_sim_result)
- ✅ v4.1.2: 修复 radius 传播 bug + FEM showcase 可视化
- ✅ v4.1.3: 双语 README 大改版 + FEM 板块 + 文件整理
  - README.md (英文): 新增 FEM 专区（物理模型、验证表、API示例、求解器表）
  - README_CN.md (中文): 镜像英文结构，专业学术语调
  - Showcase 图片: FEM dark/light + deformation summary → docs/images/
  - 根目录整理: CONTRIBUTING/SECURITY/CODE_OF_CONDUCT → .github/
  - ATTRIBUTIONS.md → docs/, sync_notebook.sh + verify_readme.py → scripts/
  - MANIFEST.in 更新, 版本升至 4.1.3
  - 通过 GitHub API 推送成功 (commit 8dda533)

### 验证结果
- README.md: 15,253 bytes, 7项检查全部通过 ✓
- README_CN.md: 11,665 bytes, 6项检查全部通过 ✓
- GitHub root: 干净（社区文件移走，仅保留核心配置+README）

### 下一步
- 更新复旦开源资源页面 (swdfz.fudan.edu.cn/kyzy/list.htm) 至 v4.1.3
- 发布 PyPI v4.1.3
- 创建 GitHub Release v4.1.3

### 版本历史
| 版本 | 日期 | 关键变更 |
|------|------|----------|
| 4.1.3 | 2026-07-23 | 双语README大改版 + FEM板块 + 文件整理 |
| 4.1.2 | 2026-07-23 | 修复 radius 传播 bug |
| 4.1.1 | 2026-07-23 | FEM 便捷API + showcase |
| 4.1.0 | 2026-07-23 | BeamFrameFEM_v6 + 3D结构 + API修复 |
| 4.0.5 | 2026-07-17 | 性能优化 |
