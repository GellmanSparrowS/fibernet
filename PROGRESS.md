# PROGRESS

## 当前状态: ✅ 教程可视化 v10 完成（变形幅度已调整）

**最后更新:** 2026-07-14  
**脚本版本:** run_tutorial_viz_v10.py  
**输出目录:** tutorials/v4_tutorial/tutorial_viz/

---

## 最近变更

### 2026-07-14 - 调整02.5图变形幅度
- **问题**: 02.5 voronoi diverse 变形太剧烈
- **修复**: 位移幅度从 ±0.4 减小到 ±0.15
- **影响**: 仅重新生成了 02_5_voronoi_diverse_dark/light.png
- **其他**: 01-10 图保持不变

### 2026-07-14 - 修复 voronoi 变形参数（v10）
- **问题**: 02.5 图只有前3条边变形，其他67条边都是直线
- **修复**: 
  - `n_disp` 从 15 改为 350（70条边 × 5个点/边）
  - 刚度从 1e4 提升到 1e5
  - 阻尼从 0.5 降到 0.3
- **结果**: 波传播明显改善，力传导正常

---

## 可视化清单（22个文件）

| 编号 | 文件名 | 内容 | 状态 |
|------|--------|------|------|
| 01 | 01_gallery_undeformed_{dark,light}.png | 12种基本单元类型 | ✅ |
| 02 | 02_gallery_deformed_{dark,light}.png | 12种单元带中间点变形 | ✅ |
| 02.5 | 02_5_voronoi_diverse_{dark,light}.png | 12种不同变形的voronoi | ✅ 已调整 |
| 03 | 03_feature_stats_{dark,light}.png | 特征统计分布 | ✅ |
| 04 | 04_simulation_stretch_{dark,light}.png | 8帧拉伸轨迹 | ✅ |
| 05 | 05_stress_distribution_{dark,light}.png | 应力分布 | ✅ |
| 06 | 06_ml_analysis_{dark,light}.png | ML分析（预测/重要性/混淆矩阵） | ✅ |
| 07 | 07_batch_stats_{dark,light}.png | 批量统计（力/能量/拉伸） | ✅ |
| 08 | 08_feature_force_correlation_{dark,light}.png | 特征-力相关性 | ✅ |
| 09 | 09_rl_reward_curves_{dark,light}.png | RL奖励曲线 | ✅ |
| 10 | 10_rl_structure_changes_{dark,light}.png | RL结构变化 | ✅ |

---

## 技术细节

### Voronoi 变形参数
- **边数**: 70条（由 seed=12345 生成）
- **中间点**: 每条边5个点 (n_pts_per_side=5)
- **总位移**: 350个 (70×5)
- **位移幅度**: ±0.15（已从±0.4减小）

### 模拟参数
- **刚度**: 1e5（高刚度确保波传播）
- **阻尼**: 0.3（低阻尼减少能量耗散）
- **时间步**: 1e-6
- **帧数**: 15000帧

### 波传播验证
结构0的位移梯度（从固定端到拉伸端）：
- x ∈ [0.0, 0.2]: 平均位移 0.01（固定端附近）
- x ∈ [0.4, 0.6]: 平均位移 0.33（中间区域）
- x ∈ [1.8, 2.0]: 平均位移 0.93（拉伸端）

---

## 下一步

✅ 教程可视化已完成，可以：
1. 查看 `tutorials/v4_tutorial/tutorial_viz/` 中的22张图片
2. 如需调整任何可视化，修改 `run_tutorial_viz_v10.py` 后重新运行
3. 将可视化集成到 Jupyter notebook 或文档中
