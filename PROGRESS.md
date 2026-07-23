# FiberNet v4 — Large Deformation FEM Test Results

## 完成时间 (2026-07-23)

### ✅ 所有测试完成

**测试配置:**
- 方法: BeamFrameFEM_v6 (梁单元有限元)
- 材料: E=1e9 Pa, ν=0.3
- 变形结构: n_pts_per_side=5, perturbation=±0.40
- 边界条件: 两侧各 10% 固定 (刚性板)
- 求解器: 线性 (|stretch-1|≤0.3), 非线性 (|stretch-1|>0.3)

**测试范围:**
- 2D: 8 单元 × 4 半径 × 4 拉伸目标 = 128 模拟
- 3D: 6 单元 × 2 半径 × 2 拉伸目标 = 24 模拟
- 总计: 152 模拟, 全部成功 ✅

### 关键发现

#### 1. 变形传导 (Deformation Propagation)
- **所有结构都是 FULL propagation** (100%)
- 变形完全传导到结构内部，没有局部化现象
- 平均传导比率: ~1.015 (内部最大位移 / 边界最大位移)
- 半径对传导的影响很小: r=0.02~0.20 时传导比率都在 1.011~1.018

#### 2. 弯曲 vs 轴向应力 (Bending vs Axial)
- **所有结构都是 BENDING-dominated** (弯曲主导)
- 弯曲主导度 (B/A = σ_bending/σ_axial):
  - 低 (B/A < 10): honeycomb (9.2), square (9.0), triangle (4.6)
  - 中 (B/A 10-100): reentrant (21.0), kagome (15.8), star (51.6), diamond (70.4), chiral (98.8)
  - 高 (B/A > 100): octet (163.0), cubic (218.0), fcc (469.5), reentrant_3d (476.9), diamond_3d (501.5), bcc (376.1)

#### 3. 力-拉伸关系 (Force-Stretch)
- 力随半径增加: r=0.02 → r=0.20 时力增加 ~100x
- 不同结构的平均力:
  - 低力: honeycomb (12.4 kN), reentrant (24.8 kN)
  - 中力: square (74.0 kN), star (72.3 kN), triangle (50.4 kN)
  - 高力: kagome (144.5 kN), reentrant_3d (96.0 kN), fcc (35.3 kN)

#### 4. 单元类型特性
- **triangle**: 传导最好 (prop=1.031), 弯曲最少 (B/A=4.6)
- **cubic**: 传导好 (prop=1.040), 中等弯曲 (B/A=218.0)
- **square**: 传导差 (prop=1.001), 弯曲少 (B/A=9.0)
- **diamond**: 传导中等 (prop=1.024), 弯曲中等 (B/A=70.4)
- **3D 结构**: 弯曲主导度远高于 2D (B/A > 100)

### 输出文件

```
output_data/deformation_test/
├── analysis_report_fem.json          # 详细分析报告 (6 KB)
└── viz/
    └── large_deformation_fem_summary.png  # 综合可视化 (7.8 MB)
```

### 脚本位置

```
scripts/deformation_test/
└── run_large_deformation_fem.py      # 主测试脚本 (1100+ 行)
```

### 运行方法

```bash
cd fibernet
source .venv/bin/activate
python scripts/deformation_test/run_large_deformation_fem.py
```

脚本支持断点续跑：如果中断，重新运行会自动跳过已完成的模拟。

### 技术细节

#### 力计算
- 2D: 使用 edge_forces (轴向力 + 剪切力) 在边界节点求和
- 3D: 使用 σ_axial × A × cos(θ) 在边界边求和
- 非线性求解器: 增量加载 (n_steps=10), 几何非线性

#### 传导分析
- 传导比率 = interior_max_disp / boundary_max_disp
- 内部节点 = 不在边界 10% 范围内的节点
- FULL: ratio > 0.5, PARTIAL: 0.2-0.5, LOCALIZED: < 0.2

#### 弯曲主导度
- B/A = max(σ_bending) / max(σ_axial)
- B/A > 2.0: 弯曲主导
- B/A < 0.5: 轴向主导
- 0.5-2.0: 混合

### 结论

BeamFrameFEM_v6 在大变形测试中表现良好：
1. **变形传导**: 所有结构都完全传导，没有局部化
2. **弯曲行为**: 所有结构都是弯曲主导，符合梁单元理论
3. **数值稳定性**: 152 个模拟全部成功，无失败
4. **几何非线性**: 大变形 (2x stretch, 0.5x compress) 需要非线性求解器

这些结果为 FiberNet 的力学性能分析提供了可靠的基准数据。
