# PROGRESS

## 当前状态: ✅ v4.0.1 已发布 PyPI + Notebook已更新

**最后更新:** 2026-07-14
**最后提交:** 356ae60
**PyPI:** https://pypi.org/project/fibernet/4.0.1/

---

## 2026-07-14 - v4.0.1 发布

### 版本变更
- `pyproject.toml`: 4.0.0 → 4.0.1
- `fibernet/__init__.py`: 4.1.0 → 4.0.1

### 核心改动
1. **模拟方法**: `stretch_test()` → `dynamics()` with rigid plate boundaries
   - 左10%节点完全固定 (rigid plate)
   - 右10%节点均匀位移 (rigid plate)
   - 80%弛豫时间, damping=0.3, 30000 steps
2. **Notebook**: 添加版本检查cell (无外部依赖), 英文为主中文为辅
3. **gen_notebook.py**: 修复6处重复行 + 全面重写模拟部分

### 模拟验证结果 (test_sim_v2.py)
- 力: 3.8 kN (合理范围)
- 拉伸比: 0.998~1.038 (边级别)
- 左边界位移: 0.000000 (完美固定)
- 右边界位移: 均匀 2.155 (铁板行为)
- 位移梯度: 左0.34 → 中1.00 → 右1.63 (合理线性梯度)

### RL验证结果
- 导入: OK (ParametricStructureEnv)
- 环境创建: n_actions=36, reward_mode=minimize_force
- 10 episodes全部成功

### Notebook结构 (41 cells: 19 MD, 22 code)
| # | 内容 |
|---|------|
| 0 | Title |
| 1-2 | 1. Setup |
| 3-4 | 2. Import + version check |
| 5-8 | 3.1 Base units + gallery |
| 9-10 | 3.2 Perturbation comparison |
| 11-14 | 3.3 Deformed gallery |
| 15-17 | 3.4 Batch + gallery |
| 18-19 | 4. Simulation (dynamics, rigid plates) |
| 20-22 | 4.2 Trajectory |
| 23-24 | 4.3 Stress distribution |
| 25-26 | 4.4 Batch stats |
| 27-28 | 5. Features |
| 29-30 | 5.1 Feature stats + KDE |
| 31-33 | 6. ML |
| 34-35 | 6.3 Correlation |
| 36-39 | 7. RL |
| 40 | 8. Summary |

---

## 用户安装方式
```
pip install fibernet --upgrade
# 或
pip install fibernet==4.0.1
```
