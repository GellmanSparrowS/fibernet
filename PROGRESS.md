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

## 2024-07-15 - Push to GitHub v4.0.1

### 完成的更新
- **版本号**: 4.0.0 → 4.0.1
- **N_STRUCTURES**: 20 → 2000 (生产环境默认)
- **ML 增强**:
  - Early stopping (800棵树, patience=80)
  - 5-fold cross-validation
  - 模型保存为 joblib (`rf_force_model.joblib`)
- **RL 增强**:
  - 300 episodes (原100)
  - 收敛检测 (窗口=50)
- **向后兼容**: 添加 `get_boundary_nodes()` helper
- **教程文本**: 英文为主,中文注释为辅

### GitHub 推送状态
- ✓ Remote HEAD: `cc9f45db2c3e`
- ✓ 25个文件已推送 (代码、配置、notebook)
- ⚠ 22个大文件跳过 (>100KB, 包括模拟数据JSON和3D验证图片)

### 本地测试验证
- ✓ 语法检查通过
- ✓ Notebook生成成功 (41 cells)
- ✓ 模拟参数: dynamics, 30000 steps, 20% ramp, 80% relaxation
- ✓ 边界条件: 10% 每侧 (刚性板)

### 用户测试反馈
- Windows 上 `_get_boundary_indices(pct=...)` 报错
- 已通过 `get_boundary_nodes()` helper 解决
- 等待用户进一步测试反馈

### 下一步
- 用户测试 N=2000 的完整流程
- 根据测试结果调整参数
- 可能需要优化大文件上传 (git-lfs 或单独仓库)

---

## 2026-07-14 - CI 全部通过 ✅

### GitHub Actions 结果
- **Run ID**: 29339679411
- **Commit**: `205886b9e397`
- **结果**: 12/12 测试任务成功, 0 失败
  - Ubuntu (3.9, 3.10, 3.11, 3.12): ✓ 全部通过
  - macOS (3.9, 3.10, 3.11, 3.12): ✓ 全部通过
  - Windows (3.9, 3.10, 3.11, 3.12): ✓ 全部通过

### 修复内容
1. **测试兼容性**:
   - 添加 `pytest.importorskip("skimage")` 到 `tests/test_3d_units.py`
   - 跳过需要 scikit-image 的 TPMS 3D 测试（除非安装了 scikit-image）

2. **依赖更新**:
   - 添加 `scikit-image>=0.19` 到 dev 依赖
   - 添加 `taichi>=1.6` 到 dev 依赖
   - 这些依赖现在会在 CI 中自动安装

3. **仓库完整性**:
   - 推送完整的 `fibernet/` 包目录（165个文件）
   - 推送 `.github/workflows/ci.yml` 工作流文件
   - 推送 `scripts/`, `tutorials/`, `examples/`, `docs/` 等目录

### 本地测试验证
- ✓ 语法检查通过
- ✓ 本地 pytest: 118 passed, 1 skipped
- ✓ Notebook 生成成功 (41 cells)
- ✓ 模拟参数: dynamics, 30000 steps, 20% ramp, 80% relaxation
- ✓ 边界条件: 10% 每侧 (刚性板)

### 版本状态
- **PyPI**: 4.0.1 ✓
- **GitHub**: `205886b9e397` ✓ (CI passing)
- **本地**: `b4ff794` ✓
- **Notebook**: 已更新到 2000 structures + enhanced ML/RL ✓

### 用户使用方式
```bash
# 安装最新版本
pip install fibernet==4.0.1

# 运行教程 notebook
jupyter notebook tutorials/v4_tutorial/fibernet_v4_tutorial_updated.ipynb
```

### 下一步
- 等待用户在 Windows 上测试 N=2000 的完整流程
- 根据用户反馈进一步优化
- 可能需要在 PyPI 上发布 4.0.2 版本（如果需要）
