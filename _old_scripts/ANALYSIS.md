# FiberNet 代码库分析与改进计划

## 1. sf_share 文件分析

### 1.1 核心工作流程（来自 zigzag_RL.ipynb）

用户的研究工作流是一个**完整的闭环系统**：

```
1. 结构生成 (ZigZagLattice)
   └─> base_pts → mirror_x/mirror_y → tile (n_cols × n_rows) → 完整结构
   
2. 特征提取 (Features.py - 94维)
   ├─ 34 结构/拓扑特征 (度分布、谱性质、聚类系数等)
   ├─ 18 孔隙特征 (面积分布、形状指标)
   └─ 42 接触/重叠特征 (边像素重叠统计)

3. 数据集生成
   └─> 批量生成结构 → 提取94维特征 → 计算力学性质 → 导出CSV

4. ML代理模型训练
   ├─ RandomForest / GradientBoosting / MLP / XGBoost
   ├─ 交叉验证 + 模型选择
   └─> 保存最佳模型 (joblib)

5. RL优化 (gymnasium + stable-baselines3)
   ├─ 动作空间: 扰动 xs (zigzag控制点)
   ├─ 奖励函数: lambda_mean (ML预测) - angle_penalty - OOD惩罚
   ├─ 算法: PPO / SAC / A2C
   └─> 多次运行实验 + 快照保存
```

### 1.2 关键发现

**数据模型不匹配**: 
- 用户使用 **weld graph** (JSON格式: `{nodes: [{id, pos}], links: [{source, target}]}`)
- 我们使用 Fiber/Centerline/Crosslink 对象
- **需要桥接层**

**特征提取缺失**:
- 用户的 `Features.py` 提取94维特征，是我们的 `analyze()` 的 20倍以上
- 我们的 `analyze()` 只返回5个基础统计量
- **必须集成完整的特征提取器**

**RL环境不完整**:
- 用户使用 `gymnasium` + `stable-baselines3` (专业级)
- 我们的 toy RL agent 无法与之兼容
- **需要提供 gymnasium-compatible 环境**

**缺少关键结构生成器**:
- ZigZag lattice (用户最常用的)
- Pattern-based generators (base_pts + mirror + tile)
- **必须添加这些生成器**

**可视化质量差**:
- 用户使用 canvas-based rendering (OpenCV) 用于特征提取
- 我们的 matplotlib 可视化缺少孔隙分析、重叠可视化
- **需要升级到专业级可视化**

## 2. 当前库的问题

### 2.1 API 扩展性问题

**问题 1: create_metamaterial() 太耦合**
```python
# 当前：一步完成所有事情
meta = fn.create_metamaterial(unit_cell, array_size, weld_threshold, **params)

# 应该是：分步骤，可组合
cell = fn.create("zigzag", base_pts=[...])
array = fn.tile(cell, repeats=(4, 10), mirror_x=True, mirror_y=True)
graph = fn.weld(array)  # 检测交叉点并焊接
features = fn.extract_features(graph)  # 94维特征
```

**问题 2: 数据结构不兼容**
```python
# 用户的 JSON 格式
{
  "nodes": [{"id": 0, "pos": [0.0, 0.0]}, ...],
  "links": [{"source": 0, "target": 1}, ...]
}

# 我们的 FiberNetwork 格式
FiberNetwork(fibers=[Fiber(centerline=...)], crosslinks=[Crosslink(...)])
```

**问题 3: 特征提取太简陋**
```python
# 当前
stats = fn.analyze(meta)
# 返回: {num_fibers, num_crosslinks, nematic_order, ...}  # 只有5个

# 用户需要
features = fn.extract_features(graph)
# 返回: 94维特征向量 (结构34 + 孔隙18 + 接触42)
```

### 2.2 测试质量问题

**当前测试只检查"是否运行"，不检查"结果是否正确"**:

```python
# 坏的测试
def test_dynamics():
    traj = fn.simulate_dynamics(meta, dt=1e-7, steps=1000)
    assert 'positions' in traj  # 只检查key存在
    
# 好的测试应该检查：
def test_dynamics_physics():
    # 1. 能量守恒（无阻尼时）
    traj_no_damp = fn.simulate_dynamics(meta, dt=1e-9, steps=1000, damping=0.0)
    E_init = compute_energy(traj_no_damp['initial_positions'])
    E_final = compute_energy(traj_no_damp['positions'])
    assert abs(E_final - E_init) / E_init < 0.01  # 能量守恒
    
    # 2. 外力导致位移
    traj_loaded = fn.simulate_dynamics(meta, dt=1e-9, steps=5000, 
                                        external_force=ext_force)
    max_disp = np.max(np.linalg.norm(traj_loaded['positions'] - 
                                      traj_loaded['initial_positions'], axis=1))
    assert max_disp > 1e-6  # 必须有位移
    
    # 3. 固定节点不动
    for node in fixed_nodes:
        assert np.allclose(traj_loaded['positions'][node], 
                          traj_loaded['initial_positions'][node])
```

## 3. 改进计划

### Phase 1: 数据模型桥接 (优先级: 高)

**目标**: 支持用户的 JSON graph 格式

```python
# 新增 API
fn.load_graph_json(path)  # 加载 {nodes, links} JSON
fn.save_graph_json(network, path)  # 导出为 JSON
fn.from_networkx(G)  # 从 NetworkX Graph 转换
fn.to_networkx(network)  # 转换为 NetworkX Graph
```

### Phase 2: 结构生成器扩展 (优先级: 高)

**目标**: 添加 ZigZag 和 pattern-based 生成器

```python
# ZigZag lattice
fn.create("zigzag", base_pts=[(0, 31.7), (75, 75), ...], 
          n_cols=4, n_rows=10, mirror_x=True, mirror_y=True)

# Pattern-based (通用)
fn.create("pattern", base_pts=[...], 
          tile_repeats=(4, 10), mirror_x=True, mirror_y=True)
```

### Phase 3: 特征提取集成 (优先级: 高)

**目标**: 集成 94 维特征提取器

```python
# 新增 API
features = fn.extract_features(network, canvas_size=1024, thick=9)
# 返回: dict with 94 keys
# - structural_34: n_node, n_edge, clustering_coef, ...
# - pore_18: largest_pore_ratio, pore_area_cv, ...
# - contact_42: contact_overlap_pixel_count, ...
```

### Phase 4: 分步 API 重构 (优先级: 中)

**目标**: 提供分步骤、可组合的 API

```python
# Step 1: 生成单元胞
cell = fn.create("zigzag", base_pts=[...])

# Step 2: 平铺成阵列
array = fn.tile(cell, repeats=(4, 10), mirror_x=True, mirror_y=True)

# Step 3: 焊接交叉点
graph = fn.weld(array, threshold=0.5)

# Step 4: 提取特征
features = fn.extract_features(graph)

# Step 5: 模拟
result = fn.simulate_mechanics(graph, strain=0.001)

# Step 6: 可视化
fn.plot_structure(graph, show_pores=True, show_overlaps=True)
```

### Phase 5: RL 环境 (优先级: 中)

**目标**: 提供 gymnasium-compatible 环境

```python
# 新增 API
env = fn.create_rl_env(
    structure_generator="zigzag",
    ml_model=trained_model,
    reward_function=lambda features: features['lambda_mean'],
    action_space="continuous",  # or "discrete"
)

# 与 stable-baselines3 兼容
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=10000)
```

### Phase 6: 测试质量提升 (优先级: 高)

**目标**: 物理正确性测试

```python
# 1. 能量守恒测试
# 2. 外力-位移响应测试
# 3. 固定节点约束测试
# 4. 特征提取正确性测试
# 5. 数据格式转换测试
# 6. 性能基准测试
```

## 4. 立即行动项

### 4.1 不发布 v1.25.0

当前版本问题太多，需要先完成上述改进。

### 4.2 先完成的 3 件事

1. **集成 Features.py** → `fibernet/analysis/features.py`
2. **添加 ZigZag 生成器** → `fibernet/gen/zigzag.py`
3. **改进测试** → 物理正确性测试

### 4.3 后续完成的 3 件事

1. **分步 API 重构** → 可组合的 API
2. **JSON graph 支持** → 数据模型桥接
3. **RL 环境** → gymnasium 兼容

## 5. 测试改进示例

### 5.1 物理正确性测试

```python
def test_mass_spring_energy_conservation():
    """无阻尼时能量应该守恒"""
    meta = fn.create_metamaterial(...)
    traj = fn.simulate_dynamics(meta, dt=1e-9, steps=1000, damping=0.0)
    
    # 计算初始和最终能量
    E_init = compute_total_energy(traj['initial_positions'], traj['edges'], 
                                   traj['rest_lengths'], traj['stiffness'])
    E_final = compute_total_energy(traj['positions'], traj['edges'], 
                                    traj['rest_lengths'], traj['stiffness'])
    
    # 能量守恒误差 < 1%
    assert abs(E_final - E_init) / E_init < 0.01

def test_external_force_causes_displacement():
    """外力应该导致位移"""
    meta = fn.create_metamaterial(...)
    
    # 固定左边界，右边界施加力
    ext_force = np.zeros((n_nodes, 3))
    ext_force[right_nodes, 0] = 1e-3  # x方向1mN
    
    traj = fn.simulate_dynamics(meta, dt=1e-9, steps=5000, 
                                 external_force=ext_force, fixed_nodes=left_nodes)
    
    # 必须有位移
    max_disp = np.max(np.linalg.norm(traj['positions'] - traj['initial_positions'], axis=1))
    assert max_disp > 1e-6, f"Expected displacement > 1e-6, got {max_disp}"

def test_fixed_nodes_dont_move():
    """固定节点不应该移动"""
    meta = fn.create_metamaterial(...)
    traj = fn.simulate_dynamics(meta, dt=1e-9, steps=5000, 
                                 external_force=ext_force, fixed_nodes=fixed_nodes)
    
    for node in fixed_nodes:
        assert np.allclose(traj['positions'][node], traj['initial_positions'][node])
```

### 5.2 特征提取测试

```python
def test_feature_extraction_correctness():
    """特征提取应该返回正确的94维向量"""
    meta = fn.create_metamaterial(...)
    features = fn.extract_features(meta, canvas_size=1024, thick=9)
    
    # 检查所有94个特征都存在
    assert len(features) == 94
    
    # 检查物理约束
    assert features['n_node'] > 0
    assert features['n_edge'] > 0
    assert 0 <= features['clustering_coef'] <= 1
    assert features['total_length'] > 0
    
    # 检查孔隙特征
    assert 0 <= features['largest_pore_ratio'] <= 1
    assert features['total_pore_count'] >= 0
```

## 6. 总结

当前库的主要问题：

1. **API 耦合度过高** - create_metamaterial() 一步完成所有事情
2. **数据模型不兼容** - 不支持用户的 JSON graph 格式
3. **特征提取缺失** - 只有5个基础统计，需要94维特征
4. **缺少关键生成器** - ZigZag、pattern-based 生成器
5. **测试质量差** - 只检查"是否运行"，不检查"是否正确"
6. **RL 环境不完整** - 无法与 stable-baselines3 兼容

**建议**: 先不发布，完成上述改进后再发布 v2.0.0。
