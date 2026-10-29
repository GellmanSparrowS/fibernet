# 简化拉伸求解器

输入为二维纤维图的有限节点坐标与非零长度边。每条边按初始长度定义一个轴向弹簧，在当前边长 L 和参考长度 L_{0} 下使用 T = k(L − L_{0})/L_{0}，沿边方向分配等大反向的节点力。度数为二的纤维链节点，以及明确标注的焊接三元组，可另外在两端节点之间形成跨节点弦弹簧；其刚度为轴向刚度乘可调比例，属于弯曲响应的离散近似，而非具有独立转角自由度的 Euler–Bernoulli 梁单元。当前弹簧参数为数值模型参数，未自动转换为材料杨氏模量、截面惯性矩或试验单位。

所有节点必须位于同一个 XY 平面。具有相同 Z 坐标的平移平面可以直接计算，输出 `frames_xy` 保留 XY 坐标；Z 不相等的三维结构会被明确拒绝。边上的非空 `internal_points` 不会自动转为力学自由度，因此也会被拒绝。计算曲线纤维前，应把采样点转为实际图节点，并以连续直边连接，使曲线形状进入参考长度、弯曲近似和接触计算。Python 求解器与 APP 缓存入口执行相同的几何检查。

可选接触通过二维均匀网格搜索半径内的非邻接节点对，排除已有边、相邻弯曲链及初始接近的制造接点；新接触节点对在间距小于接触半径时施加与重叠量成正比的排斥力。候选对按固定步数重建并以节点编号规范排序；`contact_rebuild=1` 表示每次力计算都重建候选对。初始接近对不作为新接触事件。接触是节点对近似，不包含线段–线段连续碰撞、摩擦或真实粘结失效。

左右夹持区按初始样本 x 宽度的固定比例选取整列节点，左侧固定，右侧沿 x 指定终位移。`n_increments>0` 时，总位移分成指定增量，每级先按当前横向范围施加仿射预测，再进行显式速度更新和指数阻尼松弛；默认的 `n_increments=0` 使用连续斜坡加载。输出包括坐标、边轴向应变、原始右夹持反力、轴向/弯曲近似/接触势能、动能和外部做功。原始反力可非单调，不能用展示层的单调包络代替做功或干预评价。左右夹持及所有边在输出中保留相同编号，以供时序招募和独立复算使用。

## 反力与外功

每个保存帧的原始夹持反力均在该帧坐标及位移约束施加后计算，定义为 `F = -sum(f[right_nodes, 0])`。`force_curve`、`frames_xy` 和 `edge_strain` 对应同一个构形，拉伸反力以正值表示。

外功采用相邻加载端点反力的梯形积分：

\[
\Delta W_n = \frac{F_{n-1}+F_n}{2}\,(d_n-d_{n-1}).
\]

这里 `d` 是右夹持区的施加位移。连续斜坡模式在每个积分步累计外功，保存间隔只影响输出采样；增量松弛模式在相邻加载增量之间累计。两端完全夹持的单个线性弹簧满足 `F=k(L-L0)/L0` 和 `W=k(L-L0)^2/(2L0)`。一般网络中的显式积分、阻尼和仿射预测会影响功与机械能的差额；外功曲线本身不证明准静态平衡或能量收敛。

结果元数据用 `force_sampling="recorded_configuration"` 和 `work_quadrature="trapezoidal_grip_reaction"` 标识上述定义。APP 缓存采用 `e2v13`；较早版本的反力、接触观测与外功不能作为这些定义下的新结果直接复用。共享求解器的源码签名同时参与物理标签数据集的续算检查。需要组合不同版本的力学标签或比较逆设计目标时，应使用同一求解器版本重算。

该 NumPy 求解器的单一实现同时由 Python 库与 APP 调用。跨端一致性说明两者调用相同数值模型，实验预测仍需要材料标定和独立力学验证。研究报告应列出求解器、模型参数、加载方式和分析阈值。

## 研究参考与检查点重算

删环研究的参考数据同时绑定完整研究配置、求解器版本、求解器源码哈希及分析源码哈希。AI 候选计算开始前会校验这些字段，以及参考案例键是否完整匹配当前拓扑、种子和加载方向。每个案例还须已完成并具有合格候选；没有足够同长度类别候选的配置需要先调整研究配置，不能直接进入 AI 选择。旧版本的研究结果应保留；求解器定义变化后，需要以新输出路径重新计算参考和候选标签。只修改旧 JSON 的版本或哈希字段不能产生新模型的数值结果。

使用默认的 24 个加载配置时，可分别保存 `e2v13` 的新参考与 AI 结果：

```bash
python benchmarks/constrained_cycle_intervention.py \
  --output studies/e2v13/constrained_cycle_intervention.json
python benchmarks/ai_cycle_selector.py \
  --reference studies/e2v13/constrained_cycle_intervention.json \
  --output studies/e2v13/ai_cycle_selector.json --max-new 1
```

第一条命令须完成全部参考案例；第二条先保存一个真实候选重算结果，随后使用相同路径续跑，去掉 `--max-new 1` 可完成全部候选。Python 调用可通过 `AICycleSelector(reference_path=..., study_config=...)` 显式绑定另一份配置一致的参考；该研究配置也进入 AI 检查点签名。单拓扑配置可用于有界候选生成和续跑检查；完整的逐拓扑留出评估至少需要两个不同拓扑，否则会在拟合模型前明确拒绝。

`scripts/refresh_cycle_study_checkpoints.py` 用于复算后案例数值完全相同的签名更新；它会拒绝反力或积分量已经改变的结果。新版本的研究应另存文件，并由新参考重新计算相关 AI 标签、独立模型对照和图表，避免把不同求解器版本的数值拼接到同一研究中。

## Graph IDs and replay arrays

Graph node IDs need not be contiguous after editing. The solver maps sorted
original node IDs to dense array rows and preserves graph edge insertion order.
`edges`, grip indices, contact pairs and weld triplets use these dense rows
internally; result metadata includes `node_ids` and `edge_ids` to map replay
arrays back to the input graph. Input `weld_triplets` refer to original node IDs.
Disabling contact skips candidate construction as well as contact forces; it
does not change the axial or bending model.
