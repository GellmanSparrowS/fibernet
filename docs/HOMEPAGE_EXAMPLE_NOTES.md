# Example parameters and model scope

Detailed parameters and interpretation for the animations in the [homepage](../README.md). Source files, editable figures and runnable examples are linked below.

<details>
<summary>中文</summary>

## 示例参数

### 1. 生成平面与三维纤维网络

这组已保存的 octet、diamond 和 gyroid 示例分别包含 **35/118、63/144、203/531 个节点/条边**。旋转视角呈现各固定拓扑的空间连接。[三维参数](media/three_dimensional_topologies.json) · [可编辑三维帧](media/three_dimensional_topologies_final.svg)

#### 同一种几何语言，连接不同拓扑

将同一无量纲参考纤维位移轮廓用于**正方形、六边形、环形和 kagome** 网络。几何形状随参数变化，各网络的连接关系保持不变。这组几何参数化示例在保持拓扑的同时探索形状变化。[制造几何方法](METHODS_MANUFACTURING_UNIFIED.md) · [可编辑矢量关键帧](media/spectrum_four_topologies_peak.svg) · [参数记录](media/spectrum_four_topologies.json)

### 2. 模拟网络变形，比较加载方向

这个 **3 × 3 扰动 kagome 网络包含 436 个节点和 576 条边**。蓝色表示超过正轴向应变阈值的边，橙色表示连接两侧夹持端的阈值连通分量；曲线记录已保存简化模型轨迹中的招募边比例。应变阈值统计将局部伸长与网络尺度的跨端路径联系起来。[轨迹示例](../examples/reduced_recruitment_workflow.py) · [招募分析方法](METHODS_TENSILE_RECRUITMENT.md) · [来源记录](media/tensile_recruitment_kagome.json)

在同一六边形拓扑上比较横向和纵向加载。已保存的简化模型示例分别在拉伸比 **1.08 和 1.24** 时达到所设定的跨端招募阈值。[方向比较参数](media/directional_recruitment_hexagon.json) · [可编辑比较图](media/directional_recruitment_hexagon_final.svg)

### 3. 分析纤维招募与阈值敏感性

用 **α = 0.02、0.05、0.15** 三种阈值观察同一段变形。在保持原力学轨迹不变的条件下，直接比较连通性结论对分析阈值的依赖。[分析参数](media/tensile_threshold_sensitivity.json) · [可编辑分析帧](media/tensile_threshold_sensitivity_final.svg)

### 4. 学习结构响应，探索目标驱动设计

使用[图实验输入约定](GRAPH_TRAINING_CONTRACT.md)，绑定原始图数据、按母图隔离的训练/验证/最终评估分组，以及仅由训练图计算的度统计。[完整示例](../examples/graph_training_contract.py)沿用 StructureGraph/GraphBatch，不需要学习后端。

从数据准备和响应学习进入目标曲线设计工作区。这段已保存的 FiberScope 界面导览呈现从数据到设计的使用流程。Python 库共享物理标签流、14 维特征、代理模型适配器与 CEM 目标搜索。[学习示例](../examples/physical_learning_workflow.py) · [逆向设计示例](../examples/curve_inverse_workflow.py) · [导览来源](media/module_workflow_tours.json)

#### 在连续路径约束下比较结构干预

比较完整结构、低早期应变删边、高后期应变删边与固定随机删边。已保存的 kagome 和环形研究匹配移除边长，并保留欧拉路径，展示设计选择如何改变响应。受控比较使不同结构干预对响应的影响清晰可见。[Kagome 来源](media/route_preserving_intervention_kagome.json) · [环形来源](media/route_preserving_intervention_ring.json) · [比较方法](CONSTRAINED_CYCLE_INTERVENTION_2026-09-24.md)

### 5. 构建连续制造路径

闭合欧拉路径将每条独立纤维边遍历一次，也保留平行纤维的独立身份。蓝色是已访问的边，橙色游标标明当前遍历顺序。路径回放清晰呈现结构图的遍历顺序。[生成新的平面与曲面路径](../examples/manufacturing_workflow.py) · [自定义基元](../examples/custom_cell_workflow.py) · [生成新的实体与路径](../examples/solid_export_workflow.py)

沿用当前 `StructureGraph` 导出：全偶度连通图使用 `manufacturing_from_graph` → `export_mapped_route`；含奇度节点时使用 `manufacturing_trails_from_graph` → `export_mapped_trails`。两条路径均接收连通、非空的直边图，显式声明 `m`/`mm` 和坐标系，输出 mm 坐标 CSV 与稳定 ID 映射 JSON，不修改源图拓扑。详见[闭合路径适用范围](MANUFACTURING_GRAPH_MAPPING.md#geometry-and-ownership)与[多段路径适用范围](MULTISTROKE_MANUFACTURING.md#scope-and-input-validation)。

### 6. 映射曲面纤维，准备几何与路径导出

先检查曲面上的纤维网络，再进入制造工作区准备路径与几何导出。这段已保存界面导览将曲面设计连接到可检查的路径文件与实体几何文件。[曲面与路径示例](../examples/manufacturing_workflow.py) · [实体导出示例](../examples/solid_export_workflow.py) · [导览来源](media/module_workflow_tours.json)

[完整媒体画廊、参数与可编辑关键帧](MEDIA_GALLERY.md)

### 7. 检查三维网络中的电传输

在包含 **80 个节点、162 条边**的已保存环面网络上，观察电压与电流的空间分布。相机环绕固定的合成参数稳态电学解，让场分布与图连接关系同时可见。[电学场与渲染来源](media/electrical_torus_field.json) · [电传输接口](integrations/LINEAR_TRANSPORT_USAGE.md)

## 模拟与学习过程图集

沿真实保存的状态，查看网络运动、构件失效、热传输与数据驱动研究。每个示例均链接原始数据、可编辑图件和重绘方法。

### 8. 比较 LAMMPS 与 Taichi 网络动力学

LAMMPS 展示 64 个节点、441 条键在 41.667 µs 内的运动；Taichi 展示 4,340 个节点、5,073 条边的无量纲轴向网络记录。固定视角下，**位移放大 1,000 倍**，位移读数和能量保留真实量值。这些是各后端的合成基准场景。[数据、单位与重绘方法](media/new/backends/README.md)。

### 9. 解析黏弹网络运动与能量转移

105 个节点、273 条边在 0.5 s 内振动并松弛。实际轴向应力随运动呈现；动能、储能、材料耗散与后向欧拉数值损失分别显示。**位移放大 3,000 倍**，使小应变响应清晰可见。[材料模型、保存轨迹与重绘方法](media/new/viscoelastic/README.md)。

### 10. 跟踪构件移除形成的断裂序列

含缺口的 63 节点、158 梁框架进一步移除 17 个构件，直至承载路径断开。每个保存的移除状态均与计算反力和弹性储能同步呈现。图中以**未变形参考构型**呈现准静态构件移除，将局部失效事件与网络连接的丧失联系起来。[场景、事件数据与可编辑图件](media/new/fracture/README.md)。

### 11. 跟踪复杂网络中的瞬态焦耳热

温度在 4,340 个节点、5,073 条边上传播，完整呈现 81 个保存状态。同步能量账本显示输入、储存与热库交换。这组合成参数的**固定几何单向电致加热**示例，将温度场与随时间演化的能量平衡联系起来。[热模型、收敛证据与数据](media/new/thermal/README.md)。

### 12. 检查图学习研究与 RL 环境动作

沿**归档 GINE 研究训练器**的记录，查看 40 个训练轮次，以及按母体几何划分的数据集上的最终电导预测。固定的已求解电场为学习曲线提供空间参照；研究文档说明该实现与当前公开数据和模型适配器的对应关系。[研究、数据划分、误差与重绘方法](media/new/learning/README.md)。

三个网络依次展示重置、预设动作和保存的原生响应，同时给出奖励与终止状态。这些**预设动作的 RL 环境交互**展示状态、动作与响应的接口，用于检查奖励和回合状态转换。[环境记录与模型范围](media/new/learning/README.md)。



### 工作流约定

**把几何与响应放在一起看。** 左侧环绕展示已保存的固定三维网络，右侧回放包含 436 个节点、576 条边的 kagome 网络简化模型变形轨迹。[媒体来源与渲染说明](MEDIA_GALLERY.md) · [静态海报](media/network_workflow_hero_poster.svg)

物理响应学习流程与桌面应用共享 **14 维几何特征和 3 个响应标签**。从合成数据示例开始，再用自己保存的物理标签训练响应代理模型。目标曲线设计使用无导数交叉熵搜索（CEM）；基于动作的工作流可选择强化学习环境和算法适配器。

通过界面设计几何、检查轨迹、查看招募与特征卡片、训练响应代理模型，并探索目标曲线设计。共享核心将这些计算与 Python 流程连接起来；[集成说明](INTEGRATION_REPO_MAP.md)区分共享能力与应用专属功能。界面支持英文和中文；可选的 SAC/TD3/DDPG 训练需要另行配置外部 Python 运行时。

先明确物理问题，再选择模型、单位、加载控制与结果类型。梁框架 FEM 和简化弹簧／弯曲／接触求解器代表不同模型，解读输出时应对应阅读 [FEM 方法](METHODS_BEAM_FEM_CONTRACT.md)与[简化模型方法](METHODS_REDUCED_SOLVER_UNIFIED.md)。[示例解读指南](HOMEPAGE_EXAMPLE_SCOPE.md)将各幅动图与模型、数据和解读方式对应起来。

</details>

<details open>
<summary>English</summary>

## Example parameters

### 1. Generate planar and three-dimensional networks

Explore octet, diamond and gyroid networks with **35/118, 63/144 and 203/531 nodes/edges** in these saved examples. Camera rotation reveals each fixed topology. [3D parameters](media/three_dimensional_topologies.json) · [Editable 3D frame](media/three_dimensional_topologies_final.svg)

#### One geometric language, multiple topologies

Apply one dimensionless reference-fiber displacement profile to **square, hexagon, ring, and kagome** networks. Their geometry changes while each graph retains its connectivity. This parameterized geometry study explores shape while preserving topology. [Manufacturing geometry](METHODS_MANUFACTURING_UNIFIED.md) · [Editable vector frame](media/spectrum_four_topologies_peak.svg) · [Parameters](media/spectrum_four_topologies.json)

### 2. Simulate deformation and compare loading directions

The **3 × 3 perturbed kagome network has 436 nodes and 576 edges**. Blue edges exceed the positive axial-strain threshold; orange edges belong to a thresholded component connecting both grips. The trace follows the recruited-edge fraction through the saved reduced-model trajectory. The strain-threshold statistic connects local extension to network-scale spanning paths. [Trajectory example](../examples/reduced_recruitment_workflow.py) · [Recruitment methods](METHODS_TENSILE_RECRUITMENT.md) · [Source manifest](media/tensile_recruitment_kagome.json)

Compare horizontal and vertical loading on one hexagonal topology. The saved reduced-model example reaches its specified spanning-recruitment threshold at stretch ratios **1.08 and 1.24**, respectively. [Directional parameters](media/directional_recruitment_hexagon.json) · [Editable comparison](media/directional_recruitment_hexagon_final.svg)

### 3. Analyze recruitment and threshold sensitivity

Read the same deformation through three thresholds, **α = 0.02, 0.05 and 0.15**. The panels reveal which connectivity conclusions depend on the analysis threshold while keeping the saved mechanical trajectory fixed. [Analysis parameters](media/tensile_threshold_sensitivity.json) · [Editable analysis frame](media/tensile_threshold_sensitivity_final.svg)

### 4. Learn responses and explore target-driven design

Bind graph inputs to group-isolated train/validation/final membership and training-only degree statistics with the [graph experiment input contract](GRAPH_TRAINING_CONTRACT.md). The [complete example](../examples/graph_training_contract.py) uses the existing StructureGraph/GraphBatch API without a learning backend.

Move from dataset preparation and response learning to a target-curve design workspace. This saved-interface tour follows the FiberScope workflow from data to design. The library shares the physical-label stream, 14-feature representation, surrogate adapters and CEM target search. [Learning example](../examples/physical_learning_workflow.py) · [Inverse-design example](../examples/curve_inverse_workflow.py) · [Tour sources](media/module_workflow_tours.json)

#### Compare structural interventions with a continuous-route constraint

Compare the intact structure with low-early-strain, high-late-strain and fixed-random deletion choices. The saved kagome and ring studies match removed edge length and retain an Euler route, exposing how a design choice changes the response. The controlled comparisons isolate the effect of each structural intervention. [Kagome source](media/route_preserving_intervention_kagome.json) · [Ring source](media/route_preserving_intervention_ring.json) · [Comparison methods](CONSTRAINED_CYCLE_INTERVENTION_2026-09-24.md)

### 5. Construct continuous manufacturing routes

A closed Euler route visits every independent fiber edge once, including parallel fibers. Blue marks visited edges; the orange cursor shows traversal order. The playback makes the graph traversal order explicit. [Generate new planar and curved routes](../examples/manufacturing_workflow.py) · [Custom cells](../examples/custom_cell_workflow.py) · [Generate new solids and routes](../examples/solid_export_workflow.py)

Export from the current `StructureGraph`: use `manufacturing_from_graph` → `export_mapped_route` for a connected all-even graph, or `manufacturing_trails_from_graph` → `export_mapped_trails` when odd-degree nodes are present. Both accept connected, nonempty straight-edge graphs with explicit `m`/`mm` units and a coordinate frame, producing mm-coordinate CSV plus stable-ID mapping JSON without changing the source topology. See the [closed-route limits](MANUFACTURING_GRAPH_MAPPING.md#geometry-and-ownership) and [multi-stroke limits](MULTISTROKE_MANUFACTURING.md#scope-and-input-validation).

### 6. Map fibers onto surfaces and prepare exports

Inspect a surface-bound network, then move to the manufacturing workspace for route and geometry export. This saved-interface tour connects surface design with inspectable route and solid-geometry files. [Surface and route example](../examples/manufacturing_workflow.py) · [Solid export example](../examples/solid_export_workflow.py) · [Tour sources](media/module_workflow_tours.json)

[Full media gallery, parameters and editable frames](MEDIA_GALLERY.md)

### 7. Inspect electrical transport on a three-dimensional network

Explore the spatial distribution of voltage and current on a saved **80-node, 162-edge** toroidal network. The camera moves around a fixed synthetic steady-state electrical solution, making the field and graph connectivity visible together. [Electrical field and render sources](media/electrical_torus_field.json) · [Electrical transport API](integrations/LINEAR_TRANSPORT_USAGE.md)

## Recorded simulation and learning gallery

Follow network motion, member failure, heat transport and data-driven studies through their actual saved states. Each example links to its data, editable figures and render recipe.

### 8. Compare LAMMPS and Taichi network dynamics

LAMMPS follows a 64-site, 441-bond network over 41.667 µs. Taichi shows 4,340 nodes and 5,073 edges from a nondimensional axial-network run. Fixed cameras reveal recorded motion at **1,000× displacement**, with unamplified displacement and energy readouts. These are backend-specific synthetic benchmarks. [Data, units and rendering](media/new/backends/README.md).

### 9. Resolve viscoelastic motion and energy transfer

A 105-node, 273-edge network vibrates and relaxes over 0.5 s. Actual axial stress accompanies the motion; kinetic energy, stored energy, material dissipation and backward-Euler numerical loss remain distinct. **3,000× displacement** makes the small-strain response visible. [Material model, saved trajectory and recipe](media/new/viscoelastic/README.md).

### 10. Follow a fracture sequence through member removal

A notched 63-node, 158-beam frame loses 17 additional members before its load path disconnects. Every recorded removal mask is synchronized with computed reaction and elastic energy. Shown on the **undeformed reference configuration**, the quasistatic member-removal sequence connects local failure events to loss of network connectivity. [Scenario, event data and editable poster](media/new/fracture/README.md).

### 11. Track transient Joule heating across a complex network

Temperature evolves across 4,340 nodes and 5,073 edges through all 81 saved states. The accompanying energy ledger follows input, storage and reservoir transfer. The synthetic **one-way electrical heating on fixed geometry** example connects the temperature field to a time-resolved energy balance. [Thermal model, convergence evidence and data](media/new/thermal/README.md).

### 12. Inspect a graph-learning study and RL environment actions

Explore 40 training epochs and final conductance predictions from an **archived GINE research trainer**, using parent-separated networks. The fixed solved-field panel supplies spatial context alongside the learning curves. The study documentation maps this research implementation to the current public data and model adapters. [Study, splits, errors and render recipe](media/new/learning/README.md).

Three networks show reset, prescribed action and recorded native response, together with reward and termination. These **prescribed-action RL environment interactions** expose the state–action–response interface for inspecting rewards and episode transitions. [Environment records and model scope](media/new/learning/README.md).



### Workflow conventions

**Geometry and response, side by side.** The left view orbits a fixed, saved 3D network; the right replays the recorded reduced-model deformation of a 436-node, 576-edge kagome network. [Media sources and rendering details](MEDIA_GALLERY.md) · [Static poster](media/network_workflow_hero_poster.svg)

The physical-learning workflow shares a **14-feature geometry representation and three response labels** with the desktop app. Start with the synthetic-data demo, then train a response surrogate on your saved physical labels. Use derivative-free cross-entropy search (CEM) for target curves, or the optional reinforcement-learning environments and algorithm adapters for action-based workflows.

Design geometry, inspect trajectories, view recruitment and feature cards, train response surrogates, and explore target-curve designs through the interface. The shared core keeps those calculations connected to the Python workflows; [the integration map](INTEGRATION_REPO_MAP.md) distinguishes shared and app-specific functionality. The interface supports English and Chinese. Optional SAC/TD3/DDPG training uses a separately configured external Python runtime.

Start from the physical question, then select the model, units, loading controls, and result type. The beam-frame FEM and reduced spring/bending/contact solver describe different models; use their matching [FEM](METHODS_BEAM_FEM_CONTRACT.md) and [reduced-model](METHODS_REDUCED_SOLVER_UNIFIED.md) methods when interpreting output. The [media interpretation guide](HOMEPAGE_EXAMPLE_SCOPE.md) connects each visual to its model, data and interpretation.

</details>
