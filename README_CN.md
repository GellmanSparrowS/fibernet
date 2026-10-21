[English / 中文 · 仓库首页](README.md)

<!-- locale:zh:start -->
[动态示例](#动态示例) · [功能模块](#选择研究流程) · [桌面应用](#桌面应用) · [方法与复现](#方法与复现)

**从几何、响应与连接关系出发，设计纤维网络。** FiberNet 将结构生成、力学模拟、电热传输、响应学习与连续路径构造连接为 Python 研究流程；FiberScope 将共享的几何、简化模型与分析功能带入桌面。从一个网络开始，比较设计选择，再将结果用于下一阶段。


![已保存的三维网络几何与 kagome 变形轨迹：分别标明相机运动和模拟材料运动](docs/media/network_workflow_hero.gif)

**把几何与响应放在一起看。** 左侧环绕展示已保存的固定三维网络，右侧回放包含 436 个节点、576 条边的 kagome 网络简化模型变形轨迹。[媒体来源与渲染说明](docs/MEDIA_GALLERY.md) · [静态海报](docs/media/network_workflow_hero_poster.svg)

## 快速开始

### 安装 Python 库

已发布的包版本为 **FiberNet 4.2.0**：

```bash
python -m pip install fibernet==4.2.0
```

如需使用 **4.3.0 开发版源码**及其中的新增接口，请克隆仓库，在仓库根目录安装：

```bash
git clone https://github.com/GellmanSparrowS/fibernet.git
cd fibernet
python -m pip install -e .
```

使用 PyPI 获取已发布版本，使用源码安装体验开发版接口。需要 Python 3.9 及以上版本。按任务选择可选依赖，例如在源码目录运行 `python -m pip install -e ".[ml]"` 或 `python -m pip install -e ".[manufacturing]"`。

### 生成、模拟、保存

```python
import fibernet as fn

graph = fn.pattern_2d(
    unit="honeycomb", box=(10, 10), grid=(2, 2), radius=0.05, seed=23
)
fn.show(graph, save_path="network.png")

result = fn.simulate(graph, backend="fem", mode="stretch", strain=1.01)
result.save("fem_result.json", detailed=True)
print(graph.num_nodes, graph.num_edges, result.max_displacement)
```

这里的 `strain=1.01` 表示**拉伸比**，即 1% 伸长。该显式 FEM 路径使用小应变梁框架模型和核心依赖，不需要 Taichi 或桌面应用。几何与材料参数需采用一致的单位体系，详见[梁 FEM 接口约定](docs/METHODS_BEAM_FEM_CONTRACT.md)。

在源码目录运行完整的“生成结构 → 模拟 → 保存结果 → 招募分析”流程：

```bash
python -m examples.open_source_workflow --output-dir demo_output
```

输出包括网络图、模拟结果 JSON 和分析摘要。[查看可执行示例](examples/open_source_workflow.py)。

## 动态示例

### 1. 生成平面与三维纤维网络

![从不同相机角度观察 octet、diamond 与 gyroid 纤维网络](docs/media/three_dimensional_topologies.gif)

这组已保存的 octet、diamond 和 gyroid 示例分别包含 **35/118、63/144、203/531 个节点/条边**。旋转视角呈现各固定拓扑的空间连接。[三维参数](docs/media/three_dimensional_topologies.json) · [可编辑三维帧](docs/media/three_dimensional_topologies_final.svg)

#### 同一种几何语言，连接不同拓扑

![正方形、六边形、环形与 kagome 网络使用同一参考纤维位移轮廓改变几何](docs/media/spectrum_four_topologies.gif)

将同一无量纲参考纤维位移轮廓用于**正方形、六边形、环形和 kagome** 网络。几何形状随参数变化，各网络的连接关系保持不变。这组几何参数化示例在保持拓扑的同时探索形状变化。[制造几何方法](docs/METHODS_MANUFACTURING_UNIFIED.md) · [可编辑矢量关键帧](docs/media/spectrum_four_topologies_peak.svg) · [参数记录](docs/media/spectrum_four_topologies.json)

### 2. 模拟网络变形，比较加载方向

![包含 436 个节点的 kagome 网络随拉伸变形，招募边和跨夹持端连通分量同步演化](docs/media/tensile_recruitment_kagome.gif)

这个 **3 × 3 扰动 kagome 网络包含 436 个节点和 576 条边**。蓝色表示超过正轴向应变阈值的边，橙色表示连接两侧夹持端的阈值连通分量；曲线记录已保存简化模型轨迹中的招募边比例。应变阈值统计将局部伸长与网络尺度的跨端路径联系起来。[轨迹示例](examples/reduced_recruitment_workflow.py) · [招募分析方法](docs/METHODS_TENSILE_RECRUITMENT.md) · [来源记录](docs/media/tensile_recruitment_kagome.json)

![同一六边形拓扑的横向与纵向加载](docs/media/directional_recruitment_hexagon.gif)

在同一六边形拓扑上比较横向和纵向加载。已保存的简化模型示例分别在拉伸比 **1.08 和 1.24** 时达到所设定的跨端招募阈值。[方向比较参数](docs/media/directional_recruitment_hexagon.json) · [可编辑比较图](docs/media/directional_recruitment_hexagon_final.svg)

### 3. 分析纤维招募与阈值敏感性

![在同一条已保存 kagome 轨迹上比较三种招募阈值](docs/media/tensile_threshold_sensitivity.gif)

用 **α = 0.02、0.05、0.15** 三种阈值观察同一段变形。在保持原力学轨迹不变的条件下，直接比较连通性结论对分析阈值的依赖。[分析参数](docs/media/tensile_threshold_sensitivity.json) · [可编辑分析帧](docs/media/tensile_threshold_sensitivity_final.svg)

### 4. 学习结构响应，探索目标驱动设计

![学习与逆向设计工作区的已保存界面动态导览](docs/media/learning_inverse_workflow.gif)

从数据准备和响应学习进入目标曲线设计工作区。这段已保存的 FiberScope 界面导览呈现从数据到设计的使用流程。Python 库共享物理标签流、14 维特征、代理模型适配器与 CEM 目标搜索。[学习示例](examples/physical_learning_workflow.py) · [逆向设计示例](examples/curve_inverse_workflow.py) · [导览来源](docs/media/module_workflow_tours.json)

#### 在连续路径约束下比较结构干预

![kagome 网络四种匹配的保路径结构干预](docs/media/route_preserving_intervention_kagome.gif)

![环形网络匹配的保路径结构干预](docs/media/route_preserving_intervention_ring.gif)

比较完整结构、低早期应变删边、高后期应变删边与固定随机删边。已保存的 kagome 和环形研究匹配移除边长，并保留欧拉路径，展示设计选择如何改变响应。受控比较使不同结构干预对响应的影响清晰可见。[Kagome 来源](docs/media/route_preserving_intervention_kagome.json) · [环形来源](docs/media/route_preserving_intervention_ring.json) · [比较方法](docs/CONSTRAINED_CYCLE_INTERVENTION_2026-09-24.md)

### 5. 构建连续制造路径

![橙色游标沿 kagome 制造图的闭合欧拉路径前进](docs/media/continuous_route_kagome.gif)

闭合欧拉路径将每条独立纤维边遍历一次，也保留平行纤维的独立身份。蓝色是已访问的边，橙色游标标明当前遍历顺序。平面与曲面流程保留结构图到路径的映射，可用于检查和导出；路径回放清晰呈现结构图的遍历顺序。[平面与曲面路径](examples/manufacturing_workflow.py) · [自定义基元](examples/custom_cell_workflow.py) · [实体与路径导出](examples/solid_export_workflow.py)

### 6. 映射曲面纤维，准备几何与路径导出

![曲面映射与制造导出工作区的已保存界面动态导览](docs/media/surface_export_workflow.gif)

先检查曲面上的纤维网络，再进入制造工作区准备路径与几何导出。这段已保存界面导览将曲面设计连接到可检查的路径文件与实体几何文件。[曲面与路径示例](examples/manufacturing_workflow.py) · [实体导出示例](examples/solid_export_workflow.py) · [导览来源](docs/media/module_workflow_tours.json)

[完整媒体画廊、参数与可编辑关键帧](docs/MEDIA_GALLERY.md)

### 7. 检查三维网络中的电传输

![已保存的环面网络电学场相机环绕展示](docs/media/electrical_torus_field.gif)

在包含 **80 个节点、162 条边**的已保存环面网络上，观察电压与电流的空间分布。相机环绕固定的合成参数稳态电学解，让场分布与图连接关系同时可见。[电学场与渲染来源](docs/media/electrical_torus_field.json) · [电传输接口](docs/integrations/LINEAR_TRANSPORT_USAGE.md)

## 模拟与学习过程图集

沿真实保存的状态，查看网络运动、构件失效、热传输与数据驱动研究。每个示例均链接原始数据、可编辑图件和重绘方法。

### 8. 比较 LAMMPS 与 Taichi 网络动力学

![LAMMPS 谐键网络动力学](docs/media/new/backends/lammps_saved_dynamics.gif)

![Taichi 非均匀轴向网络动力学](docs/media/new/backends/taichi_saved_dynamics.gif)

LAMMPS 展示 64 个节点、441 条键在 41.667 µs 内的运动；Taichi 展示 4,340 个节点、5,073 条边的无量纲轴向网络记录。固定视角下，**位移放大 1,000 倍**，位移读数和能量保留真实量值。这些是各后端的合成基准场景。[数据、单位与重绘方法](docs/media/new/backends/README.md)。

### 9. 解析黏弹网络运动与能量转移

![保存的广义 Maxwell 网络动力学](docs/media/new/viscoelastic/viscoelastic_network_dynamics.gif)

105 个节点、273 条边在 0.5 s 内振动并松弛。实际轴向应力随运动呈现；动能、储能、材料耗散与后向欧拉数值损失分别显示。**位移放大 3,000 倍**，使小应变响应清晰可见。[材料模型、保存轨迹与重绘方法](docs/media/new/viscoelastic/README.md)。

### 10. 跟踪构件移除形成的断裂序列

![含缺口框架的准静态构件移除序列](docs/media/new/fracture/fracture_sequence.gif)

含缺口的 63 节点、158 梁框架进一步移除 17 个构件，直至承载路径断开。每个保存的移除状态均与计算反力和弹性储能同步呈现。图中以**未变形参考构型**呈现准静态构件移除，将局部失效事件与网络连接的丧失联系起来。[场景、事件数据与可编辑图件](docs/media/new/fracture/README.md)。

### 11. 跟踪复杂网络中的瞬态焦耳热

![4340 节点网络的瞬态焦耳热](docs/media/new/thermal/transient_thermal_network.gif)

温度在 4,340 个节点、5,073 条边上传播，完整呈现 81 个保存状态。同步能量账本显示输入、储存与热库交换。这组合成参数的**固定几何单向电致加热**示例，将温度场与随时间演化的能量平衡联系起来。[热模型、收敛证据与数据](docs/media/new/thermal/README.md)。

### 12. 检查图学习研究与 RL 环境动作

![保存的 GINE 电导学习研究](docs/media/new/learning/media/gine_training_history.gif)

沿**归档 GINE 研究训练器**的记录，查看 40 个训练轮次，以及按母体几何划分的数据集上的最终电导预测。固定的已求解电场为学习曲线提供空间参照；研究文档说明该实现与当前公开数据和模型适配器的对应关系。[研究、数据划分、误差与重绘方法](docs/media/new/learning/README.md)。

![强化学习环境中的预设动作](docs/media/new/learning/media/rl_environment_fixed_actions.gif)

三个网络依次展示重置、预设动作和保存的原生响应，同时给出奖励与终止状态。这些**预设动作的 RL 环境交互**展示状态、动作与响应的接口，用于检查奖励和回合状态转换。[环境记录与模型范围](docs/media/new/learning/README.md)。

## 选择研究流程

| 研究目标 | Python 入口 | 指南或可执行示例 |
| --- | --- | --- |
| 生成二维、三维结构 | `pattern_2d`、`pattern_3d`、`StructureGraph` | [参数化生成](fibernet/gen/pattern.py) |
| 研究变形与力学响应 | 显式 FEM 或 Taichi 路径；共享简化模型使用 `ReducedBeamSolver` | [接口层次、单位与模型选择](docs/API_LAYERS.md) |
| 量化结构与招募行为 | `SnapshotFeatureExtractor`、`analyze_tensile_recruitment` | [快照特征](examples/snapshot_features_workflow.py) · [招募分析](examples/tensile_recruitment_quickstart.py) |
| 构建复用基元与连续路径 | `CustomCellRegistry`、`compile_planar`、`compile_surface` | [自定义基元方法](docs/METHODS_CUSTOM_CELL_UNIFIED.md) · [制造几何](docs/METHODS_MANUFACTURING_UNIFIED.md) |
| 从模拟响应中学习 | `PlanarPhysicalDataset`、`light_features`、`PhysicalRegressor` | [可续算标签生成](examples/physical_dataset_workflow.py) · [代理模型学习](examples/physical_learning_workflow.py) |
| 搜索目标响应结构 | `CurveInverseDesigner`；可选 `fibernet.rl` 工具 | [目标曲线搜索](examples/curve_inverse_workflow.py) · [学习与搜索方法](docs/METHODS_CURVE_INVERSE_UNIFIED.md) |

物理响应学习流程与桌面应用共享 **14 维几何特征和 3 个响应标签**。从合成数据示例开始，再用自己保存的物理标签训练响应代理模型。目标曲线设计使用无导数交叉熵搜索（CEM）；基于动作的工作流可选择强化学习环境和算法适配器。

### 开发版源码中的扩展接口

- **时变力学：**[黏弹梁框架历史](docs/VISCOELASTIC_NETWORK.md)、[轴向黏弹动力学](docs/DYNAMIC_AXIAL_VISCOELASTICITY.md)与[状态观测](docs/state_observer.md)。
- **传输与热过程：**[线性电传输](docs/integrations/LINEAR_TRANSPORT_USAGE.md)、[瞬态热网络](docs/integrations/TRANSIENT_THERMAL_USAGE.md)与[单向焦耳热](docs/ONE_WAY_JOULE_HEATING.md)。独立的[实验性电热接口](docs/EXPERIMENTAL_ELECTROTHERMAL.md)以显式系数和状态接受检查，处理固定几何、电压驱动的热反馈。
- **外部引擎：**[有界 LAMMPS 谐键动力学](docs/integrations/LAMMPS_HARMONIC_CLI.md)与[粗粒化纤维网络](docs/integrations/LAMMPS_CG_CROSSLINKED_NETWORK.md)，采用显式运行时和单位约定。
- **实验性断裂：**独立的线性静力 Timoshenko 杆件移除后端，保留稳定 ID、事件记录和可续算检查点。[模型范围与来源](docs/FRACTURE_METHODS.md)。
- **图学习：**[边感知 GINE 数据与模型适配器](docs/integrations/EDGE_GINE.md)。

通过[接口地图](docs/API_LAYERS.md)选择各模块的调用入口、输入约定与结果类型。

## 桌面应用

**希望直接通过界面操作？** 下载 [FiberScope 3.1.0 Windows x64 版](https://github.com/GellmanSparrowS/fibernet/releases/tag/v4.2.0)，完整解压 ZIP 后运行 `FiberScope.exe`。桌面可执行程序包含所需运行时，不需要本地安装 Python。

[![FiberScope 中文制造工作区](FiberScope/docs/screenshots/manufacturing-zh.png)](FiberScope/docs/screenshots/README.md)

通过界面设计几何、检查轨迹、查看招募与特征卡片、训练响应代理模型，并探索目标曲线设计。共享核心将这些计算与 Python 流程连接起来；[集成说明](docs/INTEGRATION_REPO_MAP.md)区分共享能力与应用专属功能。界面支持英文和中文；可选的 SAC/TD3/DDPG 训练需要另行配置外部 Python 运行时。

[桌面使用指南](FiberScope/README.md) · [中英文界面截图](FiberScope/docs/screenshots/README.md) · [发布说明](docs/RELEASE_4_2_0.md)

## 方法与复现

先明确物理问题，再选择模型、单位、加载控制与结果类型。梁框架 FEM 和简化弹簧／弯曲／接触求解器代表不同模型，解读输出时应对应阅读 [FEM 方法](docs/METHODS_BEAM_FEM_CONTRACT.md)与[简化模型方法](docs/METHODS_REDUCED_SOLVER_UNIFIED.md)。[示例解读指南](docs/HOMEPAGE_EXAMPLE_SCOPE.md)将各幅动图与模型、数据和解读方式对应起来。

- **可检查的媒体：**[来源记录、可编辑矢量帧与渲染命令](docs/MEDIA_GALLERY.md)。
- **研究示例：**[阈值与方向敏感性](docs/RECRUITMENT_SENSITIVITY_2026-09-24.md)、[匹配的保路径结构干预](docs/CONSTRAINED_CYCLE_INTERVENTION_2026-09-24.md)以及[同图跨模型比较](docs/INDEPENDENT_FEM_CYCLE_INTERVENTION_2026-09-24.md)。完整比较记录呈现模型依赖性与各组研究结果。
- **兼容性：**当前开发版将可选 Taichi 后端固定为 **1.7.4**。引擎应在其所属线程上初始化和使用，支持条件见[运行时生命周期约定](docs/architecture/taichi-lifecycle-audit.md)。上面的 FEM 快速开始使用核心依赖。
- **项目资料：**[示例索引](examples/README.md)、[测试](tests/test_tensile_recruitment.py)、[版本记录](CHANGELOG.md)与[能力清单](docs/CAPABILITY_AUDIT_2026-09-24.md)。

## 许可与引用

FiberNet 采用 [MIT 许可证](LICENSE)，第三方声明见[依赖与模型说明](FiberScope/THIRD_PARTY_NOTICES.md)。关于网络拓扑与逆向设计的已发表研究，请参阅 [Nature Communications 论文](https://www.nature.com/articles/s41467-026-76045-x)。
<!-- locale:zh:end -->
