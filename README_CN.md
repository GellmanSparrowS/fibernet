# FiberNet + FiberScope

[![Switch to English](docs/media/ui/language-en-inactive.svg)](README.md) [![中文 — 当前语言](docs/media/ui/language-zh-active.svg)](README_CN.md)

### 从纤维网络几何，到力学响应与结构设计。

**FiberNet** 是面向纤维网络生成、模拟、分析与逆向设计的 Python 工具库；**FiberScope** 将共享的几何、简化模型与分析流程带入桌面应用。从构建结构、观察变形、量化连接变化，到响应学习和连续路径构造，同一设计可以贯穿整个研究流程。

[快速开始](#快速开始) · [动态示例](#动态示例) · [功能模块](#选择研究流程) · [桌面应用](#桌面应用) · [方法与复现](#方法与复现)

![已保存的三维网络几何与 kagome 变形轨迹：分别标明相机运动和模拟材料运动](docs/media/network_workflow_hero.gif)

**把几何与响应放在一起看。** 左侧环绕展示已保存的固定三维网络，右侧回放包含 436 个节点、576 条边的 kagome 网络简化模型变形轨迹。左侧改变的是相机视角，而非材料状态。[媒体来源与渲染说明](docs/MEDIA_GALLERY.md) · [静态海报](docs/media/network_workflow_hero_poster.svg)

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

源码版本与 PyPI 发布版分开管理；4.3.0 尚未发布至 PyPI。需要 Python 3.9 及以上版本。按任务选择可选依赖，例如在源码目录运行 `python -m pip install -e ".[ml]"` 或 `python -m pip install -e ".[manufacturing]"`。

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

### 同一种几何语言，连接不同拓扑

![正方形、六边形、环形与 kagome 网络使用同一参考纤维位移轮廓改变几何](docs/media/spectrum_four_topologies.gif)

将同一无量纲参考纤维位移轮廓用于**正方形、六边形、环形和 kagome** 网络。几何形状随参数变化，各网络的连接关系保持不变。这是几何参数化展示，而非加载模拟。[制造几何方法](docs/METHODS_MANUFACTURING_UNIFIED.md) · [可编辑矢量关键帧](docs/media/spectrum_four_topologies_peak.svg) · [参数记录](docs/media/spectrum_four_topologies.json)

### 观察复杂网络在拉伸中逐步招募

![包含 436 个节点的 kagome 网络随拉伸变形，招募边和跨夹持端连通分量同步演化](docs/media/tensile_recruitment_kagome.gif)

这个 **3 × 3 扰动 kagome 网络包含 436 个节点和 576 条边**。蓝色表示超过正轴向应变阈值的边，橙色表示连接两侧夹持端的阈值连通分量；曲线记录已保存简化模型轨迹中的招募边比例。招募刻画的是应变阈值下的连接关系，不是完整力流。[轨迹示例](examples/reduced_recruitment_workflow.py) · [招募分析方法](docs/METHODS_TENSILE_RECRUITMENT.md) · [来源记录](docs/media/tensile_recruitment_kagome.json)

### 用一条连续路径遍历每根纤维

![橙色游标沿 kagome 制造图的闭合欧拉路径前进](docs/media/continuous_route_kagome.gif)

闭合欧拉路径将每条独立纤维边遍历一次，也保留平行纤维的独立身份。蓝色是已访问的边，橙色游标标明当前遍历顺序。平面与曲面流程保留结构图到路径的映射，可用于检查和导出；路径回放展示图遍历，而非打印机动力学。[平面与曲面路径](examples/manufacturing_workflow.py) · [自定义基元](examples/custom_cell_workflow.py) · [实体与路径导出](examples/solid_export_workflow.py)

**继续探索：**[三维拓扑动图](docs/media/three_dimensional_topologies.gif) · [阈值敏感性](docs/media/tensile_threshold_sensitivity.gif) · [不同方向加载](docs/media/directional_recruitment_hexagon.gif) · [保路径结构干预](docs/MEDIA_GALLERY.md#route-preserving-interventions)。[完整媒体画廊](docs/MEDIA_GALLERY.md)保留每幅动图的来源、参数和可编辑关键帧。

## 选择研究流程

| 研究目标 | Python 入口 | 指南或可执行示例 |
| --- | --- | --- |
| 生成二维、三维结构 | `pattern_2d`、`pattern_3d`、`StructureGraph` | [参数化生成](fibernet/gen/pattern.py) |
| 研究变形与力学响应 | 显式 FEM 或 Taichi 路径；共享简化模型使用 `ReducedBeamSolver` | [接口层次、单位与模型选择](docs/API_LAYERS.md) |
| 量化结构与招募行为 | `SnapshotFeatureExtractor`、`analyze_tensile_recruitment` | [快照特征](examples/snapshot_features_workflow.py) · [招募分析](examples/tensile_recruitment_quickstart.py) |
| 构建复用基元与连续路径 | `CustomCellRegistry`、`compile_planar`、`compile_surface` | [自定义基元方法](docs/METHODS_CUSTOM_CELL_UNIFIED.md) · [制造几何](docs/METHODS_MANUFACTURING_UNIFIED.md) |
| 从模拟响应中学习 | `PlanarPhysicalDataset`、`light_features`、`PhysicalRegressor` | [可续算标签生成](examples/physical_dataset_workflow.py) · [代理模型学习](examples/physical_learning_workflow.py) |
| 搜索目标响应结构 | `CurveInverseDesigner`；可选 `fibernet.rl` 工具 | [目标曲线搜索](examples/curve_inverse_workflow.py) · [学习与搜索方法](docs/METHODS_CURVE_INVERSE_UNIFIED.md) |

物理响应学习流程与桌面应用共享 **14 维几何特征和 3 个响应标签**。默认可运行示例使用合成数据检查接口；构建响应代理模型时，可输入自己保存的物理标签。目标曲线搜索采用无导数交叉熵方法（CEM）；可选的强化学习环境和算法适配器属于另一条流程，CEM 不等同于强化学习。

### 开发版源码中的扩展接口

- **时变力学：**[黏弹梁框架历史](docs/VISCOELASTIC_NETWORK.md)、[轴向黏弹动力学](docs/DYNAMIC_AXIAL_VISCOELASTICITY.md)与[状态观测](docs/state_observer.md)。
- **传输与热过程：**[线性电传输](docs/integrations/LINEAR_TRANSPORT_USAGE.md)、[瞬态热网络](docs/integrations/TRANSIENT_THERMAL_USAGE.md)与[单向焦耳热](docs/ONE_WAY_JOULE_HEATING.md)。独立的[实验性电热接口](docs/EXPERIMENTAL_ELECTROTHERMAL.md)以显式系数和状态接受检查，处理固定几何、电压驱动的热反馈。
- **外部引擎：**[有界 LAMMPS 谐键动力学](docs/integrations/LAMMPS_HARMONIC_CLI.md)与[粗粒化纤维网络](docs/integrations/LAMMPS_CG_CROSSLINKED_NETWORK.md)，采用显式运行时和单位约定。
- **实验性断裂：**独立的线性静力 Timoshenko 杆件移除后端，保留稳定 ID、事件记录和可续算检查点。[模型范围与来源](docs/FRACTURE_METHODS.md)。
- **图学习：**[边感知 GINE 数据与模型适配器](docs/integrations/EDGE_GINE.md)。

这些模块级流程分别定义输入与结果类型。[接口地图](docs/API_LAYERS.md)列出支持的组合，不能将它们当作可互换的 `simulate` 选择项。

## 桌面应用

**希望直接通过界面操作？** 下载 [FiberScope 3.1.0 Windows x64 版](https://github.com/GellmanSparrowS/fibernet/releases/tag/v4.2.0)，完整解压 ZIP 后运行 `FiberScope.exe`。桌面可执行程序包含所需运行时，不需要本地安装 Python。

[![FiberScope 中文制造工作区](FiberScope/docs/screenshots/manufacturing-zh.png)](FiberScope/docs/screenshots/README.md)

通过界面设计几何、检查轨迹、查看招募与特征卡片、训练响应代理模型，并探索目标曲线设计。共享核心将这些计算与 Python 流程连接起来；[集成说明](docs/INTEGRATION_REPO_MAP.md)区分共享能力与应用专属功能。界面支持英文和中文；可选的 SAC/TD3/DDPG 训练需要另行配置外部 Python 运行时。

[桌面使用指南](FiberScope/README.md) · [中英文界面截图](FiberScope/docs/screenshots/README.md) · [发布说明](docs/RELEASE_4_2_0.md)

## 方法与复现

先明确物理问题，再选择模型、单位、加载控制与结果类型。梁框架 FEM 和简化弹簧／弯曲／接触求解器代表不同模型，解读输出时应对应阅读 [FEM 方法](docs/METHODS_BEAM_FEM_CONTRACT.md)与[简化模型方法](docs/METHODS_REDUCED_SOLVER_UNIFIED.md)。数值示例不等同于实验材料标定。

- **可检查的媒体：**[来源记录、可编辑矢量帧与渲染命令](docs/MEDIA_GALLERY.md)。
- **研究示例：**[阈值与方向敏感性](docs/RECRUITMENT_SENSITIVITY_2026-09-24.md)、[匹配的保路径结构干预](docs/CONSTRAINED_CYCLE_INTERVENTION_2026-09-24.md)以及[同图跨模型比较](docs/INDEPENDENT_FEM_CYCLE_INTERVENTION_2026-09-24.md)。图示对应的研究记录同时保留模型依赖性与负结果。
- **兼容性：**当前开发版将可选 Taichi 后端固定为 **1.7.4**。引擎应在其所属线程上初始化和使用，支持条件见[运行时生命周期约定](docs/architecture/taichi-lifecycle-audit.md)。上面的 FEM 快速开始不使用 Taichi。
- **项目资料：**[示例索引](examples/README.md)、[测试](tests/test_tensile_recruitment.py)、[版本记录](CHANGELOG.md)与[能力清单](docs/CAPABILITY_AUDIT_2026-09-24.md)。

## 许可与引用

FiberNet 采用 [MIT 许可证](LICENSE)，第三方声明见[依赖与模型说明](FiberScope/THIRD_PARTY_NOTICES.md)。关于网络拓扑与逆向设计的已发表研究，请参阅 [Nature Communications 论文](https://www.nature.com/articles/s41467-026-76045-x)。
