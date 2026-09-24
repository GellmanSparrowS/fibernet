# FiberNet–FiberScope：Work 模式交接与拟投稿主线

本文件供后续 Work 模式拼图、复跑、整理引用和撰文。它是任务书和证据索引，不是投稿手稿。当前代码属于原有 `fibernet` 仓库的 4.2.0 开发分支；FiberScope 为同仓库 APP。Python 3.10 库测试 368 通过、19 跳过，APP 本地测试 25/25；本地 wheel 隔离安装及冻结 APP 的 Windows 可携带性门槛通过。库端源码测试在 GitHub CI 的 Linux、macOS、Windows 与 Python 3.9–3.12 共 12 个组合通过。尚无第三方机器安装、Linux/macOS 冻结 APP、材料实验或打印后性能验证。研究数据与图不得超出这些条件解释。

## 建议中心问题

**如何在脚本与交互应用之间保持纤维网络的图身份、几何、求解时序和研究证据一致，并用这个平台提出可反证的动态招募问题？** 文章类型以软件/方法平台为主，渗流或更准确的“正向轴向应变阈值招募与夹持贯通”作为落地实例。写作顺序建议为共同对象与 API、数值同源、可安装和可执行性、动态招募案例、干预与模型依赖边界。软件对问题的可复现支持是主张；材料增益不是当前证据支持的结论。

已发表的 [Nature Communications 论文](https://doi.org/10.1038/s41467-026-76045-x) 是制造约束拓扑与 AI 引导设计的原始贡献。新文不得把连续纤维拓扑、曲面投影、FEA–GNN–RL 或打印提升重新写为首次发现。可参考 [Squidpy](https://www.nature.com/articles/s41592-021-01358-2) 对共同表示、分析接口和教程的组织方式，以及 [napari-imagej](https://www.nature.com/articles/s41592-023-01990-0) 对 GUI 与计算生态互操作的写法；[APEX](https://www.nature.com/articles/s41524-025-01580-y) 可用于比较计算平台如何展示可执行工作流。以上仅为写作先例，不代表本平台具有同等验证广度。

## 已完成的可检验证据

| 论点 | 本地证据 | 写作限定 |
| --- | --- | --- |
| APP 与库共用科学核心 | 40 数值数组迁移前后严格一致；自定义单元坐标/边序及 JSON 读取跨端一致；制造、逆设计、数据流、快照特征各有跨端测试 | 数值相等仅针对固定输入及当前版本 |
| 可由 Python 独立运行 | 4.2.0 wheel 在 Python 3.10 隔离目录实跑自定义单元 140 边闭合路线、平行梁反力比 2、快照特征，未导入 APP；库端 12 个跨系统/版本源码 CI 作业通过 | 第三方机器的 wheel 安装和用户任务仍待验收 |
| 动态招募具有方向/阈值依赖 | 四拓扑×三基几何×双方向×接触开关 48 轨迹；六边形一个参数组 x/y 首次贯通在 λ≈1.08/1.24 | 只指应变阈值子图，不指完整力流；独立非线性梁模型未复现准确起始点 |
| 删除规则可被反事实检验 | 24 个配置的等长度、保留欧拉路线删环；408 个候选逐一删除并重算；24 个同图同夹持的简化模型/线性梁 FEM 配对复核 | 早期低应变删环在简化模型几乎无损，梁 FEM 中 11/24 反力比低于 0.95；不支持模型无关的“安全删除” |
| AI 选环基线有明确负结果 | 408 候选的拓扑家族留出随机森林与静态几何规则平均反力比 0.99999829/0.99999824 | 差约 4.8×10⁻⁸，不能宣称 AI 显著改善材料性能 |

48 轨迹与 24 干预配置的 x/y 加载共享同一个基结构。统计单位应以基几何聚类，不能把两个方向当独立试样。结构变形和受约束干预的主结果均来自数值模型，没有实验真值。与外部科学背景连接时，可分别核对 [纤维网络应变控制临界性](https://www.nature.com/articles/nphys3628)、[无序纤维网络局部刚度异质性](https://www.nature.com/articles/ncomms16096)、[组织局部应变贯通与破裂](https://www.nature.com/articles/s43856-025-00897-5)；最后一篇已经研究早期局部应变信息，不能把“初期信息提示后期区域”当作本项目独有发现。

## 主图建议与素材位置

1. **共同数据对象与跨端任务。** 节点、独立平行边、半径、参考/实际坐标、Euler 路线和 APP 参数回读；展示四种结构及真实英文界面。素材：`manuscript/FiberNet_Methods_Figures_Working.pptx` 第 1 页、`docs/media/spectrum_four_topologies_peak.svg`。
2. **可执行软件与一致性门槛。** API 任务矩阵、APP/库同参数测试、40 数值黄金数组、wheel/冻结 APP 验收；给出测试环境和排除范围。素材：PPT 第 2 页、`docs/CAPABILITY_AUDIT_2026-09-24.md`、`release_candidates/2026-09-24/manifest.json`（本地）。
3. **动态招募的方向/阈值。** 复杂 3×3 网络、时序、阈值对照与不同方向；图注明确“应变阈值贯通”。素材：PPT 第 3 页、`docs/media/tensile_recruitment_kagome_final.svg`、`docs/media/directional_recruitment_hexagon_final.svg`。
4. **受约束干预的模型边界。** 选环前后图、同长度预算、保留闭合路线，以及 24 配置逐点简化模型/独立梁配对；优先采用 PPT 第 5 页，旧第 4 页简化模型均值可作子图/补图。完整配对见 `benchmarks/results/independent_fem_cycle_intervention.json`。不要把“删边收益”作为标题。

每张主图对应一个可核验观点。结构面板应覆盖方形、六边形、圆环与笼目，不以单一简单图代替多样性；位图仅用于 APP 截图或必要的复杂形貌，结构、曲线、图例、数字和文字优先可编辑矢量。先定期刊单栏/双栏尺寸再排图，检查最终物理字号、坐标轴、图例和线宽。`manuscript/FIGURE_PLAN.md` 有更细的面板草案；现有 PPT 为工作图，不是投稿版式定稿。

## Work 模式执行顺序

1. 解压后从仓库根安装 `python -m pip install -e .`；先跑 `python -m examples.custom_cell_workflow --output-dir demo_custom_cell` 和 `python -m examples.reduced_recruitment_workflow --output-dir demo_stretch`。长任务使用脚本已有原子检查点；不要把未完成 JSON 当成完整结果。
2. 用 `python scripts/check_homepage_media.py` 审计七段主页动图。用 `python manuscript/scripts/build_figures_ppt.py` 重建五页 PPT，对照同名 JSON 的源文件 SHA；如需换算正式版心，直接修改生成脚本的字号和布局参数并在 PowerPoint 渲染检查。
3. 优先分析 `benchmarks/results/recruitment_sensitivity.json`、`constrained_cycle_intervention.json`、`ai_cycle_selector.json`、`independent_fem_cycle_intervention.json`。保留每配置的基几何 ID、方向、模型和策略，不要只画拓扑均值。补基几何聚类区间，并呈现负结果。
4. 写作前逐句核查外部文献与原始链接；已有中文工作稿 `manuscript/ZH_METHODS_DRAFT.md` 可当旧材料索引，不直接作为最终手稿。代码方法定义见 `docs/METHODS_*.md`；引用只支持它实际描述的操作。
5. 投稿前取得第三方机器/平台安装、独立结构外推、求解器收敛与力学参数敏感性、材料实验或可信外部求解器对照。没有这些数据时，结论限于软件计算一致性、可复算研究流程及模型敏感性。

## 包内定位

压缩包包含完整可运行源码、测试、示例、研究 JSON、方法文档、七段 GIF/SVG、五页可编辑 PPT、本交接文件和文件 SHA 清单。冻结 APP 的 266 MB 二进制留在本机已验收目录，压缩包只携带 APP 源码和资产；PyPI/APP 正式发布尚未执行。开源仓库保留原有 Git 历史，ZIP 用于 Work 模式移动，不作为版本控制替代。
