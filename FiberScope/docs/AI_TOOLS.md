# FiberScope AI 助手接口说明（18 个工具）

AI 助手通过 DeepSeek function-calling 驱动应用全部内部接口。每个工具在 GUI
线程执行并返回真实数据（JSON），助手据此给出含数字的中文总结。API Key 首次
输入后保存在本机用户目录（~/.fiberscope/config.json），分享软件时由对方自行填写。

## 状态与查询
- get_app_state() -> 当前单元/网格/点数/种子/扰动/位移谱、当前 tab、主题、语言、最近仿真摘要。
- list_units() -> 16 种基元清单（6 位移谱预设 + 10 经典单元）。

## 结构编辑
- set_structure(unit, grid_x, grid_y, pts, seed, perturbation) -> 设置结构参数；unit 取自 list_units。
- set_line_displacements(displacements) -> 精确设置单线位移谱，[[dx,dy],...]，值为线长比例（±1.0 内）。
- set_perturbation(value) -> 节点扰动 0..1。
- randomize_seed() -> 随机新种子。

## 仿真
- run_percolation(stretch=2.0, alpha=0.05) -> 载荷路径渗流：渗流帧、序参量 P、骨架占比。
- run_stretch(stretch=2.2, use_bending=True, use_contact=True) -> 拉伸：帧数、峰值力/刚度/韧性、接触事件数。

## 设计 / 研究扩展
- run_inverse_design(target, budget, load_best) -> 拓扑锁定的逆设计；target 为曲线
  (J/C/linear/multi) 或标量 (max_peak/min_peak/max_stiffness/min_stiffness/max_toughness)；
  优化变量=当前结构点数的单线位移谱（pts 点 -> 2*pts 值）+ 扰动；load_best=True 回传结构板块。
- run_features(groups='all'|'structure'|'pore') -> 当前结构特征中文摘要。
- set_surface(obj, unit, preset, amplitude) -> 三维曲面板块：obj 填模型名片段
  (lung/heart/vans)，unit 面内图案，preset 谱 (square/auxetic_bow/rhombic_bow/swirl)，amplitude 0..1.5。
- run_ml_training(n=40, epochs=80) -> 生成结构-性能数据集并训练代理模型，返回样本数/epoch/R2。

## 回放
- get_replay_summary() -> 探索日志摘要：记录数、agent/R0 簇曲线末值、最大新颖度、top5 发现。
- load_replay_structure(index) -> 按索引把某条记录的结构载入结构板块。

## 导出 / 系统
- export_structure(format='json'|'svg', path=None) -> 导出当前结构；path 省略时用默认存档位置。
- set_tab(tab) -> structure/percolation/stretch/features/ml/design/replay/surface。
- set_theme(mode='dark'|'light')、set_lang(lang='zh'|'en')。

## 调用示例（自然语言即可）
- “用 chiral 4x4、扰动 0.1，跑渗流和拉伸，然后以韧性最大做逆设计并导出 SVG。”
- “把当前结构铺到 heart 曲面上，幅度 1.2，用 hexagon 图案。”
- “分析当前结构的几何与孔隙特征。”


## 2.7 内置完整流程

`start_print_workflow(samples=10, iterations=20, curved=False, open_slicer=True)` 串联结构、物理标注、训练、C型逆设计、物理复核、曲面、毫米实体及导出。参数以工具实现为准。使用 `get_print_workflow_status()` 查询，`stop_print_workflow()` 取消。离线输入“准备打印”或“完整流程”亦可启动；运行时输入“停止流程”取消。界面不显示专门的流程按钮。终点是文件准备和Bambu Studio交接，不代表已经物理打印。


## 2.8 接口增补

- RunConfig.weld_intersections=False：初始二维中心线交点焊接；拆边保留纤维身份，缓存与未焊接结果隔离。实现：fslab/welding.py。
- SurfaceTab 的每个 OBJ 可设置 target_faces（12–10000）及 subdivision_levels（0–3），细分输出受 10000 面预算约束。四边面不保证严格正方形；超预算明确报错。
- DesignTab.follow_type 默认开启；model_btn 为默认关闭的模型加速复选框。模型来自 MLTab.trained_model / trained_unit，固定训练结构类型。现有模型预测标量，J/C 曲线目标使用物理仿真。
- PrintWorkflow.start(samples=10, iterations=20, curved=False, open_slicer=True, target='C', stretch=2.0, surface_model=None, create_demo=False)。录像预设使用 300 样本、60 次预算、J、金字塔。
- AI 本地指令前缀“请执行决赛录像流程”无需联网。生成学习数据支持 10–2000 个真实物理标注样本；set_surface 默认跟随当前结构。
- open_in_bambu 仅接受 STL/OBJ。独立导出仍可保存 3MF，但不作为拓竹交接格式。
- 详见 docs/METHODS_2_8.md 和 docs/finals/AI_RECORDING_2_8.md。


## 2.9 接口更新

- StructureFactory.effective_spectrum()：返回按seed和perturbation比例扰动后的共享位移谱；零值保持零。
- fslab.solid_process.build_isolated(network, settings, progress=None, stop_cb=None)：隔离进程、可取消、600秒超时，返回FiberSolid；内部入口run.py --solid-worker。
- MainWindow.tab_manufacturing：位于曲面之后。open_source(curved)选取当前网络，mount(dialog)可承接工作流成品。
- AI set_tab增加manufacturing；离线“打开制造”进入制造工作区。
- SimTab._export_gif(silent=None)：异步1600×1000 GIF，_gif.active表示进行中，cancel()中止；平均48FPS。
- 界面默认焊接开启、接触关闭，RunConfig历史默认保留；模型训练标签使用其自身显式配置。
- 构建增加Pillow12.2.0，许可证已在assets/licenses/pillow。细节及限制见docs/METHODS_2_9.md。

## 3.0 recording trace

The explicit finals-recording command parses requested unit and evaluation budget instead of fixed hexagon/60 values. Workflow stages emit actual tool arguments/results and current page state. Solid/export/slicer stages remain on the manufacturing page. See finals/AI_RECORDING_3_0.md for the tested square/300/J200 command.
