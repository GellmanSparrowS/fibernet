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
- run_features(groups='all'|'structure'|'pore'|'contact') -> 当前结构特征中文摘要。
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
- “算一下当前结构的接触组特征。”
