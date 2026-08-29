# FiberScope UI 维护文档

UI 全部位于 `studio/`，与物理核心 `fslab/` 解耦：`fslab` 不 import Qt。
视觉改动只应落在本目录 + `assets/`。

## 设计语言

- 分层表面：bg（页面）< panel（头部/状态栏）< card（卡片）< card2（内嵌控件）
- 主调 cyan（accent2）+ 辅 accent（sky）；warn=琥珀（夹持端/弯曲通道）；
  hot=红（拉伸/接触）；ok=绿（保留）
- 圆角：卡片 10、控件 7、chip 9（胶囊）
- 字体：Segoe UI / Microsoft YaHei UI；标题 19px、组标题 11px 大字距、
  统计值 17px、hint 11px

## 调色板与 QSS（studio/theme.py）

- `colors(mode)` 返回 dict；新增颜色需 dark/light 同时加
- `build_qss` 单一字符串；对象名约定：
  `#header #logo #title #tagline #chip #chip_accent #card #well #gallery
   #stat_value #stat_label #hint #explain #subtitle`
- `apply_plot_theme(pw, mode)`：pyqtgraph 图背景 + 坐标轴颜色，
  所有 PlotWidget 必须在 `set_mode` 里调用它（否则浅色主题残留深色）
- 滑块手柄用 `handle` 色（两主题皆近白），边框 accent

## 布局骨架

每个仿真 tab 三列 QSplitter：左控件（min 240 / max 320）、中画布+传输条、
右曲线（min 260 / max 400），`body.setSizes([270, 760, 330])` 定初值。
左列顺序：结构 chip 行 → 参数卡 → 运行按钮 → 进度条(QProgressBar %p%) → 状态。
右列顺序：曲线卡 → 图例说明卡（#hint 多行）→ stretch。
传输条：播放/暂停 + 倍速(0.5-4x) + 帧滑块 + 帧/拉伸比 chip；
`_on_done` 后 `_start_play()` 自动播放。
进度条由 `SimWorker.progress(int)` 驱动（engine2 `progress_cb` 每个增量回调）；
渗流 tab 在 `perc_frame<0` 时用 pg.TextItem 覆盖「窗口内未渗流」。
仿真 tab（sim_tab.py）合并渗流+拉伸：视图 combo（载荷路径渗流 / 应变·接触）
共享同一次 run；红点 = 纤维间接触事件（亮红 = 本帧新接触，淡红 = 历史）；
右列顺序：力-拉伸曲线 → 模式能量占比 → 渗流曲线（用户要求渗流在下）。

## 结构板块（structure_tab.py）

- 拥有共享 `StructureFactory`；`structure_changed` 信号 → 仿真 tab `set_spec`
- 单线编辑器 `LineEditor`：大手柄(9px/命中20px)+大画幅；下方 `DispRow`
  数值行（每点 X/Y 百分比 spinbox，与拖拽双向同步，AI 亦可精确设值）；
  参考线控制点（线长比例）→ `line_displacements`；
  structure.py 用 3×3 原型 builds 的纤维链（`chains_of`）生成位移表
  （键 = 基位置 mod CELL），周期复制到全网格 → 编辑一条线 = 编辑全部同类线
- perturbation 任意 pts 生效：seeded 均匀抖动（`_apply_jitter`，
  幅值 0.35·CELL·pert，默认 8% 即轻微可见变形），种子因此有意义
- 画廊为换行网格 QListWidget（IconMode+Wrapping，无横向滚动、不重叠），
  缩略图实时反映 pts/pert/line_displacements；默认扰动 8%（轻微变形更真实）；
  `load_spec()` 供 AI/逆设计/回放送入
- 基元本身不带随机性：voronoi（内在随机剖分）已移出于可选预设
  （探索日志仍保留其历史记录；`load_spec` 仍可载入）
- `DispRow` 数值行用 QGridLayout（每行 3 对 P_k X/Y），支持 pts=6 不溢出
- 导出：`fslab/exporter.py` export_json（spec+图）/ export_svg（按链着色矢量图）

## 画布（network_canvas.py）

- 三模式：static / percolation / strain；右上角图例 `set_legend([(hex,text)])`
- 接触标记：历史淡红(α80) + 本帧亮红；夹持端琥珀
- 像素缓存 key 含 (run, perc, static, frame, size, zoom, pan, mode, color_mode)

## 渗流语义（fslab/percolation.py，物理但直接影响视觉）

- 激活：strain > α·max；迟滞：已激活边在 strain > 0.6α·max 时保持
- 夹持连通裁剪：只保留与夹持端连通的活性连通分量 → 力路径从两端生长
- 骨架：同时连通两端的分量；BFS 深度着色（蓝→青）

## AI 助手（studio/ai_assistant.py）

- 右上角「AI 助手」按钮切换右侧面板（中央 QSplitter，可拖拽调宽，min 320）；
  × 收起、⚙ 设置；`win.ai_dock` 为旧名别名（测试用）
- 隐藏预设 SYSTEM_PROMPT（角色身份+功能总览+工作原则）；每次发送注入动态
  当前状态上下文（用户不可见）
- 反馈：运行中状态「AI 思考中…」；结束追加「完成 · tools n · Xs」；模型未给
  终答时自动补「已执行: …」摘要
- API Key 存 `~/.fiberscope/config.json`（每用户输一次；已存则永不再问）
- 工具注册表 = 全部内部接口：get_app_state/list_units/set_structure/
  set_line_displacements/set_perturbation/randomize_seed/run_percolation/
  run_stretch/run_inverse_design/export_structure/set_tab/set_theme/
  set_lang/get_replay_summary
- 工具在 GUI 线程执行（`_req` 信号 + threading.Event 桥），worker 线程等结果
- 聊天配色随主题（`colors(mode)` 角色映射）；`#aichat` 圆角卡底；
  主题切换时 `refresh_theme()` 用当前调色板重渲全部历史消息
- 回放工具：get_replay_summary 含 top_discoveries；load_replay_structure
  按 index 把记录结构送入结构板块

## 逆设计（fslab/inverse.py + design_tab.py）

- 目标：曲线 J/C/linear/multi + 标量 max/min peak/stiffness/toughness
  （`metrics_of` 只取加载相）
- stage2 CEM 优化单线点值（2*pts 维）+ perturbation；`best_spec` 可
  「发送到结构板块」（apply_structure 信号 → load_spec）
- `fixed_unit` 锁定：stage-1 只评当前结构单元（逆设计只改基元形状，
  不换拓扑）；DesignTab.set_spec 由 structure_changed 驱动
- `show_external(res, target)`：AI 工具跑完逆设计后把记录/收敛/最优曲线
  回填到本 tab，用户随时可见
- 中列含拉伸前初始结构小窗（canvas0 静态帧 0）

## 探索回放（replay_tab.py）

- 散点/事件流可点击跳转、悬停 tooltip；步骤 caption chip；
  行为簇发现曲线带三角发现标记；「载入此结构到仿真」→ load_spec
- 右列 plot 设最小高度防裁剪

## 新增 tab 流程

1. `studio/xxx_tab.py`：`__init__(mode)`、`set_mode`（调 apply_plot_theme）、
   `retranslate`（全部文案走 i18n.tr）
2. `i18n.py` 加 (zh, en) 键
3. `main.py` 注册 + retranslate + _apply_theme 列表
4. `tests/gui_smoke.py` 加回归段
5. 本文件补一段说明

## 图标与打包

- `assets/icon.ico|png`（PIL 生成脚本见 git 历史 `_tmp/make_icon.py` 思路：
  深色圆角 + 青色六边形 + 内部纤维环 + 琥珀夹持点）
- `scripts/build_exe.py` 传 `--icon` 并 `--add-data assets`；
  冻结端 `main._icon_path()` 走 `sys._MEIPASS`

## 验证

`python tests/gui_smoke.py`（离屏像素回归）+ `run.py --smoke`（含冻结版）。
改 QSS 后至少离屏截 dark/light 两主题各 tab 检查残留色。
