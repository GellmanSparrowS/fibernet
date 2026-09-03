"""Single TEXT table for all static UI strings: key -> (zh, en)."""

TEXT = {
    "app_title": ("FiberScope · 纤维网络超材料探索台", "FiberScope · Fibrous Metamaterial Explorer"),
    "tab_percolation": ("载荷路径渗流", "Load-path percolation"),
    "tab_more": ("后续模块", "More modules"),
    "tab_more_body": ("拉伸原位的、模式分解、AI 逆设计、探索回放等模块开发中。",
                      "Live stretch, mode decomposition, AI inverse design and exploration replay are under development."),
    "structure_card": ("结构 (可探索部分)", "Structure (explorable)"),
    "fixed_card": ("固定部分", "Fixed part"),
    "unit": ("单元类型", "Unit type"),
    "spectrum": ("变形谱", "Spectrum"),
    "grid": ("网格", "Grid"),
    "pts_per_side": ("每边内点数", "Points per side"),
    "perturbation": ("节点扰动", "Node jitter"),
    "seed": ("随机种子", "Seed"),
    "stretch": ("目标拉伸比", "Target stretch"),
    "quantile": ("张力阈值 \u03b1", "Tension threshold \u03b1"),
    "run_btn": ("生成并仿真", "Build & simulate"),
    "running": ("仿真中…", "Simulating..."),
    "cached": ("命中缓存", "cache hit"),
    "ready": ("就绪", "Ready"),
    "sim_failed": ("仿真失败", "Simulation failed"),
    "frame": ("帧", "Frame"),
    "strain": ("拉伸比", "Stretch"),
    "perc_curve_card": ("渗流曲线", "Percolation curve"),
    "perc_order": ("渗流序参量 P", "Order parameter P"),
    "backbone": ("骨架边占比", "Backbone fraction"),
    "perc_at": ("渗流时刻", "Percolation at"),
    "never": ("窗口内未渗流", "not percolated in window"),
    "legend_active": ("承载边", "load-bearing"),
    "legend_span": ("渗流骨架", "spanning backbone"),
    "tab_replay": ("探索回放 · 参照系", "Exploration replay · reference"),
    "tab_features": ("特征分析", "Feature analysis"),
    "tab_surface": ("三维曲面", "3D surface"),
    "tab_ml": ("机器学习", "Machine learning"),
    "replay_stats": ("统计", "Stats"),
    "replay_feed": ("发现事件流", "Discovery feed"),
    "replay_clusters": ("行为簇发现数", "Behavior clusters found"),
    "replay_novelty": ("累计最新颖度", "Running max novelty"),
    "show_agent": ("新颖性搜索 agent", "novelty-search agent"),
    "show_r0": ("R0 均匀随机", "R0 uniform random"),
    "tab_design": ("AI 逆设计", "AI inverse design"),
    "target_label": ("目标（曲线/标量）", "Target (curve/scalar)"),
    "budget": ("评估预算", "Eval budget"),
    "best_curve": ("当前最优 vs 目标", "Best vs target"),
    "convergence": ("收敛 (最优距离)", "Convergence"),
    "eval_log": ("评估日志", "Eval log"),
    "tab_stretch": ("拉伸原位 · 模式分解", "Live stretch · mode split"),
    "force_curve": ("力 - 拉伸曲线", "Force - stretch"),
    "mode_split": ("变形模式能量占比", "Mode energy split"),
    "m_axial": ("拉伸", "stretching"),
    "m_bend": ("弯曲", "bending"),
    "m_contact": ("接触", "contact"),
    "phys_bending": ("纤维弯曲刚度", "fiber bending stiffness"),
    "phys_contact": ("纤维间接触", "fiber-fiber contact"),
    "play_btn": ("播放", "Play"),
    "pause_btn": ("暂停", "Pause"),
    "theme_btn": ("浅色", "Light"),
    "lang_label": ("语言", "Lang"),
    "fixed_physics": ("物理引擎: numpy 质点-弹簧 (轴向+弯曲+接触) · 准静态增量拉伸",
                      "Physics: numpy mass-spring (axial+bending+contact) · quasi-static incremental stretch"),
    "fixed_bc": ("边界: 左端夹持固定 · 右端位移加载", "BC: left grip fixed · right grip displaced"),
    "tab_structure": ("结构生成", "Structure studio"),
    "struct_params": ("结构参数 · 可探索部分", "PARAMETERS · EXPLORABLE"),
    "struct_stats": ("实时统计", "LIVE STATS"),
    "struct_gallery": ("单元画廊", "UNIT GALLERY"),
    "struct_synced": ("已同步到各仿真板块", "synced to simulation tabs"),
    "struct_hint": ("结构在「结构生成」板块调整", "edit structure in the Structure tab"),
    "tagline": ("纤维网络超材料 · 可计算可探索环境",
                "fibrous metamaterials · a computable exploration environment"),
    "version_chip": ("V1.0", "V1.0"),
    "legend_inactive": ("未激活", "inactive"),
    "legend_load": ("承载边", "load-bearing"),
    "legend_span": ("渗流骨架", "spanning backbone"),
    "legend_grip": ("夹持端", "grip"),
    "legend_comp": ("压缩", "compression"),
    "legend_neutral": ("近零", "near zero"),
    "legend_tens": ("拉伸", "tension"),
    "legend_contact": ("新接触", "fresh contact"),
    "dice_btn": ("随机种子 ⚄", "Random seed ⚄"),
    "disp_card": ("逐点微调 · 边长百分比", "PER-POINT TUNE · % EDGE"),
    "disp_none": ("该单元的中间点不可手动调整（直梁细分）。",
                  "This unit has no adjustable intermediate points (straight beams)."),
    "disp_none_pts0": ("每边内点数为 0 时无可调点；增大点数后可编辑。",
                  "No intermediate points at pts=0; raise the count to edit."),
    "disp_clear": ("清零全部位移", "Reset all displacements"),
    "explain_card": ("图例说明", "LEGEND"),
    "explain_perc": ("灰 = 未激活；浅蓝 = 与夹持端连通的承载边（力只沿连续路径传导）；"
                     "蓝→青 = 渗流骨架（颜色深浅表示距两端深度）；琥珀 = 夹持端。"
                     "仿真完成后从第 0 帧自动播放；迟滞阈值防止载荷路径闪烁。",
                     "gray = inactive; dim blue = load-bearing edges anchored at the grips "
                     "(force flows along continuous paths only); blue->cyan = spanning backbone "
                     "(shade = depth from the grips); amber = grips. Autoplays from frame 0; "
                     "hysteresis keeps the load path stable."),
    "explain_stretch": ("红 = 拉伸 / 青 = 压缩（线宽 ∝ 应变）；红点 = 纤维间接触事件"
                        "（亮红 = 本帧新接触，淡红 = 历史接触）；琥珀 = 夹持端。"
                        "右列为力-拉伸曲线与 拉伸/弯曲/接触 三通道能量占比。",
                        "red = tension / cyan = compression (width ~ strain); red dots = "
                        "fiber-fiber contact events (bright = fresh this frame, dim = "
                        "history); amber = grips. Right: force-stretch curve and "
                        "axial/bend/contact energy split."),
    "explain_replay": ("离线探索回放：新颖性搜索 agent 与 R0 均匀随机在同预算下各 120 次评估。"
                      "散点 = 行为空间投影（蓝 agent / 灰 R0 / 橙框 = 当前）；点击散点或左侧事件流可跳转，"
                      "悬停查看细节；右列 = 行为簇发现曲线（三角 = 发现新簇时刻）与累计最新颖度。"
                      "「载入此结构到仿真」把当前记录的结构送入结构板块继续探索。",
                      "Offline exploration replay: novelty-search agent vs R0 random, 120 evals "
                      "each at equal budget. Scatter = behavior-space projection (blue agent / "
                      "gray R0 / orange ring = current); click a point or a feed entry to jump, "
                      "hover for details; right = cluster-discovery curve (triangles mark new "
                      "clusters) and running max novelty. The load button sends the current "
                      "structure into the structure studio."),
    "initial_struct": ("初始结构（拉伸前）", "Initial structure (pre-stretch)"),
    "stretched_struct": ("拉伸后结构", "Stretched structure"),
    "apply_struct": ("发送到结构板块", "Send to structure studio"),
    "design_cfg": ("优化设置", "OPTIMIZER SETUP"),
    "behav1": ("行为维度 1", "behavior dim 1"),
    "behav2": ("行为维度 2", "behavior dim 2"),
    "edit_unit": ("编辑基元 · 拖拽节点", "Edit unit · drag nodes"),
    "edit_done": ("完成 · 返回总览", "Done · back to preview"),
    "edit_line": ("编辑基元 · 拖拽单线", "Edit unit · drag one line"),
    "edit_line_hint": ("拖动参考线上的控制点（占线长比例）；位移谱会按各自方向复制到每一条纤维线，整个网格的基元同步变化；下方为基元预览。",
                        "Drag control points on the reference line (fraction of line length). The profile is replicated onto every fiber line in its own orientation; the unit preview below updates live."),
    "prim_preview": ("基元预览 · 单胞", "Unit preview · 1 cell"),
    "export_json_btn": ("导出 JSON", "Export JSON"),
    "export_svg_btn": ("导出 SVG", "Export SVG"),
    "ai_btn": ("AI 助手", "AI Assistant"),
    "ai_title": ("AI 助手 · 全接口操控", "AI Assistant · full control"),
    "ai_key_hint": ("输入你的 DeepSeek API Key。",
                    "Enter your DeepSeek API key."),
    "ai_placeholder": ("让 AI 帮你设计/仿真/导出…", "Ask the AI to design/simulate/export..."),
    "disp_row": ("逐点数值 · 线长百分比 (X/Y 对)", "Per-point values · % edge (X/Y pairs)"),
    "ai_thinking": ("AI 思考中…", "AI thinking…"),
    "ai_done": ("完成", "done"),
    "ai_fallback": ("已执行: ", "Executed: "),
    "tab_sim": ("原位仿真 · 渗流与拉伸", "Live sim · percolation & stretch"),
    "view_label": ("视图", "View"),
    "view_perc": ("载荷路径渗流", "Load-path percolation"),
    "view_strain": ("应变 · 接触", "Strain · contacts"),
    "replay_load": ("载入此结构到仿真", "Load this structure into sim"),
    "replay_toggle": ("探索回放 ▾", "Exploration replay ▾"),
    "exported": ("已导出", "Exported"),
    "edit_hint": ("直接拖动空心圆点（一条纤维线上的点同色）；松手后周期复制到全网格。"
                  "虚线为原始位置。",
                  "Drag the open handles (points on one fiber line share a color); "
                  "release to apply periodically to the full grid. Dots = original."),
}

_LANG = "zh"


def set_lang(lang: str):
    global _LANG
    _LANG = "en" if lang.lower().startswith("en") else "zh"


def get_lang() -> str:
    return _LANG


def tr(key: str) -> str:
    zh, en = TEXT[key]
    return zh if _LANG == "zh" else en
