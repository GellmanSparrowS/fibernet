# FiberScope PROGRESS

## Status (2026-09-03 · M20 便携副本确认)
- [x] 依赖检查：onedir 全量 DLL 已随包，缺依赖扫描 = 0
- [x] 最小 PATH（仅 System32/Windows）冒烟 PASS，确认不依赖 ml310 环境
- [x] 构建便携副本 zip：FiberScope_V1.0_win64.zip（含 exe + _internal + python310.dll + icuuc.dll）
- [x] 清理暂存目录与 smoke 文件

## Issues / notes
- 直接只拷 FiberScope.exe 会缺 DLL；必须连同整个 FiberScope 文件夹（或解压 zip）使用

## Status (2026-09-03 · M19 收尾两文件夹)
- [x] 六边形与蜂窝只保留 hexagon，移除 honeycomb
- [x] 提交文件夹 A：code/APP/data/README/figures（代码+EXE）
- [x] 提交文件夹 B：AI4R_OPEN_ml-bio_文档（仅 PPT+docx+PDF）
- [x] 文档仅用 figures/2 六张新图，图注无空格，标题/正文宋体
- [x] exe 重建 + 冻结冒烟 PASS；提交包 code/APP 同步

## Next
- [ ] git commit / 清理 smoke 文件

## Issues / notes
- 旧 zip 仍未重打包（等用户确认）

## Status (2026-09-03 · M18 结构合并 / 变形 / 逆设计 / AI / 文档)
- [x] 方形谱预设合并进 square（BASE_UNIT_KEYS=11；square 下拉变形谱 6 项）
- [x] kagome -> voronoi；经典单元按“每条原始线”旋转位移谱变形
- [x] 画廊 1×1 与编辑预览/主视图一致
- [x] 大网络画布性能：>800 节点不再逐点画圆
- [x] 逆设计：删 multi 目标；CEM 宽幅探索后收缩；LD_AMP 0.45；低预算可跑
- [x] AI 助手：内置 SKILL_DOC + get_skill_doc 工具；工具调用自动跳 tab
- [x] 版本 V1.0
- [x] 复赛文档 docx+PDF 重渲染，替换 figures/2 面板图（8 图）
- [x] selftest / engine2 / inverse / gui_smoke 全绿；冻结冒烟 PASS

## Next
- [ ] 提交包 code/APP 已同步（本轮）
- [ ] git commit + 清理

## Issues / notes
- 接触计算存在（engine2 罚函数 + features 交叉统计）；几何接触为零属正常（直梁无交叉）
- 旧 zip 仍未重打包（等用户确认）

## Status (2026-09-03 · M17 一致性 / 指纹 / 逆设计 / AI 跳转)
- [x] 画廊改回 1×1，与编辑基元预览一致（P1 谱系 + classic 单元）
- [x] kagome 替换为 voronoi（fibernet 周期 Voronoi，可编辑每线）
- [x] 结构指纹：左侧小结构图 + 右侧雷达图平衡展示，标签限 3 字
- [x] AI 逆设计：目标曲线/标量说明在启动时立即显示；fixed_pts 默认 5；LD_AMP 0.45
- [x] AI 助手工具调用前自动跳转对应 tab
- [x] 版本 V1.0
- [x] selftest / engine2 / inverse / gui_smoke 全绿

## Next
- [ ] 重建 exe + 冻结冒烟
- [ ] 提交包同步 + git commit + 清理

## Issues / notes
- voronoi 依赖 scipy.spatial（fibernet 函数级导入，冻结包已含）
- 旧 zip 仍未重打包（等用户确认）

## Status (2026-09-03 · M16 结构/UI 收尾优化)
- [x] 单元显示名去下划线（UNIT_DISPLAY 中英映射，combo/画廊/导出/spec_text 统一）
- [x] P1 谱差异加大（bow 0.56 / zigzag 0.48 / swirl 0.46 / pinwheel 0.42）
- [x] 画廊改为 2×2 周期铺贴（kagome/honeycomb/triangle 等更易辨认）
- [x] LineEditor 滚轮缩放 + 基元预览固定比例 + 虚线参考方形（100%=整条边可见）
- [x] 能量占比图例彩色区分（轴向蓝/弯曲琥珀/接触红，富文本）
- [x] ML loss 曲线修复：appendData 失效 -> setData 缓冲
- [x] 特征指纹雷达图（8 维归一化）+ 实时更新
- [x] 特征计算对齐参考 Features.py（orient/degree 熵原始公式、非加权 anisotropy）
- [x] AI 逆设计：探索回放折叠为小窗（默认隐藏），收敛曲线最小高度 + autoRange
- [x] 三维曲面：仅保留 3 个 500 OBJ，新增导入 OBJ、复位视图，before 窗 250×190
- [x] selftest / engine2 / inverse / gui_smoke 全绿

## Next
- [ ] 重建 exe + 冻结冒烟
- [ ] 提交包 code/ + APP/ 同步
- [ ] git commit / 清理过程文件

## Issues / notes
- ast.parse 对带 BOM 文件报 FEFF，实际 py_compile/import 正常
- 旧 zip 仍未重打包（等用户确认）

## Status (2026-09-03 · M15 APP 优化 + 收尾)
- [x] 默认浅色主题
- [x] 基元零内置扰动：structure.py n_intermediate_for 改节点数差法
- [x] 编辑位移范围 ±100%（边长比例）
- [x] 仿真固定视口（左→右拉伸，不整体放大）
- [x] 右栏紧凑 + 科学计数法 SciAxis
- [x] 逆设计 pts 联动（2*pts+1 维，decode_line_params(x, pts)）
- [x] Tab 顺序重排：structure/sim/features/ml/design(+replay)/surface
- [x] AI 助手 SSL 重试 + 不校验证书回退
- [x] 接口文档 docs/AI_TOOLS.md
- [x] 三维曲面 apply_structure 跟随当前结构 + before/after 对照（Curie，已接 main.py）
- [x] 特征仪表盘 35 标量 + 直方图卡片（Laplace，features.py/features_tab.py）
- [x] P1 结构生成逐步可视化 notebook（Godel，E:/GOAI/复赛/P1_Gen_dataset_regular_net.ipynb，25 cells 已执行）
- [x] ICU 修复：打包 Windows system icuuc.dll（Qt6Core ucnv_open 入口点错误根因）
- [x] selftest / selftest_engine2 / selftest_inverse / gui_smoke 全绿

## Next
- [x] 重建 exe + 冻结冒烟 PASS（dist + 提交包 APP 双跑）
- [x] 提交包 code/ + APP/ 同步（含 icuuc.dll、notebook、AI_TOOLS.md）
- [x] 更新 files_registry.json（30 项）/ PROGRESS / 清理 _qa、_tmp、smoke 产物
- [x] git commit（本提交之后）

## ICU 修复细节
- 根因：PyInstaller 收集了 anaconda 的 icuuc.dll（仅导出 ucnv_open_58 版本后缀符号），
  Qt6Core 需要纯 ucnv_* 符号 -> 启动报「入口点 ucnv_open 无法定位」。
- 方案：build_exe.py 删除收集到的 icu*.dll，改为打包 Windows System32 icuuc.dll
  （自包含，导出 Qt6Core 所需的全部 20 个符号），复制后清除只读属性。

## Issues / notes
- Laplace 子代理限流中断，但其 fslab/features.py 与 studio/features_tab.py 已完整、语法通过并验证可运行
- 旧 zip（8/29）早于 ICU 修复，需等用户确认后重打包


## Status (2026-09-03 · M14 最终轮 APP 大改)
- [x] S1 structure.py v4 双路径（P1 谱系 6 单元 + 经典 10 单元，零内置扰动）
- [x] S2 结构 Tab：默认 square、pert=0、画廊显示基元本形
- [x] S6 AI 助手全量重写：气泡+工具卡、思考动画、完成反馈、主题重渲
- [x] 特征分析 Tab（fslab/features.py + studio/features_tab.py，16 特征中文名+3 直方图+区域选择+批量）
- [x] 三维曲面 Tab（fslab/surface3d.py + studio/surface_tab.py，6 OBJ、拖动旋转、<7ms 重算）
- [x] 机器学习 Tab（fslab/mlmodel.py + studio/ml_tab.py，断点数据集+动态 Loss+早停+R2 散点）
- [x] main.py 接线 7 Tab + AI 面板加宽 + i18n 键
- [x] 修复 CLASSIC_UNITS 缺 reentrant/chiral/star/diamond（gui_smoke chiral 接触断言失败根因）
- [x] AI 新工具对齐真实 API：run_features/set_surface(驱动 Tab 控件)/run_ml_training
- [x] 测试全绿：selftest / gui_smoke / selftest_engine2 / selftest_inverse + AI 三工具离屏验证

## Next
- [x] 重建 exe + 冻结冒烟 PASS（ICU 冲突修复：剔除 anaconda icuuc/icudt78）
- [x] 提交包 code/ + APP/ 同步（robocopy /MIR）、README 增七板块表
- [x] build_doc.py 增补（七板块/§2.3 扩展工具/§3.1 双路径生成/17 工具表/§7.3/§8）重渲染 13 页 QA 通过
- [x] 过程文件清理、git 提交 320dc40 / 2ef2cd6
- [ ] zip 重打包（等用户确认）

## Issues / notes
- 子代理交付的 API 与初规格有出入，已在 ai_assistant 侧对齐（FEATURE_ZH 为 key->中文；MLP(hidden,seed)+r2_score 模块函数）
- OBJ 路径绝对指向 E:\GOAI\复赛\*.obj（只读），打包后评委机器路径不存在 → 曲面 Tab 已做缺文件容错？待查


## Status (2026-08-30 凌晨)
- M1-M13 完成；源码测试全绿；exe 重建+冻结冒烟进行中
- 提交材料齐备：可运行环境(exe) + 探索日志(data/) + 参照系(agent vs R0) + README/复现

## 本轮已完成（用户第六轮反馈 M13）
- [x] AI 逆设计结果回填逆设计板块：DesignTab.show_external（日志/收敛/最优曲线/状态）
- [x] DispRow 改 QGridLayout（每行 3 对 P_k X/Y），pts=6 不溢出不缺盒
- [x] 基元去内在随机性：voronoi 移出于可选预设（pert=0 仍乱的唯一单元）；
      探索日志保留历史记录，load_spec/load_replay_structure 仍可载入
- [x] AI 聊天主题切换重渲：refresh_theme() 用当前调色板重放历史消息
      （修复深色下发的消息切浅色后过浅）
- [x] 文档/注册表同步

## git
- 待提交 M13
- 历史: M9 4d694d0 / M10-M12 9ce73df

## Next
- [ ] 冻结冒烟确认 + 提交 + 汇报
- [ ] zip dist/FiberScope 提交包（可选）

## Issues / notes
- chiral 的 len_cv 高属设计本身（手性韧带长短交替），周期规整，非随机
- AI 面板旧名 win.ai_dock 保留为别名（gui_smoke 依赖）

## Status (2026-08-29 晚 · 复赛文档轮)
- [x] 复赛说明文档全量重写（构建器存 E:\GOAI\复赛\build_doc.py）：摘要+八章+附录清单，图 1–5、表 2 个，共 12 页
- [x] 渲染 QA（Word→PDF→逐页 PNG 目检）通过；第 7 页留白用 §5.3 预算分配对照段填充（数字来自 exploration_log：接受率 13.3%、R0 max novelty 0.61、单元访问分布）
- [x] README 图号改 1–5、参照系章节改 §5.1；删除多余图 AI助手 力学模拟.png；清理 _doc_state.txt
- [x] 提交包 AI4R_OPEN_ml-bio 齐备（docx/README/APP/code/data/figures），等用户确认后打包 zip

## Next
- [ ] 用户确认文档后打包 AI4R_OPEN_ml-bio.zip
- [ ] 如需要：按复赛要求清单再核一遍提交项
