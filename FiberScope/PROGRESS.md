# FiberScope PROGRESS

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
