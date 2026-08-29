# FiberScope · 纤维网络超材料交互探索台

GOAI 赛道三（开放探索赛题）复赛交付物：最小可运行探索环境的交互前端。
科学问题：Eulerian 可制造性约束下规则纤维网络的非线性力学响应谱
（详见 `../初赛/问题定义文档 杨云浩.docx` 与 REFINe 手稿）。

## 运行

```
conda activate ml310            # Python 3.10, 依赖见下
python run.py                   # 交互界面
python run.py --theme light --lang en
python run.py --smoke           # 离屏冒烟
```

或打包版：`python scripts/build_exe.py` → `dist/FiberScope/FiberScope.exe --smoke`。

依赖：numpy, networkx, PySide6, pyqtgraph, (开发/测试: PIL)；
物理核心 `fslab/engine2.py` 为纯 numpy，无 taichi/torch 依赖。
结构生成复用本地 `../fibernet`（MIT，自有项目；经 `FIBERNET_ROOT` 环境变量可改路径）。

## 五个标签页

0. 结构生成（独立板块）：单元画廊（12 种拓扑缩略图）、参数面板、
   实时结构预览与节点/边统计；结构规格实时同步到各仿真标签页。
1. 载荷路径渗流：拉伸过程中承载边从两个夹持端逐渐变蓝、最终渗流连接；
   右侧小窗为渗流序参量 P 与骨架边占比曲线；底部传输条
   （播放/暂停 + 倍速 + 帧滑块）在仿真完成后自动播放动态过程。
2. 拉伸原位 · 模式分解：应变着色画布（红=拉伸/青=压缩/红点=新接触），
   力-拉伸曲线与 弯曲/拉伸/接触 三通道能量占比堆叠图。
3. AI 逆设计：两阶段优化（12 种拓扑筛选 → CEM 位移场精调）实时收敛，
   目标曲线 vs 当前最优曲线 + 评估日志流。
4. 探索回放 · 参照系：回放 `data/exploration_log.jsonl`
   （新颖性搜索 agent vs R0 均匀随机，各 120 次评估，同预算对照）。

## 环境设计（固定 / 可探索 / 反馈）

- 固定：engine2 质点-弹簧物理（轴向弹簧 + 度-2 节点邻次近邻弯曲弹簧 +
  节点接触排斥）；左端夹持、右端位移加载（与 fibernet 引擎同口径的
  百分位夹持）；dt=1e-5、16000 步、能量账本（轴向/弯曲/接触/动能/输入功）。
- 可探索：结构生成板块的单元类型（12）、网格、每边内点数、节点扰动、
  种子；仿真板块的目标拉伸比、张力阈值 α、物理开关（弯曲/接触）。
- 反馈：逐帧轨迹、边应变、夹持反力、三通道能量、接触事件、渗流序参量。

## 复现

- 随机种子：所有标签页种子控件默认 7；探索日志生成 `python scripts/run_exploration.py --budget 120 --seed 0`（断点续跑：按 JSONL 行数恢复）。
- 仿真缓存：`_cache/*.npz`，原子写入（tmp+rename），参数哈希命名；删缓存即重算。
- 测试：`tests/selftest.py`（渗流）、`tests/selftest_engine2.py`（物理验证：
  Gibson-Ashby 弯曲/拉伸对比、接触激活、能量守恒 <1%、确定性）、
  `tests/selftest_inverse.py`（逆设计改进）、`tests/gui_smoke.py`（离屏像素回归）。

## 已验证的物理信号（示例）

- 渗流相变：square/triangle/kagome/reentrant/diamond 在拉伸窗口内出现
  P=0 平台→突升→饱和；honeycomb/chiral（弯曲/旋转主导）窗口内不渗流
  ——拉伸主导 vs 弯曲主导的载荷传递差异。
- 接触硬化：chiral 单元旋转扫掠产生新接触（最多数十对），接触能量通道激活。
- 逆设计：拓扑筛选 + 场精调逐级降低目标曲线距离。

## 基于已有项目的说明

- 原项目：fibernet（https://github.com/GellmanSparrowS/fibernet, MIT, v4.0.5），
  仅复用其结构生成器 `gen.pattern`（纯 numpy/networkx）。
- 本仓库新增：engine2 物理核心、渗流分析、逆设计、探索管线、全部 GUI 与测试。
