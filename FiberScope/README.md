# FiberScope 3.1

## Start here (English)

FiberScope is the desktop interface; FiberNet is the Python library. The
[4.2.0 release](https://github.com/GellmanSparrowS/fibernet/releases/tag/v4.2.0)
contains the documented FiberScope 3.1 Windows x64 build. Extract the entire
ZIP and launch `FiberScope.exe`; it does not need Python. The current source
checkout may contain newer library features than this desktop release.

For source use, run these commands from the repository root in a Python 3.10
virtual environment:

```bash
python -m pip install -e .
python -m pip install -r FiberScope/requirements.txt
python FiberScope/run.py --lang en --theme light
```

The pinned desktop dependencies were verified on Windows with Python 3.10.
A successful headless check on another platform does not establish native
window, graphics-driver, or packaged-executable compatibility there.

### First session

1. In Structure studio, choose a small topology and grid; set the seed to
   make the geometry reproducible
2. Open Live sim and choose a target stretch ratio (1.01 means 1% extension)
3. Select Build & simulate, then use playback to inspect the trajectory
4. In the tensile-recruitment view, vary the strain threshold, select
   Build & simulate again, and inspect which recruited components span both grips. Colors indicate thresholded
   edge strain and connectivity, not complete force transfer
5. Use Export to save results. The displayed reaction belongs to the reduced
   spring/bending/contact model and is not an experimentally calibrated force

The library's beam-FEM shortcut is a different model from this desktop reduced
solver. See the [model guide](../docs/METHODS_REDUCED_SOLVER_UNIFIED.md) and
[interface screenshots](docs/screenshots/README.md). Screenshots describe the
captured release and may retain older labels.

Core generation and simulation work offline. Online AI services need separate
configuration. Optional external SAC/TD3/DDPG training needs the dependencies
in `requirements-rl.txt` and a configured `FIBERSCOPE_TRAIN_PYTHON`; it is not
bundled into the executable.

### Source checks

```bash
python FiberScope/run.py --smoke
python FiberScope/scripts/audit_ui_layout.py --capture-dir demo_layout
python FiberScope/scripts/audit_simulation_ui.py --capture-dir demo_simulation_ui
```

These require the desktop dependencies, including Qt. The smoke command runs
offscreen workflows and exports; the layout audit captures selected pages. The simulation UI audit runs a small
network, checks playback/loop/speed controls, and verifies English/Chinese
control widths at two window sizes.
Inspect captures separately for clipping and readability. Neither command
alone certifies a Windows executable built on another machine.

## 中文指南


纤维网络材料交互设计软件：结构生成 → 拉伸仿真 → 特征分析 → 机器学习 → 逆向设计 → 三维曲面 → 制造导出。

## 直接运行

从 [GitHub Releases](https://github.com/GellmanSparrowS/fibernet/releases/tag/v4.2.0) 下载 FiberScope 3.1 Windows x64 压缩包，完整解压后运行 FiberScope.exe，无需安装 Python。AI 联网对话需自行配置服务；核心生成、仿真、学习和确定性录像流程可离线使用。SAC、TD3、DDPG 的外部训练依赖另行配置 Python 运行时，未打入桌面压缩包。

[魔搭在线体验](https://modelscope.cn/studios/GellmanSparrow/FiberScope) · [决赛录像指令](docs/finals/AI_RECORDING_3_0.md)

## 源码运行

在仓库根目录安装并运行（Python 3.10）：

~~~bash
python -m pip install -r FiberScope/requirements.txt
python FiberScope/run.py
~~~

源码支持本仓库布局，也支持原来的 FiberScope 与 fibernet 两个并列目录。自定义位置可设置 FIBERNET_ROOT，指向包含 fibernet 包的目录。

## 测试与构建

~~~bash
python FiberScope/scripts/run_tests.py
python FiberScope/scripts/build_exe.py
~~~

构建仅在 Windows 上执行；另需安装 requirements-dev.txt。完整外部 RL 使用 requirements-rl.txt 和 FIBERSCOPE_TRAIN_PYTHON；默认 CEM 和监督模型不需要外部 RL 环境。

## 3.0 重点

- 随机种子选择十个共享位移参数，幅度滑条实时缩放；手动编辑后继续按比例调整。
- 圆环由四圆弧和四连接段采样，保留连续制造所需的逻辑纤维身份。
- 制造随来源自动二维/三维，来源与路径播放同行、导出同行。
- AI 录像流程按方形、300 个物理样本、J 型目标 200 次预算执行，实际切换七页并显示工具结果。
- 默认焊接锚点、关闭纤维间接触；GIF 1600×1000、平均48FPS。
- 曲面保留立体比例；打印尺寸统一为毫米，三轴最大250 mm，默认纤维直径2 mm。

## 文档

- [3.0 方法](docs/METHODS_3_0.md)
- [扩展 API](docs/open_source/API.md)
- [AI 工具](docs/AI_TOOLS.md)
- [第三方说明](THIRD_PARTY_NOTICES.md)

制作：复旦大学高分子科学系 杨云浩。致谢世界人工智能开源大赛。
