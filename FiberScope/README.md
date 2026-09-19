# FiberScope 3.0

纤维网络材料交互设计软件：结构生成 → 拉伸仿真 → 特征分析 → 机器学习 → 逆向设计 → 三维曲面 → 制造导出。

## 直接运行

从 [GitHub Releases](https://github.com/GellmanSparrowS/fibernet/releases/tag/fiberscope-v3.0.0) 下载 Windows x64 版，解压并运行 FiberScope.exe。无需安装 Python。AI 联网对话需自行配置服务；核心生成、仿真、学习和确定性录像流程可离线使用。

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
