<table>
<tr><td align="center">
<img src="docs/media/ui/homepage-cover-v2.svg" alt="FiberNet and FiberScope" width="820">
</td></tr>
</table>

# FiberNet

Generate fiber networks, simulate their response, and reuse the same structure for analysis, learning, or fabrication.

[Start here](#quick-start) · [PyPI](https://pypi.org/project/fibernet/) · [Examples](examples/README.md) · [中文](#fibernet-zh)

<a name="fibernet-zh"></a>

<details>
<summary>中文</summary>

<!-- locale:zh:start -->
## 快速开始

需要 Python 3.9 或更高版本。安装 [PyPI 正式版](https://pypi.org/project/fibernet/) 后即可运行内置示例，不用下载源码、准备输入文件或使用 GPU。

```bash
python -m pip install fibernet
python -m fibernet.examples.basic_usage --output-dir fibernet_demo
```

示例会生成纤维网络，计算小变形响应，并保存结构、图片和模拟结果。

<details>
<summary>从源码运行仓库里的算例</summary>

**开发版源码：4.24.0**

```bash
git clone https://github.com/GellmanSparrowS/fibernet.git
cd fibernet
python -m pip install -e .
python -m examples.open_source_workflow --output-dir demo_output
```

其他流程和可选依赖见[算例索引](examples/README.md)。仓库算例可能比 PyPI 正式版更新。

</details>

## 按研究问题选算例

- **生成二维或三维纤维网络**：[生成参数与示例](docs/METHODS_3D_GENERATION_PARAMETERS.md)
- **算变形、比较材料，或模拟熔合接点**：[梁模型](docs/API_LAYERS.md) · [三维实体力学](docs/TET4_STATIC.md)
- **看整个三维网络受力后电阻怎么变**：[360 杆三维力电算例](docs/DEFORMED_TRANSPORT_PREPARATION.md)
- **分析网络、构建数据集或训练代理模型**：[网络分析](examples/snapshot_features_workflow.py) · [物理量学习流程](examples/physical_learning_workflow.py)
- **沿用同一个 `StructureGraph` 导出制造路径**：[闭合路径](docs/MANUFACTURING_GRAPH_MAPPING.md) · [多段路径](docs/MULTISTROKE_MANUFACTURING.md)

力电算例在变形后的几何上重算电阻，不包含材料本征压阻，也不把电学结果反馈给力学。其他模块见[功能与接口导航](docs/api/WORKFLOW_MAP.md)。

<img src="docs/media/three_dimensional_topologies.gif" alt="从不同角度观察固定的 octet、diamond 和 gyroid 纤维网络" width="750">

*从不同角度看 octet、diamond 和 gyroid 网络。[更多动图及来源](docs/MEDIA_GALLERY.md)。*

## FiberScope 桌面应用

[下载 FiberScope Windows x64 版](https://github.com/GellmanSparrowS/fibernet/releases/tag/v4.2.0) · [桌面指南](FiberScope/README.md)

完整解压 ZIP 后打开 `FiberScope.exe`。桌面程序和 Python 包独立安装。

[更新记录](CHANGELOG.md) · [参与贡献](.github/CONTRIBUTING.md) · [MIT 许可证](LICENSE)
<!-- locale:zh:end -->

</details>

<details open>
<summary>English</summary>

<!-- locale:en:start -->
## Quick start

Requires Python 3.9 or newer. Install the [published package](https://pypi.org/project/fibernet/) and run the built-in example. No checkout, input files, or GPU are needed.

```bash
python -m pip install fibernet
python -m fibernet.examples.basic_usage --output-dir fibernet_demo
```

The example generates a network, solves its small-deformation response, and saves the graph, a figure, and simulation results.

<details>
<summary>Run examples from a source checkout</summary>

**Development checkout: 4.24.0**

```bash
git clone https://github.com/GellmanSparrowS/fibernet.git
cd fibernet
python -m pip install -e .
python -m examples.open_source_workflow --output-dir demo_output
```

See the [example index](examples/README.md) for other workflows and optional dependencies. The repository examples may be ahead of the published package.

</details>

## Choose a workflow

- **Generate a 2D or 3D network**: [Generation parameters and examples](docs/METHODS_3D_GENERATION_PARAMETERS.md)
- **Simulate deformation, compare materials, or model welded joints**: [Beam models](docs/API_LAYERS.md) · [3D solid mechanics](docs/TET4_STATIC.md)
- **See how a complete 3D network's resistance changes under load**: [360-member mechanoelectric example](docs/DEFORMED_TRANSPORT_PREPARATION.md)
- **Analyze structure, build a dataset, or train a surrogate**: [Network analysis](examples/snapshot_features_workflow.py) · [Physical learning workflow](examples/physical_learning_workflow.py)
- **Export the same `StructureGraph` for fabrication**: [Closed routes](docs/MANUFACTURING_GRAPH_MAPPING.md) · [Multiple strokes](docs/MULTISTROKE_MANUFACTURING.md)

The mechanoelectric example recomputes resistance on the deformed geometry; it does not model intrinsic piezoresistivity or electrical feedback into mechanics. For other modules, see the [module and API guide](docs/api/WORKFLOW_MAP.md).

<img src="docs/media/three_dimensional_topologies.gif" alt="Camera views of fixed octet, diamond, and gyroid fiber networks" width="750">

*Octet, diamond, and gyroid networks shown from different angles. [More animations and their sources](docs/MEDIA_GALLERY.md).*

## FiberScope desktop app

[Download FiberScope for Windows x64](https://github.com/GellmanSparrowS/fibernet/releases/tag/v4.2.0) · [Desktop guide](FiberScope/README.md)

Extract the entire ZIP and open `FiberScope.exe`. The desktop app is separate from the Python package.

[Changelog](CHANGELOG.md) · [Contributing](.github/CONTRIBUTING.md) · [MIT license](LICENSE)
<!-- locale:en:end -->

</details>
