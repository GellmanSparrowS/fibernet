<table>
<tr><td align="center">
<img src="docs/media/ui/homepage-cover-v2.svg" alt="FiberNet and FiberScope" width="820">
</td></tr>
</table>

# FiberNet

A Python library for generating fiber networks and studying how their architecture affects deformation and transport. Reuse the same network for analysis, learning, and fabrication planning.

[Quick start](#quick-start) · [Examples](examples/README.md) · [中文](#fibernet-zh)

<a name="fibernet-zh"></a>

<details>
<summary>中文</summary>

<!-- locale:zh:start -->
## 快速开始

用 Python 3.9 或更高版本，从 [PyPI](https://pypi.org/project/fibernet/) 安装。内置算例会生成一个网络，计算小变形响应，并把结构、图和结果存到 `fibernet_demo/`。

```bash
python -m pip install fibernet
python -m fibernet.examples.basic_usage --output-dir fibernet_demo
```

<img src="docs/media/three_dimensional_topologies.gif" alt="从不同角度观察固定的 octet、diamond 和 gyroid 三维纤维网络" width="750">

*三种固定拓扑的旋转预览。[打开动图画廊](docs/MEDIA_GALLERY.md)*

## 接下来可以做什么

- **建立与加载网络：**生成[二维或三维网络](docs/METHODS_3D_GENERATION_PARAMETERS.md)，在[三维实体网格](docs/TET4_STATIC.md)上计算小变形响应。
- **看受力后电阻怎么变：**[360 杆网络](docs/DEFORMED_TRANSPORT_PREPARATION.md)和[含熔合节点的三通路实体网络](examples/README.md#predicting-strain-induced-resistance-in-a-fused-solid-lattice)都在变形后的几何上重算电阻。孔隙流动可从[变形前后流量对比](examples/README.md#how-solid-deformation-changes-pore-flow)入手。
- **继续分析、学习或制造：**用同一个 `StructureGraph` [训练模型](examples/physical_learning_workflow.py)，或生成[闭合路径](docs/MANUFACTURING_GRAPH_MAPPING.md)和[多段路径](docs/MULTISTROKE_MANUFACTURING.md)。

<details>
<summary>从源码运行仓库算例</summary>

**开发版源码：4.35.0**

```bash
git clone https://github.com/GellmanSparrowS/fibernet.git
cd fibernet
python -m pip install -e .
python -m examples.open_source_workflow --output-dir demo_output
```

[更多算例与可选依赖](examples/README.md) · [流程索引](docs/api/WORKFLOW_MAP.md) · [接口与求解器](docs/API_LAYERS.md)

</details>

[FiberScope 桌面应用](FiberScope/README.md) · [更新记录](CHANGELOG.md) · [参与贡献](.github/CONTRIBUTING.md) · [MIT 许可证](LICENSE)
<!-- locale:zh:end -->

</details>

<details open>
<summary>English</summary>

<!-- locale:en:start -->
## Quick start

Install from [PyPI](https://pypi.org/project/fibernet/) with Python 3.9 or newer. The built-in example generates a network, computes its small-deformation response, and saves the structure, a figure, and results in `fibernet_demo/`.

```bash
python -m pip install fibernet
python -m fibernet.examples.basic_usage --output-dir fibernet_demo
```

<img src="docs/media/three_dimensional_topologies.gif" alt="Rotating views of fixed octet, diamond, and gyroid fiber networks" width="750">

*Camera views of three fixed topologies. [Open the animation gallery](docs/MEDIA_GALLERY.md).*

## Choose a workflow

- **Build and load a network:** generate a [2D or 3D network](docs/METHODS_3D_GENERATION_PARAMETERS.md), or compute small-deformation response on a [3D solid mesh](docs/TET4_STATIC.md).
- **See how loading changes resistance:** recompute resistance on the deformed geometry of a [360-member network](docs/DEFORMED_TRANSPORT_PREPARATION.md) or a [three-path solid network with fused joints](examples/README.md#predicting-strain-induced-resistance-in-a-fused-solid-lattice). For fluids, [compare flow before and after deformation](examples/README.md#how-solid-deformation-changes-pore-flow).
- **Reuse the network:** use the same `StructureGraph` to [train models](examples/physical_learning_workflow.py) or plan [closed](docs/MANUFACTURING_GRAPH_MAPPING.md) and [multi-stroke routes](docs/MULTISTROKE_MANUFACTURING.md).

<details>
<summary>Run repository examples from source</summary>

**Development checkout: 4.35.0**

```bash
git clone https://github.com/GellmanSparrowS/fibernet.git
cd fibernet
python -m pip install -e .
python -m examples.open_source_workflow --output-dir demo_output
```

[More examples and optional dependencies](examples/README.md) · [Workflow map](docs/api/WORKFLOW_MAP.md) · [API layers and solvers](docs/API_LAYERS.md)

</details>

[FiberScope desktop app](FiberScope/README.md) · [Changelog](CHANGELOG.md) · [Contributing](.github/CONTRIBUTING.md) · [MIT license](LICENSE)
<!-- locale:en:end -->

</details>
