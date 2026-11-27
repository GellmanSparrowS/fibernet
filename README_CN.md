[English homepage](README.md#quick-start)

# FiberNet 中文指南

<!-- locale:zh:start -->
## 快速开始

用 Python 3.9 或更高版本，从 [PyPI](https://pypi.org/project/fibernet/) 安装。内置算例会生成一个网络，计算小变形响应，并把结构、图和结果存到 `fibernet_demo/`。

```bash
python -m pip install fibernet
python -m fibernet.examples.basic_usage --output-dir fibernet_demo
```

## 接下来可以做什么

- **建立与加载网络：**生成[二维或三维网络](docs/METHODS_3D_GENERATION_PARAMETERS.md)，在[三维实体网格](docs/TET4_STATIC.md)上计算小变形响应。
- **看受力后电阻怎么变：**[360 杆网络](docs/DEFORMED_TRANSPORT_PREPARATION.md)和[含熔合节点的三通路实体网络](examples/README.md#predicting-strain-induced-resistance-in-a-fused-solid-lattice)都在变形后的几何上重算电阻。孔隙流动可从[变形前后流量对比](examples/README.md#how-solid-deformation-changes-pore-flow)入手。
- **继续分析、学习或制造：**用同一个 `StructureGraph` [训练模型](examples/physical_learning_workflow.py)，或生成[闭合路径](docs/MANUFACTURING_GRAPH_MAPPING.md)和[多段路径](docs/MULTISTROKE_MANUFACTURING.md)。


### 力学动图

<img src="docs/media/homepage/woven-fracture.webp" alt="夹具从两侧拉伸预裂织物网络，纤维伸长后撕裂逐步扩展" width="960">

**裂口在拉伸下扩展。** 动图重放简化弹簧网络计算。[查看动图说明](docs/MEDIA_GALLERY.md#tensile-fracture-of-a-notched-weave) · [更多动图](docs/MEDIA_GALLERY.md)。

### 从取向场生成纤维束

<img src="docs/media/homepage/orientation-to-bundle.webp" alt="平面取向矩阵逐格绘出并逐层展开，随后生成三维纤维束" width="960">

**从二维排列方向出发。** 将平面矩阵扩展成多层三维取向场，再沿取向场生成纤维。[查看动图说明](docs/MEDIA_GALLERY.md#from-orientation-field-to-fibers)。

用 AI 客户端做拓扑分析，可从 [本地 MCP 快速上手](adapters/mcp/README.md) 开始（Linux，需配套安装）。

<details>
<summary>从源码运行仓库算例</summary>

**开发版源码：4.37.0**

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
