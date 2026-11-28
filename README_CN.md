[English homepage](README.md#quick-start) · [在仓库首页播放动图](README.md#animations)

# FiberNet 中文指南

<!-- locale:zh:start -->
## 快速开始

Python 3.9+ · [PyPI](https://pypi.org/project/fibernet/) · 算例输出：`fibernet_demo/`

```bash
python -m pip install fibernet
python -m fibernet.examples.basic_usage --output-dir fibernet_demo
```

## 接下来可以做什么

- **建模与模拟：**[二维/三维生成](docs/METHODS_3D_GENERATION_PARAMETERS.md) · [三维力学](docs/TET4_STATIC.md)。
- **变形与输运：**[纤维网络电阻](docs/DEFORMED_TRANSPORT_PREPARATION.md) · [三维实体电阻](examples/README.md#predicting-strain-induced-resistance-in-a-fused-solid-lattice) · [孔隙流动](examples/README.md#how-solid-deformation-changes-pore-flow)。
- **用 `StructureGraph` 学习与制造：**[模型训练](examples/physical_learning_workflow.py) · [闭合路径](docs/MANUFACTURING_GRAPH_MAPPING.md) · [多段路径](docs/MULTISTROKE_MANUFACTURING.md)。

[动图画廊](docs/MEDIA_GALLERY.md) · [本地 MCP（Linux）](adapters/mcp/README.md)

<details>
<summary>从源码运行仓库算例</summary>

**开发版源码：4.42.1**

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
