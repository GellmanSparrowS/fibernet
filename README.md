# FiberNet + FiberScope

[![English — selected](docs/media/ui/language-en-active.svg)](README.md) [![切换到中文](docs/media/ui/language-zh-inactive.svg)](README_CN.md)

### From network geometry to mechanical response and design.

**FiberNet** is a Python toolkit for fiber-network generation, simulation, analysis, and inverse design. **FiberScope** brings the shared geometry, reduced-model, and analysis workflows into a desktop application. Build a structure, follow its deformation, measure how connectivity evolves, and carry the same design into learning or continuous-route construction.

[Quick start](#quick-start) · [Animated examples](#animated-examples) · [Modules](#choose-a-workflow) · [Desktop app](#desktop-application) · [Methods and reproducibility](#methods-and-reproducibility)

![Saved 3D network geometry and recorded kagome deformation: camera motion and simulated material motion are labeled separately](docs/media/network_workflow_hero.gif)

**Geometry and response, side by side.** The left view orbits a fixed, saved 3D network; the right replays the recorded reduced-model deformation of a 436-node, 576-edge kagome network. The 3D view changes the camera, not the material. [Media sources and rendering details](docs/MEDIA_GALLERY.md) · [Static poster](docs/media/network_workflow_hero_poster.svg)

## Quick start

### Install the Python library

The published package is **FiberNet 4.2.0**:

```bash
python -m pip install fibernet==4.2.0
```

For the **4.3.0 development checkout** and its newer APIs, clone this repository and install from its root:

```bash
git clone https://github.com/GellmanSparrowS/fibernet.git
cd fibernet
python -m pip install -e .
```

The checkout version and PyPI release are separate; 4.3.0 is not yet a published PyPI release. Python 3.9+ is required. Optional dependencies are selected by workflow, for example `python -m pip install -e ".[ml]"` or `python -m pip install -e ".[manufacturing]"` in a checkout.

### Generate, simulate, and save

```python
import fibernet as fn

graph = fn.pattern_2d(
    unit="honeycomb", box=(10, 10), grid=(2, 2), radius=0.05, seed=23
)
fn.show(graph, save_path="network.png")

result = fn.simulate(graph, backend="fem", mode="stretch", strain=1.01)
result.save("fem_result.json", detailed=True)
print(graph.num_nodes, graph.num_edges, result.max_displacement)
```

Here `strain=1.01` is a **stretch ratio**: 1% extension. This explicit FEM route uses a small-strain beam-frame model and the core dependencies, without Taichi or the desktop app. Choose consistent geometry and material units; see the [beam-FEM contract](docs/METHODS_BEAM_FEM_CONTRACT.md).

For the complete generation → simulation → saved result → recruitment-analysis workflow, run from the checkout:

```bash
python -m examples.open_source_workflow --output-dir demo_output
```

It writes a network figure, simulation JSON, and an analysis summary. [Read the executable example](examples/open_source_workflow.py).

## Animated examples

### One geometric language, multiple topologies

![Square, hexagon, ring, and kagome networks morph under the same reference-fiber profile](docs/media/spectrum_four_topologies.gif)

Apply one dimensionless reference-fiber displacement profile to **square, hexagon, ring, and kagome** networks. Their geometry changes while each graph retains its connectivity. This is a geometry demonstration, not a loading simulation. [Manufacturing geometry](docs/METHODS_MANUFACTURING_UNIFIED.md) · [Editable vector frame](docs/media/spectrum_four_topologies_peak.svg) · [Parameters](docs/media/spectrum_four_topologies.json)

### Watch a complex network recruit during stretch

![A recorded 436-node kagome network deforms while recruited edges and grip-spanning components evolve](docs/media/tensile_recruitment_kagome.gif)

The **3 × 3 perturbed kagome network has 436 nodes and 576 edges**. Blue edges exceed the positive axial-strain threshold; orange edges belong to a thresholded component connecting both grips. The trace follows the recruited-edge fraction through the saved reduced-model trajectory. Recruitment is a strain-threshold connectivity statistic, not a complete force-flow map. [Trajectory example](examples/reduced_recruitment_workflow.py) · [Recruitment methods](docs/METHODS_TENSILE_RECRUITMENT.md) · [Source manifest](docs/media/tensile_recruitment_kagome.json)

### Trace every fiber in a continuous route

![An orange cursor traverses a closed Euler route through a kagome manufacturing graph](docs/media/continuous_route_kagome.gif)

A closed Euler route visits every independent fiber edge once, including parallel fibers. Blue marks visited edges; the orange cursor shows traversal order. The planar and surface workflows preserve the graph-to-route mapping for inspection and export. Route playback shows graph traversal, not printer dynamics. [Planar and curved routes](examples/manufacturing_workflow.py) · [Custom cells](examples/custom_cell_workflow.py) · [Solid and route export](examples/solid_export_workflow.py)

**Explore more:** [3D topology gallery](docs/media/three_dimensional_topologies.gif) · [Threshold sensitivity](docs/media/tensile_threshold_sensitivity.gif) · [Directional loading](docs/media/directional_recruitment_hexagon.gif) · [Route-preserving interventions](docs/MEDIA_GALLERY.md#route-preserving-interventions). Every animation retains its source/parameter manifest and editable keyframe in the [full media gallery](docs/MEDIA_GALLERY.md).

## Choose a workflow

| Your goal | Python entry point | Guide or runnable example |
| --- | --- | --- |
| Generate 2D/3D structures | `pattern_2d`, `pattern_3d`, `StructureGraph` | [Pattern generation](fibernet/gen/pattern.py) |
| Study deformation and mechanics | Explicit FEM or Taichi route; `ReducedBeamSolver` for the shared reduced model | [API layers, units, and model selection](docs/API_LAYERS.md) |
| Quantify structure and recruitment | `SnapshotFeatureExtractor`, `analyze_tensile_recruitment` | [Snapshot features](examples/snapshot_features_workflow.py) · [Recruitment](examples/tensile_recruitment_quickstart.py) |
| Build reusable cells and continuous routes | `CustomCellRegistry`, `compile_planar`, `compile_surface` | [Custom-cell methods](docs/METHODS_CUSTOM_CELL_UNIFIED.md) · [Manufacturing](docs/METHODS_MANUFACTURING_UNIFIED.md) |
| Learn from simulated responses | `PlanarPhysicalDataset`, `light_features`, `PhysicalRegressor` | [Resumable labels](examples/physical_dataset_workflow.py) · [Surrogate learning](examples/physical_learning_workflow.py) |
| Search for a target response | `CurveInverseDesigner`; optional `fibernet.rl` tools | [Target-curve search](examples/curve_inverse_workflow.py) · [Learning and search methods](docs/METHODS_CURVE_INVERSE_UNIFIED.md) |

The physical-learning workflow shares a **14-feature geometry representation and three response labels** with the desktop app. Its default runnable demo uses synthetic data to check the API; supply your own saved physical labels for a response surrogate. Target-curve search uses a derivative-free cross-entropy method (CEM). Optional reinforcement-learning environments and algorithm adapters are a separate workflow, not another name for CEM.

### Additional APIs in the development checkout

- **Time-dependent mechanics:** [viscoelastic beam-frame histories](docs/VISCOELASTIC_NETWORK.md), [axial viscoelastic dynamics](docs/DYNAMIC_AXIAL_VISCOELASTICITY.md), and [state observation](docs/state_observer.md).
- **Transport and heating:** [linear electrical transport](docs/integrations/LINEAR_TRANSPORT_USAGE.md), [transient thermal networks](docs/integrations/TRANSIENT_THERMAL_USAGE.md), and [one-way Joule heating](docs/ONE_WAY_JOULE_HEATING.md). The separate [experimental electrothermal API](docs/EXPERIMENTAL_ELECTROTHERMAL.md) handles fixed-geometry, voltage-driven thermal feedback with explicit coefficients and acceptance checks.
- **External-engine integration:** [bounded LAMMPS harmonic dynamics](docs/integrations/LAMMPS_HARMONIC_CLI.md) and [coarse-grained fiber networks](docs/integrations/LAMMPS_CG_CROSSLINKED_NETWORK.md), with explicit runtime and unit contracts.
- **Experimental fracture:** a standalone linear-static Timoshenko member-removal backend with stable IDs, event records, and resumable checkpoints. [Model scope and provenance](docs/FRACTURE_METHODS.md).
- **Graph learning:** [edge-aware GINE data and model adapters](docs/integrations/EDGE_GINE.md).

These module-level routes have distinct inputs and result types. The [API map](docs/API_LAYERS.md) identifies supported combinations; they are not interchangeable `simulate` selectors.

## Desktop application

**Prefer to work visually?** Download [FiberScope 3.1.0 for Windows x64](https://github.com/GellmanSparrowS/fibernet/releases/tag/v4.2.0), extract the complete ZIP, and run `FiberScope.exe`. The desktop executable includes its runtime and does not need a local Python installation.

[![FiberScope's English manufacturing workspace](FiberScope/docs/screenshots/manufacturing-en.png)](FiberScope/docs/screenshots/README.md)

Design geometry, inspect trajectories, view recruitment and feature cards, train response surrogates, and explore target-curve designs through the interface. The shared core keeps those calculations connected to the Python workflows; [the integration map](docs/INTEGRATION_REPO_MAP.md) distinguishes shared and app-specific functionality. The interface supports English and Chinese. Optional SAC/TD3/DDPG training uses a separately configured external Python runtime.

[Desktop guide](FiberScope/README.md) · [English and Chinese screenshots](FiberScope/docs/screenshots/README.md) · [Release notes](docs/RELEASE_4_2_0.md)

## Methods and reproducibility

Start from the physical question, then select the model, units, loading controls, and result type. The beam-frame FEM and reduced spring/bending/contact solver describe different models; use their matching [FEM](docs/METHODS_BEAM_FEM_CONTRACT.md) and [reduced-model](docs/METHODS_REDUCED_SOLVER_UNIFIED.md) methods when interpreting output. Numerical demonstrations are not experimental material calibration.

- **Media you can inspect:** [source manifests, editable vector frames, and render commands](docs/MEDIA_GALLERY.md).
- **Research examples:** [threshold/direction sensitivity](docs/RECRUITMENT_SENSITIVITY_2026-09-24.md), [matched route-preserving interventions](docs/CONSTRAINED_CYCLE_INTERVENTION_2026-09-24.md), and [same-graph cross-model comparisons](docs/INDEPENDENT_FEM_CYCLE_INTERVENTION_2026-09-24.md). These retain model-specific and negative findings alongside the visual examples.
- **Compatibility:** this development checkout pins the optional Taichi backend to **1.7.4**. Initialize and use an engine on its owning thread; see the [runtime lifecycle contract](docs/architecture/taichi-lifecycle-audit.md) for supported configurations. The FEM quick start above does not use Taichi.
- **Project reference:** [examples index](examples/README.md), [tests](tests/test_tensile_recruitment.py), [release history](CHANGELOG.md), and [capability audit](docs/CAPABILITY_AUDIT_2026-09-24.md).

## License and attribution

FiberNet uses the [MIT license](LICENSE). Third-party acknowledgments are in [the notices](FiberScope/THIRD_PARTY_NOTICES.md). For the published scientific work on network topology and inverse design, see the [Nature Communications article](https://www.nature.com/articles/s41467-026-76045-x).
