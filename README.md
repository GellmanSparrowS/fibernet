<table>
<tr><td align="center">
<img src="docs/media/ui/homepage-cover-v2.svg" alt="FiberNet + FiberScope" width="1000">
<h3>From network geometry to mechanical response and design.</h3>
<p>Generate structures, simulate their response, analyze networks, and connect learning with fabrication workflows.</p>
<p>
<a href="README.md"><img src="docs/media/ui/homepage-language-en-active-v2.svg" alt="English — selected" width="180" height="54"></a>
<a href="README_CN.md"><img src="docs/media/ui/homepage-language-zh-inactive-v2.svg" alt="切换到中文" width="180" height="54"></a>
<a href="#quick-start"><img src="docs/media/ui/homepage-quickstart-en-v2.svg" alt="Quick start" width="220" height="54"></a>
</p>
</td></tr>
</table>

[Animated examples](#animated-examples) · [Modules](#choose-a-workflow) · [Desktop app](#desktop-application) · [Methods and reproducibility](#methods-and-reproducibility)

**FiberNet** is a Python toolkit for fiber-network generation, simulation, analysis, and inverse design. **FiberScope** brings the shared geometry, reduced-model, and analysis workflows into a desktop application. Build a structure, follow its deformation, measure how connectivity evolves, and carry the same design into learning or continuous-route construction.


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

### 1. Generate planar and three-dimensional networks

![Octet, diamond and gyroid fiber networks shown from changing camera angles](docs/media/three_dimensional_topologies.gif)

Explore octet, diamond and gyroid networks with **35/118, 63/144 and 203/531 nodes/edges** in these saved examples. Camera rotation reveals each fixed topology. [3D parameters](docs/media/three_dimensional_topologies.json) · [Editable 3D frame](docs/media/three_dimensional_topologies_final.svg)

#### One geometric language, multiple topologies

![Square, hexagon, ring, and kagome networks morph under the same reference-fiber profile](docs/media/spectrum_four_topologies.gif)

Apply one dimensionless reference-fiber displacement profile to **square, hexagon, ring, and kagome** networks. Their geometry changes while each graph retains its connectivity. This is a geometry demonstration, not a loading simulation. [Manufacturing geometry](docs/METHODS_MANUFACTURING_UNIFIED.md) · [Editable vector frame](docs/media/spectrum_four_topologies_peak.svg) · [Parameters](docs/media/spectrum_four_topologies.json)

### 2. Simulate deformation and compare loading directions

![A recorded 436-node kagome network deforms while recruited edges and grip-spanning components evolve](docs/media/tensile_recruitment_kagome.gif)

The **3 × 3 perturbed kagome network has 436 nodes and 576 edges**. Blue edges exceed the positive axial-strain threshold; orange edges belong to a thresholded component connecting both grips. The trace follows the recruited-edge fraction through the saved reduced-model trajectory. Recruitment is a strain-threshold connectivity statistic, not a complete force-flow map. [Trajectory example](examples/reduced_recruitment_workflow.py) · [Recruitment methods](docs/METHODS_TENSILE_RECRUITMENT.md) · [Source manifest](docs/media/tensile_recruitment_kagome.json)

![Horizontal and vertical loading of the same hexagonal topology](docs/media/directional_recruitment_hexagon.gif)

Compare horizontal and vertical loading on one hexagonal topology. The saved reduced-model example reaches its specified spanning-recruitment threshold at stretch ratios **1.08 and 1.24**, respectively. [Directional parameters](docs/media/directional_recruitment_hexagon.json) · [Editable comparison](docs/media/directional_recruitment_hexagon_final.svg)

### 3. Analyze recruitment and threshold sensitivity

![Three recruitment thresholds evaluated on the same saved kagome trajectory](docs/media/tensile_threshold_sensitivity.gif)

Read the same deformation through three thresholds, **α = 0.02, 0.05 and 0.15**. The panels reveal which connectivity conclusions depend on the analysis threshold while keeping the saved mechanical trajectory fixed. [Analysis parameters](docs/media/tensile_threshold_sensitivity.json) · [Editable analysis frame](docs/media/tensile_threshold_sensitivity_final.svg)

### 4. Learn responses and explore target-driven design

![Animated tour of the saved learning and inverse-design workspaces](docs/media/learning_inverse_workflow.gif)

Move from dataset preparation and response learning to a target-curve design workspace. This tour presents saved FiberScope interface views; it is a workflow explanation rather than a training-history replay. The library shares the physical-label stream, 14-feature representation, surrogate adapters and CEM target search. [Learning example](examples/physical_learning_workflow.py) · [Inverse-design example](examples/curve_inverse_workflow.py) · [Tour sources](docs/media/module_workflow_tours.json)

#### Compare structural interventions with a continuous-route constraint

![Four matched route-preserving interventions on a kagome network](docs/media/route_preserving_intervention_kagome.gif)

![Matched route-preserving interventions on a ring network](docs/media/route_preserving_intervention_ring.gif)

Compare the intact structure with low-early-strain, high-late-strain and fixed-random deletion choices. The saved kagome and ring studies match removed edge length and retain an Euler route, exposing how a design choice changes the response. These are saved controlled interventions, not an RL training sequence. [Kagome source](docs/media/route_preserving_intervention_kagome.json) · [Ring source](docs/media/route_preserving_intervention_ring.json) · [Comparison methods](docs/CONSTRAINED_CYCLE_INTERVENTION_2026-09-24.md)

### 5. Construct continuous manufacturing routes

![An orange cursor traverses a closed Euler route through a kagome manufacturing graph](docs/media/continuous_route_kagome.gif)

A closed Euler route visits every independent fiber edge once, including parallel fibers. Blue marks visited edges; the orange cursor shows traversal order. The planar and surface workflows preserve the graph-to-route mapping for inspection and export. Route playback shows graph traversal, not printer dynamics. [Planar and curved routes](examples/manufacturing_workflow.py) · [Custom cells](examples/custom_cell_workflow.py) · [Solid and route export](examples/solid_export_workflow.py)

### 6. Map fibers onto surfaces and prepare exports

![Animated tour of saved surface mapping and manufacturing export workspaces](docs/media/surface_export_workflow.gif)

Inspect a surface-bound network, then move to the manufacturing workspace for route and geometry export. The saved-interface tour connects the surface and fabrication tools; route/solid files describe geometry and do not constitute a printer execution record. [Surface and route example](examples/manufacturing_workflow.py) · [Solid export example](examples/solid_export_workflow.py) · [Tour sources](docs/media/module_workflow_tours.json)

[Full media gallery, parameters and editable frames](docs/MEDIA_GALLERY.md)

### 7. Inspect electrical transport on a three-dimensional network

![Camera orbit around a saved torus-network electrical field](docs/media/electrical_torus_field.gif)

Explore the spatial distribution of voltage and current on a saved **80-node, 162-edge** toroidal network. The camera moves around a fixed synthetic steady-state electrical solution, making the field and graph connectivity visible together. [Electrical field and render sources](docs/media/electrical_torus_field.json) · [Electrical transport API](docs/integrations/LINEAR_TRANSPORT_USAGE.md)

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

