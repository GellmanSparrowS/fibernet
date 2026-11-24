# FiberNet 4.22.0

**Generate fiber-network graphs, measure their structure, and run reproducible small-deformation simulations in Python.**

FiberNet connects native graph generation, topology and planar snapshot analysis, beam-FEM simulation, and JSON persistence. FiberScope is the separate desktop application in the same [repository](https://github.com/GellmanSparrowS/fibernet).

## Changes in 4.22.0

Solve 1 to 8 ordered force cases on one Tet4 solid with
`fibernet.sim.tet4.solve_static_tet4_load_cases`. The cases share the
mesh, material assignments, and zero-displacement restraints. The solver
sets up the system once and checks reactions, residuals, and work and
energy for each case. It returns results in input order; if any case
fails, it returns no partial tuple.

The default dependencies are unchanged. See the
[4.22.0 release notes](https://github.com/GellmanSparrowS/fibernet/blob/main/docs/RELEASE_4_22_0.md)
and the worked example in the matching source distribution.

## Install

Python 3.9 or newer is required. The core dependencies are NumPy, SciPy, NetworkX, and Matplotlib.

```bash
python -m pip install "fibernet==4.22.0"
```

Run the installed example from any directory:

```bash
python -m fibernet.examples.basic_usage --output-dir fibernet_demo
```

It generates its own network and writes `network.json`, `network.png`,
`fem_result.json`, and `summary.json`. No source checkout or input data is needed.

### Run the welded Tet4 junction example from a released version

From an empty directory, download the matching source distribution to get the
standalone example without cloning the repository:

```bash
python -m pip download --no-deps --no-binary=fibernet "fibernet==4.22.0"
python -c "import pathlib, tarfile; p='examples/tet4_static_welded_junction.py'; t=tarfile.open('fibernet-4.22.0.tar.gz'); pathlib.Path(p.rsplit('/', 1)[-1]).write_bytes(t.extractfile('fibernet-4.22.0/' + p).read()); t.close()"
python tet4_static_welded_junction.py --tier quick
python tet4_static_welded_junction.py --tier reference
```

Both tiers check force and moment balance and agreement of strain energy and
half external work. The JSON includes the junction's share of total strain energy. The reference
tier has 5,424 tetrahedra; if your
environment rejects it at the factor limit, use the smaller quick tier.
These are comparisons on each discrete mesh, not mesh convergence.

To open the two quick-tier results in ParaView, add Meshio and write VTUs
to a new directory:

```bash
python -m pip install "meshio==5.3.5"
python -c "from pathlib import Path; Path('welded-vtu').mkdir()"
python tet4_static_welded_junction.py --tier quick --write-vtu-dir welded-vtu
```

This writes `welded-quick-homogeneous.vtu` and
`welded-quick-stiffer_core.vtu`. Select the same `stress_Pa` Cell Data
component and color range in both files. Use Warp by Vector with
`displacement_m` only if you also want a deformed view. Files are
published individually without overwrite; the pair is not transactional.

The example below needs no Torch, Taichi, GPU, or desktop application. Install optional dependencies for the workflow you choose:

```bash
python -m pip install "fibernet[ml]==4.22.0"             # scikit-learn, pandas, tqdm
python -m pip install "fibernet[manufacturing]==4.22.0"  # solid geometry tools
python -m pip install "fibernet[fracture]==4.22.0"       # experimental fracture backend
python -m pip install "fibernet==4.22.0" "meshio==5.3.5" # Tet4 mesh I/O and result export
```

Installing an extra supplies dependencies; model definitions and input requirements are in the corresponding methods documentation.

### Work from source instead

Use a source checkout to run the repository's `examples/` modules:

```bash
git clone https://github.com/GellmanSparrowS/fibernet.git
cd fibernet
python -m pip install -e .
python -m examples.open_source_workflow --output-dir demo_output
```

For source extras, use `python -m pip install -e ".[ml]"` or `python -m pip install -e ".[fracture]"`. A checkout follows its checked-out revision and can differ from a PyPI release. Check `fibernet.__version__` and `fibernet.__file__` in the Python environment running your script.

## Generate, analyze, simulate, and save

```python
from pathlib import Path
import numpy as np
import fibernet as fn
from fibernet.api import analyze, export, load
from fibernet.analysis import SnapshotFeatureExtractor

output = Path("demo_output")
output.mkdir(exist_ok=True)

# SI units: coordinates and radius in m, E in Pa.
graph = fn.pattern_2d(
    unit="honeycomb", box=(10, 10), grid=(2, 2), radius=0.05, seed=23
)
assert (graph.num_nodes, graph.num_edges) == (18, 22)

topology = analyze(graph, kind="topology")
snapshot = SnapshotFeatureExtractor(include_contact=False).extract_graph(graph)
print(topology)

export(graph, str(output / "network.json"))
restored_graph = load(str(output / "network.json"))
assert isinstance(restored_graph, fn.StructureGraph)
assert list(graph.nodes) == list(restored_graph.nodes)
assert list(graph.edges) == list(restored_graph.edges)

result = fn.simulate(
    graph, backend="fem", mode="stretch", stretch_ratio=1.01,
    nonlinear=False, E=1e9, nu=0.3,
)
result.save(str(output / "fem_result.json"), detailed=True)
restored_result = fn.SimResult.load(str(output / "fem_result.json"))
assert np.isclose(result.max_displacement, restored_result.max_displacement)
print(graph.num_nodes, graph.num_edges, result.max_displacement)
```

This small honeycomb example has 18 nodes and 22 edges. The source workflow also saves `network.png` and a recruitment `summary.json`. The generation, linear-FEM result round trip, snapshot descriptors, and native graph JSON round trip have been exercised together on this fixed example.

`stretch_ratio=1.01` means 1% extension. The explicit `nonlinear=False` selects the small-deformation beam solver. SI convention: coordinates and radius in m; E in Pa; forces in N. Displacement is in meters; units are not converted automatically. Choose material parameters for your own problem. See the [FEM model and output definitions](https://github.com/GellmanSparrowS/fibernet/blob/main/docs/METHODS_BEAM_FEM_CONTRACT.md).

## Choose an analysis

- **Graph connectivity:** `analyze(graph, kind="topology")` returns `TopologyMetricsV1`, retaining physical edges and stable identities.
- **Planar snapshots:** `SnapshotFeatureExtractor` provides structure, bounded-pore, and distribution descriptors. Contact descriptors are an explicit option. See [snapshot definitions](https://github.com/GellmanSparrowS/fibernet/blob/main/docs/METHODS_SNAPSHOT_FEATURES_UNIFIED.md).
- **Paths and geometry:** `analyze_geometry_paths` uses explicit length units and coordinate frames. `analyze_spectrum` provides a bounded graph-Laplacian analysis. See [native analysis and legacy summary](https://github.com/GellmanSparrowS/fibernet/blob/main/docs/LEGACY_ANALYSIS_SUMMARY.md).
- **Tensile recruitment:** `analyze_tensile_recruitment` measures thresholded tensile connectivity on supplied edge-strain frames. See the [recruitment method](https://github.com/GellmanSparrowS/fibernet/blob/main/docs/METHODS_TENSILE_RECRUITMENT.md).

## Moving from legacy interfaces

Use `StructureGraph` for explicit node/edge incidence and native JSON for preserving it. `fibernet.api.export(graph, "network.json")` and `load(...)` retain the native representation; they do not silently convert it to `FiberNetwork`.

`analyze(FiberNetwork)` returns the twelve-field legacy endpoint summary for its supported nonperiodic, crosslink-free domain. It uses arc lengths and endpoint connectivity; interior crossings are not inferred as junctions. Crosslinked inputs and ambiguous endpoint projections require explicit native graph construction. [Input and result contract](https://github.com/GellmanSparrowS/fibernet/blob/main/docs/LEGACY_ANALYSIS_SUMMARY.md).

Legacy `ml.FeatureExtractor` supports inventory counts when **all five feature-group flags are disabled**. The flags still default to enabled; requesting a legacy group raises `NotImplementedError`, rather than returning a complete historical vector or fabricated zeros. `GraphFeatureExtractor` and `SnapshotFeatureExtractor` have different schemas and must not be substituted into an existing trained model without validation. [Counts-only example and migration details](https://github.com/GellmanSparrowS/fibernet/blob/main/docs/LEGACY_ML_FEATURE_BOUNDARY.md).

Legacy `FiberNetwork` LAMMPS export (`to_lammps`, `api.export(..., format="lammps")`, and inferred `.data`/`.lammps` output) now raises `NotImplementedError` before opening the destination, including for straight, unlinked chains. Preserve centerlines and declared crosslinks with JSON instead; storage is not a solver model. See the [legacy LAMMPS export boundary](https://github.com/GellmanSparrowS/fibernet/blob/main/docs/LEGACY_LAMMPS_EXPORT.md).

Legacy Abaqus/ANSYS beam export accepts its documented straight-fiber domain; Gmsh export writes geometry-only line meshes. These exports do not qualify external solver execution or supply a complete mechanical model. The legacy FEA-LAMMPS template is disabled. The [explicit StructureGraph LAMMPS backend](https://github.com/GellmanSparrowS/fibernet/blob/main/docs/integrations/LAMMPS_HARMONIC_CLI.md) is a separate integration. See [legacy FEA export scope](https://github.com/GellmanSparrowS/fibernet/blob/main/docs/LEGACY_FEA_EXPORT.md).

## Documentation and support

[API layers and simulation modes (verified source)](https://github.com/GellmanSparrowS/fibernet/blob/0586e23ad57d2edff577ab64c7f33910b091fc2f/docs/API_LAYERS.md)

[Source quick start](https://github.com/GellmanSparrowS/fibernet/blob/main/docs/quickstart.rst) · [Examples](https://github.com/GellmanSparrowS/fibernet/tree/main/examples) · [Issues](https://github.com/GellmanSparrowS/fibernet/issues) · [MIT license](https://github.com/GellmanSparrowS/fibernet/blob/main/LICENSE)

For a bug report, include the Python and FiberNet versions, the command or minimal example, and the complete traceback. For headless image examples, set `MPLBACKEND=Agg`.
