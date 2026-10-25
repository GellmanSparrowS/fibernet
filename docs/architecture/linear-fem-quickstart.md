# Linear FEM quickstart: actual loads and boundary conditions

From the source checkout, with its base NumPy/SciPy/Matplotlib dependencies:

```sh
python -m examples.linear_fem_quickstart --output-dir demo_linear_fem
```

For a solve, choose a **new directory**; even an empty existing directory is refused. Importing
the example performs no solve, plotting or file creation. It needs no meshio,
scikit-learn, PyTorch or Taichi, and never opens a window.

To redraw existing saved evidence without another solve or graph generation:

```sh
python -m examples.linear_fem_quickstart --plot-only --output-dir demo_linear_fem
```

This calls `draw_from_saved(output_dir)` in the example, reads its four JSON
files, and replaces **only** the PNG/SVG. Copy the old images first if needed.
The regular solve path also calls this renderer after saving the result. It is
an example-local function, not a new library API or general experiment schema.

The [example](../../examples/linear_fem_quickstart.py) makes one 10 mm square with
four nodes and four beam elements. The caller declares SI units, global
`E=1e6 Pa`, `nu=0.3`, and radius `1e-4 m`. Both left nodes are fully clamped
(`ux=uy=rz=0`); both right nodes receive `[1e-5, 0] N` and have free rotations.
The graph's Material record describes the same values, but the FEM consumes
explicit constructor E/nu, not that record.

## Trace the public API

1. `fn.pattern_2d(..., grid=(1, 1), n_pts_per_side=0)` creates the graph
2. `graph.to_numpy()` yields `(N,3)` positions, `(M,2)` edges and `(M,)` radii.
   Record `sorted(graph.nodes)` and `list(graph.edges)` as the input row IDs
3. Convert ID-keyed force/BC choices to input rows. `solve_2d` takes `edges.T`,
   `positions[:, :2]`, exact `(N,2)` translational forces, and clamped row indices.
   Applied nodal moments are unsupported; do not add a third force column
4. `BeamFrameFEM(E=..., nu=...).solve_2d(...)` executes one linear solve with
   explicit `damping=1e-6`. The root `fn.simulate(..., backend='fem')` convenience
   route only exposes stretch loading; explicit forces/BCs intentionally use the
   public module API rather than an invented root keyword or new helper
5. `to_sim_result(raw, graph, node_ids=..., edge_ids=...)` preserves row identity;
   `result.save(..., detailed=True)` and `fn.SimResult.load(...)` preserve fields

## Inspect the six outputs

- `graph.json`: native geometry, IDs, radii and descriptive material
- `experiment.json`: caller-maintained units, solver parameters, ID-keyed loads
  and clamps, plus exact force array and row maps; JSON object ID keys are strings
- `result.json`: detailed public SimResult output, including stable-ID diagnostics
- `summary.json`: measured translations and separately dimensioned residual summaries
- `linear_fem_quickstart.png` and `.svg`: A4 portrait, with undeformed geometry,
  node IDs, actual declared clamps and load directions above two concise text
  columns; arrow lengths are schematic

The example reloads the graph and result through public APIs and asserts
identity, displacement, axial-force and diagnostic equality without solving again.
These assertions check the software journey, not physical validity.

The original bounded reference execution used **2.735 CPU seconds** (2.564 user, 0.171
system; 2.735 s wall) including imports and both plots, under an 18 CPU-second
limit with single-thread BLAS. The A4 revision redraws those saved values without
another solve. The original execution produced max translation
`3.183097848626393e-6 m`, max absolute free force residual
`3.1830978486162807e-12 N`, and max absolute free moment residual
`1.985011265211617e-24 N m`. This is one tiny API check, not a performance
benchmark or a tolerance requirement. A read-only home directory caused
Matplotlib/Fontconfig cache warnings; temporary-cache fallback still produced
both images. On restricted hosts, set `MPLCONFIGDIR` to a writable directory.

The A4 plot-only redraw used **2.465 CPU seconds** (2.326 user, 0.139 system;
2.467 s wall) under a 10 CPU-second limit. Its subprocess replaced FEM
construction/solve methods and graph generators with fail-fast sentinels:
none was called, and all four JSON files remained byte-identical. The SVG is
210 × 297 mm. Import safety and existing-output refusal were checked separately
from the unchanged numerical results; no second solve was performed.

## Read numerical diagnostics honestly

`physical_free_residual` is the actual `K_physical @ u - applied_load` restricted
to free DOFs. The summary reports max absolute translational entries in **N**
and max absolute rotational entries in **N m** separately, never one mixed-unit
norm. Constrained reactions remain in result metadata. `regularization` records
the actual coefficient; it is a numerical mixed-equation correction, not a
physical support or universally dimensioned spring. `physical_equilibrium`,
`load_compatibility`, and `solution_uniqueness` stay **`not_evaluated`** even when
residuals are small. A finished solve or successful reload certifies none of them.

## API friction visible in this journey

- Stable node IDs still require a manual ID-to-row dictionary, force-array fill,
  clamp-row conversion, edge transpose, and explicit adapter maps
- Geometry/result files alone do not record the whole experiment; units and
  applied loads/BCs need the caller's separate manifest
- `render_graph(show_boundary=True)` highlights graph boundary flags, not the
  solver's actual constraints. This example disables it and overlays its declared
  clamps and loads on the public `render_graph(ax=...)` result
- Diagnostic availability is distinct from an equilibrium/stability verdict;
  consumers must separate force and moment units and retain `not_evaluated`

See [row identity](raw-array-identity-migration.md),
[force shapes](fem-force-shape-contract.md), and
[diagnostic semantics](fem-linear-diagnostics-usage.md).

## A representative 81-node static network

From an installed development build, run:

```bash
python -m fibernet.examples.linear_static_network --output-dir static_network
```

From a checkout, the same implementation is also available as
`python -m examples.linear_fem_quickstart --case network81 --output-dir static_network`.
The default four-node example is unchanged. Both commands require a new output
directory and run exactly two fixed load cases, F and 2F. Only the base NumPy,
SciPy, NetworkX and Matplotlib dependencies are needed; no native provider or
learning backend is selected.

This synthetic case contains an 8×8 triangular cell array (81 nodes, 208 straight
members) over 0.2 m × 0.2 m. Circular radii are 0.5 mm and the explicitly supplied
Young's modulus is 1 GPa. All three planar DOFs of the nine left nodes are clamped.
The nine right nodes receive nonuniform forces in N, with zero applied nodal
moments. `case.json` records the exact recipe, units and checks. Graph material
records match the explicit modulus; the solver does not infer it from metadata.

The existing public `solve_thermoelastic_network_snapshot` API is used with
uniform 300 K reference/current temperature and zero expansion coefficient, so
thermal strain is zero. This is a linear welded Euler–Bernoulli frame with no
numerical regularization or added geometric stiffness. It is not a buckling,
mesh-convergence or experimental material validation.

The example checks free-DOF residuals, global forces and moments (including
support couples and force lever arms), small displacement/rotation, doubled
response and fourfold energy. An independent analytic integral of linear axial
strain and cubic Hermite curvature is compared with half the external work;
it does not reuse the solver stiffness matrix. Failed checks raise rather than
changing the parameters or retrying.

Outputs include `graph.json`, `case.json`, both full `snapshot_F*.json` files,
stable-ID nodal CSV, per-member energy JSON, `summary.json`, and A4 editable SVG/PNG under `figures/`.
The graph is reloaded through `StructureGraph.load_json`; snapshot JSON is only
checked for field readback equality, not accepted as a native restart format.
Failure leaves partial new output for inspection and best-effort `failure.json`,
without deleting files or replacing the original exception.

Redraw saved F fields into another **new** figure directory without generating
or solving:

```bash
python -m fibernet.examples.linear_static_network --output-dir static_network \
  --plot-only --figure-dir redrawn_figures
```

The upper panel colors true-scale geometry by displacement magnitude in µm.
The lower panel explicitly magnifies displacement ×10,000 for visibility; it
does not change the numerical result. Force arrows in both panels are schematic
and located at the undeformed reference nodes. Missing or inconsistent saved
fields fail rather than trigger another solve. Existing figure directories are
refused, so repeat rendering cannot silently overwrite user files.
