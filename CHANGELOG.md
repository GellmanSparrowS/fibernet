## [Unreleased] - Legacy trim crosslink preservation

- Preserve and remap surviving crosslinks in `core.transform.trim_to_box`.
  Whole-fiber mode keeps original attachments; sampled mode remaps attachments
  on retained vertices and intact original segments. Geometry selection and
  public signatures are unchanged. Crosslink-based derived results intentionally
  change from the old empty-link output; see the
  [compatibility and migration contract](docs/architecture/legacy-trim-crosslinks.md).

## [Unreleased] - Reduced-model configuration admission

- Reject nonfinite, malformed and out-of-range reduced-model controls at
  config/solver entry and before desktop cache lookup. Revalidate mutable
  configs at run entry, and normalize admitted NumPy scalars for float64
  coefficient arithmetic and JSON metadata. Public fields and ordinary
  Python-float numerical behavior are unchanged; see the
  [configuration and migration contract](docs/architecture/reduced-config-contract.md).

## [4.4.2] - 2026-11-13

See [4.4.2 release notes and migration](docs/RELEASE_4_4_2.md) for the complete
changes since published 4.4.1: detached StructureGraph metadata snapshots,
geometry/force corrections, per-edge FEM materials and stricter admission.
Shared transport graph validation, float64 recruitment thresholds and the
versioned recruitment replay archive are also included. Historical Unreleased
entries below remain as development history.

## [Unreleased] - Finite-width contact scaling

- Normalize contact raster geometry and width together so a change of length
  unit does not erase short-segment contacts through an absolute projection
  floor. Preserve field names, adjacency exclusions and raster budgets.
- Restore pixel spacing and overlap area to caller units, reject unrepresentable
  output scales, and add exact-pixel and public-route regressions. See
  [finite-width contact units](docs/FINITE_WIDTH_CONTACT_SCALING.md).

## [Unreleased] - Per-member cylinder geometry

- Preserve each represented edge's radius-length association in 3D cylinder
  volume and lateral-area sums, correcting five derived geometry features for
  heterogeneous radii, including native octet networks. Keep feature names,
  ordering, default missing radii and historical cap/floor conventions.
- Add analytical, generated-structure and scaling tests plus a runnable
  generation-to-FEM-to-JSON/CSV/VTK workflow. See the
  [radius workflow guide](docs/heterogeneous_radius_workflow.rst) for scope.

## [Unreleased] - Consistent scalar graph scaling

- Treat NumPy real numeric scalars and zero-dimensional numeric arrays like
  Python scalar factors when resizing StructureGraph radii as well as geometry.
  Keep vector coordinate-only behavior, material values and stable IDs unchanged.
- Reject masked scalar factors before reading hidden values. Add scalar parity,
  copy/serialization and analytical 2-D/3-D axial/bending FEM checks. See the
  [geometric scaling guide](docs/graph_scaling.rst) for units and vector semantics.

## [Unreleased] - Per-edge linear beam materials

- Add explicit graph-material selection for linear 2-D/3-D beam FEM using the
  existing StructureGraph. Preserve uniform defaults; validate and snapshot
  per-edge isotropic properties through assembly, recovery and result metadata.
- Add heterogeneous series, parallel, network, bending/torsion and ownership
  checks. Document material precedence, explicit units and axial-proxy energy.

## [Unreleased] - Static network workflow checks

- Add real small-network static FEM, saved-result analysis and VTK workflow
  tests with an independent energy-Hessian oracle. Document stored material
  precedence, numerical regularization and axial-proxy versus full beam energy.
  No solver law or public API changes.

## [Unreleased] - Legacy VTK attribute admission

- Reject malformed field names, nonnumeric/complex/nonfinite attributes and
  float32-overflowing values before legacy VTK export opens its destination.
  Invalid attributes no longer truncate an existing output or emit invalid
  records. Public signatures, coordinate and valid attribute bytes are
  unchanged. See [attribute ordering and precision](docs/io/legacy_vtk_attributes.md).

## [Unreleased] - Beam FEM radius admission

- Reject nonpositive, boolean and non-real beam radii before FEM graph dispatch,
  and reject unrepresentable circular sections instead of returning invalid
  stresses. Use float64 section arithmetic to prevent NumPy integer-power
  overflow. Graph storage, explicit material precedence, units and public
  signatures are unchanged. See the [FEM radius contract](docs/METHODS_BEAM_FEM_CONTRACT.md#circular-section-radius-admission).

## [Unreleased] - Nested masked learning geometry

- Reject masked coordinates and internal-point rows nested in lists, tuples or
  object arrays before creating a `GraphBatch`, including all-false masks.
  NumPy conversion previously could silently expose hidden internal-point
  values as valid geometry and length features. Ordinary numeric snapshots,
  v1/v2 schemas, ID mappings and public signatures remain unchanged. See the
  [graph-data contract](docs/architecture/graph-data-usage.md).

## [Unreleased] - Native 3D cell dimension validation

- Reject invalid `pattern_3d` box dimensions before built-in or registered
  factory work, including zero-width cells, negative extents and complex values
  that previously could collapse, reflect or silently change the generated
  geometry. Valid scalar types, argument precedence and zero-grid behavior are
  preserved. See the [3D dimension contract](docs/METHODS_3D_GENERATION_PARAMETERS.md).

## [Unreleased] - Unprofiled harmonic output admission

- Reject unprofiled LAMMPS harmonic tasks whose mandatory trajectory atom rows
  alone exceed the explicit aggregate or per-file output budget, before runtime
  validation or launch. This lower bound is not a size forecast or native-engine
  qualification. Public signatures, graph/SI unit semantics, named-profile
  forecasts and runtime watchdogs are unchanged. See the
  [admission scope](docs/integrations/LAMMPS_HARMONIC_CLI.md#explicit-runtime-and-lifecycle).

## [Unreleased] - Manufacturing provider requirement

- Require `manifold3d>=3.2.0` in the manufacturing/full extras and check
  `Manifold.simplify` before constructing native geometry. Version 3.1.0 exposes
  `to_mesh64` but lacks `simplify`, so the old unversioned dependency could fail
  late with `AttributeError`. Public signatures, meshing and export algorithms
  are unchanged. See [provider requirements and validation](docs/solid_validation.md#surface-extraction-precision).

## [Unreleased] - Kagome lattice perturbation

- Honor positive `perturbation` and its `seed` in `lattice_2d(topology="kagome")`,
  including `create("lattice_2d", ...)`. Apply displacement after existing tile
  connectivity is built and before rotation. Zero perturbation, public
  signatures and other lattice types are unchanged. Historical Kagome calls
  silently ignored positive perturbation; see the
  [usage and migration contract](docs/architecture/lattice-perturbation-contract.md).

## [Unreleased] - Ionic bond force direction

- Correct `IonicBond.compute_force` to return force on the first endpoint,
  matching Spring and Covalent crosslinks and the negative first-endpoint
  energy gradient. Opposite charges attract and like charges repel. Nonzero
  returned forces and newly stored state forces reverse sign; magnitudes,
  energies, constructor signatures and rupture thresholds are unchanged.
  External callers using the previous sign must follow the
  [endpoint and migration contract](docs/architecture/crosslink-force-contract.md).

## [Unreleased] - transformed tiling boundary labels

- Recompute `tile_with_transforms` node flags on the nominal outer XY planes
  after per-cell transformation and translation. Keep the first welded
  representative's labels and leave both Z flags false. Geometry and public
  signatures are unchanged. Boundary/internal selections and derived features
  intentionally change from the previous all-false labels; see the
  [usage and compatibility contract](docs/architecture/transformed-tiling-boundaries.md).

## [4.4.1] - 2026-11-09

See [4.4.1 release notes and migration](docs/RELEASE_4_4_1.md) for changes
since 4.4.0, including the CSV, k-core and rotation fixes. Historical Unreleased
entries below remain as development history.

## [Unreleased] - Scale-invariant 3D rotation axes

- Normalize 3D rotation axes without overflowing or underflowing their norm.
  Large and tiny finite axis magnitudes now preserve rigid geometry. Reject
  malformed, complex, nonfinite and zero axes before copying a 3D graph.
  Public signatures, default-axis rotation and 2D axis behavior are unchanged.

## [Unreleased] - 3D graph k-core correction

- Restore `GraphFeatureExtractor3D` k-core values by computing the unweighted
  mean degree locally, including isolated nodes. `kcore_frac` retains its
  maximum-core-number / mean-degree formula and can exceed one. Names, order
  and signatures are unchanged, but historical all-zero k-core values change.
  See [migration guidance](docs/architecture/graph-feature-3d-kcore.md) before
  combining corrected feature values with existing datasets or models.

## [Unreleased] - CSV training input preservation

- Reject model destinations that identify the input CSV entry before training,
  including symlink, case-insensitive and Unicode-normalization aliases. Distinct
  hardlink entries remain supported; ambiguous same-parent aliases require readable
  directory entries and exact spellings. Native Dataset training and model
  serialization are unchanged.

## [Unreleased] - LAMMPS dual-reference displacement

- Add `compare_lammps_node_displacements` in `fibernet.analysis.geometry_paths`
  to compare final LAMMPS positions with the loaded initial graph and an explicit
  reference graph. Align by stable node IDs, verify initial units and graph digest,
  and return detached metre-valued arrays. Reference correspondence and coordinate
  frame remain caller responsibilities. This is geometry-only post-processing.
- Reject silent zero displacement caused by FTZ/DAZ floating-point modes,
  including when distinct normal coordinates have a subnormal difference.

## [Unreleased] - LAMMPS stable edge history

- Add `EdgeLengthHistory` and `edge_length_history` in the explicit
  `fibernet.integrations.lammps_backend` module. Extract aligned step/time/SI
  endpoint-distance tuples by stable StructureGraph edge ID from a prepared run
  and parsed result, including sampled CG trajectories. This is geometry-only
  post-processing, not native output authentication or physical validation.

## [Unreleased] - LAMMPS projection numeric range

- Use scaled Euclidean norms for projected bond lengths and maximum displacement.
  Compute stretch statistics without directly squaring extreme values, and
  translate tightly clustered stretches before scaling their spread.
  Reject nonfinite projected quantities and nonzero force/stretch values that
  underflow to zero. Public signatures and graph/SI unit conventions are unchanged.
  Regression evidence uses synthetic protocol fixtures, not native engine runs.

## [Unreleased] - snapshot pore units

- Normalize planar pore geometry by the active edge-endpoint span before
  detecting bounded centerline faces, then return pore areas in caller units
  squared and perimeters in caller units. For well-conditioned geometry, pore
  results are less sensitive to length units; approximate tolerances and
  near-degenerate geometry limitations remain.
  Structure/contact calculations and the caller-unit straightness cutoff are
  unchanged. Historical pore results near the old tolerances may change.

## [Unreleased] - snapshot open-chain straightness

- Include both boundary nodes when measuring the chord divided by full path
  length of an open chain containing degree-2 nodes. Exclude closed walks and
  retain the existing single-edge exclusion and 1e-12 caller-unit length cutoff.
  Affected straightness values intentionally change; public signatures and
  other feature calculations are unchanged by this correction. Pore unit
  normalization is documented separately.

## [Unreleased] - tutorial pipeline artifact recovery

- Restore persisted native graphs instead of regenerating them when the tutorial
  pipeline resumes. Validate generation inputs, saved simulation summaries, and
  feature-file bindings before continuing; reject missing or inconsistent files
  without overwriting them. Legacy checkpoints without artifact bindings require
  a fresh data directory. This does not restore interrupted solver state.

## [4.4.0] - 2026-11-06

See [4.4.0 release notes and migration](docs/RELEASE_4_4_0.md) for the changes
since 4.3.0. Historical Unreleased entries below are retained as development
history and include changes already shipped in earlier releases.

## [Unreleased] - offline LAMMPS saved-run reader

- Add `read_saved_lammps_run` to reconstruct existing saved preparation files
  and parse captured output without runtime admission or engine execution.
- Require a caller-supplied manifest digest, exact current-compiler regeneration,
  bounded POSIX regular-file reads and explicit nonauthentication semantics.
  Returned SI output is protocol-checked, not certified as original engine data
  or physically valid; compiler drift is rejected without automatic migration.

## [Unreleased] - gyroid factory parameters

- Forward `lattice_3d(topology="gyroid")` parameters `grid_size`, `radius`, and
  `material` to the existing backend instead of silently discarding them.
  Calls without these options now use the documented factory defaults
  `(3, 3, 3)` and `0.1`, rather than the backend defaults `(2, 2, 2)` and `0.15`.
  To retain the previous default geometry and radius, pass `grid_size=(2, 2, 2)`
  and `radius=0.15` explicitly. Direct `gyroid_lattice_3d` defaults are unchanged.
- This routes parameters to the existing TPMS-inspired fiber-network generator;
  it does not produce a solid or shell gyroid or qualify mechanical behavior.
  The existing backend does not use `threshold`, and the factory's `seed` and
  `perturbation` do not affect this route. Those behaviors are unchanged.

## [Unreleased] - checked graph prediction

- Add `predict_gnn` for existing CPU scalar FiberGNN models and GraphBatch v1,
  checking caller-declared raw XYZ+dimension compatibility while preserving
  graph identity and prior module modes. Low-level forward remains unchanged.
- Return graph-aligned model-space predictions with an explicit caller-supplied
  target name and unknown target-unit/transform statuses.

## [Unreleased] - raw beam-resultant availability

- Expose the existing component contract on raw 2-D, 3-D and incremental 2-D beam results. Preserve numeric outputs and approximate qualifications; unrecovered 3-D shear slots remain explicitly unavailable.
- Document caller-qualified field-learning targets and retain unknown legacy component availability.

## [Unreleased] - explicit Taichi geometry policy

- Breaking input acceptance: graph-based spring/Taichi calls now reject nonempty edge-owned `internal_points` by default. Explicit `geometry_policy="endpoint_chord"` retains the existing endpoint interpretation without modifying the graph or claiming curved-fiber mechanics.
- Forward the policy through direct/facade, streaming batch, Dataset and RL callers; bind batch/Dataset resumes to the policy and helper source. Actual-node straight-edge segmentation remains accepted, and generator/sampling defaults are unchanged.
- Preserve geometry provenance in ordinary result metadata. Legacy results are not backfilled. See `docs/TAICHI_GEOMETRY_POLICY.md` for migration and scope.

## [Unreleased] - RL evaluation isolation

- Add keyword-only `eval_env` to `fibernet.rl.algorithms.train_rl`, enabling automatic evaluation independently of model saving.
- Reject shared known in-process training/evaluation environment objects before model or callback construction. Remote resources and custom wrappers still require caller-managed isolation.
- Save-only calls now warn and skip unsafe automatic evaluation on the training environment, while retaining checkpoint/final saves. Provide an independent `eval_env` to restore best-model selection.
- Document migration and bounded validation in `docs/architecture/rl-evaluation-isolation.md`; add inert contract regressions without real model training or simulation.
- Correct root/easy `train_rl` documentation to identify random-policy rollout statistics, not policy learning, and clarify batch damping's existing metadata-only behavior. Runtime and exports are unchanged; source-bound batch resume may require a fresh output directory after this documentation change.

## [Unreleased] - thermal reconstruction budget

- Bound thermal capacity reconstruction with outward-rounded `C*rho`, where `rho` is the current returned-temperature reconstruction radius; retain separate field-evaluation allowances and unchanged algebraic/precision thresholds.
- Identify this numerical acceptance revision with `thermal_capacity_reconstruction_profile: binary64-rn-capacity-rho-1` in transient and coupled electrothermal provenance. Public call signatures and result layouts are unchanged; numerical bounds and acceptance outcomes may differ.
- Preserve explicit refusal for lost increments, unrepresentable products and heat scales too small for the precision budget. Smaller time steps are not guaranteed to be representable; this correction does not establish trajectory convergence or coupled-energy accuracy.
- Include self-contained synthetic acceptance/refusal tests with exact rational equation checks; no external capture file is required.

## [Unreleased] - runtime compatibility

- Use ASCII status labels in easy training and batch helpers so decorative symbols do not break narrow redirected consoles; no locale or stream configuration is changed
- Read UTF-8 Python source fixtures explicitly in documentation/exporter tests; keep graph JSON loading on its historical default text codec
- Test invalid UTF-8 batch input with a complete graph and independent UTF-8/cp1252 disk-decoding oracles, separating decoder failures from stub-engine failures

- Temporarily restrict TaichiEngine and accel/full/dev extras to Taichi 1.7.4 for the version-verified runtime-lifetime repair; other versions fail explicitly
- Same-owner engines share the runtime; owning worker threads are supported, cross-owner use and unverifiable external initialization fail safely
- Non-Taichi backends are unaffected; broader Taichi versions require explicit lifecycle matrix validation

## [4.3.0] - 2026-10-07

- Add experimental `fibernet.sim.fracture` Config/Solver/Result API and dedicated material-complete JSON CLI
- Preserve frozen linear Timoshenko, sparse equilibrium and near-tie removal kernels with source-hash provenance
- Add explicit units/grips, strict geometry/material validation, pre-removal event traces, null free-fragment poses and hash-bound resumable equilibrated checkpoints
- Add analytical/synthetic regression tests and a runnable public example; retain Python >=3.9 and old `simulate` semantics
- Add optional `fracture` dependencies (psutil, threadpoolctl); do not bundle LAMMPS or private research inputs/results
- Replace hard-coded 4.2.0 PyPI CI check with an explicit, version-parameterized post-publication verification workflow
- Scope: uncalibrated small-displacement static software; no automatic contact, finite-separation proof or fracture-toughness claim

## [4.1.4] - 2026-07-23

### Changed
- Rename `BeamFrameFEM_v6` → `BeamFrameFEM` (public API alias; `BeamFrameFEM_v6` still available for backward compatibility)
- Split Simulation section into Mass-Spring and FEM subsections with comparison table (physics, joints, radius effect, speed, use cases)
- Reorder showcase images: generation → mass-spring → trajectory → FEM → ML → RL
- Remove Performance section from README

### Added
- `BeamFrameFEM` lazy import in `fibernet.ml` (importable as `from fibernet.ml import BeamFrameFEM`)
- FEM API section with linear/nonlinear solver explanation
- Mass-Spring vs FEM comparison table in both README.md and README_CN.md

## [4.1.3] - 2026-07-23

### Documentation
- Bilingual README overhaul: added dedicated Beam Frame FEM section with showcase images, API examples, and verification results
- README.md and README_CN.md updated to v4.1.2 feature set (26 unit types, dual simulation, FEM physics model)
- Added FEM showcase images (dark + light themes) to `docs/images/`

### Improved
- Reorganized root-level files: community health files (CONTRIBUTING, SECURITY, CODE_OF_CONDUCT) moved to `.github/`, ATTRIBUTIONS.md moved to `docs/`, utility scripts moved to `scripts/`
- Updated MANIFEST.in for new file locations

## [4.1.2] - 2026-07-23

### Fixed
- **Critical**: `graph_to_fem_input()` now correctly extracts per-edge radius from StructureGraph
  - Root cause: iterating `graph.edges` (Dict) gave keys (ints) instead of SEdge values
  - FEM results now properly reflect fiber diameter changes (welded joint physics)
- `to_sim_result()` radius access fixed for energy calculation

### Improved
- FEM showcase: restructured layout (5 rows) fixing middle-row panel overflow
- Dark theme: increased edge linewidth (0.6→2.0), switched to brighter colormaps (magma/RdYlBu)
- Radius variation row now shows physical stress dependence on fiber diameter

## [4.1.1] - 2026-07-23

### Added
- `BeamFrameFEM_v6.graph_to_fem_input()`: StructureGraph to FEM input conversion
- `BeamFrameFEM_v6.stretch_test()`: One-liner uniaxial FEM stretch test
- `BeamFrameFEM_v6.to_sim_result()`: Convert FEM dict to SimResult
- `render_fem_stress()`: FEM stress visualization (dark/light themes)

### Fixed
- solve_2d and solve_3d: Invalid node index now raises clear ValueError
- solve_3d: Now returns reactions, edge_forces, K matrix (matching solve_2d)

### API Improvement
- Basic FEM analysis: ~10 lines reduced to 3 lines

## [4.1.0] - 2026-07-23

### Added
- **BeamFrameFEM_v6**: Production-grade beam frame finite element method
  - `solve_2d()`: Linear 2D beam frame analysis with axial + bending + shear
  - `solve_2d_nonlinear()`: Geometrically nonlinear solver (co-rotational incremental)
  - `solve_3d()`: 3D beam frame analysis
  - Full stress decomposition: axial, bending, total stresses per element
  - Reaction force computation at fixed boundaries
  - Moment distribution along beams
  - Validated against analytical cantilever, displacement BC, and nonlinear benchmarks

- **Large Deformation Test Suite**: Comprehensive FEM validation framework
  - 152 simulations across 14 structure types (8 2D + 6 3D)
  - 4 fiber radii (0.02, 0.05, 0.10, 0.20)
  - Stretch targets: 2.0x, 1.5x, 0.7x, 0.5x
  - Deformation propagation analysis (boundary vs interior displacement)
  - Bending vs axial dominance quantification
  - Checkpoint/resume support for long-running tests
  - Key finding: 100% FULL deformation propagation, all structures BENDING-dominated

- **14 3D unit types** (up from initial set):
  `bcc`, `chiral_3d`, `cubic`, `diamond_3d`, `fcc`, `gyroid`, `hcp`, `iwp`,
  `lidinoid`, `neovius`, `octet`, `reentrant_3d`, `schwarz_d`, `schwarz_p`

- **12 2D unit types**: `chiral`, `cross`, `diamond`, `hexagon`, `honeycomb`,
  `kagome`, `missing_rib`, `reentrant`, `square`, `star`, `triangle`, `voronoi`

### Fixed
- `easy.py`: `simulate()` default mode changed from `"tension"` (invalid) to `"stretch"` (correct)
- `easy.py`: Docstring examples updated to match actual API

### Changed
- Version bumped to 4.1.0 (minor release: new FEM module + 3D structures)

## [4.0.5] - 2026-07-17

### Fixed
- Vectorized `compute_detailed()` edge length computation (same pattern as trajectory save)

### Changed
- `dynamics()` now supports `num_steps=0` for kernel warmup (field allocation + kernel compilation without simulation)
- Notebook (Cell 19) updated: kernel warmup added, JSON save restored with tqdm progress

## [4.0.4] - 2026-07-17

### Fixed
- **Critical: Progressive slowdown during batch simulation (kernel recompilation)**
  - Moved `@ti.kernel def substep()` out of `dynamics()` method body
  - Kernel now compiled once per `(dim, num_nodes, num_edges)` and cached alongside fields
  - Before: Each `dynamics()` call recompiled the kernel → Taichi LLVM accumulated modules → progressive slowdown
  - After: Kernel cached → constant per-structure time across hundreds of calls
  - Test: 50 structures, first-10 avg = 6.60s, last-10 avg = 7.09s (ratio 1.07x — no slowdown)

### Technical Details
- File: `fibernet/sim/accelerated.py`
- Change: Kernel definition moved inside the `_field_cache` allocation block, stored as `cached['substep']`
- Impact: Eliminates ~0.5-1s kernel compilation overhead per call after the first
- Combined with 4.0.2 (vectorized save) and 4.0.3 (field cache): total ~16x speedup from original

## [4.0.3] - 2026-07-17

### Fixed
- **Critical: Taichi SNode exhaustion causing simulation hang at ~128 structures**
  - Added `_field_cache` to `TaichiEngine` class
  - Reuses Taichi fields across calls with same `(dim, num_nodes, num_edges)`
  - Before: Each `dynamics()` call allocated 14 Taichi fields → SNode limit (~1024) hit at ~128 calls → process hangs indefinitely
  - After: Fields cached and reused → tested 150+ structures without hang
  - Memory usage reduced significantly (no field accumulation)

### Technical Details
- File: `fibernet/sim/accelerated.py`
- Change: Added class-level `_field_cache` dict keyed by `(dim, num_nodes, num_edges)`
- Impact: Fixes critical hang bug, improves memory efficiency
- Test: Successfully ran 150 structures sequentially (previously hung at #128)

## [4.0.2] - 2026-07-16

### Performance
- **16x speedup** in simulation trajectory saving
  - Vectorized edge length computation in `dynamics()` method
  - Replaced Python list comprehension with NumPy vectorized operation
  - Before: ~93s per structure (30000 steps, save_interval=500)
  - After: ~27s per structure with same parameters
  - With recommended parameters (8000 steps): ~5.8s per structure

### Fixed
- Simulation appearing to "hang" with many structures
  - Root cause: excessive overhead in trajectory save loop
  - Each save took ~1.17s due to Python loop over all edges
  - Now <1ms per save with vectorized computation

### Technical Details
- File: `fibernet/sim/accelerated.py`
- Change: `np.array([np.linalg.norm(...) for e in range(n)])` → `np.linalg.norm(..., axis=1)`
- Impact: Pure performance optimization, no API or numerical changes
- All 118 existing tests pass

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.18.0] - 2026-07-05

### Added
- **Mesh export module** (`fibernet/io/mesh_export.py`)
  - Export to STL (ASCII and binary)
  - Export to Wavefront OBJ
  - Export to Stanford PLY
  - Cylindrical mesh generation for fibers
  - Configurable polygon approximation (n_sides)

### Tests
- 9 new tests in `tests/test_mesh_export.py`
- Total: 752 passing, 6 skipped

## [1.17.0] - 2026-07-05

### Added
- **Network comparison module** (`fibernet/analysis/comparison.py`)
  - `NetworkFingerprint`: structural fingerprint computation
  - `NetworkComparator`: pairwise distances, clustering, similarity search
  - `compare_networks()`: compare multiple networks
  - `network_similarity()`: pairwise similarity score
- Example 14: advanced structural analysis workflow

### Tests
- 14 new tests in `tests/test_comparison.py`
- Total: 743 passing, 6 skipped

## [1.16.0] - 2026-07-05

### Added
- **Homogenization module** (`fibernet/analysis/homogenization.py`)
  - `EffectiveElasticProperties`: E, nu, G computation (2D/3D)
  - `EffectiveThermalProperties`: thermal conductivity, CTE
  - `EffectiveElectricalProperties`: electrical conductivity
  - `compute_effective_properties()`: comprehensive analysis

### Tests
- 13 new tests in `tests/test_homogenization.py`
- Total: 729 passing, 6 skipped

## [1.15.0] - 2026-07-05

### Added
- **Spatial statistics module** (`fibernet/analysis/spatial.py`)
  - `SpatialStatistics`: Ripley's K, pair correlation, nearest neighbor
  - `OrientationAnalysis`: nematic order, orientation histograms
  - `LengthAnalysis`: length distributions, statistics, fitting
  - `ConnectivityAnalysis`: degree distribution, mean connectivity
  - `AnisotropyAnalysis`: fabric tensor, anisotropy index
  - `compute_spatial_statistics()`: comprehensive analysis

### Tests
- 22 new tests in `tests/test_advanced_statistics.py`
- Total: 716 passing, 6 skipped

## [1.14.0] - 2026-07-05

### Added
- **Material database** (`fibernet/materials.py`)
  - 29 pre-defined materials (polymers, metals, ceramics, biological, carbon)
  - `get_material()`: quick material lookup
  - `list_materials()`: list all available materials
  - `compare_materials()`: material selection tool
  - `m()`: shortcut for get_material
- **Unit conversion utilities** (`fibernet/units.py`)
  - Length: m, mm, um, nm, cm, km, in, ft
  - Force: N, kN, mN, µN, nN, lbf, dyne
  - Pressure: Pa, kPa, MPa, GPa, TPa, bar, atm, psi, ksi
  - Temperature: K, C, F, R
  - Energy: J, kJ, mJ, cal, kcal, eV, BTU, kWh

### Tests
- 33 new tests in `tests/test_materials.py` and `tests/test_units.py`
- Total: 694 passing, 6 skipped

## [1.13.0] - 2026-07-05

### Added
- **Fractal network generators** (`fibernet/gen/fractal.py`)
  - Sierpinski triangle
  - Koch curve
  - Fractal tree
  - Hilbert curve
- **Gradient network generators** (`fibernet/gen/gradient.py`)
  - Density gradient (linear, exponential, sinusoidal)
  - Property gradient (material properties varying spatially)
  - Multi-zone networks (custom regions with different properties)
- Example 13: publication-ready research workflow

### Tests
- 29 new tests in `tests/test_fractal.py` and `tests/test_gradient.py`
- Total: 687 passing, 6 skipped

## [1.12.0] - 2026-07-04

### Added
- Network topology analysis module
- Comprehensive topology metrics

### Tests
- Total: 658 passing

## [1.11.0] - 2026-07-04

### Added
- Visualization module (2D/3D plotting)
- Design of Experiments (DOE) module
  - Grid search
  - Latin hypercube sampling
  - Random sampling
  - Sensitivity analysis

### Tests
- Total: 635 passing

## [1.10.0] - 2026-07-04

### Added
- Fatigue analysis module
- Creep analysis module
- Diffusion/transport module

### Tests
- Total: 607 passing

## [1.9.0] - 2026-07-03

### Added
- Benchmark suite for performance testing
- Enhanced multiscale testing

### Tests
- Total: 585 passing

## [1.8.0] - 2026-07-03

### Added
- Electromagnetic analysis
- Molecular dynamics integration
- Open-source tool integrations
- Research case study examples

### Tests
- Total: 575 passing

## [1.7.0] - 2026-07-02

### Added
- Graph neural network (GNN) module
- Permeability analysis
- Uncertainty quantification (UQ)
- Coefficient of thermal expansion (CTE)

### Tests
- Total: 544 passing

## [1.6.0] - 2026-07-02

### Added
- Incremental FEM solver
- Buckling analysis module

### Tests
- Total: 477 passing

## [1.5.0] - 2026-07-01

### Added
- Nonlinear mechanics module
- Damage mechanics
- Viscoelasticity

### Tests
- Total: 448 passing

## [1.4.0] - 2026-07-01

### Added
- Periodic boundary conditions (PBC)
- Transforms module
- Advanced generators (chiral, woven, hierarchical)

### Tests
- Total: 410 passing

## [1.3.0] - 2026-06-30

### Added
- Machine learning module
  - Property prediction
  - Feature engineering
- Percolation analysis
- Integration tests

### Tests
- Total: 370 passing

## [1.2.0] - 2026-06-30

### Added
- Taichi-accelerated FEM solver
- I/O modules (VTK, LAMMPS, GMSH, PDB, XYZ)
- Pandas integration

### Tests
- Total: 320 passing

## [1.1.0] - 2026-06-29

### Added
- Simulation modules (thermal, fracture, etc.)
- Core data structures
- Basic generators

### Tests
- Total: 250 passing

## [1.0.0] - 2026-06-28

### Added
- Initial release
- Core fiber network data structures
- Basic random network generators
- Simple FEM solver
- Sphinx documentation framework
- GitHub Actions CI/CD
- pyproject.toml for pip installation

### Tests
- Total: 180 passing
