# FiberNet Examples

Start with the installed [PyPI demo](../README.md#quick-start) for a complete
network-generation and small-deformation run. Most workflows below use a
**source checkout**. The two commands in the next section also run from an
installation of this checkout, with no separate example archive.

### Run the one-way 3D examples from any directory

Install this checkout into a Python 3.10+ environment, then run either example
from outside the repository:

```sh
python -m pip install ".[solid-iterative,pore-flow]"
python -m fibernet.examples.tet4_iterative_mechanoelectric_sweep
python -m fibernet.examples.tet4_solid_to_pore_flow --matched-force
```

The first returns the electrical resistance at five grip displacements on a
28,512-tetrahedron network; add `--refinement 5` for 132,000 tetrahedra. The
second compares pore flow through a single scaffold at matched grip forces
with two different elastic moduli. Both print JSON to standard output. Install
only `solid-iterative` for the first or `pore-flow` for the second.

## Choose a 3D simulation

| What you want to calculate | Start here | What you get |
| --- | --- | --- |
| Load an existing Gmsh solid and inspect its deformation | [Tagged mesh with rigid grips](#solve-a-tagged-gmsh-mesh-with-rigid-grips) | Grip reaction, nodal fields and optional ParaView output |
| Compare stiffness changes at a given force on the same network | [Resistance in a fused solid lattice](#predicting-strain-induced-resistance-in-a-fused-solid-lattice) | Computed grip reaction and geometric resistance response for each material layout |
| Compare many loading locations without solving for every full field | [Small-force resistance by load location](#compare-where-a-small-force-changes-resistance) | Ranked initial resistance slope for each patch |
| Measure how loading changes flow through a pore | [Solid deformation and pore flow](#how-solid-deformation-changes-pore-flow) | Steady reference and deformed flow at the same pressure gradient |

For a finite electrical response at specific grip strokes, use the
[five-point sweep](#predicting-strain-induced-resistance-in-a-fused-solid-lattice).
For steady flow without a mechanical solve, start with the
[periodic-pore example](#steady-periodic-flow-through-a-3d-pore).

## Start here: explicit linear mechanics

```bash
python -m examples.linear_fem_quickstart --output-dir /a/new/private/linear-example
```

One four-node 2D square: declare nodal forces and full clamps, solve with the
public beam-frame module, inspect numerical diagnostics, and save/read back the
result. The figure shows the undeformed network with the actual loads and
constraints. The output directory must be new. See the short
[linear FEM guide](../docs/architecture/linear-fem-quickstart.md) for dependencies,
units and the limits of the residual/status fields. No Torch/Taichi or ML fitting
is needed. Reuse saved data without another solve with
`python -m examples.linear_fem_quickstart --output-dir EXISTINGDIR --plot-only`.
The PNG/SVG layout is A4 portrait. This does not validate the historical scripts below.

## Short Python API example

Use an explicit linear beam model for 1% grip extension. FiberNet does not
automatically convert geometry or material units.

```python
import fibernet as fn

# SI units: coordinates and radius in m; E in Pa; forces in N.
graph = fn.pattern_2d(
    unit="honeycomb", box=(10, 10), grid=(2, 2), radius=0.05, seed=23
)
result = fn.simulate(
    graph, backend="fem", mode="stretch", stretch_ratio=1.01,
    nonlinear=False, E=1e9, nu=0.3,
)
print(graph.num_nodes, graph.num_edges, result.max_displacement)
```

See the [beam FEM guide](../docs/METHODS_BEAM_FEM_CONTRACT.md) for model choices
and input units.

## Inferring mixed normal strains from a 3D network

From a source checkout:

```bash
python -m examples.mechanoelectric_mixed_strain_3d
```

Use three sequential DC resistance measurements to recover the signed x, y and z
normal strains applied to the same 360-rod specimen. The JSON includes the 3×3
calibration matrix and compares full-matrix recovery with independent per-axis
readouts on four held-out tension/compression combinations.

Calibration and prediction both prescribe translations on the same boundary nodes;
interior nodes and rotations remain free. Rod areas and conductivities are fixed,
so resistance changes come from the computed changes in rod lengths. This is a
noiseless simulation of prescribed normal strains, not shear recovery or an
experimental resolution estimate.

## Recovering normal and shear strains with diagonal electrodes

```bash
python -m examples.mechanoelectric_tensor_readout_3d
```

Three axial electrical readouts cannot distinguish six independent strain
components. This example uses six additional diagonal electrode pairs, tagged by
node ID on the same 360-rod specimen, then reconstructs held-out mixtures using
only the six diagonal circuits. The output lists each electrode's node IDs, the
calibration matrix, and the errors relative to the axial-only control.

All load cases use the same boundary fixture. The three shear inputs use the
engineering convention, for example `gamma_xy = 2 epsilon_xy`. Circuits are driven
one at a time. Rod area and conductivity are fixed. The recovered quantities
are prescribed small boundary strains, not local strains or measured sensor
resolution.

## Solve a tagged Gmsh mesh with rigid grips

Use [the named-grip example](../docs/TET4_VTU_BUNDLE.md#run-a-tagged-mesh-above-the-banded-solver-limits)
to import a one- or two-material solid, prescribe a grip stroke and write nodal
results with stable IDs. Add `--write-vtu` to open the deformation in ParaView,
or omit the mesh input to run on a generated 28,512-cell fixture.

## Predicting strain-induced resistance in a fused solid lattice

From a source checkout:

```bash
python -m examples.tet4_mechanoelectric_looped_network
python -m examples.tet4_mechanoelectric_looped_network --matched-force
python -m examples.tet4_mechanoelectric_looped_network --geometry-breakdown
```

Three axial paths share four crosslinks in a 3D, square-section lattice.
Compare uniform stiffness, uniformly doubled stiffness, and one axial half
at twice the stiffness, all at fixed conductivity. The same electrode
node IDs follow the deformed solid, so the measured resistance change
comes from geometry. The output includes the tensile grip reaction,
unstrained resistance, and the normalized resistance slope near zero strain.

With `--matched-force`, linear elasticity sets a different grip displacement
for each material layout. The example re-solves both signs and checks that
the computed grip reactions match −0.5 N and +0.5 N. The normalized resistance
slope is then reported per newton rather than per unit strain. The grip is
still rigid; this mode is not a prescribed uniform traction on a free cap.

The `--geometry-breakdown` mode separates stretch along the local current
from transverse contraction using undeformed Joule-power weights. Their sum
matches the small-force resistance slope. These are signed geometric
contributions, not an intrinsic change in conductivity.

This example uses the bounded Tet4 solvers, a 387-node synthetic lattice,
and a small prescribed grip displacement. It does not model intrinsic
piezoresistivity, contact resistance, or electrical feedback into mechanics.

For a finer connected lattice and a five-point clamped stroke sweep,
install `fibernet[solid-iterative]` from a checkout and run:

~~~sh
python -m examples.tet4_iterative_mechanoelectric_sweep
~~~

The example does one elastic solve and reuses its displacement shape under
proportional strokes, with a separate DC solve on all five geometries.
Use this shortcut only for the same linear material, fixed grips and no preload.

The default uses 28,512 tetrahedra. Pass `--refinement 4` or
`--refinement 5` to run the same geometry, grip strokes, materials, and
electrodes on 67,584 or 132,000 tetrahedra. The meshes are generated locally;
no mesh download is needed. Use the default for a smoke test and the larger
tiers to measure runtime or check sensitivity to mesh resolution. All tiers
report node and tetrahedron counts in the JSON.

If you only need the initial slope, run:

~~~sh
python -m examples.tet4_iterative_mechanoelectric_sweep --initial-slope
~~~

This variant uses one undeformed DC solve to get the slope of relative
resistance per micrometre of grip travel. It also reports the signed
current-aligned stretch and transverse contraction contributions at fixed
conductivity and electrode voltages. The same refinement option works for
the initial slope. Use the five-point sweep for resistance values at finite
displacements.

## Compare where a small force changes resistance

From a checkout with the `solid-iterative` dependencies:

~~~sh
python -m examples.tet4_mechanoelectric_patch_screen --verify
~~~

The example ranks twelve equal-area exterior patches of a synthetic
three-dimensional lattice. Each candidate uses the same total force
direction with uniform Cartesian traction over its own patch. Geometry,
elasticity, electrode voltages and conductivity stay fixed. The output is
the initial relative resistance slope per newton. Patch locations change
the moment about the held face, which is reported along with the
centroid and area.

One electrical reference solve and one mechanical adjoint give all the
initial scalar slopes. The optional check replays three patches with
independent mechanical solutions and two deformed electrical states each.
For full displacement fields or finite resistance curves use forward
solves. This example does not model contact or a material conductivity
change.

## 3D porosity before and after loading

From the source checkout:

```bash
python -m examples.mechano_porosity_3d --strain 0.005 --samples 768 --axis x
```

Compress and stretch the same 360-rod, mixed-stiffness 3D specimen using the
linear beam solver. Then measure the void inside a fixed box, in mm³. Finite
straight rods have fixed radii; overlaps at fused junctions count only once.
Both sampling offsets typically show about +0.006% void volume under 0.5%
compression and −0.006% under 0.5% tension for this specimen.

The ray integrals are analytic along each ray and sampled in the two transverse
coordinates. Increase `--samples` or change `--axis` to check numerical sensitivity;
the spread between the two offsets is not an error bound. No CAD tools or fluid
solver are required. Void volume does not give connected porosity or permeability.

## Steady periodic flow through a 3D pore

From a source checkout, using Python 3.10 or newer:

```bash
python -m pip install "fibernet[pore-flow]"
python -m examples.tet4_periodic_pore
```

Compare a square duct with and without a solid obstacle using the same
imposed pressure gradient. The solver computes steady Stokes flow on a
tetrahedral fluid mesh, separately from the geometric porosity example.
For a matching package and checkout, see the
[4.25.0 release guide](../docs/RELEASE_4_25_0.md#solve-a-small-periodic-pore-flow-problem).

## Separate geometry and conductivity effects

On a two-terminal Tet4 mesh, take the displacement and node IDs from a
mechanics solve and supply both conductivity tables in S/m:

```python
from fibernet.sim.tet4_resistance_attribution import compare_tet4_resistance

parts = compare_tet4_resistance(
    mesh, mechanics.displacement_m, node_ids=mechanics.node_ids,
    reference_conductivity_S_per_m_by_material=sigma_before,
    changed_conductivity_S_per_m_by_material=sigma_after,
    electrode_voltage_V_by_id=electrodes,
)
```

The result contains four resistances in ohms. The shape, material and
interaction log ratios sum to the combined log ratio. The supplied scalar
conductivities are a material input; the mesh topology and electrodes stay fixed.

## How solid deformation changes pore flow

For up to eight deformed meshes compared with the same pore, call
`fibernet.sim.tet4_pore.solve_matched_pore_geometry_cases` with an ordered
list or tuple of `Tet4Bundle` instances. Use the same stable IDs, ordered
tetrahedra, material labels, viscosity and pressure-gradient driver.
The routine solves the reference geometry once and solves each deformed
geometry independently. Each still needs the supported periodic caps
and a cell volume ratio between 0.75 and 1.25 relative to the reference.

For a solid enclosed by a pore with an exactly matching wall triangulation,
prepare the transfer once and apply each solid displacement in meters:

```python
from fibernet.sim.tet4_pore_interface import prepare_conforming_pore_wall

wall = prepare_conforming_pore_wall(solid, pore)
reshaped = [wall.deform(u.displacement_m, node_ids=u.node_ids) for u in solid_results]
```

Both input meshes use the same coordinate unit. The pore's outer box stays
fixed, and the harmonic extension factorization is reused across the loads.

Using the same Python 3.10+ source checkout and `pore-flow` extra:

```bash
python -m examples.tet4_solid_to_pore_flow
```

Solve small-strain elasticity in all three bonded paths under a prescribed
grip displacement. Transfer the solid boundary displacements to matching pore
mesh nodes, keep the duct exterior fixed, and recompute steady Stokes flow at
the same pressure gradient. The example prints the computed grip reaction,
bulk and end-face flow changes, and the pore cell-volume ratio range. It is a
synthetic voxel scaffold and a one-way reference/deformed-geometry calculation;
the fluid does not load the solid.

Add `--matched-force` to run the same network at 1 and 2 GPa, with
unchanged Poisson ratio, viscosity, and pressure gradient. A calibration solve
determines which grip displacement produces the requested force; the example
then verifies the actual reactions at ±0.15 N and reports the centered
fractional bulk-flow change per newton. The stiffer case deforms roughly
half as much at equal force, so its small-load fractional response is
approximately halved in this fixture.

## Notched-network fracture scan

From a source checkout with `.[fracture]` installed:

```bash
python -m examples.fracture_scan --output-dir fracture_scan
```

Run three seeded, 63-node notched networks at two matched loading horizons.
The new directory receives complete case inputs/results and CSV/JSON summaries
of peak force, observed path loss and the work/elastic-energy ledger. Unobserved
path-loss thresholds remain null. See the [fracture scan guide](../docs/FRACTURE_METHODS.md#reproducible-notched-network-scan)
for units, reproduction and outcome states.

## Historical quick-start commands

```bash
# Selected historical commands below require their own capability review
python examples/01_mechanical_analysis.py
python examples/02_structure_comparison.py
python examples/04_buckling_analysis.py
python examples/05_chiral_braided_structures.py
python examples/06_uncertainty_quantification.py
```

## Examples Overview

| Example | Description | Key Modules |
|---------|-------------|-------------|
| 01 | Mechanical analysis of random networks | `FiberFEM`, `MorphologyAnalyzer` |
| 02 | Structure comparison (random vs lattice) | `TopologyAnalyzer`, generators |
| 03 | Historical CTE/thermal-deformation notice; unsupported | Original source archived; no current CTE backend |
| 04 | Buckling analysis | `BucklingAnalyzer` |
| 05 | Chiral & braided structures | `chiral_metamaterial`, `braided_rope` |
| 06 | Uncertainty quantification | `monte_carlo_ensemble` |

## Current explicit transient thermal example

```bash
python -m examples.transient_thermal_explicit
```

Run from the project root. The default 0.02-second interval is a quick entry-point demo, not full relaxation or new time-accuracy validation. This independent example supplies heat capacities (J/K), edge conductances (W/K), initial/fixed temperatures (K), sources (W), and time (s) explicitly. It uses the isolated current module; it is **not** a replacement for old steady-state conductivity or CTE calculations. No effective conductivity, thermal expansion or deformation is computed. See [thermal usage](../docs/integrations/TRANSIENT_THERMAL_USAGE.md) and [migration notes](../docs/integrations/THERMAL_MIGRATION.md).

## Historical thermal callers

`03_thermal_analysis.py` is now an import-safe unsupported notice; its byte-preserved original is in `_archived/03_thermal_analysis.py` and should not be executed.

These four multi-feature scripts are unchanged this round, but contain unsupported historical thermal calls and are **not supported as complete runnable workflows**:
- `research_case_study.py`: old `api.simulate_thermal` / effective-conductivity call
- `comprehensive_demo.py`: unavailable root `simulate_thermal` call
- `full_workflow.py`: unavailable root `simulate_thermal` call
- `integration_examples.py`: unavailable `ThermalSolver` import/calls

Their other paths have not been reviewed by this migration. Do not infer that skipping the thermal section makes the entire script valid. Old `api.simulate_thermal` and thermal configuration creation/execution fail explicitly without silently selecting a different model.

## Available Generators

### 2D Structures
- `random_straight_2d` - Random straight fibers
- `square_lattice_2d` - Square lattice
- `triangular_lattice_2d` - Triangular lattice
- `kagome_lattice_2d` - Kagome lattice
- `honeycomb_lattice_2d` - Honeycomb structure
- `electrospun_network` - Electrospun mat simulation
- `paper_network` - Paper-like random network

### 3D Structures
- `random_straight_3d` - 3D random network
- `cubic_lattice_3d` - Cubic lattice
- `diamond_lattice_3d` - Diamond lattice
- `octet_truss_3d` - Octet truss structure
- `woven_3d_orthogonal` - 3D orthogonal weave
- `foam_like_3d` - Foam-like cellular structure

### Special Structures
- `chiral_metamaterial` - Chiral/auxetic structures
- `braided_rope` - Braided fiber ropes
- `plain_weave_2d` - Plain weave textile
- `twill_weave_2d` - Twill weave textile
- `satin_weave_2d` - Satin weave textile
- `twisted_bundle` - Twisted fiber bundles
- `hierarchical_bundle` - Hierarchical fiber bundles
- `biomimetic_collagen` - Collagen-like networks
- `biomimetic_fibrin` - Fibrin-like networks
- `fractal_network` - Fractal fiber networks
- `gradient_density_network` - Graded density networks
- `kirigami_structure` - Kirigami-inspired structures
- `auxetic_structure` - Negative Poisson's ratio

## Research Applications

### Composite Materials
```python
# Generate fiber-reinforced composite
net = gen.fiber_reinforced_composite(
    fiber_volume_fraction=0.6,
    fiber_alignment=0.8,
)
```

### Biological Tissues
```python
# Generate collagen-like network
net = gen.biomimetic_collagen(
    density=0.3,
    branching_probability=0.2,
)
```

### Filtration Media
```python
# Generate electrospun filter
net = gen.electrospun_network(
    num_fibers=1000,
    fiber_diameter_range=(0.1, 1.0),
)
```

### Metamaterials
```python
# Generate auxetic structure
net = gen.auxetic_structure(
    cell_type='re-entrant',
    num_cells=(5, 5),
)
```

## Fixed 81-node static network

`python -m fibernet.examples.linear_static_network --output-dir static_network`
runs a synthetic 208-member planar welded frame at F and 2F, checks balance and
independent strain energy, and saves complete fields plus an A4 SVG/PNG.
The installed-package entry point and the checkout alias
`python -m examples.linear_fem_quickstart --case network81 --output-dir static_network`
use one implementation. Output directories must be new. See the
[linear FEM guide](../docs/architecture/linear-fem-quickstart.md#a-representative-81-node-static-network)
for SI inputs, model limits and saved-only rendering.
