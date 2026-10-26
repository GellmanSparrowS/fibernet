# FiberNet Examples

Examples have different support status. Historical scripts may depend on unavailable APIs; the index is not a blanket claim that every example runs.

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
