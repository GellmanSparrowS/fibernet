# FiberNet Examples

Research-ready examples demonstrating FiberNet capabilities.

## Quick Start

```bash
# Run any example
python examples/01_mechanical_analysis.py
python examples/02_structure_comparison.py
python examples/03_thermal_analysis.py
python examples/04_buckling_analysis.py
python examples/05_chiral_braided_structures.py
python examples/06_uncertainty_quantification.py
```

## Examples Overview

| Example | Description | Key Modules |
|---------|-------------|-------------|
| 01 | Mechanical analysis of random networks | `FiberFEM`, `MorphologyAnalyzer` |
| 02 | Structure comparison (random vs lattice) | `TopologyAnalyzer`, generators |
| 03 | Thermal analysis (CTE) | `ThermalAnalyzer` |
| 04 | Buckling analysis | `BucklingAnalyzer` |
| 05 | Chiral & braided structures | `chiral_metamaterial`, `braided_rope` |
| 06 | Uncertainty quantification | `monte_carlo_ensemble` |

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
