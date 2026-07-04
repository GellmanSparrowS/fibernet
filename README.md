# FiberNet

**A comprehensive Python toolkit for fiber network structure generation, simulation, and analysis.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-202%20passing-green.svg)]()
[![Version](https://img.shields.io/badge/version-0.4.0-blue.svg)]()
[![GitHub](https://img.shields.io/github/stars/GellmanSparrowS/fibernet?style=social)](https://github.com/GellmanSparrowS/fibernet)

FiberNet enables researchers to generate, analyze, simulate, and visualize fiber network structures for applications in materials science, biomechanics, polymer physics, and composites engineering.

## Features

| Category | Capabilities |
|----------|-------------|
| **Generation** | 50+ generators: random, ordered, chiral, woven, hierarchical, biomimetic, CNT, paper, textile, electrospun |
| **Simulation** | Mechanical (linear/nonlinear), dynamics, fracture, thermal, electromagnetic, fluid flow, acoustic |
| **Analysis** | Morphology, topology, spectral, pore structure, anisotropy, networkx graph analysis, statistics |
| **ML Integration** | Feature extraction, property prediction with sklearn, dataset generation |
| **I/O** | JSON, LAMMPS, VTK, GMSH, PDB, XYZ formats |
| **Acceleration** | Taichi CPU/GPU parallel computing |
| **Units** | SI, CGS, micro, nano, molecular unit systems |

## Installation

```bash
pip install fibernet

# With all optional dependencies
pip install fibernet[full]

# Development install
git clone https://github.com/GellmanSparrowS/fibernet.git
cd fibernet && pip install -e ".[dev]"
```

## Quick Start

```python
import fibernet as fn
from fibernet import gen

# Generate a random 2D fiber network
net = gen.random_straight_2d(
    num_fibers=100,
    fiber_length=15.0,
    box_size=(50, 50),
    radius=0.1,
    seed=42
)

# Quick analysis
results = fn.analyze(net)
print(f"Nematic order: {results['nematic_order']:.3f}")

# Run mechanical simulation
mech = fn.simulate_mechanics(net, strain=0.01, axis=0)
print(f"Effective modulus: {mech['modulus']:.2e} Pa")

# Export to VTK for Paraview visualization
fn.export(net, "network.vtk")
```

## Network Generators

### Disordered Networks
- `random_straight_2d` / `random_straight_3d` — Random straight fibers
- `random_walk_fibers` — Random walk (polymer-like)
- `random_curved_fibers` — Curved fibers with persistence
- `poisson_voronoi_2d` / `_3d` — Voronoi-based networks
- `electrospun_mat` — Electrospun nanofiber mats

### Ordered Structures
- `square_lattice_2d` / `cubic_lattice_3d` — Regular lattices
- `honeycomb_lattice_2d` — Honeycomb structure
- `triangular_lattice_2d` — Triangular lattice
- `diamond_lattice_3d` — Diamond cubic
- `octet_truss_3d` — Octet truss

### Chiral & Special
- `double_helix` — DNA-like double helix
- `triple_helix` — Collagen-like triple helix
- `chiral_braided` — Braided chiral structures
- `auxetic_reentrant` — Auxetic (negative Poisson ratio)

### Woven & Textile
- `plain_weave_2d` / `plain_weave_3d` — Plain weave
- `satin_weave` — Satin weave pattern
- `twill_weave` — Twill weave pattern
- `textile_weave` — General weave with crimp

### Biomimetic
- `biomimetic_collagen` — Collagen-like D-banding
- `cnt_network_2d` / `cnt_network_3d` — Carbon nanotube networks
- `paper_network` — Cellulose paper fibers
- `fiber_reinforced_composite` — Composite structures

### Hierarchical
- `hierarchical_bundle` — Multi-scale bundled fibers
- `gyroid_scaffold` — Gyroid triply periodic surface
- `foam_scaffold` — Stochastic foam structure
- `voronoi_fiber` — Voronoi fiber networks

## Simulation Capabilities

### Mechanical (Linear)
```python
from fibernet.sim.mechanical import FiberFEM

fem = FiberFEM(net, segments_per_fiber=5)
result = fem.apply_uniaxial_strain(strain=0.01, axis=0)
print(f"Energy: {result.energy:.4e} J")
print(f"Modulus: {fem.effective_modulus(strain=0.001, axis=0):.2e} Pa")
```

### Nonlinear Mechanics
```python
from fibernet.sim.nonlinear import NonlinearFEM, BilinearPlasticity

model = BilinearPlasticity(E=1e9, sigma_y=1e7, Et=1e8)
fem = NonlinearFEM(net, constitutive_model=model)
strains, stresses, energies = fem.stress_strain_curve(axis=0, max_strain=0.05)
```

Constitutive models: Neo-Hookean, Mooney-Rivlin, Arruda-Boyce, Bilinear Plasticity, Power-Law Hardening, Maxwell/Kelvin-Voigt Viscoelasticity

### Fluid Flow
```python
from fibernet.sim.fluid import DarcySolver

solver = DarcySolver(net)
porosity = solver.compute_porosity()
K = solver.kozeny_carman_permeability()
print(f"Permeability: {K:.4e} m²")
```

### Thermal
```python
thermal = fn.simulate_thermal(net, T_hot=100, T_cold=0)
print(f"Conductivity: {thermal['conductivity']:.2f} W/(m·K)")
```

### Acoustic
```python
from fibernet.sim.acoustic import AcousticSolver

solver = AcousticSolver(net, segments_per_fiber=5)
result = solver.compute_modes(num_modes=20)
print(f"Fundamental frequency: {result.fundamental_frequency():.2f} Hz")
```

## Machine Learning

```python
from fibernet.ml.features import extract_features
from fibernet.ml.predictor import PropertyPredictor

# Extract structural features
features = extract_features(net)

# Train property predictor
predictor = PropertyPredictor(model_type='random_forest')
predictor.fit(networks, properties)

# Predict
predicted = predictor.predict(new_network)
```

## Transformations

```python
from fibernet.core.transform import rotate, scale, mirror, merge, tile

rotated = rotate(net, angle=0.785, axis=[0, 0, 1])  # 45° rotation
scaled = scale(net, factor=2.0)
mirrored = mirror(net, axis=0)
merged = merge([net1, net2, net3])
tiled = tile(net, repeats=(2, 2, 1))  # 2×2 supercell
```

## Periodic Boundary Conditions

```python
from fibernet.core.pbc import PeriodicBox, apply_pbc, compute_rdf

wrapped, box = apply_pbc(net, box_size=[50, 50, 50])
r, g = compute_rdf(wrapped, box, r_max=20, num_bins=50)
```

## Unit Systems

```python
from fibernet.utils.units import UnitConverter

# Convert between unit systems
E_si = 200e9  # Pa
E_cgs = UnitConverter.from_si(E_si, 'stress', to_unit='CGS')  # dyn/cm²
```

## I/O Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| JSON | `.json` | Native serialization |
| LAMMPS | `.lammps` | Molecular dynamics |
| VTK | `.vtk` | Paraview/VisIt visualization |
| GMSH | `.msh` | FEM meshing |
| PDB | `.pdb` | Molecular visualization |
| XYZ | `.xyz` | Simple atomic coordinates |

## Project Structure

```
fibernet/
├── core/           # Data structures (Fiber, Network, Material, PBC, Crosslinks)
├── gen/            # Network generators (50+)
├── sim/            # Simulation engines (9 physics modules)
├── analysis/       # Structural analysis (morphology, topology, spectral)
├── io/             # I/O format support (6 formats)
├── ml/             # Machine learning integration
├── viz/            # Visualization
├── utils/          # Utilities (units, validation)
└── api.py          # High-level convenience API
```

## Testing

```bash
pytest tests/ -v          # Run all 202 tests
pytest tests/ --cov       # With coverage
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. We welcome:
- New generators
- Simulation models
- Analysis tools
- Bug fixes and improvements

## License

MIT License. See [LICENSE](LICENSE).

## Citation

If you use FiberNet in your research, please cite:

```bibtex
@software{fibernet2026,
  title = {FiberNet: A Comprehensive Fiber Network Simulation Toolkit},
  author = {FiberNet Contributors},
  year = {2026},
  url = {https://github.com/GellmanSparrowS/fibernet}
}
```
