# FiberNet 🧶

**A comprehensive Python toolkit for generation, simulation, and analysis of fiber network structures.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Homepage](https://img.shields.io/badge/Homepage-ml--biomat.com-orange)](https://ml-biomat.com/)

## Overview

FiberNet is a research-grade toolkit designed for studying fiber network structures across multiple scales and physics. It provides tools for:

- **Generation**: Create diverse 2D/3D fiber networks — from random Mikado models to chiral metamaterials
- **Simulation**: Mechanical (FEM), dynamics (MD/Brownian), fracture, thermal, electromagnetic
- **Analysis**: Topology, morphology, effective properties, percolation
- **Visualization**: 2D plots, 3D rendering, animations

Built for materials science researchers targeting high-impact publications (Nature Materials, Advanced Materials, etc.).

## Installation

```bash
# Basic installation
pip install fibernet

# With GPU acceleration (Taichi)
pip install fibernet[gpu]

# With 3D visualization (PyVista)
pip install fibernet[viz3d]

# Full installation
pip install fibernet[all]

# Development
pip install fibernet[dev]
```

## Quick Start

```python
import fibernet as fn
from fibernet import gen, sim, viz, analysis

# Generate a random 3D fiber network
network = gen.random_straight_3d(
    num_fibers=200,
    fiber_length=15.0,
    box_size=(50, 50, 50),
    radius=0.2,
    seed=42,
)

# Analyze morphology
morph = analysis.MorphologyAnalyzer(network)
print(morph.full_report())

# Run mechanical simulation
fem = sim.FiberFEM(network, segments_per_fiber=5)
result = fem.apply_uniaxial_strain(strain=0.01, axis=0)
print(f"Max displacement: {result.max_displacement():.6f}")
print(f"Effective modulus: {fem.effective_modulus(0.001, axis=0):.2e} Pa")

# Visualize
viz.plot_network_2d(network, save_path="network.png")
```

## Structure Types

### Disordered Networks
```python
# 2D Mikado model
net = gen.random_straight_2d(num_fibers=500, fiber_length=10, box_size=(50, 50))

# 3D random with preferred orientation
net = gen.random_straight_3d(num_fibers=300, orientation_bias=[1,0,0], orientation_spread=0.3)

# Random walk (semi-flexible polymers)
net = gen.random_walk_fibers(num_fibers=50, persistence_length=5.0)
```

### Ordered Lattices
```python
# Square lattice
net = gen.square_lattice_2d(spacing=5.0, grid_size=(10, 10))

# Honeycomb
net = gen.honeycomb_lattice_2d(cell_size=3.0)

# Octet truss (3D lightweight structure)
net = gen.octet_truss_3d(spacing=5.0, grid_size=(3, 3, 3))
```

### Chiral & Complex Structures
```python
# Double helix (DNA-like)
net = gen.double_helix(helix_radius=5, pitch=3, num_turns=5)

# Braided rope
net = gen.braided_rope(num_strands=4, rope_radius=3)

# Twisted bundle
net = gen.twisted_bundle(num_fibers=19, twist_angle=np.pi/4)

# Chiral metamaterial
net = gen.chiral_metamaterial(unit_cell_size=10, grid_size=(3, 3, 3))
```

### Woven Structures
```python
# Plain weave
net = gen.plain_weave_2d(spacing=2.0, grid_size=(20, 20))

# Twill weave (2/2)
net = gen.twill_weave_2d(twill_pattern=(2, 2))

# 3D orthogonal woven
net = gen.woven_3d_orthogonal(grid_size=(5, 5, 3))
```

### Hierarchical & Multi-scale
```python
# Multi-level hierarchical bundle
net = gen.hierarchical_bundle(levels=3, fibers_per_level=[7, 7, 7])

# Gradient density network
net = gen.gradient_density_network(density_gradient="gaussian")

# Fractal branching
net = gen.fractal_network(iterations=4, branch_factor=3)
```

## Simulation Capabilities

### Mechanical (FEM)
```python
fem = sim.FiberFEM(network)
result = fem.solve_static(forces=F, fixed_nodes=[0, 1, 2])
E_eff = fem.effective_modulus(strain=0.001, axis=0)
strains, stresses = sim.stress_strain_curve(network, max_strain=0.05)
```

### Dynamics
```python
dyn = sim.FiberDynamics(network, dt=1e-6, temperature=300)
result = dyn.run_verlet(num_steps=10000)
```

### Fracture
```python
frac = sim.FiberFracture(network, failure_criterion="max_stress")
result = frac.run_progressive_failure(max_strain=0.1)
```

### Thermal
```python
thermal = sim.ThermalSolver(network)
result = thermal.solve_steady_state(T_hot=100, T_cold=0)
print(f"Effective conductivity: {result.effective_conductivity:.2f} W/(m·K)")
```

### Electromagnetic
```python
em = sim.EMSolver(network)
result = em.solve_conductivity(voltage=1.0, axis=0)
print(f"Effective conductivity: {result.effective_conductivity:.2e} S/m")
```

### Multi-physics Coupling
```python
# Thermo-mechanical
tm = sim.ThermoMechanical(network)
result = tm.solve(delta_T=100)

# Piezoresistive
pz = sim.PiezoResistive(network, gauge_factor=2.0)
strains, dR = pz.resistance_vs_strain()
```

## Materials Database

```python
from fibernet.core.material import get_material, list_materials

# Built-in materials
print(list_materials())
# ['actin', 'aluminum', 'carbon_fiber', 'cellulose', 'collagen', 'cnt',
#  'copper', 'fibrin', 'glass_fiber', 'graphene_sheet', 'kevlar', 'nylon',
#  'pdms', 'pla', 'pva', 'silk', 'steel', 'titanium', 'uHMWPE', ...]

# Custom material
from fibernet.core.material import Material
my_mat = Material(name="my_fiber", density=1500, youngs_modulus=50e9, poissons_ratio=0.25)
```

## I/O Formats

```python
# JSON
network.save_json("my_network.json")
net = fn.FiberNetwork.load_json("my_network.json")

# HDF5 (for large networks)
network.save_hdf5("my_network.h5")

# VTK (for ParaView)
from fibernet.utils import export_vtk
export_vtk(network, "network.vtk")

# CSV (for data analysis)
from fibernet.utils import export_csv
export_csv(network, "fibers.csv")

# STL (for 3D printing)
from fibernet.utils import export_stl
export_stl(network, "network.stl")
```

## Architecture

```
fibernet/
├── core/          # Data structures (Fiber, FiberNetwork, Material)
├── gen/           # Network generators (25+ structure types)
├── sim/           # Simulation engines (6 physics domains)
├── analysis/      # Topology, morphology, properties
├── viz/           # 2D/3D visualization
├── utils/         # I/O, geometry utilities
└── examples/      # Example scripts and notebooks
```

## Citing FiberNet

If you use FiberNet in your research, please cite:

```bibtex
@software{fibernet2026,
  title = {FiberNet: A Comprehensive Toolkit for Fiber Network Research},
  author = {ML-BioMat Lab},
  year = {2026},
  url = {https://github.com/GellmanSparrowS/fibernet}
}
```

## Dependencies

### Required
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- NetworkX ≥ 3.0
- Matplotlib ≥ 3.7
- h5py ≥ 3.8
- PyYAML ≥ 6.0

### Optional
- **Taichi** ≥ 1.6 (GPU-accelerated simulations)
- **PyVista** ≥ 0.40 (3D rendering)
- **scikit-learn** ≥ 1.2 (Advanced analysis)

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see our GitHub repository for guidelines.

---

Built with ❤️ by [ML-BioMat Lab](https://ml-biomat.com/)
