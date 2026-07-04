# FiberNet

**A comprehensive Python toolkit for fiber network structure generation, simulation, and analysis.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-344%20passing-green.svg)]()
[![Version](https://img.shields.io/badge/version-0.9.0-blue.svg)]()
[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)]()
[![GitHub](https://img.shields.io/github/stars/GellmanSparrowS/fibernet?style=social)](https://github.com/GellmanSparrowS/fibernet)

FiberNet enables researchers to generate, simulate, and analyze fiber network structures for applications in materials science, biomechanics, polymer physics, composites engineering, and more. Designed as a research-grade tool with emphasis on reproducibility and extensibility.

**Homepage**: [https://ml-biomat.com/](https://ml-biomat.com/)

## Key Features

| Category | Capabilities |
|----------|-------------|
| **Generation** | 50+ generators: random, ordered, chiral, woven, hierarchical, biomimetic, CNT, paper, textile, electrospun |
| **Simulation** | FEM (linear/nonlinear), mass-spring dynamics, fracture, thermal, electromagnetic, acoustic, DMA |
| **Crosslinks** | Rigid, spring, breakable, friction, bonded, covalent, hydrogen bond, ionic, physical entanglement |
| **Analysis** | Morphology, topology, spectral, pore structure, anisotropy, stress-strain, networkx graph analysis |
| **ML Integration** | Feature extraction, property prediction with sklearn, dataset generation |
| **I/O** | JSON, YAML, LAMMPS, VTK, GMSH, PDB, XYZ, pandas formats |
| **Acceleration** | Taichi CPU/GPU parallel FEM, parallel contact detection |
| **Visualization** | matplotlib 3D, pyvista interactive, stress coloring, damage evolution, animations |
| **Reproducibility** | YAML config system, ensemble generation, convergence studies, config hashing |
| **Units** | SI, CGS, micro, nano, molecular unit systems |

## Installation

```bash
# Basic installation
pip install fibernet

# With all optional dependencies (pyvista, sklearn, pandas, networkx)
pip install fibernet[full]

# Development install
git clone https://github.com/GellmanSparrowS/fibernet.git
cd fibernet && pip install -e ".[dev]"
```

## Quick Start

### Generate a Fiber Network

```python
from fibernet import gen

# Random 2D fiber network
net = gen.random_straight_2d(
    num_fibers=100,
    fiber_length=15.0,
    box_size=(50, 50),
    radius=0.1,
    seed=42
)

print(f"Fibers: {net.num_fibers}")
print(f"Crosslinks: {net.num_crosslinks}")

# 3D random network
net_3d = gen.random_straight_3d(
    num_fibers=200,
    fiber_length=20.0,
    box_size=(50, 50, 50),
    seed=42
)

# Ordered lattice
lattice = gen.square_lattice_2d(spacing=5.0, grid_size=(10, 10))

# Chiral network
chiral = gen.chiral_network_2d(
    num_fibers=50,
    fiber_length=15.0,
    box_size=(50, 50),
    chirality=0.5,
    seed=42
)

# Woven structure
woven = gen.woven_2d(
    warp_count=10,
    weft_count=10,
    spacing=5.0,
    weave_pattern='plain'
)
```

### Mechanical Simulation (FEM)

```python
from fibernet.sim import FiberFEM

# Create FEM solver
fem = FiberFEM(net, segments_per_fiber=5)

# Apply uniaxial strain
result = fem.apply_uniaxial_strain(strain=0.01, axis=0)

print(f"Energy: {result.energy:.4e} J")
print(f"Stress: {result.stress}")

# Extract stress-strain curve
from fibernet.analysis import extract_stress_strain
curve = extract_stress_strain(net, strain_range=(0, 0.05), num_steps=10)
print(f"Young's modulus: {curve.youngs_modulus:.2e} Pa")
print(f"Poisson's ratio: {curve.poissons_ratio:.3f}")
```

### GPU-Accelerated FEM with Taichi

```python
from fibernet.sim import TaichiFEMSolver
import numpy as np

solver = TaichiFEMSolver(arch="cpu", num_threads=4)

# Solve beam network
result = solver.solve_beam_network(
    node_positions=node_positions,
    elements=elements,
    youngs_modulus=1e9,
    radii=radii,
    fixed_nodes=[0, 1],
    applied_forces=applied_forces,
)

# Progressive damage simulation
damage = solver.progressive_damage(
    node_positions=node_positions,
    elements=elements,
    youngs_modulus=1e9,
    radii=radii,
    fixed_nodes=[0],
    strain_range=(0, 0.1),
    num_steps=20,
)
```

### Dynamic Mechanical Analysis (DMA)

```python
from fibernet.sim import GeneralizedMaxwell, frequency_sweep, temperature_sweep

# Create viscoelastic model (Prony series)
model = GeneralizedMaxwell(
    E_inf=1e9,           # Equilibrium modulus
    E_i=[5e8, 3e8, 2e8], # Relaxation moduli
    tau_i=[0.01, 0.1, 1.0]  # Relaxation times
)

# Frequency sweep
freq_result = frequency_sweep(model, freq_range=(0.01, 100), num_points=50)
print(f"Storage modulus range: {freq_result.storage_modulus.min()/1e9:.2f} - {freq_result.storage_modulus.max()/1e9:.2f} GPa")
print(f"Crossover frequency: {freq_result.crossover_frequency} rad/s")

# Temperature sweep
temp_result = temperature_sweep(model, temp_range=(250, 350), frequency=1.0)
print(f"Glass transition Tg: {temp_result.glass_transition_temperature:.1f} K")
```

### Advanced Crosslink Models

```python
from fibernet.core.crosslinks import (
    CovalentBond, HydrogenBond, PhysicalEntanglement, IonicBond,
    create_crosslink
)

# Covalent bond (Morse potential)
covalent = CovalentBond(stiffness=1e9, bond_energy=5e-19)

# Hydrogen bond (angle-dependent, reformable)
h_bond = HydrogenBond(strength=1e7, energy=3e-20, reform_probability=0.1)

# Physical entanglement (sliding contact)
entanglement = PhysicalEntanglement(friction=1e-6, slip_force=1e-6)

# Ionic bond (screened Coulomb)
ionic = IonicBond(charge1=1.6e-19, charge2=-1.6e-19, debye_length=1e-9)

# Factory function
crosslink = create_crosslink('covalent', stiffness=1e9)
```

### Thermal Simulation

```python
from fibernet.sim import ThermalSolver

solver = ThermalSolver(net)
result = solver.steady_state(temperature_diff=100.0, direction=0)
print(f"Effective conductivity: {result.effective_conductivity:.4e} W/(m·K)")
```

### Statistical Ensemble Generation

```python
from fibernet.utils.ensemble import generate_ensemble, ensemble_analysis
from fibernet.analysis.morphology import MorphologyAnalyzer

# Generate ensemble of networks
ensemble = generate_ensemble(
    gen.random_straight_2d,
    num_networks=50,
    base_seed=42,
    num_fibers=100,
    fiber_length=10.0,
    box_size=(50, 50)
)

# Run analysis on ensemble
def analyze(net):
    morph = MorphologyAnalyzer(net)
    return {
        'nematic_order': morph.nematic_order_parameter(),
        'porosity': morph.porosity(),
    }

stats = ensemble_analysis(ensemble, analyze)
print(f"Nematic order: {stats['nematic_order']['mean']:.3f} ± {stats['nematic_order']['std']:.3f}")
```

### Reproducible Experiments with Config

```python
from fibernet.utils.config import create_template_config, run_from_config

# Create experiment config
config = create_template_config('experiment.yaml', template_type='mechanical')

# Run experiment from config
results = run_from_config('experiment.yaml')
print(f"Young's modulus: {results['stress_strain'].youngs_modulus:.2e} Pa")
```

### Visualization

```python
from fibernet.viz import (
    visualize_3d_matplotlib,
    visualize_3d_pyvista,
    visualize_network_stress,
    visualize_damage_evolution
)

# Static 3D plot
fig, ax = visualize_3d_matplotlib(net, show_crosslinks=True)

# Interactive 3D (pyvista)
plotter = visualize_3d_pyvista(net, fiber_radius=0.1)

# Stress-colored visualization
import numpy as np
stress = np.random.uniform(0, 1e6, net.num_fibers)
fig, ax = visualize_network_stress(net, stress, cmap='coolwarm')

# Damage evolution plots
fig = visualize_damage_evolution(damage_result)
```

### Export to Simulation Software

```python
from fibernet.io import to_vtk, to_lammps, to_gmsh, to_pdb

# VTK (ParaView)
to_vtk(net, 'network.vtk')

# LAMMPS (molecular dynamics)
to_lammps(net, 'network.lammps')

# GMSH (finite element meshing)
to_gmsh(net, 'network.geo')

# PDB (Protein Data Bank)
to_pdb(net, 'network.pdb')
```

## Project Structure

```
fibernet/
├── core/              # Core data structures (Fiber, FiberNetwork, Crosslinks)
├── gen/               # 50+ network generators
├── sim/               # Simulation engines (FEM, dynamics, thermal, EM, DMA)
├── analysis/          # Analysis tools (morphology, topology, stress-strain)
├── materials/         # Materials database
├── io/                # I/O formats (VTK, LAMMPS, PDB, GMSH, XYZ)
├── ml/                # Machine learning integration
├── viz/               # Visualization (matplotlib, pyvista)
├── utils/             # Utilities (config, ensemble, units, parallel)
└── transforms/        # Network transformations (rotate, scale, tile)
```

## Documentation

Full documentation with API reference and tutorials: [https://fibernet.readthedocs.io/](https://fibernet.readthedocs.io/)

## Citation

If you use FiberNet in your research, please cite:

```bibtex
@software{fibernet2025,
  title = {FiberNet: A Comprehensive Python Toolkit for Fiber Network Generation, Simulation, and Analysis},
  author = {FiberNet Contributors},
  year = {2025},
  url = {https://github.com/GellmanSparrowS/fibernet},
  version = {0.9.0}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with NumPy, SciPy, NetworkX, matplotlib, and Taichi
- Inspired by research in computational materials science and biomechanics
- Supported by the [ML-BioMat](https://ml-biomat.com/) research group
