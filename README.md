# FiberNet

**A comprehensive Python toolkit for fiber network structure generation, simulation, and analysis.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-575%20passing-green.svg)]()
[![Version](https://img.shields.io/badge/version-1.13.0-blue.svg)]()
[![CI/CD](https://github.com/GellmanSparrowS/fibernet/actions/workflows/ci.yml/badge.svg)](https://github.com/GellmanSparrowS/fibernet/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/fibernet/badge/?version=latest)](https://fibernet.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)]()
[![GitHub](https://img.shields.io/github/stars/GellmanSparrowS/fibernet?style=social)](https://github.com/GellmanSparrowS/fibernet)

FiberNet enables researchers to generate, simulate, and analyze fiber network structures for applications in materials science, biomechanics, polymer physics, composites engineering, and more. Designed as a research-grade tool with emphasis on reproducibility and extensibility.

**Homepage**: [https://ml-biomat.com/](https://ml-biomat.com/)

## Key Features

| Category | Capabilities |
|----------|-------------|
| **Generation** | 57 generators: random, ordered, chiral, woven, hierarchical, biomimetic, CNT, paper, textile, electrospun |
| **Simulation** | FEM, dynamics, fracture, damage/fatigue, thermal, electromagnetic, acoustic, fluid, rheology, DMA, multi-scale |
| **Crosslinks** | Rigid, spring, breakable, friction, bonded, covalent, hydrogen bond, ionic, physical entanglement |
| **Analysis** | Morphology, topology, spectral, pore structure, anisotropy, percolation, multi-scale homogenization |
| **ML Integration** | Feature extraction, GNN models, property prediction, dataset generation |
| **I/O** | JSON, YAML, LAMMPS, VTK, GMSH, PDB, XYZ, pandas, HDF5 formats |
| **Acceleration** | Taichi CPU/GPU parallel FEM, parallel contact detection |
| **Visualization** | matplotlib 3D, pyvista interactive, plotly web, stress coloring, damage evolution, animations |
| **Reproducibility** | YAML config system, ensemble generation, convergence studies, config hashing |
| **Units** | SI, CGS, micro, nano, molecular unit systems |

## Installation

```bash
# Basic installation
pip install fibernet

# With all optional dependencies (pyvista, sklearn, pandas, networkx, plotly)
pip install fibernet[full]

# Development install
git clone https://github.com/GellmanSparrowS/fibernet.git
cd fibernet && pip install -e ".[dev]"
```

## Quick Start

### High-Level API

```python
import fibernet as fn

# Create a random 2D network
net = fn.create("random_2d", num_fibers=100, fiber_length=10.0, box_size=(30, 30), seed=42)

# Analyze structure
stats = fn.analyze(net)
print(f"Fibers: {stats['num_fibers']}, Order: {stats['nematic_order']:.3f}")

# Run mechanical simulation
result = fn.simulate_mechanics(net, strain=0.01)

# Transform
net_scaled = fn.scale(net, factor=2.0)
net_rotated = fn.rotate(net, angle=0.785, axis=[0, 0, 1])

# Visualize
fn.plot(net)

# Export
fn.export(net, "network.json", format="json")
fn.export(net, "network.vtk", format="vtk")
```

### Network Generation

```python
from fibernet import gen

# Random 2D fiber network
net = gen.random_straight_2d(
    num_fibers=100, fiber_length=15.0, box_size=(50, 50),
    radius=0.1, seed=42
)

# 3D random network
net_3d = gen.random_straight_3d(
    num_fibers=200, fiber_length=20.0, box_size=(50, 50, 50), seed=42
)

# Ordered lattices
square = gen.square_lattice_2d(spacing=5.0, grid_size=(10, 10))
honeycomb = gen.honeycomb_lattice_2d(cell_size=5.0, grid_size=(10, 10))
triangular = gen.triangular_lattice_2d(spacing=5.0, grid_size=(10, 10))
cubic = gen.cubic_lattice_3d(spacing=5.0, grid_size=(5, 5, 5))
octet = gen.octet_truss_3d(spacing=5.0, grid_size=(3, 3, 3))

# Specialized structures
chiral = gen.chiral_network_2d(
    num_fibers=50, fiber_length=15.0, box_size=(50, 50),
    chirality=0.5, seed=42
)
woven = gen.woven_2d(warp_count=10, weft_count=10, spacing=5.0)
helix = gen.single_helix(radius=5.0, pitch=2.0, turns=3, num_points=100)
dna = gen.double_helix(radius=5.0, pitch=2.0, turns=3)

# Biomimetic
collagen = gen.biomimetic_collagen(num_fibers=100, box_size=(50, 50), seed=42)
fibrin = gen.biomimetic_fibrin(num_fibers=100, box_size=(50, 50), seed=42)
electrospun = gen.electrospun(num_fibers=200, box_size=(50, 50), seed=42)
```

### Mechanical Simulation (FEM)

```python
from fibernet.sim import FiberFEM

# Create FEM solver
fem = FiberFEM(net, segments_per_fiber=5)

# Apply uniaxial strain
result = fem.apply_uniaxial_strain(strain=0.01, axis=0)
print(f"Energy: {result.energy:.4e} J")
print(f"Max stress: {result.max_stress:.2e} Pa")

# Compute effective modulus
E_eff = fem.effective_modulus(strain=0.01, axis=0)
print(f"Effective modulus: {E_eff:.2e} Pa")
```

### GPU-Accelerated FEM with Taichi

```python
from fibernet.sim import TaichiFEMSolver

solver = TaichiFEMSolver(arch="cpu", num_threads=4)
result = solver.solve_beam_network(
    node_positions=node_positions, elements=elements,
    youngs_modulus=1e9, radii=radii,
    fixed_nodes=[0, 1], applied_forces=applied_forces,
)

# Progressive damage simulation
damage = solver.progressive_damage(
    node_positions=node_positions, elements=elements,
    youngs_modulus=1e9, radii=radii,
    fixed_nodes=[0], strain_range=(0, 0.1), num_steps=20,
)
```

### Damage Mechanics and Fatigue

```python
from fibernet.sim import DamageMechanicsSolver, FatigueSolver

# Progressive failure under monotonic loading
damage_solver = DamageMechanicsSolver(
    net, youngs_modulus=1e9, tensile_strength=1e8
)
result = damage_solver.progressive_failure(max_strain=0.1, num_steps=100)
print(f"Peak load: {result.peak_load:.2e}")
print(f"Energy absorbed: {result.energy_absorbed:.2e}")
print(f"Residual stiffness: {damage_solver.residual_stiffness():.2e} Pa")

# Fatigue life prediction
fatigue = FatigueSolver(
    net, youngs_modulus=1e9, tensile_strength=1e8, fatigue_limit=3e7
)
N_f = fatigue.compute_cycles_to_failure(stress_amplitude=5e7)
sn_curve = fatigue.generate_sn_curve(stress_range=(0.3, 0.9))

# Miner's rule for variable amplitude
load_history = [(4e7, 1000), (6e7, 500), (8e7, 100)]
cumulative_damage = fatigue.miners_rule(load_history)
```

### Fracture Mechanics

```python
from fibernet.sim import CrackPropagationSolver

solver = CrackPropagationSolver(net, fracture_toughness=100.0)
tip = solver.initialize_crack(
    tip_position=np.array([15.0, 15.0, 0.0]),
    tip_direction=np.array([1.0, 0.0, 0.0]),
    initial_length=2.0,
)
```

### Thermal Simulation

```python
from fibernet import simulate_thermal

result = simulate_thermal(net, T_hot=100.0, T_cold=0.0)
print(f"Temperature field shape: {result['temperatures'].shape}")
print(f"Heat flux: {result['heat_flux']:.4e} W/m²")
```

### Rheology (Fiber Suspensions)

```python
from fibernet.sim import FiberSuspensionRheology

rheo = FiberSuspensionRheology(
    net, fluid_viscosity=1.0, aspect_ratio=20.0,
    volume_fraction=0.01, interaction_parameter=0.01
)

# Effective viscosity
eta = rheo.compute_effective_viscosity(shear_rate=10.0)
print(f"Effective viscosity: {eta:.4f} Pa·s")

# Jeffery orbit for single fiber
orbit = rheo.jeffery_orbit(
    initial_orientation=np.array([1.0, 0.0, 0.0]),
    shear_rate=1.0, total_time=10.0, num_steps=500
)
print(f"Jeffery orbit period: {orbit.period:.4f} s")

# Orientation evolution (Folgar-Tucker)
a_history = rheo.orientation_evolution(
    shear_rate=1.0, total_time=5.0, num_steps=50
)
```

### Electromagnetic Simulation

```python
from fibernet.sim import EMSolver
from fibernet.core import Material

# Create conductive network
mat = Material(name="carbon", electrical_conductivity=1e6)
net_cond = gen.random_straight_2d(
    num_fibers=100, fiber_length=12.0, box_size=(30, 30),
    material=mat, seed=42
)

solver = EMSolver(net_cond)
result = solver.solve_conductivity(voltage=1.0, axis=0)
print(f"Effective conductivity: {result.effective_conductivity:.2e} S/m")
print(f"Percolating: {result.is_percolating}")

# Percolation analysis
volumes, probs = solver.percolation_analysis(num_samples=20)
```

### Percolation Analysis

```python
from fibernet.analysis import PercolationAnalyzer

analyzer = PercolationAnalyzer(net)
result = analyzer.analyze()
print(f"Percolates: {result.percolates}")
print(f"Largest cluster: {result.largest_cluster_size}")
print(f"Percolation probability: {result.percolation_probability:.3f}")
print(f"Effective conductivity: {result.effective_conductivity:.4f}")

# Cluster statistics
clusters = analyzer.cluster_analysis()
print(f"Number of clusters: {clusters['n_clusters']}")
```

### Multi-Scale Homogenization

```python
from fibernet.sim import HomogenizationSolver, compute_effective_properties

# Full homogenization
solver = HomogenizationSolver(
    net, fiber_youngs_modulus=1e9, fiber_poissons_ratio=0.3,
    fiber_thermal_conductivity=0.5, fiber_density=1000.0
)
props = solver.homogenize()
print(f"Effective Young's modulus: {props.effective_youngs_modulus:.2e} Pa")
print(f"Porosity: {props.porosity:.4f}")
print(f"Is isotropic: {props.is_isotropic}")

# Quick convenience function
props = compute_effective_properties(net)
```

### Dynamic Mechanical Analysis (DMA)

```python
from fibernet.sim import GeneralizedMaxwell, frequency_sweep, temperature_sweep

model = GeneralizedMaxwell(
    E_inf=1e9, E_i=[5e8, 3e8, 2e8], tau_i=[0.01, 0.1, 1.0]
)

freq_result = frequency_sweep(model, freq_range=(0.01, 100), num_points=50)
temp_result = temperature_sweep(model, temp_range=(250, 350), frequency=1.0)
print(f"Glass transition Tg: {temp_result.glass_transition_temperature:.1f} K")
```

### Fluid Flow Through Porous Media

```python
from fibernet.sim import DarcySolver, PoreNetworkModel

pore_model = PoreNetworkModel(net)
solver = DarcySolver(net, fluid_viscosity=1e-3, pressure_gradient=1e4)
result = solver.solve()
print(f"Permeability: {result['permeability']:.4e} m²")
```

### Machine Learning Integration

```python
from fibernet.ml import FeatureExtractor, GNNFeatureExtractor

# Extract structural features
extractor = FeatureExtractor()
features = extractor.extract_features(net)
print(f"Feature vector: {features.shape}")

# GNN-compatible graph data
gnn = GNNFeatureExtractor(
    node_features=['position', 'degree'],
    edge_features=['length', 'angle']
)
graph_data = gnn.extract_graph(net)
```

### Statistical Ensemble Generation

```python
from fibernet.utils.ensemble import generate_ensemble, ensemble_analysis
from fibernet.analysis import MorphologyAnalyzer

ensemble = generate_ensemble(
    gen.random_straight_2d, num_networks=50, base_seed=42,
    num_fibers=100, fiber_length=10.0, box_size=(50, 50)
)

def analyze(net):
    morph = MorphologyAnalyzer(net)
    return {
        'nematic_order': morph.nematic_order_parameter(),
        'porosity': morph.porosity(),
    }

stats = ensemble_analysis(ensemble, analyze)
print(f"Nematic order: {stats['nematic_order']['mean']:.3f} ± {stats['nematic_order']['std']:.3f}")
```

### Visualization

```python
# Matplotlib (static)
from fibernet.viz import visualize_3d_matplotlib
fig, ax = visualize_3d_matplotlib(net, show_crosslinks=True)

# Plotly (interactive web)
from fibernet.viz import visualize_interactive, visualize_stress_field, export_html
fig = visualize_interactive(net, color_by='orientation', title="My Network")
export_html(fig, "network.html", auto_open=True)

# PyVista (interactive 3D)
from fibernet.viz import visualize_3d_pyvista
plotter = visualize_3d_pyvista(net, fiber_radius=0.1)

# Stress-colored visualization
import numpy as np
stress = np.random.uniform(0, 1e6, net.num_fibers)
fig = visualize_stress_field(net, stress, title="Stress Distribution")
```

### Export to Simulation Software

```python
from fibernet.io import to_vtk, to_lammps, to_gmsh, to_pdb

to_vtk(net, 'network.vtk')         # ParaView
to_lammps(net, 'network.lammps')   # LAMMPS MD
to_gmsh(net, 'network.geo')        # GMSH meshing
to_pdb(net, 'network.pdb')         # Protein Data Bank
```

### Unit Systems

```python
from fibernet.utils.units import convert_network

# Convert to micrometers
net_micro = convert_network(net, from_unit='si', to_unit='micro')
```

## Examples

FiberNet includes several example scripts in the `examples/` directory:

- **`basic_usage.py`** — Quick start with network generation and analysis
- **`full_workflow.py`** — Complete pipeline: generate → analyze → simulate → export
- **`ml_example.py`** — Machine learning integration with feature extraction and prediction
- **`comprehensive_demo.py`** — Showcase of all major features

Run any example:
```bash
python examples/full_workflow.py
```

## Project Structure

```
fibernet/
├── core/              # Core data structures (Fiber, FiberNetwork, Material, Crosslinks)
├── gen/               # 57 network generators
├── sim/               # Simulation engines
│   ├── mechanical.py  # FEM (linear/nonlinear)
│   ├── accelerated.py # Taichi GPU-accelerated FEM
│   ├── thermal.py     # Heat conduction
│   ├── electromagnetic.py # Electrical conductivity
│   ├── acoustic.py    # Acoustic wave propagation
│   ├── fracture.py    # Crack propagation
│   ├── damage.py      # Damage mechanics, fatigue
│   ├── rheology.py    # Fiber suspension rheology
│   ├── fluid.py       # Darcy flow, pore network
│   ├── multiscale.py  # Homogenization, RVE
│   └── viscoelastic.py # DMA, Maxwell, Kelvin-Voigt
├── analysis/          # Analysis tools
│   ├── morphology.py  # Structure analysis
│   └── percolation.py # Percolation analysis
├── ml/                # Machine learning (features, GNN)
├── io/                # I/O formats (VTK, LAMMPS, PDB, GMSH, XYZ, HDF5)
├── viz/               # Visualization (matplotlib, pyvista, plotly)
├── utils/             # Utilities (config, ensemble, units, parallel)
├── api.py             # High-level convenience API
└── transforms/        # Network transformations
```

## Tutorials

Jupyter notebook tutorials are available in the `tutorials/` directory:

1. **Getting Started** (`01_getting_started.ipynb`) - Basic network generation and analysis
2. **Mechanical Simulation** (`02_mechanical_simulation.ipynb`) - FEM and stress-strain analysis
3. **Machine Learning** (`03_machine_learning.ipynb`) - Feature extraction and property prediction

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_core.py tests/test_generators.py
pytest tests/test_integration.py -v
```

## Documentation

Full documentation with API reference: [https://fibernet.readthedocs.io/](https://fibernet.readthedocs.io/)

## Citation

If you use FiberNet in your research, please cite:

```bibtex
@software{fibernet2025,
  title = {FiberNet: A Comprehensive Python Toolkit for Fiber Network Generation, Simulation, and Analysis},
  author = {FiberNet Contributors},
  year = {2025},
  url = {https://github.com/GellmanSparrowS/fibernet},
  version = {1.8.0}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with NumPy, SciPy, NetworkX, matplotlib, pyvista, plotly, and Taichi
- Inspired by research in computational materials science and biomechanics
- Supported by the [ML-BioMat](https://ml-biomat.com/) research group
