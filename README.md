# FiberNet

**A comprehensive Python toolkit for fiber network structure generation, simulation, and analysis.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

FiberNet provides tools for:
- **Generating** 2D/3D fiber networks (ordered, disordered, chiral, woven, hierarchical, biomimetic)
- **Transforming** networks (mirror, rotate, scale, merge, tile, pattern)
- **Simulating** physics (mechanics, dynamics, fracture, thermal, electromagnetic)
- **Analyzing** structure (topology, morphology, spectral, pore distribution)
- **Accelerating** computations with Taichi (CPU/GPU parallel)

## Installation

```bash
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# For development
pip install -e ".[dev]"
```

## Quick Start

```python
import fibernet as fn
from fibernet import gen
from fibernet.sim.mechanical import FiberFEM
from fibernet.analysis import MorphologyAnalyzer

# Generate a 2D random fiber network
net = gen.random_straight_2d(num_fibers=100, fiber_length=15, box_size=(50, 50), seed=42)
print(f"Network: {net.num_fibers} fibers, {net.num_crosslinks} crosslinks")

# Analyze morphology
morph = MorphologyAnalyzer(net)
print(f"Mean fiber length: {morph.mean_fiber_length():.2f}")
print(f"Nematic order: {morph.nematic_order_parameter():.3f}")

# Run mechanical simulation
fem = FiberFEM(net, segments_per_fiber=5)
result = fem.apply_uniaxial_strain(strain=0.001, axis=0)
print(f"Effective modulus: {fem.effective_modulus():.2e} Pa")
```

## Structure Generators

### Disordered Networks
- `random_straight_2d` / `random_straight_3d` - Mikado model
- `random_walk_fibers` - Random walk polymer chains
- `oriented_random_2d` - Partially aligned fibers
- `poisson_line_network_2d` - Poisson line process

### Ordered Lattices
- `square_lattice_2d` / `triangular_lattice_2d` / `honeycomb_lattice_2d`
- `cubic_lattice_3d` / `octet_truss_3d` / `kagome_lattice_2d`
- `diamond_lattice_3d` - Tetrahedral bonding

### Chiral Structures
- `single_helix` / `double_helix` - DNA-like helices
- `braided_rope` - Multi-strand braiding
- `twisted_bundle` - Twisted fiber bundles
- `chiral_metamaterial` - Chiral lattice structures

### Woven Structures
- `plain_weave_2d` / `twill_weave_2d` / `satin_weave_2d`
- `woven_3d_orthogonal` - 3D orthogonal woven

### Hierarchical Structures
- `hierarchical_bundle` - Multi-level fiber bundles
- `gradient_density_network` - Spatially varying density
- `core_shell_fiber` - Core-shell composite fibers
- `fractal_network` - Self-similar fractal structures

### Advanced Generators
- `voronoi_network_2d` / `voronoi_network_3d` - Voronoi tessellation
- `electrospun_network` - Electrospun nanofiber mats (4 deposition modes)
- `meltblown_network` - Melt-blown nonwovens
- `biomimetic_collagen` - Type-I collagen with D-banding
- `biomimetic_fibrin` - Fibrin clot with branching
- `defected_lattice` - Lattices with controlled defects
- `composite_network` - Multi-material composites
- `graded_network` - Spatially graded properties
- `auxetic_structure` - Negative Poisson's ratio
- `kirigami_structure` - Cut-pattern metamaterials

### Variant Generators
- `lattice_2d_to_3d` - Extrude 2D to 3D with vertical bonds
- `curved_lattice` - Apply curvature deformation
- `multi_radius_network` - Bimodal/uniform/normal/power-law radius
- `variable_stiffness_network` - Spatially varying stiffness
- `gyroid_infill` - TPMS gyroid structures
- `foam_like_3d` - Open-cell foam with curved struts

## Network Transformations

```python
from fibernet.core.transform import mirror, rotate, scale, translate, merge, tile, create_pattern

# Basic transforms
mirrored = mirror(net, axis=0)
rotated = rotate(net, angle=np.pi/4, axis=[0, 0, 1])
scaled = scale(net, factor=2.0)
translated = translate(net, offset=[10, 10, 0])

# Merge multiple networks
composite = merge([net1, net2], offsets=[np.zeros(3), np.array([50, 0, 0])])

# Tile for periodic structures
tiled = tile(net, repeats=(3, 3, 1))

# Create patterns (circular, linear, grid, spiral)
patterned = create_pattern(net, pattern_type="circular", num_units=8, radius=30)
```

## Simulation Engines

### Mechanical (FEM)
```python
from fibernet.sim.mechanical import FiberFEM, stress_strain_curve

fem = FiberFEM(net, segments_per_fiber=5)
result = fem.apply_uniaxial_strain(strain=0.01, axis=0)
print(f"Max displacement: {result.max_displacement()}")
print(f"Effective modulus: {fem.effective_modulus():.2e} Pa")
```

### Dynamics
```python
from fibernet.sim.dynamics import FiberDynamics

dyn = FiberDynamics(net, dt=1e-6, damping=0.01)
dyn.run(steps=1000, temperature=300)
```

### Fracture
```python
from fibernet.sim.fracture import FiberFracture

frac = FiberFracture(net, strength_mean=1e9, strength_std=1e8)
result = frac.progressive_failure(max_strain=0.1)
```

### Thermal
```python
from fibernet.sim.thermal import ThermalSolver

thermal = ThermalSolver(net)
result = thermal.solve_steady_state(T_hot=100, T_cold=0, axis=0)
print(f"Effective conductivity: {result.effective_conductivity:.2f} W/(m·K)")
```

### Electromagnetic
```python
from fibernet.sim.electromagnetic import EMSolver

em = EMSolver(net)
result = em.solve_conductivity(axis=0)
print(f"Effective conductivity: {result.effective_conductivity:.2e} S/m")
```

### Taichi Acceleration
```python
from fibernet.sim.accelerated import TaichiEngine

engine = TaichiEngine(arch="cpu", num_threads=8)
forces = engine.parallel_force_computation(positions, rest_lengths, stiffness, edges)
```

## Analysis Tools

```python
from fibernet.analysis import MorphologyAnalyzer, TopologyAnalyzer
from fibernet.analysis.advanced import SpectralAnalyzer, PoreAnalyzer, AnisotropyAnalyzer

# Morphology
morph = MorphologyAnalyzer(net)
print(f"Nematic order: {morph.nematic_order_parameter():.3f}")
print(f"Mean tortuosity: {morph.mean_tortuosity():.3f}")

# Topology
topo = TopologyAnalyzer(net)
print(f"Mean degree: {topo.degree_statistics()['mean']:.2f}")
print(f"Connected: {topo.is_connected()}")

# Spectral analysis
spectral = SpectralAnalyzer(net)
print(f"Spectral gap: {spectral.spectral_gap():.4f}")
print(f"Spectral entropy: {spectral.spectral_entropy():.4f}")

# Pore distribution
pore = PoreAnalyzer(net)
stats = pore.pore_size_statistics()
print(f"Mean pore size: {stats['mean']:.3f}")

# Anisotropy
aniso = AnisotropyAnalyzer(net)
print(f"Anisotropy index: {aniso.anisotropy_index():.3f}")
```

## Visualization

```python
from fibernet.viz.plot2d import plot_network_2d
from fibernet.viz.render3d import render_network_3d

# 2D plot
fig = plot_network_2d(net, color_by="orientation")
fig.savefig("network_2d.png", dpi=150)

# 3D render
plotter = render_network_3d(net, color_by="stress")
plotter.screenshot("network_3d.png")
```

## Built-in Materials

21 pre-defined materials:
- Carbon fiber, Glass fiber, Kevlar, Dyneema
- Collagen, Fibrin, Silk, Cellulose
- Steel, Aluminum, Titanium, Copper
- CNT, Graphene, Polymer (generic)
- Nylon, Polyester, Polypropylene
- Ceramic, Bio-glass

## Examples

```bash
python examples/basic_usage.py           # Simple introduction
python examples/advanced_generators.py   # Advanced generators
python examples/transformations.py       # Transform operations
python examples/full_workflow.py         # Complete research workflow
```

## Project Structure

```
fibernet/
├── core/           # Data structures (Fiber, Network, Material, Transform)
├── gen/            # Structure generators (disordered, ordered, chiral, woven, hierarchical, advanced, variants)
├── sim/            # Simulation engines (mechanical, dynamics, fracture, thermal, electromagnetic, coupling, accelerated)
├── analysis/       # Analysis tools (topology, morphology, properties, advanced)
├── viz/            # Visualization (2D plots, 3D rendering, animations)
├── utils/          # Utilities (geometry, I/O)
└── examples/       # Example scripts
```

## Testing

```bash
python -m pytest tests/ -v
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use FiberNet in your research, please cite:

```bibtex
@software{fibernet2026,
  title = {FiberNet: A Comprehensive Toolkit for Fiber Network Simulation},
  author = {FiberNet Contributors},
  year = {2026},
  url = {https://github.com/GellmanSparrowS/fibernet}
}
```
