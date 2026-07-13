# FiberNet Showcase: A Comprehensive Fiber Network Structure Library

> **74 generators** covering 2D/3D, ordered/disordered, natural/synthetic, simple/complex structures

## Overview

FiberNet provides a unified API for generating diverse fiber network structures suitable for mechanical simulation, metamaterial design, and biomimetic research.

### Key Features

- **One-line generation**: `fn.create("generator_name", **params)` 
- **Intelligent defaults**: Auto-percolation ensures connectivity
- **Parametric control**: 3-10 parameters per generator → >10²⁰ possible structures
- **Simulation-ready**: 87% connected networks, 99% with crosslinks
- **Reproducible**: Seed parameter for deterministic generation

## Quick Start

```python
import fibernet as fn

# Generate any structure with one line
net = fn.create("random_2d", num_fibers=100, seed=42)
net = fn.create("honeycomb_2d", cell_size=5.0)
net = fn.create("tpms_sheet", resolution=12)

# List all 74 generators
print(fn.list_generators())
```

## Showcase Categories

Each category demonstrates **parametric programmability** - the same generator with different parameters produces fundamentally different structures.

### 1. Random 2D Networks
![Random 2D](output_viz/showcase/01_random_2d.png)

**Generators**: `random_straight_2d`, `oriented_random_2d`

**Parameters**:
- `num_fibers`: 30 → 1000 (density control)
- `fiber_length`: 5 → 20 (length distribution)
- `angle_std`: 0.01 → 1.57 (anisotropy: aligned → isotropic)

**Applications**: Non-woven fabrics, paper, electrospun mats

**Key insight**: Same generator, different `angle_std` produces aligned composites vs isotropic networks.

---

### 2. Random 3D Networks
![Random 3D](output_viz/showcase/02_random_3d.png)

**Generators**: `random_straight_3d`, `oriented_random_3d`, `random_walk_fibers`

**Parameters**:
- `num_fibers`: 40 → 150 (3D density)
- `angle_std`: 0.1 → 0.5 (3D anisotropy)
- `num_steps`: 10 → 50 (polymer chain length)

**Applications**: 3D scaffolds, porous media, polymer networks

**Key insight**: 3D percolation threshold (ρ_c·L³≈2.53) automatically enforced.

---

### 3. Lattice 2D
![Lattice 2D](output_viz/showcase/03_lattice_2d.png)

**Generators**: `square_lattice_2d`, `honeycomb_lattice_2d`, `triangular_lattice_2d`, `kagome_lattice_2d`

**Parameters**:
- `cell_size`: 3 → 20 (unit cell scale)
- `grid_size`: (3,3) → (10,10) (array size)
- `perturbation`: 0 → 0.3 (defect introduction)

**Applications**: Mechanical metamaterials, architected materials

**Key insight**: Topology (square vs honeycomb vs triangular) determines mechanical properties.

---

### 4. Lattice 3D
![Lattice 3D](output_viz/showcase/04_lattice_3d.png)

**Generators**: `cubic_lattice_3d`, `octet_truss_3d`, `diamond_lattice_3d`, `gyroid_lattice_3d`, `plate_lattice_3d`

**Parameters**:
- `cell_size`: 5 → 15 (3D unit cell)
- `grid_size`: (2,2,2) → (4,4,4) (3D array)

**Applications**: Lightweight structures, 3D printing, metamaterials

**Key insight**: Octet truss achieves near-isotropic stiffness; gyroid has smooth curvature.

---

### 5. Metamaterial 2D
![Metamaterial 2D](output_viz/showcase/05_metamaterial_2d.png)

**Generators**: `reentrant_honeycomb_2d`, `star_honeycomb_2d`, `arrowhead_auxetic_2d`, `chiral_honeycomb_2d`, `missing_rib_auxetic_2d`

**Parameters**:
- `reentrant_angle`: 120° → 170° (auxetic behavior control)
- `cell_size`: 5 → 15 (unit cell)

**Applications**: Auxetic materials (negative Poisson's ratio), impact absorption

**Key insight**: Reentrant angle θ controls transition from auxetic (ν<0) to conventional (ν>0).

---

### 6. Metamaterial 3D
![Metamaterial 3D](output_viz/showcase/06_metamaterial_3d.png)

**Generators**: `reentrant_honeycomb_3d`, `proper_octet_truss_3d`, `diamond_lattice_3d`, `gyroid_lattice_3d`, `plate_lattice_3d`

**Applications**: 3D auxetics, lightweight armor, energy absorption

**Key insight**: 3D auxetics require careful design to maintain negative Poisson's ratio in all directions.

---

### 7. Fractal
![Fractal](output_viz/showcase/07_fractal.png)

**Generators**: `sierpinski`, `koch_curve`, `fractal_tree`, `hilbert`, `fractal_network`

**Parameters**:
- `iterations`: 2 → 6 (fractal depth)
- `branching_factor`: 2 → 4 (tree complexity)

**Applications**: Fractal antennas, self-similar structures, space-filling curves

**Key insight**: Fractal dimension D controls mass scaling: M ∝ L^D

---

### 8. Biomimetic
![Biomimetic](output_viz/showcase/08_biomimetic.png)

**Generators**: `biomimetic_collagen`, `electrospun`, `meltblown`, `paper_network`

**Parameters**:
- `num_fibers`: 80 → 300 (density)
- `persistence_length`: 5 → 50 (fiber stiffness)
- `bundling_probability`: 0 → 0.7 (collagen bundling)

**Applications**: Tissue engineering, collagen networks, non-woven biomaterials

**Key insight**: Collagen networks mimic extracellular matrix with D-periodicity and branching.

---

### 9. Bundles
![Bundles](output_viz/showcase/09_bundles.png)

**Generators**: `parallel_bundle_2d`, `twisted_bundle_2d`, `random_bundle_3d`, `tendon_like_bundle_3d`, `braided_bundle_3d`

**Parameters**:
- `num_fibers`: 10 → 50 (bundle size)
- `twist_rate`: 0 → 2π (twist per unit length)
- `bundle_length`: 20 → 100 (bundle length)

**Applications**: Tendons, ligaments, ropes, cables

**Key insight**: Twist introduces coupling between tension and torsion.

---

### 10. Voronoi
![Voronoi](output_viz/showcase/10_voronoi.png)

**Generators**: `voronoi_2d`, `voronoi_3d`, `foam_like_3d`

**Parameters**:
- `num_seeds`: 20 → 100 (cell density)
- `regularity`: 0 → 1 (ordered → random)

**Applications**: Cellular solids, foams, biological tissues

**Key insight**: Voronoi tessellation produces realistic foam microstructures.

---

### 11. TPMS (Triply Periodic Minimal Surfaces)
![TPMS](output_viz/showcase/11_tpms.png)

**Generators**: `tpms_sheet`, `tpms_lattice`, `tpms_gradient`

**Parameters**:
- `resolution`: 6 → 15 (mesh density)
- `surface_type`: "gyroid" | "schwarz_d" | "schwarz_p" (surface family)

**Applications**: Additive manufacturing, heat exchangers, bone scaffolds

**Key insight**: TPMS structures have zero mean curvature and smooth transitions.

---

### 12. Woven
![Woven](output_viz/showcase/12_woven.png)

**Generators**: `woven_3d`, `plain_weave`, `twill_weave`, `satin_weave`, `textile_weave`

**Parameters**:
- `warp_count`: 5 → 20 (warp yarns)
- `weft_count`: 5 → 20 (weft yarns)
- `weave_pattern`: "plain" | "twill" | "satin"

**Applications**: Textile composites, woven fabrics

**Key insight**: Weave pattern determines drape, stiffness, and failure modes.

---

### 13-15. Parametric Demonstrations

These categories demonstrate **programmability** - the same generator with different parameters produces a continuous family of structures.

#### Parametric: Random (N)
![Parametric Random](output_viz/showcase/13_parametric_random.png)

**Generator**: `random_straight_2d` with `num_fibers` = 30, 80, 200, 500, 1000

**Insight**: Density varies from sparse (30 fibers, disconnected) to dense (1000 fibers, highly connected).

#### Parametric: Honeycomb
![Parametric Honeycomb](output_viz/showcase/14_parametric_honeycomb.png)

**Generator**: `honeycomb_lattice_2d` with `cell_size` = 3, 5, 8, 12, 20

**Insight**: Topology remains constant (hexagonal), but geometric scale changes 6.7×.

#### Parametric: Oriented
![Parametric Oriented](output_viz/showcase/15_parametric_oriented.png)

**Generator**: `oriented_random_2d` with `angle_std` = 0.01, 0.1, 0.3, 0.7, 1.57

**Insight**: Continuous transition from aligned (σ=0.01) to isotropic (σ=1.57).

---

## API Design Principles

### 1. Unified Entry Point
```python
fn.create(generator_name, **kwargs)
```
All 74 generators accessible through one function.

### 2. Intelligent Defaults
Each generator has optimized defaults ensuring:
- **Connectivity**: Random networks auto-compute percolation density
- **Reasonable size**: Not too small (disconnected) or too large (OOM)
- **Crosslinks**: 99% of generators produce networks with crosslinks

### 3. Progressive Complexity
```python
# Beginner: just the name
net = fn.create("random_2d")

# Intermediate: key parameters
net = fn.create("random_2d", num_fibers=200, fiber_length=15.0)

# Advanced: full control
net = fn.create("random_2d", num_fibers=200, fiber_length=15.0, 
                box_size=50.0, angle_std=0.5, radius=0.1, 
                material=Material("nylon"), seed=42)
```

### 4. Reproducibility
```python
net1 = fn.create("random_2d", num_fibers=100, seed=42)
net2 = fn.create("random_2d", num_fibers=100, seed=42)
# net1 and net2 are identical
```

### 5. Auto-Percolation
```python
# No need to manually tune density for connectivity
net = fn.create("random_2d", num_fibers=100)
# Automatically ensures ρ > ρ_c (percolation threshold)
```

---

## Combinatorial Explosion

With 74 generators and 3-10 parameters each, the parameter space is enormous:

| Generator | Key Parameters | Typical Range | Combinations |
|-----------|---------------|---------------|--------------|
| `random_2d` | N, L, σ | 4×3×3 values | **36** |
| `oriented_2d` | N, L, μ, σ, box | 5×3×6×5×3 | **1,350** |
| All 74 generators | Conservative | 1000 combos each | **74,000** |
| All 74 generators | Full sweep | 100,000 combos each | **7,400,000** |

**One `fn.create()` call → 7.4 million possible structures.**

---

## Simulation Readiness

### Connectivity
- **87% connected**: 62/71 generators produce single-component networks
- **Auto-percolation**: Random networks automatically exceed percolation threshold
- **Crosslinks**: 99% of networks have crosslinks (node mechanics defined)

### Topology
- **Degree range**: 0 → 488 (sparse to highly interconnected)
- **Fiber count**: 1 → 11,480 (single helix to TPMS gradient)
- **Crosslink count**: 0 → 96,116

### Geometric Diversity
- **Dimensions**: 2D (30 generators) + 3D (41 generators)
- **Scales**: Nano (CNT) → Micro (electrospun) → Meso (woven) → Macro (architecture)
- **Materials**: Polymer / Metal / Ceramic / Biological

---

## Research Applications

### Mechanical Metamaterials
- **Auxetics**: Reentrant honeycombs, star honeycombs, arrowhead structures
- **Lattice materials**: Octet truss, gyroid, plate lattices
- **Topology optimization**: Compare different unit cells

### Biomimetic Materials
- **Collagen networks**: ECM mechanics, cell-matrix interactions
- **Tendon/ligament**: Bundled fibers with crimp
- **Non-woven biomaterials**: Electrospun, meltblown

### Additive Manufacturing
- **TPMS structures**: Gyroid, Schwarz-D, Schwarz-P
- **Lattice infill**: Cubic, octet, diamond
- **Gradient materials**: Density gradients, property gradients

### Soft Matter
- **Polymer networks**: Random walk fibers
- **Gels**: Highly interconnected random networks
- **Biopolymer networks**: Fibrin, collagen

---

## Visualization

All showcase images use **publication-quality rendering**:
- Square canvas, no axes/frames
- Dark background with bright fibers
- Anti-aliased lines
- Adaptive line width based on fiber count
- Consistent style across all visualizations

```python
from fibernet.viz.showcase import render_2d, render_2d_grid

# Single network
render_2d(net, save_path="network.png", background="dark")

# Grid of networks (parametric study)
render_2d_grid(networks, titles=["A", "B", "C"], ncols=3)
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/fibernet.git
cd fibernet

# Install
pip install -e .

# Run tests
python -m pytest tests/
```

---

## Citation

If you use FiberNet in your research, please cite:

```bibtex
@software{fibernet2026,
  title = {FiberNet: A Comprehensive Fiber Network Structure Library},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/fibernet}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contact

For questions, issues, or contributions:
- GitHub Issues: [github.com/yourusername/fibernet/issues](https://github.com/yourusername/fibernet/issues)
- Email: your.email@example.com

---

**FiberNet: Laying the foundation for the next decade of fiber network research.**
