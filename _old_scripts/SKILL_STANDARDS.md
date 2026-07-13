# FiberNet Skill Standards

> Quality standards and design principles for FiberNet development

## 1. Visualization Standards

### Publication-Quality Rendering

All visualizations must meet these standards:

#### Canvas Design
- **Square aspect ratio**: 1:1 (e.g., 4×4, 8×8 inches)
- **No axes or frames**: Clean, minimalist presentation
- **No tick marks or labels**: Focus on structure, not coordinates
- **Background**: Dark (#0a0a0a) for 2D, can use light (#f5f5f5) for 3D

#### Color Scheme
- **2D fibers**: Bright green (#00ff88) on dark background
- **3D tubes**: Material-based coloring or orientation-based
- **Crosslinks**: Bright red-pink (#ff3366) when shown
- **Consistency**: Same generator → same color across visualizations

#### Line Rendering
- **Anti-aliased**: Always enable `antialiased=True`
- **Adaptive line width**: 
  - ≤50 fibers: 2.0pt
  - 50-1000 fibers: linear interpolation
  - ≥1000 fibers: 0.2pt
- **Alpha transparency**: 0.9 for 2D, 0.7 for 3D (depth cueing)

#### Layout
- **Grid visualization**: 1×5 layout for parametric studies
- **Spacing**: `pad_inches=0.1`, `tight_layout(pad=0.5)`
- **Titles**: White text on dark background, fontsize=9, bold
- **Subtitles**: Italic, fontsize=8, parameter values

#### 3D Specific
- **Viewing angle**: Azimuth=30°, Elevation=20° (isometric-like)
- **Tube radius**: Use actual fiber radius or 0.1 as default
- **Smooth shading**: Always enable `smooth_shading=True`
- **Point size**: Crosslinks as spheres, point_size=5

### Code Example
```python
from fibernet.viz.showcase import render_2d, ShowcaseStyle

# Correct usage
render_2d(net, 
          background='dark',
          color='#00ff88',
          line_width=ShowcaseStyle.compute_line_width(net.num_fibers),
          save_path='output.png')
```

---

## 2. Generator Design Principles

### Essential Generators Only

**DO NOT** create redundant generators. Ask:
1. Can this be achieved by parameter variation on an existing generator?
2. Is the topology fundamentally different?
3. Does it serve a distinct research application?

**Example**: 
- ✅ `honeycomb_lattice_2d` with `cell_size` parameter (different scales)
- ❌ `honeycomb_lattice_2d_small`, `honeycomb_lattice_2d_large` (redundant)

### Parameter Design

Each generator should have **3-10 parameters** covering:

#### Required Parameters
1. **Size control**: `num_fibers`, `cell_size`, `grid_size`
2. **Geometric control**: `fiber_length`, `radius`, `angle`
3. **Statistical control**: `angle_std`, `perturbation`, `seed`

#### Optional Parameters
4. **Material**: `material=Material("nylon")`
5. **Connectivity**: `ensure_connected=True`
6. **Advanced**: `threshold_factor`, `regularity`

### Parameter Ranges

Parameters should have **sensible defaults** and **documented ranges**:

```python
def random_straight_2d(
    num_fibers: int = 100,        # Range: 10 → 10000
    fiber_length: float = 10.0,   # Range: 1 → 100
    box_size: float = 50.0,       # Range: 10 → 500
    radius: float = 0.1,          # Range: 0.01 → 1.0
    seed: Optional[int] = None,   # Any integer for reproducibility
) -> FiberNetwork:
    """
    Generate random straight fiber network.
    
    Parameters
    ----------
    num_fibers : int
        Number of fibers (10-10000). Default 100.
    fiber_length : float
        Fiber length (1-100). Default 10.0.
    """
```

### Auto-Percolation

Random network generators **must** auto-compute percolation density:

```python
# 2D percolation threshold: ρ_c·L² ≈ 5.71
rho_c_2d = 5.71 / (fiber_length ** 2)

# Ensure user-requested density exceeds threshold
actual_density = max(requested_density, 1.5 * rho_c_2d)
```

**Why**: Users shouldn't need to manually tune density for connectivity.

### Crosslink Detection

All generators **must** detect crosslinks:

```python
# After generating fibers
network = detect_crosslinks(fibers, tolerance=1e-6)

# Verify
assert network.num_crosslinks > 0 or is_single_fiber
```

**Why**: 99% of networks should have crosslinks for simulation readiness.

---

## 3. API Design Principles

### Unified Entry Point

**All** generators accessible through one function:

```python
net = fn.create(generator_name, **kwargs)
```

**Never** expose module internals:
```python
# ❌ Bad
from fibernet.gen.disordered import random_straight_2d
net = random_straight_2d(num_fibers=100)

# ✅ Good
net = fn.create("random_straight_2d", num_fibers=100)
```

### Generator Registry

All generators registered in `fibernet/gen/__init__.py`:

```python
GENERATOR_REGISTRY = {
    "random_straight_2d": disordered.random_straight_2d,
    "honeycomb_lattice_2d": ordered.honeycomb_lattice_2d,
    # ...
}

def list_generators() -> List[str]:
    return sorted(GENERATOR_REGISTRY.keys())
```

### Progressive Complexity

API should support three levels of usage:

#### Level 1: Beginner (name only)
```python
net = fn.create("random_2d")
# Uses all defaults, ensures connectivity
```

#### Level 2: Intermediate (key parameters)
```python
net = fn.create("random_2d", 
                num_fibers=200, 
                fiber_length=15.0)
```

#### Level 3: Advanced (full control)
```python
net = fn.create("random_2d",
                num_fibers=200,
                fiber_length=15.0,
                box_size=50.0,
                angle_std=0.5,
                radius=0.1,
                material=Material("nylon"),
                seed=42,
                ensure_connected=True,
                threshold_factor=2.0)
```

### Reproducibility

All stochastic generators **must** accept `seed` parameter:

```python
net1 = fn.create("random_2d", num_fibers=100, seed=42)
net2 = fn.create("random_2d", num_fibers=100, seed=42)
# net1 and net2 are identical
```

### Error Handling

Generators should fail gracefully:

```python
try:
    net = fn.create("random_2d", num_fibers=100)
except PercolationError as e:
    print(f"Failed to achieve connectivity: {e}")
    # Suggest increasing num_fibers
```

---

## 4. Testing Standards

### Unit Tests

Each generator **must** have tests for:

#### Basic Functionality
```python
def test_random_2d_basic():
    net = fn.create("random_straight_2d", num_fibers=100)
    assert net.num_fibers == 100
    assert net.dimension == 2
    assert net.num_crosslinks > 0
```

#### Connectivity
```python
def test_random_2d_connected():
    net = fn.create("random_straight_2d", num_fibers=100)
    assert net.is_connected()
```

#### Reproducibility
```python
def test_random_2d_seed():
    net1 = fn.create("random_straight_2d", num_fibers=100, seed=42)
    net2 = fn.create("random_straight_2d", num_fibers=100, seed=42)
    assert networks_equal(net1, net2)
```

#### Parameter Validation
```python
def test_random_2d_invalid_params():
    with pytest.raises(ValueError):
        fn.create("random_straight_2d", num_fibers=-1)
    with pytest.raises(ValueError):
        fn.create("random_straight_2d", fiber_length=0)
```

### Integration Tests

Test generator → simulation pipeline:

```python
def test_random_2d_to_simulation():
    net = fn.create("random_straight_2d", num_fibers=100)
    sim = LinearElasticSimulation(net)
    results = sim.run(strain=0.01)
    assert results.stress_tensor is not None
```

### Performance Tests

Benchmark generation speed:

```python
def test_random_2d_performance():
    # Should generate 1000 fibers in < 1 second
    with Timer() as t:
        net = fn.create("random_straight_2d", num_fibers=1000)
    assert t.elapsed < 1.0
```

### Coverage

**Minimum 80% code coverage** for all generators.

---

## 5. Documentation Standards

### Docstrings

Every generator **must** have comprehensive docstring:

```python
def random_straight_2d(
    num_fibers: int = 100,
    fiber_length: float = 10.0,
    box_size: float = 50.0,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """
    Generate 2D random straight fiber network.
    
    Creates a network of straight fibers with random positions and
    orientations. Automatically ensures percolation (connectivity).
    
    Parameters
    ----------
    num_fibers : int, optional
        Number of fibers (10-10000). Default 100.
    fiber_length : float, optional
        Length of each fiber (1-100). Default 10.0.
    box_size : float, optional
        Size of square domain (10-500). Default 50.0.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    FiberNetwork
        Generated network with crosslinks detected.
    
    Examples
    --------
    >>> net = fn.create("random_straight_2d", num_fibers=100)
    >>> print(f"Fibers: {net.num_fibers}, Crosslinks: {net.num_crosslinks}")
    Fibers: 100, Crosslinks: 332
    
    Notes
    -----
    Uses 2D percolation theory (ρ_c·L² ≈ 5.71) to ensure connectivity.
    Actual density = max(requested, 1.5 × ρ_c).
    
    See Also
    --------
    oriented_random_2d : Anisotropic variant
    random_walk_fibers : Curved fibers
    
    References
    ----------
    .. [1] Pike & Seager, Phys. Rev. B 10, 1421 (1974).
    """
```

### Gallery Examples

Each generator should have a gallery example in `examples/`:

```python
# examples/random_2d_gallery.py
import fibernet as fn
from fibernet.viz import render_2d_grid

# Generate parametric study
networks = [
    fn.create("random_straight_2d", num_fibers=n)
    for n in [50, 100, 200, 500]
]

# Visualize
render_2d_grid(networks, 
               titles=["N=50", "N=100", "N=200", "N=500"],
               save_path="random_2d_parametric.png")
```

### Tutorials

Create tutorials for common workflows:

1. **Getting Started**: Generate first network
2. **Parametric Study**: Vary parameters systematically
3. **Custom Materials**: Define material properties
4. **Simulation Integration**: Connect to FEA/MD
5. **Advanced Visualization**: Custom rendering

---

## 6. Code Quality Standards

### Type Hints

All functions **must** have type hints:

```python
def random_straight_2d(
    num_fibers: int = 100,
    fiber_length: float = 10.0,
    seed: Optional[int] = None,
) -> FiberNetwork:
    ...
```

### Naming Conventions

- **Generators**: `snake_case` (e.g., `random_straight_2d`)
- **Classes**: `PascalCase` (e.g., `FiberNetwork`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `PERCOLATION_THRESHOLD_2D`)
- **Private functions**: `_leading_underscore` (e.g., `_detect_crosslinks`)

### File Organization

```
fibernet/
├── core/
│   ├── network.py          # FiberNetwork class
│   ├── fiber.py            # Fiber class
│   ├── crosslink.py        # Crosslink class
│   └── material.py         # Material class
├── gen/
│   ├── __init__.py         # Registry
│   ├── disordered.py       # Random networks
│   ├── ordered.py          # Lattices
│   ├── metamaterials.py    # Auxetics, etc.
│   ├── biomimetic.py       # Biological networks
│   └── ...
├── viz/
│   ├── __init__.py
│   ├── showcase.py         # Publication-quality viz
│   └── interactive.py      # Interactive viz (future)
├── sim/
│   ├── __init__.py
│   ├── elastic.py          # Elastic simulation
│   └── ...
└── utils/
    ├── __init__.py
    └── geometry.py         # Geometric utilities
```

### Linting

Use `black` for formatting:
```bash
black fibernet/
```

Use `flake8` for linting:
```bash
flake8 fibernet/
```

---

## 7. Research Application Guidelines

### When to Use Which Generator

| Application | Recommended Generator | Key Parameters |
|-------------|----------------------|----------------|
| Non-woven fabrics | `random_straight_2d` | `num_fibers`, `angle_std` |
| Paper | `random_straight_2d` | `num_fibers`, `fiber_length` |
| Electrospun mats | `electrospun` | `num_fibers`, `persistence_length` |
| Collagen networks | `biomimetic_collagen` | `bundling_probability` |
| Tendons | `tendon_like_bundle_3d` | `crimp_amplitude` |
| Metamaterials | `reentrant_honeycomb_2d` | `reentrant_angle` |
| Lightweight structures | `octet_truss_3d` | `cell_size`, `grid_size` |
| 3D printing | `tpms_sheet` | `resolution`, `surface_type` |
| Foams | `voronoi_3d` | `num_seeds`, `regularity` |

### Parametric Study Design

**DO**: Systematically vary one parameter at a time

```python
# Good: Clear parametric study
densities = [50, 100, 200, 500, 1000]
networks = [fn.create("random_straight_2d", num_fibers=n) for n in densities]
render_2d_grid(networks, titles=[f"N={n}" for n in densities])
```

**DON'T**: Randomly vary multiple parameters

```python
# Bad: Confusing visualization
networks = [
    fn.create("random_straight_2d", num_fibers=50, fiber_length=5),
    fn.create("random_straight_2d", num_fibers=200, fiber_length=20),
    fn.create("random_straight_2d", num_fibers=1000, fiber_length=10),
]
```

### Simulation Readiness Checklist

Before using a network in simulation:

- [ ] **Connected**: `net.is_connected() == True`
- [ ] **Crosslinks**: `net.num_crosslinks > 0`
- [ ] **Reasonable size**: 100 < `num_fibers` < 10000
- [ ] **Material defined**: All fibers have material properties
- [ ] **Geometry valid**: No zero-length fibers, no overlapping fibers

---

## 8. Performance Guidelines

### Memory Management

**DO**: Clean up large networks after use

```python
net = fn.create("random_straight_2d", num_fibers=10000)
# ... use network ...
del net
gc.collect()
```

**DON'T**: Keep many large networks in memory

```python
# Bad: Memory leak
networks = [fn.create("random_straight_2d", num_fibers=10000) for _ in range(100)]
```

### Parallelization

**DO**: Use multiprocessing for parametric studies

```python
from multiprocessing import Pool

def generate_network(n):
    return fn.create("random_straight_2d", num_fibers=n)

with Pool(4) as p:
    networks = p.map(generate_network, [100, 200, 500, 1000])
```

### Caching

**DO**: Cache expensive computations

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def generate_cached(num_fibers: int, seed: int) -> FiberNetwork:
    return fn.create("random_straight_2d", num_fibers=num_fibers, seed=seed)
```

---

## 9. Contribution Guidelines

### Adding a New Generator

1. **Check for redundancy**: Can existing generators achieve this?
2. **Design parameters**: 3-10 parameters with sensible defaults
3. **Implement generator**: Follow coding standards
4. **Add tests**: Unit, integration, performance
5. **Write docstring**: Comprehensive with examples
6. **Create gallery example**: Visual demonstration
7. **Update registry**: Add to `GENERATOR_REGISTRY`
8. **Update docs**: Add to README, tutorials

### Pull Request Checklist

- [ ] Code follows style guide (black, flake8)
- [ ] All tests pass (pytest)
- [ ] Coverage ≥ 80%
- [ ] Docstrings complete
- [ ] Gallery example added
- [ ] Documentation updated
- [ ] Performance benchmarked
- [ ] No redundant generators

---

## 10. Future Directions

### Planned Features

1. **Curved fibers**: Bezier curve-based fibers
2. **Field-guided networks**: Orientation field-guided generation
3. **Hierarchical structures**: Multi-scale networks
4. **Interactive visualization**: WebGL-based 3D viewer
5. **Machine learning**: Generative models for inverse design

### Research Opportunities

1. **Inverse design**: Given properties → generate structure
2. **Multi-objective optimization**: Pareto-optimal structures
3. **Defect engineering**: Controlled defect introduction
4. **Dynamic networks**: Time-evolving structures
5. **Multi-physics**: Coupled thermal-mechanical-electrical

---

## Summary

**FiberNet Skill Standards** ensure:

1. ✅ **Publication-quality visualizations** (dark background, no axes, square canvas)
2. ✅ **Essential generators only** (no redundancy, clear parametric control)
3. ✅ **Unified API** (`fn.create()` for all generators)
4. ✅ **Comprehensive testing** (unit, integration, performance)
5. ✅ **Complete documentation** (docstrings, examples, tutorials)
6. ✅ **High code quality** (type hints, linting, organization)
7. ✅ **Research-ready** (simulation integration, application guidelines)

**Goal**: Lay the foundation for the next decade of fiber network research.

---

**Last Updated**: 2026-07-09  
**Version**: 1.0  
**Maintained by**: FiberNet Development Team
