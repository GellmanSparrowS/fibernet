# FiberNet API Reference

**Version**: 1.25.0  
**Last Updated**: 2026-07-08

## Quick Start

```python
import fibernet as fn

# 1. Create a metamaterial (unit cell → tile → weld)
meta = fn.create_metamaterial(
    unit_cell="reentrant_honeycomb_2d",
    array_size=(3, 3),
    reentrant_angle=150,
)

# 2. Run mechanics (beam FEM)
result = fn.simulate_mechanics(meta, strain=0.001)
print(f"E = {result['modulus']:.2e} Pa")

# 3. Run dynamics (mass-spring, Taichi-accelerated)
traj = fn.simulate_dynamics(meta, dt=1e-7, steps=5000, backend="taichi")

# 4. Visualize
fn.plot_metamaterial(meta, save_path="structure.png")
fn.plot_dynamics(traj, save_path="dynamics.png")
```

---

## Core API Functions

### Structure Generation

#### `create(generator, **kwargs)`

Create a fiber network using a registered generator.

```python
# Basic lattices
net = fn.create("square_2d", num_fibers=100)
net = fn.create("honeycomb_2d", cell_size=10.0)
net = fn.create("triangular_2d", spacing=5.0)

# 3D structures
net = fn.create("cubic_3d", size=(10, 10, 10))
net = fn.create("octet_3d", cell_size=5.0)

# Random networks
net = fn.create("random_2d", num_fibers=200, fiber_length=10.0)
net = fn.create("random_walk", num_steps=1000)

# Metamaterials
net = fn.create("reentrant_honeycomb_2d", reentrant_angle=150)
net = fn.create("chiral_honeycomb_2d", node_radius=3.0)
net = fn.create("star_honeycomb_2d", star_angle=60)
```

**Available generators**: Use `fn.list_generators()` to see all 40+ registered generators.

---

#### `create_metamaterial(unit_cell, array_size, weld_threshold, **cell_params)`

Create a metamaterial from a unit cell array with welded crosslinks.

**Workflow**:
1. Generate parameterized unit cell
2. Tile into array (Nx × Ny)
3. Weld intersections with crosslinks
4. Return complete graph ready for simulation

```python
# Re-entrant honeycomb (auxetic when angle > 90°)
meta = fn.create_metamaterial(
    unit_cell="reentrant_honeycomb_2d",
    array_size=(3, 3),
    reentrant_angle=150,
    cell_height=10,
    cell_width=10,
)

# Chiral honeycomb
meta = fn.create_metamaterial(
    unit_cell="chiral_honeycomb_2d",
    array_size=(4, 4),
    node_radius=3.0,
    ligament_length=8.0,
)

# Star-shaped
meta = fn.create_metamaterial(
    unit_cell="star_honeycomb_2d",
    array_size=(3, 3),
    star_angle=60,
)
```

**Unit cell types**:
- `"reentrant_honeycomb_2d"` : Re-entrant honeycomb (auxetic when angle > 90°)
- `"chiral_honeycomb_2d"` : Chiral honeycomb with rotating nodes
- `"star_honeycomb_2d"` : Star-shaped honeycomb
- `"arrowhead_auxetic_2d"` : Arrowhead auxetic structure
- `"hierarchical_lattice_2d"` : Multi-scale hierarchical lattice
- `"missing_rib_auxetic_2d"` : Missing-rib auxetic

**Parameters**:
- `unit_cell` (str): Unit cell type
- `array_size` (tuple): Array dimensions (Nx, Ny). Minimum recommended: (3, 3)
- `weld_threshold` (float): Distance threshold for auto-detecting intersections (default 0.5)
- `**cell_params`: Unit cell parameters (varies by cell type)

**Returns**: `FiberNetwork` with welded crosslinks and metadata

---

### Transformations

```python
# Mirror
net2 = fn.mirror(net, axis=0)  # Mirror about x-axis

# Rotate
net3 = fn.rotate(net, angle=np.pi/4, axis=[0, 0, 1])

# Scale
net4 = fn.scale(net, factor=2.0)

# Translate
net5 = fn.translate(net, offset=[10, 0, 0])

# Merge multiple networks
merged = fn.merge([net1, net2, net3])

# Tile periodically
tiled = fn.tile(net, repeats=(3, 3, 1), spacing=[20, 20, 0])
```

---

### Simulation

#### `simulate_mechanics(network, strain, axis, model)`

Run mechanical simulation using beam FEM.

```python
result = fn.simulate_mechanics(
    network=meta,
    strain=0.001,        # 0.1% strain
    axis=0,              # x-direction
    model="linear",      # "linear", "bilinear", "neo_hookean"
)

# Access results
E = result['modulus']           # Young's modulus [Pa]
stress_max = result['max_stress']  # Maximum stress [Pa]
energy = result['energy']       # Strain energy [J]
displacements = result['displacements']  # Node displacements [m]
```

**Returns**: dict with keys:
- `modulus`: Young's modulus [Pa]
- `max_stress`: Maximum stress [Pa]
- `max_displacement`: Maximum displacement [m]
- `energy`: Strain energy [J]
- `displacements`: Node displacement array
- `node_positions`: Node position array
- `fem`: FiberFEM object for advanced analysis

---

#### `simulate_dynamics(network, dt, steps, damping, backend)`

Run mass-spring dynamics simulation.

**Model**:
- Crosslinks → point masses
- Fiber segments between crosslinks → springs
- Spring stiffness: `k = E * A / L` (axial rigidity)
- Node mass: distributed fiber mass based on connectivity

```python
traj = fn.simulate_dynamics(
    network=meta,
    dt=1e-7,              # Time step [s]
    steps=5000,           # Number of steps
    damping=0.05,         # Velocity damping coefficient
    backend="taichi",     # "taichi" (GPU/CPU parallel) or "numpy"
    save_interval=100,    # Save trajectory every N steps
)

# Access results
positions = traj['positions']      # Final positions [m]
trajectory = traj['trajectory']    # List of position snapshots
energy = traj['energy']            # Final energy [J]
edges = traj['edges']              # Spring connectivity
rest_lengths = traj['rest_lengths']  # Spring rest lengths [m]
stiffness = traj['stiffness']      # Spring stiffness [N/m]
```

**Returns**: dict with keys:
- `positions`: Final node positions (N, 3)
- `trajectory`: List of position snapshots for animation
- `velocities`: Final node velocities (N, 3)
- `forces`: Final node forces (N, 3)
- `energy`: Final strain energy [J]
- `time_seconds`: Simulation time [s]
- `edges`: Spring connectivity (M, 2)
- `rest_lengths`: Spring rest lengths (M,)
- `stiffness`: Spring stiffness (M,)
- `initial_positions`: Initial node positions (N, 3)

---

#### `simulate_thermal(network, T_hot, T_cold, axis)`

Run thermal conduction simulation.

```python
result = fn.simulate_thermal(
    network=meta,
    T_hot=100,      # Hot side temperature [°C]
    T_cold=0,       # Cold side temperature [°C]
    axis=0,         # Heat flow direction
)

k = result['conductivity']  # Thermal conductivity [W/(m·K)]
temps = result['temperatures']  # Node temperatures [°C]
```

---

### Analysis

#### `analyze(network)`

Analyze network structure.

```python
stats = fn.analyze(meta)

print(f"Fibers: {stats['num_fibers']}")
print(f"Crosslinks: {stats['num_crosslinks']}")
print(f"Nematic order: {stats['nematic_order']:.3f}")
print(f"Mean length: {stats['mean_length']:.2f} mm")
print(f"Connected: {stats['is_connected']}")
```

**Returns**: dict with keys:
- `num_fibers`: Number of fibers
- `num_crosslinks`: Number of crosslinks
- `dimension`: Network dimension (2 or 3)
- `nematic_order`: Nematic order parameter [0, 1]
- `mean_length`: Mean fiber length [mm]
- `total_length`: Total fiber length [mm]
- `mean_tortuosity`: Mean tortuosity
- `num_nodes`: Number of graph nodes
- `num_edges`: Number of graph edges
- `mean_degree`: Mean node degree
- `is_connected`: Is graph connected?
- `num_components`: Number of connected components

---

### Visualization

#### `plot_metamaterial(network, show_unit_cells, show_crosslinks, colormap, save_path)`

Professional visualization for metamaterial structures.

```python
fn.plot_metamaterial(
    meta,
    show_unit_cells=True,
    show_crosslinks=True,
    colormap="viridis",
    save_path="metamaterial.png",
)
```

**Features**:
- Fibers colored by orientation angle
- Crosslink points highlighted
- Unit cell boundaries (if metadata available)
- Professional styling with legends

---

#### `plot_dynamics(result, show_forces, colormap, save_path)`

Visualize mass-spring dynamics trajectory.

```python
fn.plot_dynamics(
    traj,
    show_forces=False,
    colormap="viridis",
    save_path="dynamics.png",
)
```

**Features**:
- Multi-frame trajectory visualization
- Springs colored by strain
- Optional force vector overlay
- Time annotations

---

#### `plot_stress_strain(result, show_modulus, save_path)`

Plot stress-strain curve from mechanics simulation.

```python
fn.plot_stress_strain(
    result,
    show_modulus=True,
    save_path="stress_strain.png",
)
```

---

#### `plot(network, **kwargs)`

Quick plot of any network.

```python
fn.plot(meta, color_by="orientation")
```

---

### Export / Import

```python
# Export to various formats
fn.export(meta, "structure.json")
fn.export(meta, "structure.vtk")
fn.export(meta, "structure.lammps")

# Load from file
meta = fn.load("structure.json")
```

**Supported formats**: json, lammps, vtk, vtp, xyz, pdb, msh

---

## Registry Pattern

FiberNet uses a **registry pattern** for extensibility:

### List Available Generators

```python
generators = fn.list_generators()
print(f"Available: {len(generators)}")
for g in generators[:10]:
    print(f"  - {g}")
```

### Register Custom Generator

```python
@fn.register_generator("my_custom_lattice")
def make_my_lattice(cell_size=10.0, **kwargs):
    net = fn.FiberNetwork()
    # Build your custom structure
    return net

# Use it
net = fn.create("my_custom_lattice", cell_size=15.0)
```

### Register Custom Backend

```python
@fn.register_backend("my_fem_solver")
def run_my_fem(network, strain=0.01, **kwargs):
    # Your custom simulation
    return {"modulus": 1e9, "energy": 100.0}

# Use it
result = fn.simulate(network, backend="my_fem_solver", strain=0.001)
```

---

## Data Structures

### `FiberNetwork`

```python
net = fn.create_metamaterial(...)

# Access fibers
for fiber in net.fibers:
    print(f"Fiber {fiber.fiber_id}: length={fiber.length:.2f}")
    print(f"  Centerline: {fiber.centerline.shape}")  # (N, 3)
    print(f"  Radius: {fiber.radius:.3f}")
    print(f"  Material: {fiber.material.name}")

# Access crosslinks
for cl in net.crosslinks:
    print(f"Crosslink: fibers {cl.fiber_i}-{cl.fiber_j}")
    print(f"  Position: {cl.position}")
    print(f"  Type: {cl.crosslink_type}")

# Bounding box
bb_min, bb_max = net.bounding_box()
print(f"Size: {bb_max - bb_min}")

# Metadata (for metamaterials)
if hasattr(net, 'metadata'):
    print(f"Unit cell: {net.metadata['unit_cell']}")
    print(f"Array size: {net.metadata['array_size']}")
```

### `Material`

```python
from fibernet import Material

mat = Material(
    name="steel",
    density=7800,           # kg/m^3
    youngs_modulus=200e9,   # Pa
    poissons_ratio=0.3,
    yield_strength=250e6,   # Pa
)

# Compute derived properties
K = mat.bulk_modulus()
mu, lam = mat.get_lame_parameters()
```

### `Fiber`

```python
from fibernet import Fiber

fiber = Fiber(
    centerline=np.array([[0, 0, 0], [10, 0, 0]]),
    radius=0.2,
    material=mat,
)

print(f"Length: {fiber.length:.2f}")
print(f"Direction: {fiber.direction}")
print(f"Curvature: {fiber.curvature()}")
```

---

## Common Workflows

### 1. Parametric Study

```python
import numpy as np

angles = np.linspace(100, 170, 15)
results = []

for angle in angles:
    meta = fn.create_metamaterial(
        unit_cell="reentrant_honeycomb_2d",
        array_size=(3, 3),
        reentrant_angle=angle,
    )
    
    result = fn.simulate_mechanics(meta, strain=0.001)
    results.append({
        'angle': angle,
        'modulus': result['modulus'],
        'energy': result['energy'],
    })

# Analyze results
import pandas as pd
df = pd.DataFrame(results)
print(df)
```

---

### 2. ML Surrogate Model

```python
from sklearn.ensemble import RandomForestRegressor

# Generate dataset
X = []  # Features: [angle, cell_height, cell_width, radius]
y = []  # Target: log10(E)

for angle in np.linspace(100, 170, 10):
    for height in [8, 10, 12]:
        meta = fn.create_metamaterial(
            unit_cell="reentrant_honeycomb_2d",
            array_size=(3, 3),
            reentrant_angle=angle,
            cell_height=height,
            cell_width=10,
        )
        
        result = fn.simulate_mechanics(meta, strain=0.001)
        X.append([angle, height, 10, 0.2])
        y.append(np.log10(result['modulus']))

# Train model
X = np.array(X)
y = np.array(y)

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Predict
E_pred = 10 ** model.predict([[150, 10, 10, 0.2]])[0]
print(f"Predicted E: {E_pred:.2e} Pa")
```

---

### 3. Reinforcement Learning

```python
# Use trained ML model as reward
def reward(params):
    X = np.array([[params['angle'], params['cell_height'], 
                   params['cell_width'], params['radius']]])
    log_E = model.predict(X)[0]
    return log_E  # Maximize stiffness

# Simple RL loop
best_reward = -np.inf
best_params = None

for episode in range(100):
    # Sample parameters
    params = {
        'angle': np.random.uniform(100, 170),
        'cell_height': np.random.uniform(8, 12),
        'cell_width': 10,
        'radius': 0.2,
    }
    
    r = reward(params)
    if r > best_reward:
        best_reward = r
        best_params = params

print(f"Best angle: {best_params['angle']:.1f}°")
print(f"Best E: {10**best_reward:.2e} Pa")
```

---

## Testing

Run the full test suite:

```bash
pytest tests/ -v
```

Run specific test modules:

```bash
pytest tests/test_api.py -v
pytest tests/test_metamaterial.py -v
pytest tests/test_dynamics.py -v
```

---

## Performance Tips

1. **Use Taichi backend for dynamics**: 10-100× faster than numpy
   ```python
   traj = fn.simulate_dynamics(meta, backend="taichi")
   ```

2. **Smaller array sizes for quick tests**:
   ```python
   meta = fn.create_metamaterial(array_size=(2, 2))  # Fast
   ```

3. **Reduce trajectory save frequency**:
   ```python
   traj = fn.simulate_dynamics(meta, save_interval=500)  # Save less often
   ```

4. **Use linear mechanics for quick estimates**:
   ```python
   result = fn.simulate_mechanics(meta, model="linear")  # Fast
   ```

---

## Troubleshooting

### "Network has no crosslinks"

**Problem**: `simulate_dynamics()` requires crosslinks.

**Solution**: Use `create_metamaterial()` which automatically welds intersections:
```python
meta = fn.create_metamaterial(...)  # Has crosslinks
```

Or manually add crosslinks:
```python
net.auto_crosslink(threshold=0.5)
```

---

### "Unknown generator"

**Problem**: Generator name not recognized.

**Solution**: Check available generators:
```python
print(fn.list_generators())
```

---

### Taichi initialization error

**Problem**: Taichi already initialized or GPU not available.

**Solution**: Use CPU backend:
```python
traj = fn.simulate_dynamics(meta, backend="numpy")  # Pure Python fallback
```

---

## Version History

### 1.25.0 (2026-07-08)
- Added `create_metamaterial()` workflow
- Added `simulate_dynamics()` with Taichi acceleration
- Added `plot_metamaterial()`, `plot_dynamics()`, `plot_stress_strain()`
- Registry pattern for extensibility
- Fixed mass-spring system builder

### 1.24.0 (previous)
- Basic API: `create()`, `simulate_mechanics()`, `analyze()`, `plot()`
- Beam FEM solver
- 40+ structure generators

---

## API Design Philosophy

FiberNet is designed for **extensibility**:

1. **Registry pattern**: Easy to add new generators and backends
2. **Modular architecture**: Separate gen/, sim/, viz/, analysis/ modules
3. **Third-party integration**: LAMMPS, GROMACS, NetworkX, PyVista
4. **Multiple simulation backends**: Beam FEM, mass-spring, Taichi, future truss FEM
5. **Professional visualization**: Publication-ready plots
6. **ML/RL ready**: Structured data for machine learning workflows

---

## Contact & Support

- **Documentation**: https://fibernet.readthedocs.io
- **GitHub**: https://github.com/GellmanSparrowS/fibernet
- **Issues**: https://github.com/GellmanSparrowS/fibernet/issues
- **Email**: support@ml-biomat.com

---

## Citation

If you use FiberNet in your research, please cite:

```bibtex
@software{fibernet2026,
  title={FiberNet: A Comprehensive Toolkit for Fiber Network Structure Research},
  author={ML-BioMat Lab},
  year={2026},
  url={https://github.com/GellmanSparrowS/fibernet}
}
```
