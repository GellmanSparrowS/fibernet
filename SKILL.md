# FiberNet API Reference

**Version**: 2.1.0  
**Last Updated**: 2026-07-08

## Quick Start

```python
import fibernet as fn

# 1. Create a structure
net = fn.create("reentrant_honeycomb_2d", reentrant_angle=150)

# 2. Check connectivity (all generators now produce fully connected networks)
print(f"Fibers: {net.num_fibers}, Crosslinks: {net.num_crosslinks}")
print(f"Density: {net.density():.4f}")

# 3. Run mechanics (beam FEM)
result = fn.simulate_mechanics(net, strain=0.001)
print(f"E = {result['modulus']:.2e} Pa")

# 4. Visualize (publication-quality, no nodes)
fig, ax = fn.viz.plot(net, color_by="orientation", theme="dark")
fn.viz.save_figure(fig, "structure.png", dpi=300)
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

# Metamaterials (fully connected via node-based construction)
net = fn.create("reentrant_honeycomb_2d", reentrant_angle=150)
net = fn.create("chiral_honeycomb_2d", node_radius=3.0)
net = fn.create("star_honeycomb_2d", star_angle=60)

# Fractals (fully connected via shared nodes)
net = fn.create("sierpinski", iterations=4)
net = fn.create("fractal_tree", iterations=6)
net = fn.create("hilbert", order=4)
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

### Network Utilities

#### `connect_components(max_gap=None, strategy="nearest")`

Bridge disconnected components by adding short fibers between nearest points.

```python
net = fn.create("some_generator")
n_bridges = net.connect_components(max_gap=50.0)
print(f"Added {n_bridges} bridging fibers")
```

**Parameters**:
- `max_gap` (float): Maximum gap distance to bridge (default: 10 × mean fiber length)
- `strategy` (str): "nearest" bridges nearest points, "centroid" bridges component centroids

**Returns**: Number of bridges added

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

# Access trajectory
positions = traj['positions']      # (n_steps, n_nodes, 3)
velocities = traj['velocities']    # (n_steps, n_nodes, 3)
energies = traj['energies']        # (n_steps,) total energy
```

**Returns**: dict with keys:
- `positions`: Node position history
- `velocities`: Node velocity history
- `energies`: Total energy history
- `n_steps`: Number of steps completed

---

## Visualization (Publication-Quality)

### Quick Plot

```python
import fibernet as fn
from fibernet.viz import plot, plot_3d, save_figure

net = fn.create("reentrant_honeycomb_2d")

# 2D plot with orientation coloring
fig, ax = plot(net, color_by="orientation", theme="dark", title="Re-entrant")
save_figure(fig, "reentrant.png", dpi=300)

# 3D plot
net3d = fn.create("diamond_lattice_3d")
fig, ax = plot_3d(net3d, theme="dark", elevation=25, azimuth=-60)
save_figure(fig, "diamond_3d.png", dpi=300)
```

### Color Modes

```python
# Uniform color
plot(net, color_by="uniform")

# By orientation (angle)
plot(net, color_by="orientation", colormap="hsv")

# By fiber length
plot(net, color_by="length", colormap="viridis")

# By fiber radius
plot(net, color_by="radius", colormap="plasma")

# Custom scalar data
custom_data = np.random.rand(net.num_fibers)
plot(net, color_by="custom", color_data=custom_data)
```

### Themes

```python
# Light theme (default)
plot(net, theme="light")

# Dark theme
plot(net, theme="dark")

# Publication theme (white background, minimal)
plot(net, theme="publication")

# Blueprint theme (blue on dark)
plot(net, theme="blueprint")
```

### Comparison Plots

```python
from fibernet.viz import plot_comparison

nets = [fn.create(name) for name in ["square_2d", "honeycomb_2d", "kagome_2d"]]
labels = ["Square", "Honeycomb", "Kagome"]

fig, axes = plot_comparison(nets, labels=labels, color_by="orientation", ncols=3)
save_figure(fig, "comparison.png", dpi=300)
```

### Statistics Panels

```python
from fibernet.viz import plot_statistics

fig = plot_statistics(net, theme="dark")
save_figure(fig, "statistics.png", dpi=250)
```

---

## Data Model

### FiberNetwork Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_fibers` | int | Number of fibers |
| `num_crosslinks` | int | Number of crosslinks |
| `dimension` | int | 2 or 3 |
| `total_length` | float | Sum of all fiber lengths |
| `total_volume` | float | Sum of all fiber volumes |
| `density()` | float | Volume/area fraction (handles 2D correctly) |
| `mean_fiber_length` | float | Average fiber length |
| `mean_radius` | float | Average fiber radius |

### Graph Model

| Model | Use Case | Key Object |
|-------|----------|------------|
| **Graph** (`nx.Graph`) | Welding, tiling, feature extraction | Nodes have `pos`, edges = fiber segments |
| **FiberNetwork** | FEM simulation, dynamics | `Fiber` + `Material` objects |

Both models interoperate via `to_networkx()` / `from_networkx()`.

---

## Graph Operations (`fibernet.graph`)

### `weld_graph(G, tolerance=1e-6)`

Detect all edge crossings and insert junction nodes at intersection points.

```python
import fibernet as fn

gen = fn.RegularNetworkGenerator(side_length=10, num_points_per_side=2, tiling=3)
G = gen.generate()
G_welded = fn.weld_graph(G)  # crossings become nodes
```

### `find_intersections(G)`

Return crossing points without modifying the graph.

```python
ix = fn.find_intersections(G)
print(f"{len(ix)} crossing points detected")
```

### `merge_coincident_nodes(G, tolerance=0.5)`

Merge nodes closer than `tolerance`.

```python
G_clean = fn.merge_coincident_nodes(G, tolerance=0.1)
```

---

## I/O Functions

### `save_graph_json(G, path)` / `load_graph_json(path)`

Save/load graphs in JSON format (D3.js compatible).

### `to_networkx(net)` / `from_networkx(G)`

Convert between `FiberNetwork` ↔ `nx.Graph`.

```python
G = fn.to_networkx(net)          # FiberNetwork → Graph
net = fn.from_networkx(G)         # Graph → FiberNetwork
```

---

## Feature Extraction (94 Dimensions)

### `fn.extract_features(G_or_net, canvas_size=512)`

Extract a comprehensive 94-dimensional feature vector.

| Feature Group | Count | Examples |
|---------------|-------|---------|
| Structural | 34 | `n_node`, `n_edge`, `mean_degree`, `density`, `n_components` |
| Pore | 18 | `porosity`, `mean_pore_area`, `max_pore_area`, `specific_surface` |
| Contact | 42 | `n_weld`, `weld_density`, `mean_coordination`, `anisotropy` |

```python
features = fn.extract_features(G_welded, canvas_size=256)
print(f"Weld count: {features['n_weld']}")
print(f"Porosity: {features['porosity']:.4f}")
```

---

## Complete Workflow Example

```python
import fibernet as fn
from fibernet.viz import plot, save_figure

# 1. Generate metamaterial
net = fn.create("reentrant_honeycomb_2d", reentrant_angle=150, grid_size=(5, 5))

# 2. Verify connectivity
print(f"Fibers: {net.num_fibers}, Crosslinks: {net.num_crosslinks}")
print(f"Connected: {net.num_components == 1}")

# 3. Visualize
fig, ax = plot(net, color_by="orientation", theme="dark")
save_figure(fig, "structure.png", dpi=300)

# 4. Extract features
features = fn.extract_features(net)
print(f"Density: {features['density']:.4f}")

# 5. Run mechanics
result = fn.simulate_mechanics(net, strain=0.001)
print(f"Young's modulus: {result['modulus']:.2e} Pa")

# 6. Save
fn.save_graph_json(fn.to_networkx(net), "network.json")
```

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
