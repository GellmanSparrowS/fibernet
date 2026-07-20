# Simulation Engine

FiberNet uses a mass-spring dynamics model implemented on the Taichi framework for GPU-accelerated simulation. The `TaichiEngine` class provides both low-level dynamics control and high-level stretch test interfaces.

## Physical Model

```
F_spring  = k × (L - L₀) / L₀ × direction
F_damping = -c × v_rel · direction × direction × L₀
F_drag    = -γ × v
```

- **Nodes**: point masses with position and velocity
- **Edges**: linear springs with configurable stiffness and rest length
- **Boundary**: Dirichlet BC (fixed nodes) during stretch
- **Integration**: explicit Verlet with damping and air drag

## Two-Phase Simulation

Each stretch test runs in two phases:

1. **Relaxation** (first `ramp_fraction` of steps): energy minimization, no external loading
2. **Loading** (remaining steps): controlled displacement of boundary nodes to target stretch ratio

## API

### Stretch Test (Recommended)

```python
engine = fn.TaichiEngine()
r = engine.stretch_test(
    g,
    target_stretch=1.5,      # stretch ratio
    stiffness=1e5,            # spring constant (N/m)
    damping=0.3,              # damping ratio
    num_steps=5000,           # total steps
    ramp_fraction=0.2,        # 20% relaxation + 80% loading
    save_interval=1000,       # trajectory save interval
    auto_steps=True,          # auto-calculate from graph diameter
)
```

### Low-Level Dynamics

```python
r = engine.dynamics(
    g,
    fixed_nodes=[0, 1, 2],
    displacement_schedule=schedule,
    stiffness=1e5,
    damping=0.3,
    dt=1e-4,
    num_steps=5000,
    save_interval=500,
    max_nodes=20000,          # memory guard
)
```

### Force Computation

```python
forces = engine.compute_forces(
    positions,      # (N, dim) node positions
    rest_lengths,   # (E,) rest lengths
    stiffness,      # (E,) per-edge stiffness
    edges,          # (E, 2) connectivity
)
```

## Result Object

`SimResult` contains:

| Field | Type | Description |
|-------|------|-------------|
| `max_force` | float | Maximum edge force (N) |
| `max_stretch` | float | Maximum edge stretch ratio |
| `mean_stretch` | float | Mean stretch |
| `edge_forces` | array | Per-edge forces (E,) |
| `edge_stretches` | array | Per-edge stretch ratios (E,) |
| `positions_trajectory` | list | List of (N, 3) position arrays |
| `energy` | float | Total elastic energy |
| `displacements` | array | Final node displacements (N, 3) |

### Serialization

```python
r.save("result.json", detailed=True)   # with trajectory
r2 = fn.SimResult.load("result.json")  # restore
```

## Performance Considerations

- **Field caching**: Taichi fields are cached per `(dim, num_nodes, num_edges)` to avoid SNode exhaustion
- **Kernel caching**: compiled kernels are reused across calls with matching signatures
- **Memory guard**: `max_nodes` parameter warns or blocks oversized structures
- **Auto steps**: `auto_steps=True` calculates steps from graph diameter for appropriate wave propagation

## 3D Simulation

The engine supports both 2D and 3D structures transparently:

```python
g = fn.pattern_3d(unit="cubic", box=(10, 10, 10), grid=(2, 2, 2))
r = engine.stretch_test(g, target_stretch=1.3)
```
