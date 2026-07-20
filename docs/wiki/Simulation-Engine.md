# Simulation Engine

Mass-spring dynamics on the Taichi framework for GPU-accelerated simulation. Accessed via `fn.TaichiEngine()`.

## Physical Model

Point masses (nodes) connected by linear springs (edges), solved via explicit Verlet integration with dashpot damping and air drag.

## Simulation Phases

Each run has two phases:
1. **Relaxation** — energy minimization, no external loading
2. **Loading** — controlled displacement of boundary nodes to target stretch ratio

The split is controlled by `ramp_fraction` (default 0.2).

## Interfaces

| Method | Purpose |
|--------|---------|
| `stretch_test()` | High-level displacement-controlled uniaxial stretch |
| `dynamics()` | Low-level: custom fixed nodes, displacement schedules, external forces |
| `compute_forces()` | Single-step force computation (used internally, also available standalone) |

`stretch_test` auto-calculates step count from graph diameter when `auto_steps=True`. 3D structures get proportionally more steps.

## Result

Returns `SimResult` containing forces, stretches, trajectory, energy, and displacements. Supports JSON serialization with `save()`/`load()`.

## Performance

- **Field caching**: Taichi fields cached per `(dim, num_nodes, num_edges)` — avoids SNode exhaustion across repeated calls
- **Kernel caching**: compiled kernels reused for matching signatures
- **Memory guard**: `max_nodes` parameter warns/blocks oversized structures
- **Auto steps**: step count derived from graph diameter for proper wave propagation

## Dimension Support

2D and 3D structures are handled transparently — the engine detects dimension from the input `StructureGraph`.
