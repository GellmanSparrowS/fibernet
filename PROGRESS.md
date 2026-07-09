# FiberNet v3 — Refactoring Progress

## Architecture

```
fibernet/
├── core/
│   ├── structure_graph.py    ← StructureGraph: universal data structure
│   ├── transforms.py         ← translate, rotate, mirror, scale, compose
│   ├── tiling.py             ← tile_2d, tile_3d with node welding
│   ├── material.py           ← Material properties
│   ├── fiber.py              ← Fiber (legacy compat)
│   └── network.py            ← FiberNetwork (legacy compat)
├── gen/
│   └── pattern.py            ← Pattern Engine: Base Unit + Transform + Tiling
├── sim/
│   ├── fem.py                ← Beam FEM solver (Euler-Bernoulli, scipy.sparse)
│   └── rl_env.py             ← Gymnasium RL environment
├── viz/
│   └── render.py             ← Publication-quality 2D/3D visualization
├── ml/
│   └── dataset_v2.py         ← ML dataset generation pipeline
└── __init__.py               ← Unified top-level API
```

## All Phases Complete ✅

### Phase 1: Core Foundation
- **StructureGraph**: NumPy-native, spatial-hash node merging, edge deduplication,
  boundary flags, internal points, fingerprinting, JSON/networkx/numpy conversion
- **Transforms**: translate, rotate (2D/3D), mirror (x/y/z), scale, compose
- **Tiling**: tile_2d/tile_3d with automatic welding, tile_with_transforms, fit_unit_to_box

### Phase 2: Structure Generation
- **Pattern Engine**: `pattern_2d()` / `pattern_3d()` unified API
- **11 built-in 2D units**: square, triangle, hexagon, honeycomb, kagome,
  reentrant (auxetic), chiral, star, cross, missing_rib, diamond
- **3 built-in 3D units**: cubic, octet, diamond_3d
- All 11 2D units produce connected structures when tiled
- Deterministic: no randomness unless explicitly seeded
- Edge discretization: N internal points per edge for deformation

### Phase 3: Simulation
- **BeamFEM**: Euler-Bernoulli beam FEM with scipy.sparse assembly
- Uniaxial tension, shear test, stress-strain curve
- Effective E*, ν*, G* extraction
- Deformed graph output for visualization
- Reentrant verified auxetic (negative Poisson's ratio)

### Phase 4: Visualization
- **render.py**: render_graph, render_graph_3d, render_deformation, render_gallery
- 4 themes: dark, light, blueprint, publication
- Color modes: uniform, orientation, length, stress, strain, custom
- Glow effects, depth-based 3D alpha, edge discretization rendering
- 9 showcase images in `output_viz/`

### Phase 5: ML / RL
- **dataset_v2.py**: parameter sweep → FEM labeling → numpy/JSON export
- Feature extraction (18 topological + geometric features)
- Checkpoint/resume for long generations
- **rl_env.py**: Gymnasium-compatible FiberNetworkEnv
  - Action: choose unit, grid, radius
  - Reward: distance to target E*, ν*

### Phase 6: Integration
- Unified top-level API: `import fibernet as fn`
- 9/9 integration tests pass in ~2 seconds
- Clean git history with rollback points

## Quick Start

```python
import fibernet as fn

# Generate
g = fn.pattern_2d(unit="honeycomb", box=(10, 10), grid=(5, 5), n_internal=8)

# Simulate
fem = fn.BeamFEM(g)
result = fem.uniaxial_tension(strain=0.01)
print(f"E* = {result.effective_youngs_modulus:.2e} Pa")

# Visualize
fig = fn.render_graph(g, theme="dark", color_by="orientation")
fig.savefig("honeycomb.png", dpi=200)

# ML
ds = fn.generate_dataset(units=["honeycomb", "square"], save_dir="datasets/")

# RL
env = fn.FiberNetworkEnv(target_E=1e5, target_nu=-0.3)
```

## Tests

```bash
cd fibernet && source .venv/bin/activate
python tests/test_integration_v3.py
```

## Git Log

```
pre-refactor → 1a → 1b → 1c → 2a → 2b → 3a → 4 → 5 → 6 (integration)
```

---
**Status**: All phases complete. 9/9 tests passing.
**Last updated**: 2026-07-09
