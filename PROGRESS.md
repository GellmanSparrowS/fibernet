# FiberNet Refactoring Progress

## Architecture

```
StructureGraph (core)
  ├── pattern_2d/3d (gen)  ← Base Unit + Transform + Tiling + Welding
  ├── transforms (core)    ← translate, rotate, mirror, scale
  ├── tiling (core)        ← tile_2d, tile_3d with node welding
  ├── fem (sim)            ← Taichi beam FEM solver
  ├── render (viz)         ← Publication-quality 2D/3D visualization
  ├── ml/dataset (ml)      ← Dataset generation pipeline
  └── rl_env (sim)         ← Reinforcement learning environment
```

## Completed

### Phase 1a: StructureGraph Core ✅
- File: `fibernet/core/structure_graph.py`
- NumPy-native node/edge storage
- Spatial hashing for node merging
- Edge deduplication (shared edges between cells)
- Edge discretization (n_internal points per edge)
- Boundary flags for periodic structures
- Conversions: networkx, numpy, FiberNetwork, JSON
- Connectivity checks, fingerprinting

### Phase 1b: Geometric Transforms ✅
- File: `fibernet/core/transforms.py`
- translate, rotate (2D/3D), mirror (x/y/z), scale
- compose() for chaining
- Correct boundary flag flipping on mirror
- Internal points transform support

### Phase 1c: Tiling + Welding ✅
- File: `fibernet/core/tiling.py`
- tile_2d/tile_3d with automatic node welding
- tile_with_transforms for per-cell transforms
- fit_unit_to_box for normalizing arbitrary units
- Boundary flag detection on outer perimeter

### Phase 2a: Pattern Engine ✅
- File: `fibernet/gen/pattern.py`
- Unified API: pattern_2d(unit, box, grid, ...)
- 11 built-in units, all produce connected structures
- Custom points with fit_to_box and boundary_mode
- Deterministic: no randomness unless seed+perturbation set
- 3D units: cubic, octet, diamond_3d

### Phase 2b: Unit Connectivity Fixes ✅
- All 11 built-in units now properly connected when tiled
- Kagome: star pattern through center
- Reentrant: shared center nodes
- Triangle: rhombus cell for triangular lattice

## In Progress

### Phase 2c: Edge Discretization ✅ (built into pattern engine)
- Every edge supports n_internal points for deformation
- discretize_edges() method for post-hoc discretization

## Next Steps

### Phase 3a: Taichi Beam FEM Solver
- Beam element FEM with Taichi acceleration
- Mechanical property extraction (E, ν, σ-ε curve)
- Deformation output for visualization
- Uniaxial tension/compression tests

### Phase 3b: Checkpoint/Resume
- Save/load simulation state
- Memory-safe batch processing

### Phase 4: Visualization
- Publication-quality renderer
- Deformed structure visualization
- Showcase generation

### Phase 5: ML/RL
- Dataset pipeline
- GNN features
- RL environment

## Git History
```
pre-refactor → 1a (StructureGraph) → 1b (transforms) → 1c (tiling) → 2a (pattern engine) → 2b (unit fixes)
```

---
**Last updated**: 2026-07-09
**Status**: Phase 2 complete, starting Phase 3 (Taichi FEM)
