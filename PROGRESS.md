# FiberNet v4.1.0 — Release Complete

## Status: v4.1.0 Released ✅ (2026-07-23)

### PyPI
- **Published**: https://pypi.org/project/fibernet/4.1.0/
- Install: `pip install fibernet==4.1.0`

### GitHub
- **Release**: https://github.com/GellmanSparrowS/fibernet/releases/tag/v4.1.0
- **Commit**: https://github.com/GellmanSparrowS/fibernet/commit/41ece8f92919

### Tests
- 312/312 passing
- 0 failures, 1 warning (taichi locale deprecation — upstream issue)

---

## What's New in v4.1.0

### BeamFrameFEM_v6 (New FEM Module)
- `solve_2d()`: Linear 2D beam frame FEM (axial + bending + shear)
- `solve_2d_nonlinear()`: Geometrically nonlinear co-rotational solver
- `solve_3d()`: 3D beam frame analysis
- Full stress decomposition per element
- Validated: cantilever, displacement BC, nonlinear — all match analytical

### Large Deformation Test Suite
- 152 FEM simulations (128 2D + 24 3D), all passing
- 8 2D units × 4 radii × 4 stretch targets
- 6 3D units × 2 radii × 2 stretch targets
- Key finding: 100% FULL deformation propagation, all BENDING-dominated

### 3D Unit Types (14 total)
`bcc`, `chiral_3d`, `cubic`, `diamond_3d`, `fcc`, `gyroid`, `hcp`, `iwp`,
`lidinoid`, `neovius`, `octet`, `reentrant_3d`, `schwarz_d`, `schwarz_p`

### 2D Unit Types (12 total)
`chiral`, `cross`, `diamond`, `hexagon`, `honeycomb`, `kagome`,
`missing_rib`, `reentrant`, `square`, `star`, `triangle`, `voronoi`

### Bug Fixes
- `easy.py`: `simulate()` default mode `"tension"` → `"stretch"`
- Version consistency: `__init__.py`, `version.py`, `pyproject.toml` all at 4.1.0

### API Audit Results
- 62/62 top-level exports verified ✓
- All 12 2D units generate correctly ✓
- All 14 3D units generate correctly ✓
- TaichiEngine.dynamics + stretch_test working ✓
- SimResult save/load roundtrip ✓
- GraphFeatureExtractor (94 features) ✓
- GraphFeatureExtractor3D (60 features) ✓
- ML utilities (8 functions, lazy import) ✓
- RL utilities (7 functions, lazy import) ✓
- Visualization (12 render functions) ✓
- Transforms + Tiling ✓
