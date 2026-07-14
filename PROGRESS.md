# v4 Tutorial Visualization - Progress Report

## Status: ✅ v10 COMPLETE

**Last Updated:** 2026-07-14  
**Script:** `scripts/run_tutorial_viz_v10.py`  
**Output:** `tutorials/v4_tutorial/tutorial_viz/` (22 PNGs)

---

## v10 Changes (from v9)

### Fixed Issues
1. **Simulation propagation**: Increased stiffness from 1e3→1e4, steps 8000→15000, ramp 70%→50%
2. **Fig 04/05**: Now uses voronoi with `n_pts_per_side=5` (5 nodes per edge)
3. **Added fig 02.5**: 12 voronoi structures with diverse intermediate point displacements (n_pts=5, amplitude ±0.4)

---

## Generated Visualizations (22 files)

| # | Name | Description |
|---|------|-------------|
| 01 | gallery_undeformed | 12 base unit types (3×4 grid, undeformed) |
| 02 | gallery_deformed | 12 base unit types with n_pts=2 displacements |
| 02.5 | voronoi_diverse | 12 voronoi with diverse n_pts=5 displacements |
| 03 | feature_stats | Top 20 features by variance |
| 04 | simulation_stretch | Single voronoi 8-frame trajectory (n_pts=5) |
| 05 | stress_distribution | Edge stress (original vs deformed, n_pts=5) |
| 06 | ml_analysis | ML predictions, confusion matrix, OOB loss |
| 07 | batch_stats | Force/energy/stretch distributions |
| 08 | force_feature_importance | Correlation analysis |
| 09 | rl_reward | Real RL training (50 episodes) |
| 10 | rl_structure_changes | Top 8 structure changes |

Each visualization has dark + light theme versions.

---

## Simulation Parameters (v10)

- Stiffness: 1e4
- Damping: 0.5
- Steps: 15000 (50% ramp + 50% hold)
- Target stretch: 1.5x
- Box: (1.0, 1.0), Grid: (2,2)
- Voronoi: n_internal=5, n_pts_per_side=5

### Results (20 voronoi structures)
- Force range: ~12k - 162k N
- Mean force: ~61k N
- Max stretch range: 2.2 - 17.2
- Displacement propagation: smooth gradient (left→right)

### Propagation Analysis (Structure 0)
- Nodes: 1672, Edges: 1728
- Zero displacement: 5.1% (all in left boundary)
- Displacement gradient: 0.0002 (left) → 0.947 (right)
- Linear increase across structure

---

## Known Issues

1. **Force magnitude**: Still high (12k-162k N). User requested "tens of Newtons"
2. **Some structures have extreme stretch**: max_stretch up to 17.2
3. **RL rewards**: Range -810k to -284k (negative force values)

---

## Next Steps

1. Review visualizations
2. Consider adjusting stiffness/stretch for better force scaling
3. Commit to git
