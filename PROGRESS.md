# v4 Tutorial Visualization - Progress Report

## Status: ✅ v9 COMPLETE

**Last Updated:** 2026-07-14  
**Script:** `scripts/run_tutorial_viz_v9.py`  
**Output:** `tutorials/v4_tutorial/tutorial_viz/` (20 PNGs)

---

## v9 Changes (from v8)

### Fixed Issues
1. **01/02 Gallery**: Changed from 20 voronoi → 12 unit types (3×4 grid)
   - 01: 12 base unit types (undeformed, no intermediate points)
   - 02: 12 base unit types (deformed with `n_pts_per_side=2`, auto displacements)
   
2. **04 Simulation**: Changed from 20 structures → 1 structure 8-frame trajectory (2×4 grid)
   - Shows single voronoi structure's stretch trajectory over time
   
3. **09 RL**: Changed from fake rewards → real RL training
   - Uses `create_rl_environment()` with `ParametricStructureEnv`
   - 50 episodes of real training with random policy

4. **Force Scaling**: Reduced stiffness from 1e5 → 1e3
   - Force range: 3.8k - 109k N (still high, may need further reduction)

---

## Generated Visualizations (20 files)

### 01: Gallery Undeformed (2 files: dark + light)
- **Content:** 12 base unit types in default state
- **Grid:** 3×4 layout
- **Units:** chiral, cross, diamond, hexagon, honeycomb, kagome, missing_rib, reentrant, square, star, triangle, voronoi

### 02: Gallery Deformed (2 files: dark + light)
- **Content:** Same 12 units with intermediate point displacements
- **Parameters:** `n_pts_per_side=2`, `seed=42` (auto-generated displacements)
- **Shows:** Curved/wavy beam geometry

### 03: Feature Statistics (2 files: dark + light)
- **Content:** Top 20 features by variance (from 94 total)
- **Data:** 20 voronoi structures, 26 valid features (68 removed as invalid)
- **CSV:** `data/voronoi_features.csv`

### 04: Simulation Trajectory (2 files: dark + light)
- **Content:** Single voronoi structure's 8-frame stretch trajectory
- **Grid:** 2×4 layout (8 frames)
- **Shows:** Progressive deformation from initial to final state

### 05: Stress Distribution (2 files: dark + light)
- **Content:** Edge stress visualization (original vs deformed)
- **Coloring:** Stretch ratio (RdYlGn_r colormap)
- **Range:** Stretch min-max displayed

### 06: ML Analysis (2 files: dark + light)
- **Content:** 4-panel ML analysis
  - Predictions vs Actual (R², RMSE)
  - Top 15 Feature Importances
  - Confusion Matrix (binary classification)
  - OOB Error vs Number of Trees

### 07: Batch Statistics (2 files: dark + light)
- **Content:** 4-panel batch simulation statistics
  - Force distribution
  - Force by structure index
  - Energy distribution
  - Stretch distribution

### 08: Force-Feature Importance (2 files: dark + light)
- **Content:** Correlation analysis
  - Top 15 force-feature correlations
  - Scatter plot of top correlated feature

### 09: RL Reward Curves (2 files: dark + light)
- **Content:** Real RL training results
  - Episode rewards over 50 episodes
  - Moving average (window=5)
  - Reward distribution histogram
- **Reward range:** -415k to -30k (negative force)

### 10: RL Structure Changes (2 files: dark + light)
- **Content:** Top 8 structures by force diversity
- **Overlay:** Gray=original, Colored=deformed
- **Grid:** 2×4 layout

---

## Simulation Results

**Parameters:**
- Stiffness: 1e3 (reduced from 1e5)
- Damping: 0.5
- Steps: 8000 (70% ramp + 30% hold)
- Target stretch: 1.5x
- Box: (1.0, 1.0)
- Grid: (2, 2)

**Results (20 voronoi structures):**
- Force range: 3,828 - 109,183 N
- Mean force: 10,722 N
- Valid features: 26 / 94

---

## Known Issues

1. **Force magnitude still high**: User requested "tens of Newtons" but got 3.8k-109k N
   - May need to reduce stiffness further (try 1e2 or 1e1)
   - Or adjust box size / target stretch

2. **Feature count low**: Only 26 valid features out of 94
   - 68 features removed (all NaN, all zero, or zero variance)
   - May need more diverse structures for better feature variation

---

## Next Steps

1. **Review visualizations** in `tutorials/v4_tutorial/tutorial_viz/`
2. **Adjust force scaling** if needed (reduce stiffness to 1e2 or 1e1)
3. **Commit changes** to git
4. **Generate more structures** if feature diversity is insufficient

---

## Technical Details

**Script:** `scripts/run_tutorial_viz_v9.py` (714 lines)
**Data:** `tutorials/v4_tutorial/data/voronoi_*_sim.json` (20 files)
**Features:** `tutorials/v4_tutorial/data/voronoi_features.csv` (26 valid features)

**Themes:** Each visualization generated in both dark and light themes
**Checkpoint:** Simulation results saved with checkpoint support
**Memory:** gc.collect() called every 5 simulations to prevent OOM

---

## Git Status

Ready to commit after review.
