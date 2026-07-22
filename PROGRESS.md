# FiberNet ML/RL API Benchmark Progress

## Status: Phase 2 Complete (Baseline) ✅
**Last Updated**: 2026-07-22

## Completed
- [x] Phase 1: API Correctness & Robustness (6/6 passed)
- [x] Phase 2: Synthetic Data Pipeline (baseline results)
- [x] Phase 2b: NetworkX + Scipy Graph Analysis

## Phase 2 Results

### Dataset
- 50 synthetic fiber networks (7 unit types: honeycomb, kagome, reentrant, diamond, square, triangle, voronoi)
- Size range: 9-2028 nodes
- Ground truth: DifferentiableSpringNetwork (E=1e9, ν=0.3)
- Compliance range: [0.008, 347.5] (4 orders of magnitude)

### GNN Graph-Level (Compliance Prediction)
- **R² = 0.036** (poor - baseline)
- MAE = 12.0, RMSE = 19.0
- 24K params, 20 training samples
- **Issue**: Too few samples for model complexity

### PINN_GNN Node-Level (Displacement Fields)
- val_loss = 0.85 (normalized MSE)
- correlation ≈ 0 (mean collapse)
- 239K params, 40 training samples
- **Issue**: Model too large, physics loss not helping
- **Training time**: 550s for 30 epochs (too slow)

### Graph-Level Verification (NetworkX Analysis)
- **Algebraic connectivity vs compliance**: corr = -0.10 (weak)
- **Avg degree vs compliance**: corr = -0.08 (weak)
- **Triangle lattice**: highest clustering (0.48), highest degree (4.9)
- **Diamond**: highest algebraic connectivity (0.66) = most rigid
- **Voronoi**: lowest connectivity (0.005), largest diameter (54)
- All graphs fully connected (reachability = 1.0)

### Bugs Found & Fixed
1. **Edge indexing**: `graph_from_structure` returns bidirectional edges (2× undirected)
   Fix: Use `edge_index.shape[1]` instead of `len(g.edges)`
2. **Log transform**: Compliance spans 4 orders of magnitude
   Fix: Use `log1p` for GNN training, `expm1` for prediction
3. **Displacement normalization**: PINN_GNN collapsing to mean
   Fix: Normalize targets with dataset statistics
4. **Name error**: `u_mean_val` → `u_mean`

## Key Insights
1. **GNN needs more data**: 50 samples insufficient for 24K params
2. **PINN_GNN needs better training**: physics loss not improving, correlation ≈ 0
3. **Graph topology weakly predicts compliance**: corr < 0.15
4. **NetworkX analysis reveals structure-property relationships**:
   - Triangle lattice: high clustering, high degree → moderate compliance
   - Diamond: high connectivity → low compliance (rigid)
   - Voronoi: low connectivity → variable compliance

## Next Steps
- [ ] Phase 3: Complexity scaling (timing/memory for different graph sizes)
- [ ] Phase 4: Cross-module integration (GNN+ODE, PINN+Conservative)
- [ ] Phase 5: Simulation extensions (Taichi, JAX)
- [ ] Phase 6: Statistical summary

## Files
- `benchmarks/phase2_synthetic_pipeline.py` (789 lines)
- `benchmarks/results/phase2_results.json`
- `benchmarks/results/phase2b_graph_analysis.json`

## Known Issues
- PINN_GNN training too slow (550s for 30 epochs)
- GNN graph-level prediction needs more data
- Physics loss not improving PINN_GNN performance
