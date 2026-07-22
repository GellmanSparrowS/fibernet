# FiberNet ML/RL API Benchmark Progress

## Status: Phase 1 Complete ✅
**Last Updated**: 2026-07-22

## Completed
- [x] Git initialized with baseline commit
- [x] Phase 1: API Correctness & Robustness (6/6 passed, 2.5s)

## Phase 1 Results Summary

### Test Structure
- 5 real fiber network structures: honeycomb 2x2 (18n/22e), honeycomb 5x5 (90n/130e),
  kagome 6x6 (169n/312e), reentrant 3x3 (54n/66e), voronoi 3x3 (504n/649e)

### Module Results
| Module | Status | Key Metrics |
|--------|--------|-------------|
| GFlowNet | ✅ PASS | 13K params, TB+DB training, reward shaping |
| DiffPhysics | ✅ PASS | 90-node spring: disp=0.017, stress=2.9MPa, grad 99.2% nonzero |
| | | 18-node beam: disp=0.014, rot=0.088 |
| | | Optimizer improvement: 7.71 |
| PINN_GNN | ✅ PASS | 46K params, force_balance=0.001, constitutive=0.29 |
| | | Multi-structure: different preds for 5 structures |
| NeuralODE | ✅ PASS | RK4 err=6e-08, DOPRI5 err=6e-08, Maxwell err=1.1e-05 |
| ConservativeNN | ✅ PASS | Hamiltonian drift=0, momentum ||ΣF||=1.7e-08 |
| | | Div-free max=5.6e-09, energy drift=2.2e-06 |
| Existing | ✅ PASS | GNN/Diffusion/GAN backward compatible |

### Bugs Found & Fixed
1. **Singular stiffness matrix** (DiffPhysics): `torch.linalg.lstsq` returns zeros
   for near-singular float32 matrices without gradient tracking.
   Fix: Custom `_robust_solve()` using `torch.linalg.pinv` with adaptive rcond.
2. **Beam network zero output** (DiffPhysics): Default damping too large for bending stiffness.
   Fix: Use explicit low damping (1e-6) for beam tests.
3. **PINN_GNN shape mismatch**: n_outputs=2 with 1D labels.
   Fix: Set n_outputs=1 in benchmark test.
4. **Constitutive loss explosion** (PINN_GNN): E=1e9 causes loss ~1e17.
   Fix: Normalize constitutive loss by E.

## Files
- Benchmark script: `benchmarks/phase1_api_correctness.py` (678 lines)
- Results: `benchmarks/results/phase1_results.json`
- Checkpoint: `benchmarks/results/phase1_checkpoint.json` (supports --resume)

## Next Steps
- [ ] Phase 2: Synthetic fiber data end-to-end pipeline tests
- [ ] Phase 3: Complexity scaling (small/medium/large graph timing)
- [ ] Phase 4: Cross-module integration pipelines
- [ ] Phase 5: Performance/memory profiling
- [ ] Phase 6: Statistical summary report

## Known Issues / Limitations
- Beam solver needs explicit low damping for small structures (EI << EA)
- GFlowNet needs more iterations for good exploration (5 iters too few)
- Fatigue model converges fast (10 cycles) - may need parameter tuning
- Creep power-law model shows very small strain increase (ratio ~1.0)
