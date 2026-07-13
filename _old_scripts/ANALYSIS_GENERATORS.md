# FiberNet Generator Module Analysis Report
## Date: 2026-07-08

## 1. Overview

Total code: ~8900 lines across 20 modules in `fibernet/gen/`

| Category | Count | Status |
|----------|-------|--------|
| Registered (fn.create) | 40 | 39/40 OK (field_guided timeout) |
| Extra gen.* | 54 | 41 OK, 10 FAIL, 3 WRONG_TYPE |
| Class-based | 7 | 0/7 OK |
| **Total** | **101** | **80 OK, 20 FAIL** |

## 2. Registered Generators (fn.create) — Detailed Data

| Name | Fibers | CL | Components | Connected | Time |
|------|--------|------|------------|-----------|------|
| arrowhead_auxetic_2d | 125 | 460 | 1 | Y | 0.00s |
| biomimetic_collagen | 131 | 155 | 1 | Y | 0.63s |
| biomimetic_fibrin | 244 | 249 | 2 | N | 0.32s |
| braided_rope | 3 | 732 | 1 | Y | 0.51s |
| chiral_honeycomb_2d | 262 | 736 | 1 | Y | 0.01s |
| cubic_3d | 144 | 528 | 1 | Y | 0.00s |
| diamond_lattice_3d | 333 | 825 | 1 | Y | 0.03s |
| double_helix | 2 | 51 | 1 | Y | 0.00s |
| electrospun | 200 | 646 | 1 | Y | 0.01s |
| field_guided | — | — | — | — | >60s TIMEOUT |
| fractal_tree | 31 | 45 | 1 | Y | 0.00s |
| gyroid_lattice_3d | 9858 | 59358 | 1 | Y | 0.32s |
| helix | 1 | 0 | 1 | Y | 0.00s |
| hierarchical_lattice_2d | 183 | 920 | 1 | Y | 0.00s |
| hilbert | 63 | 62 | 1 | Y | 0.00s |
| honeycomb_2d | 600 | 2688 | 1 | Y | 0.01s |
| kagome_2d | 504 | 2088 | 1 | Y | 0.01s |
| missing_rib_auxetic_2d | 73 | 248 | 1 | Y | 0.00s |
| octet_3d | 441 | 6444 | 1 | Y | 0.01s |
| oriented_2d | 100 | 44 | 63 | N | 0.01s |
| oriented_3d | 100 | 525 | 87 | N | 0.03s |
| plain_weave | 40 | 400 | 1 | Y | 0.01s |
| plate_lattice_3d | 360 | 4080 | 1 | Y | 0.01s |
| proper_octet_truss_3d | 441 | 6444 | 1 | Y | 0.01s |
| random_2d | 100 | 98 | 25 | N | 0.01s |
| random_3d | 100 | 215 | 91 | N | 0.02s |
| random_walk | 50 | 42 | 48 | N | 16.52s |
| reentrant_honeycomb_2d | 185 | 452 | 1 | Y | 0.00s |
| reentrant_honeycomb_3d | 360 | 2756 | 1 | Y | 0.01s |
| sierpinski | 81 | 237 | 1 | Y | 0.00s |
| square_2d | 220 | 598 | 1 | Y | 0.01s |
| star_honeycomb_2d | 200 | 412 | 1 | Y | 0.00s |
| tpms_gradient | 18540 | 176876 | 4 | N | 1.47s |
| tpms_lattice | 3196 | 121128 | 1 | Y | 0.23s |
| tpms_sheet | 8688 | 75980 | 1 | Y | 0.34s |
| triangular_2d | 324 | 1459 | 1 | Y | 0.01s |
| twill_weave | 40 | 400 | 1 | Y | 0.01s |
| twisted_bundle | 7 | 0 | 7 | N | 0.64s |
| voronoi_2d | 118 | 199 | 2 | N | 0.01s |
| voronoi_3d | 699 | 1358 | 60 | N | 0.21s |

## 3. Key Issues

### P0: Critical
1. **Random networks fragmented**: random_2d/3d, oriented_2d/3d have 25-91 components at defaults.
   - Root cause: density below percolation threshold
   - 2D threshold: λ_c = πL²N/A ≈ 5.7. Current: π×100×100/2500 = 2.0 (way below!)
2. **20+ generators return 0 crosslinks**: bundles, laminates, gradient, specialized —
   useless for simulation
3. **fiber_reinforced_composite**: produces 3.8M fibers → OOM

### P1: High
4. **Class generators return nx.Graph**: RegularNetworkGenerator, ZigZagGenerator
5. **Single Fiber returns**: sinusoidal_fiber_2d, helical_fiber_3d, arc_fiber_2d
6. **Duplicate**: octet_3d == proper_octet_truss_3d (identical output)
7. **Missing required params**: 7 generators lack usable defaults

### P2: Medium
8. **Slow**: field_guided (>60s), random_walk (16.5s for 50 fibers)
9. **Large structures**: gyroid (9858 fib), tpms_sheet (8688), tpms_gradient (18540)
10. **Not registered**: 54 gen.* generators not in fn.create()

## 4. What Works Well
- All ordered lattices: connected, correct topology, good parametric control
- All metamaterials: connected, physically meaningful structures
- Fractals, woven: connected with proper crosslinks
- TPMS (lattice/sheet): connected (gradient has 4 components)
- fn.create() registry: clean, extensible pattern
- FiberGraph infrastructure: solid foundation

## 5. Fix Strategy
See implementation commits for details.
