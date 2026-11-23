# FiberNet 4.16.0

## Prescribed heat generation and cooling on Tet4 solids

Run steady conduction with a known volumetric source. The new experimental
`solve_steady_conduction_tet4_with_source` API takes a signed source density
in W/m³ for each stable cell ID. Set positive values to generate heat,
negative values to remove it, and explicit zero in every unloaded cell.
The existing source-free entry point is unchanged.

This synthetic benchmark divides a 3 mm tetrahedron (along each coordinate
axis) into four tetrahedra using its centroid. The four outer corners stay
at 300 K. Uniform heat generation at 8 × 10⁶ W/m³ raises the centroid to
303 K; the four corners each remove 9 mW, balancing 36 mW generated inside.

```python
from itertools import combinations

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4_conduction import solve_steady_conduction_tet4_with_source

points = [[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3], [.75, .75, .75]]
tetra = [(*face, 4) for face in combinations(range(4), 3)]
node_ids = [11, 17, 23, 31, 41]
cell_ids = [5, 7, 9, 13]
mesh = Tet4Bundle(
    np.array(points), np.array(tetra), np.array(node_ids), np.array(cell_ids),
    np.full(4, 1), {1: {"young_modulus_pa": 1e8, "poisson_ratio": .2}}, "mm",
)
result = solve_steady_conduction_tet4_with_source(
    mesh,
    conductivity_W_per_m_K_by_material={1: .25},
    prescribed_K_by_id=dict.fromkeys(node_ids[:4], 300.),
    source_W_per_m3_by_cell=dict.fromkeys(cell_ids, 8e6),
)
np.testing.assert_allclose(result.temperature_K, [300, 300, 300, 300, 303])
np.testing.assert_allclose(result.heat_into_solid_W, [-.009, -.009, -.009, -.009, 0])
np.testing.assert_allclose(result.total_source_W, .036)
np.testing.assert_allclose(result.balance_W, 0, atol=1e-16)
print(f"Centroid: {result.temperature_K[4]:.1f} K")
print(f"Generated power: {result.total_source_W * 1e3:.1f} mW")
```

Cap heat is negative when it leaves the solid. Generation and cap reactions
are in watts; the source and terminal pairings are in W K, not energy.
This API prescribes a known source, rather than computing Joule heating
or temperature-dependent generation. See the
[steady Tet4 conduction guide](TET4_STEADY_CONDUCTION.md) for model limits
and the result fields.
