# FiberNet 4.17.0

## Prescribed inward heat flux on Tet4 exterior faces

The experimental `solve_steady_conduction_tet4_with_surface_flux` API
applies one uniform heat flux to selected exterior triangles of a
four-node tetrahedral mesh. Each face is specified by three stable node
IDs; their order does not matter. Positive W/m² heats the solid and
negative W/m² cools it.

This synthetic bar has a length of 3 mm, a cross-section of 2 mm², and a
conductivity of 0.25 W/(m K). Hold the left cap at 300 K and apply
100 W/m² inward on the right cap. The right cap reaches 301.2 K and
supplies 0.2 mW, which the left cap removes.

```python
from itertools import permutations, product

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4_conduction import solve_steady_conduction_tet4_with_surface_flux

grid = list(product(range(2), repeat=3))
row = {point: i for i, point in enumerate(grid)}
tets = []
for order in permutations(range(3)):
    corner = [0, 0, 0]
    tet = [row[tuple(corner)]]
    for axis in order:
        corner[axis] += 1
        tet.append(row[tuple(corner)])
    tets.append(tet)
ids = np.arange(8) + 101
mesh = Tet4Bundle(
    np.asarray(grid) * [3., 1., 2.], np.asarray(tets), ids, np.arange(6),
    np.ones(6, dtype=int), {1: {"young_modulus_pa": 1e8, "poisson_ratio": .2}}, "mm",
)
a, b, c, d = map(int, ids[4:])
result = solve_steady_conduction_tet4_with_surface_flux(
    mesh,
    conductivity_W_per_m_K_by_material={1: .25},
    prescribed_K_by_id=dict.fromkeys(map(int, ids[:4]), 300.),
    faces_by_node_id=[(a, b, d), (a, c, d)],
    inward_flux_W_per_m2=100.,
)
np.testing.assert_allclose(result.temperature_K[4:], 301.2, rtol=0, atol=1e-12)
np.testing.assert_allclose(result.total_surface_W, 2e-4, rtol=1e-12)
np.testing.assert_allclose(result.heat_into_solid_W.sum(), -2e-4, rtol=1e-12)
np.testing.assert_allclose(result.balance_W, 0, atol=1e-15)
print(f"Right cap: {result.temperature_K[4]:.1f} K")
print(f"Applied power: {result.total_surface_W * 1e3:.3f} mW")
```

Applied power and prescribed-temperature reactions are reported separately.
This surface-only entry point does not combine with volumetric sources in
the same solve. See the [steady conduction guide](TET4_STEADY_CONDUCTION.md)
for supported inputs, sign conventions, and limits.
