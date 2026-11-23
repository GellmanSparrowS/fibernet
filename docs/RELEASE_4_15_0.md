# FiberNet 4.15.0

## Steady heat conduction on a Tet4 mesh

The experimental `solve_steady_conduction_tet4` Python API computes a
steady, source-free temperature field from prescribed nodal temperatures
and one isotropic conductivity per material. It returns absolute nodal
temperatures, cellwise Fourier flux, and discrete heat entering the solid
through the prescribed nodes. It uses the base FiberNet install, with no
VTU or Meshio requirement.

This synthetic bar has a 1 mm² cross-section and two sections in series:
1 mm at 1 W/(m K), followed by 2 mm at 4 W/(m K). Prescribe all nodes on
both end caps; the other faces are insulated in the weak form.

```python
from itertools import permutations, product

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4_conduction import solve_steady_conduction_tet4

grid = list(product(range(4), range(2), range(2)))
row = {point: i for i, point in enumerate(grid)}
cells, phases = [], []
for x in range(3):
    for order in permutations(range(3)):
        corner = [x, 0, 0]
        tet = [row[tuple(corner)]]
        for axis in order:
            corner[axis] += 1
            tet.append(row[tuple(corner)])
        cells.append(tet)
        phases.append(17 if x == 0 else 9)

points = np.asarray(grid, dtype=float)
ids = np.arange(len(grid), dtype=np.uint64) + 100
hot, cold = points[:, 0] == 0, points[:, 0] == 3
boundary = {int(i): 330.0 for i in ids[hot]}
boundary.update({int(i): 300.0 for i in ids[cold]})
bundle = Tet4Bundle(
    points, np.array(cells), ids, np.arange(len(cells)), np.array(phases),
    {i: {"young_modulus_pa": 1e8, "poisson_ratio": 0.3} for i in (17, 9)},
    "mm",
)
result = solve_steady_conduction_tet4(
    bundle, conductivity_W_per_m_K_by_material={17: 1.0, 9: 4.0},
    prescribed_K_by_id=boundary,
)
caps_W = [result.heat_into_solid_W[hot].sum(), result.heat_into_solid_W[cold].sum()]
np.testing.assert_allclose(result.temperature_K, np.repeat([330, 310, 305, 300], 4))
np.testing.assert_allclose(result.flux_W_per_m2, np.tile([20_000., 0., 0.], (18, 1)), atol=3e-9)
np.testing.assert_allclose(caps_W, [0.02, -0.02])
np.testing.assert_allclose(result.terminal_pairing_W_K, 0.6)
print("heat into solid at hot/cold caps (W):", caps_W)
```

Heat into the solid is positive at the hot cap and negative at the cold
cap, in watts. Cell flux is in W/m². The terminal pairing is
`Q * (T_hot - T_cold) = 0.6 W K`, not energy in joules. The mesh records
require elastic constants, but conduction does not use them.

See the [steady conduction guide](TET4_STEADY_CONDUCTION.md) for bounds
and field definitions. The solver has no volumetric sources, specified
heat flux, transient response, or thermoelastic feedback. The discretization
does not guarantee pointwise flux continuity or a discrete maximum
principle on arbitrary meshes.
