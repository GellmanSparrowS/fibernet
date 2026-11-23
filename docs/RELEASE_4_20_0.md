# FiberNet 4.20.0

## Steady convection on tetrahedral solids

Use `solve_steady_conduction_tet4_with_convection` to model heat exchange
with an environment at a known temperature. Unlike a prescribed heat flux,
convection responds to the computed surface temperature. A single solve
can also include signed volumetric heat and prescribed-temperature nodes.

This synthetic bar is 3 mm long with a 1 mm² cross-section. It generates
60 µW of heat. The left end is held at 300 K; the right end exchanges
heat with a 300 K ambient at 500 W/(m² K); the sides are insulated.
The right end removes about 25.714 µW and the left end about 34.286 µW.

```python
from itertools import permutations, product

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4_conduction import solve_steady_conduction_tet4_with_convection

grid = list(product(range(7), range(3), range(3)))
row = {point: i for i, point in enumerate(grid)}
points = np.array(grid, dtype=float) * .5  # mm
cells = []
for start in product(range(6), range(2), range(2)):
    for order in permutations(range(3)):
        corner = list(start)
        tet = [row[tuple(corner)]]
        for axis in order:
            corner[axis] += 1
            tet.append(row[tuple(corner)])
        cells.append(tet)
node_ids = 100 + np.arange(len(points))
cell_ids = 200 + np.arange(len(cells))
mesh = Tet4Bundle(
    points, np.array(cells), node_ids, cell_ids,
    np.ones(len(cells), dtype=int),
    {1: {"young_modulus_pa": 1e8, "poisson_ratio": .2}}, "mm",
)
faces = []
for y, z in product(range(2), repeat=2):
    n0, n1, n2, n3 = [int(node_ids[row[(6, j, k)]]) for j, k in
                      [(y, z), (y + 1, z), (y, z + 1), (y + 1, z + 1)]]
    faces.extend(((n0, n1, n3), (n0, n2, n3)))
left = points[:, 0] == 0
result = solve_steady_conduction_tet4_with_convection(
    mesh, conductivity_W_per_m_K_by_material={1: .25},
    prescribed_K_by_id=dict.fromkeys(map(int, node_ids[left]), 300.),
    source_W_per_m3_by_cell=dict.fromkeys(map(int, cell_ids), 2e4),
    faces_by_node_id=faces, convection_W_per_m2_K=500., ambient_K=300.,
)
exchanges = np.array([result.total_source_W, result.total_convective_heat_into_solid_W,
                      result.heat_into_solid_W[left].sum()])
np.testing.assert_allclose(exchanges, 1e-6 * np.array([60., -180./7., -240./7.]), rtol=1e-12)
np.testing.assert_allclose(result.balance_W, 0., atol=1e-15)
print(f"Generated: {exchanges[0] * 1e6:.3f} µW")
print(f"Convection removed: {-exchanges[1] * 1e6:.3f} µW")
print(f"Prescribed face removed: {-exchanges[2] * 1e6:.3f} µW")
```

Positive heat enters the solid; negative heat leaves it. The reference
values check integrated exchange against the one-dimensional steady solution;
they do not imply pointwise agreement on a coarse mesh. The coefficient is
an input, not a prediction of the fluid flow. See the
[steady conduction guide](TET4_STEADY_CONDUCTION.md) for units and bounds.
