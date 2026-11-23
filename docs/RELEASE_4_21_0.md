# FiberNet 4.21.0

## One-call steady DC heating and surface convection

Use `solve_dc_thermal_tet4_with_convection` to calculate direct-current
power and pass its Joule heating into steady conduction on the same
tetrahedral mesh. Choose electrical and thermal conductivity separately
by material. The result includes the electrical fields and the thermal
fields, with separate convection and prescribed-temperature exchanges.

In this 3 mm bar with a 1 mm² cross-section, 0.3 V across 2 S/m generates
60 µW. The left end is held at 300 K, the right end exchanges heat with a
300 K ambient at 500 W/(m² K), and the sides are insulated. The convection
boundary removes approximately 25.714 µW; the held end removes 34.286 µW.

```python
from itertools import permutations, product

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4_dc_thermal import solve_dc_thermal_tet4_with_convection

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
right = points[:, 0] == 3.
electrodes = {**dict.fromkeys(map(int, node_ids[left]), 0.),
              **dict.fromkeys(map(int, node_ids[right]), .3)}
result = solve_dc_thermal_tet4_with_convection(
    mesh, conductivity_S_per_m_by_material={1: 2.},
    electrode_voltage_V_by_id=electrodes,
    conductivity_W_per_m_K_by_material={1: .25},
    prescribed_K_by_id=dict.fromkeys(map(int, node_ids[left]), 300.),
    faces_by_node_id=faces, convection_W_per_m2_K=500., ambient_K=300.,
)
heat = result.thermal
np.testing.assert_array_equal(result.electrical.joule_power_W_by_cell,
                              heat.integrated_source_W_by_cell)
exchanges = np.array([heat.total_source_W, heat.total_convective_heat_into_solid_W,
                      heat.heat_into_solid_W[left].sum()])
np.testing.assert_allclose(exchanges, 1e-6 * np.array([60., -180./7., -240./7.]), rtol=1e-10)
np.testing.assert_allclose(heat.balance_W, 0., atol=1e-15)
print(f"Generated: {exchanges[0] * 1e6:.3f} µW")
print(f"Convection removed: {-exchanges[1] * 1e6:.3f} µW")
print(f"Prescribed face removed: {-exchanges[2] * 1e6:.3f} µW")
```

The calculation is one-way: temperature does not affect the electrical
conductivity. The reference values verify the **integrated** heat
exchanges against the one-dimensional steady solution. See the
[DC Joule heating guide](TET4_DC_JOULE.md) for units and model assumptions.
