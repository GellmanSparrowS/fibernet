# FiberNet 4.18.0

## Steady DC and one-way Joule heating on Tet4 solids

Use `solve_dc_tet4` for signed nodal electrode voltages, cell current
density, and cell Joule power. Use `solve_dc_thermal_tet4` to turn that
power into a thermal source on the same tetrahedral mesh. Electrical and
thermal conductivities use separate material maps.

This synthetic bar has area 1 mm² and two sections in series: 1 mm at
2 S/m and 2 mm at 8 S/m. The applied 0.3 V drives 0.4 mA and dissipates
120 µW. Both thermal caps are held at 300 K; they remove 80 and 40 µW.

```python
from itertools import permutations, product

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4_dc_thermal import solve_dc_thermal_tet4

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
        phases.append(1 if x == 0 else 2)
points = np.asarray(grid, dtype=float)
node_ids = np.arange(len(grid)) + 100
left, right = points[:, 0] == 0, points[:, 0] == 3
volts = dict.fromkeys(map(int, node_ids[left]), 0.)
volts.update(dict.fromkeys(map(int, node_ids[right]), .3))
mesh = Tet4Bundle(
    points, np.array(cells), node_ids, np.arange(len(cells)), np.array(phases),
    {m: {"young_modulus_pa": 1e8, "poisson_ratio": .2} for m in (1, 2)}, "mm",
)
r = solve_dc_thermal_tet4(
    mesh, conductivity_S_per_m_by_material={1: 2., 2: 8.},
    electrode_voltage_V_by_id=volts,
    conductivity_W_per_m_K_by_material={1: .25, 2: .25},
    prescribed_K_by_id=dict.fromkeys(volts, 300.),
)
current_A = r.electrical.electrode_current_into_solid_A[right].sum()
cap_W = [r.thermal.heat_into_solid_W[cap].sum() for cap in (left, right)]
np.testing.assert_allclose(current_A, 4e-4, rtol=1e-12)
np.testing.assert_allclose(r.electrical.total_joule_power_W, 120e-6, rtol=1e-12)
np.testing.assert_allclose(cap_W, [-80e-6, -40e-6], rtol=1e-12)
np.testing.assert_allclose(r.thermal.balance_W, 0, atol=1e-15)
print(f"Current: {current_A * 1e3:.3f} mA")
print(f"Joule heat: {r.electrical.total_joule_power_W * 1e6:.1f} µW")
```

Positive electrode current enters the solid; negative heat at a prescribed
thermal cap leaves it. This example verifies current and power balances,
not the local peak temperature on a coarse mesh. Electrical properties do
not change with temperature in this one-way model. See the
[DC and Joule heating guide](TET4_DC_JOULE.md) for supported inputs and limits.
