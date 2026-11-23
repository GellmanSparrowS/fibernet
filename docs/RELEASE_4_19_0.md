# FiberNet 4.19.0

## Internal heat generation with simultaneous surface cooling

Use `solve_steady_conduction_tet4_with_mixed_loads` when a solid generates
heat internally and exchanges heat through selected exterior triangles.
One solve reports volume heating, surface heating or cooling, and
prescribed-temperature heat exchange separately.

This synthetic 1 mm cube has six tetrahedra. The left face is held at
300 K. The right face removes 6 mW, matching the 6 mW generated inside.
The other faces are insulated. Despite the zero total load, the opposite
face is cooler than the prescribed face in this discrete solution.

```python
from itertools import permutations, product

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4_conduction import solve_steady_conduction_tet4_with_mixed_loads

points = np.array(list(product((0., 1.), repeat=3)))
cells = np.array([(0, a, a + b, 7) for a, b, _ in permutations((4, 2, 1))])
node_ids = 100 + np.arange(len(points))
cell_ids = 200 + np.arange(len(cells))
mesh = Tet4Bundle(
    points, cells, node_ids, cell_ids, np.ones(len(cells), dtype=int),
    {1: {"young_modulus_pa": 1e8, "poisson_ratio": .2}}, "mm",
)
right = [tuple(map(int, node_ids[idx])) for idx in ([4, 5, 7], [4, 6, 7])]
result = solve_steady_conduction_tet4_with_mixed_loads(
    mesh,
    conductivity_W_per_m_K_by_material={1: 1.},
    prescribed_K_by_id=dict.fromkeys(map(int, node_ids[:4]), 300.),
    source_W_per_m3_by_cell=dict.fromkeys(map(int, cell_ids), 6e6),
    faces_by_node_id=right,
    inward_flux_W_per_m2=-6000.,
)
np.testing.assert_allclose([result.total_source_W, result.total_surface_W],
                           [.006, -.006], rtol=1e-12)
np.testing.assert_allclose(result.temperature_difference_K[4:],
                           [-48/13, -75/26, -75/26, -63/26], atol=1e-12, rtol=0)
np.testing.assert_allclose(result.balance_W, 0., atol=1e-15)
print(f"Volume heating: {result.total_source_W * 1e3:+.3f} mW")
print(f"Surface heating: {result.total_surface_W * 1e3:+.3f} mW")
print(f"Minimum change from 300 K: {result.temperature_difference_K.min():+.3f} K")
```

Positive heat enters the solid; negative heat leaves it. The applied nodal
load and the integrated source and surface loads are separate outputs.
The temperatures are compared with the analytic discrete reference for this
six-element mesh, not a continuum or mesh-convergence claim. See the
[steady conduction guide](TET4_STEADY_CONDUCTION.md) for the input bounds
and the heat and temperature-pairing ledgers.
