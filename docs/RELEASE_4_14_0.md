# FiberNet 4.14.0

## Inspect a three-dimensional thermal snapshot

The optional Tet4 solve-and-write API exports one prescribed-temperature
snapshot to a new VTU file. The mesh, absolute nodal temperature, displacement,
cell-mean stress and integrated cell elastic energy all come from the same
solve. Open the VTU in ParaView, then color by temperature or apply
**Warp By Vector** to the displacement.

Install the core and optional VTU provider:

```bash
python -m pip install "fibernet==4.14.0" "meshio==5.3.5"
```

Run this small synthetic check in a writable directory on a file system
supporting hard links, with no existing `thermal_snapshot.vtu`:

```python
import numpy as np
from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.io.tet4_results import solve_prescribed_temperature_tet4_to_vtu

node_ids = (10, 20, 30, 40)
mesh = Tet4Bundle(
    points=np.vstack((np.zeros((1, 3)), np.eye(3))),
    tetrahedra=np.array([[0, 1, 2, 3]]),
    node_ids=np.array(node_ids),
    cell_ids=np.array([101]),
    material_ids=np.array([1]),
    materials={1: {"young_modulus_pa": 1e8, "poisson_ratio": 0.0}},
    length_unit="mm",
)
result = solve_prescribed_temperature_tet4_to_vtu(
    "thermal_snapshot.vtu", mesh,
    temperature_K_by_id=dict.fromkeys(node_ids, 310.0),
    stress_free_temperature_K=300.0,
    alpha_per_K_by_material={1: 1e-5},
    forces_by_id={},
    fixed_dofs_by_id=dict.fromkeys(node_ids, ("x", "y", "z")),
)
np.testing.assert_allclose(
    result.mean_stress_Pa[0], -10_000.0 * np.eye(3),
)
print("thermal_snapshot.vtu:", result.elastic_energy_J, "J")
```

The fully held cell has 10 kPa of compressive normal stress in all three
axes. In the VTU, coordinates and displacement are in metres, temperature is
in kelvin, stress is in pascals and `elastic_energy_J` is integrated per cell.
The file is a visualization diagnostic, not a restart or interchange
bundle, and the API does not solve heat transfer.

The [Tet4 guide](TET4_THERMOELASTIC.md) also links the source example that
compares two thermal-expansion assignments on one three-cylinder junction.
