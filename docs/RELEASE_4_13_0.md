# FiberNet 4.13.0

## Prescribed-temperature thermoelasticity for 3D solids

Use the new experimental Tet4 Python API for one small-strain elastic
snapshot on a conforming tetrahedral mesh. Supply absolute nodal temperatures,
a stress-free temperature, mechanical restraints and thermal expansion by
material. The result includes displacement, support reaction, cell-mean
elastic stress and stored elastic energy. Material assignments and outputs
follow stable mesh IDs.

### Quick comparison on a synthetic two-material junction

These two 1 mm tetrahedra share a face. Keep geometry, stiffness, temperature
and restraints fixed, and replace one material's thermal expansion
coefficient. Both cells have Poisson ratio zero and all displacements are
held, so each tensile-positive normal stress has the independent check
`-E * alpha * (T - T_ref)`.

```python
import numpy as np
from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4_thermoelastic import solve_prescribed_temperature_tet4

node_ids = (10, 20, 30, 40, 50)
mesh = Tet4Bundle(
    points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                     [0, 0, 1], [0, 0, -1]], dtype=float),
    tetrahedra=np.array([[0, 1, 2, 3], [0, 2, 1, 4]]),
    node_ids=np.array(node_ids),
    cell_ids=np.array([101, 102]),
    material_ids=np.array([1, 2]),
    materials={i: {"young_modulus_pa": 1e8, "poisson_ratio": 0.0}
               for i in (1, 2)},
    length_unit="mm",
)

for alpha, expected_kPa in (({1: 1e-5, 2: 1e-5}, [-10., -10.]),
                            ({1: 1e-5, 2: -2e-5}, [-10., 20.])):
    result = solve_prescribed_temperature_tet4(
        mesh,
        temperature_K_by_id=dict.fromkeys(node_ids, 310.0),
        stress_free_temperature_K=300.0,
        alpha_per_K_by_material=alpha,
        fixed_dofs_by_id=dict.fromkeys(node_ids, ("x", "y", "z")),
        forces_by_id={},
    )
    sigma_zz_kPa = result.mean_stress_Pa[:, 2, 2] / 1e3
    np.testing.assert_allclose(sigma_zz_kPa, expected_kPa)
    print(result.cell_ids.tolist(), sigma_zz_kPa.tolist())
```

Cell 101 remains in 10 kPa compression. Cell 102 switches from 10 kPa
compression to 20 kPa tension. These are synthetic material coefficients; the
calculation illustrates the effect of changing one coefficient at a fixed
mesh, not a fitted material or an interface-strength prediction.

See the [Tet4 guide](TET4_THERMOELASTIC.md) for input units and result
interpretation. This API takes a prescribed temperature field. It does not
solve heat transfer or replace the [3D beam workflow](THERMOELASTIC_3D.md).
