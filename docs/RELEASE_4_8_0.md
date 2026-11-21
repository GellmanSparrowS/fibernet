# FiberNet 4.8.0

Check the Tet4 factor budget before solving, and convert a known surface
traction to the point-load mapping the solver already accepts. Both
helpers work on an in-memory Tet4 bundle without Meshio.

## Run a one-cell traction example

After installing `fibernet==4.8.0`, run this block. The coordinates describe
a 3 mm right tetrahedron. Hold the three nodes on the bottom face, and
apply a constant global traction to its slanted exterior face.

```python
import numpy as np
from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4 import inspect_static_tet4_admission, solve_static_tet4
from fibernet.sim.tet4_loads import nodal_forces_from_traction

tip = 2**64 - 1
bundle = Tet4Bundle(
    points=[[0., 0., 0.], [3., 0., 0.], [0., 3., 0.], [0., 0., 3.]],
    tetrahedra=[[0, 1, 2, 3]],
    node_ids=[31, 97, 2**63 + 1, tip],
    cell_ids=[20],
    material_ids=[8],
    materials={8: {"young_modulus_pa": 1e9, "poisson_ratio": 0.25}},
    length_unit="mm",
)
held = {node: ("x", "y", "z") for node in (31, 97, 2**63 + 1)}
budget = inspect_static_tet4_admission(bundle, fixed_dofs_by_id=held)
print("Factor entries:", budget.factor_entries, "/", budget.factor_entry_limit)
if not budget.within_factor_budget:
    raise SystemExit("This mesh and restraint layout exceed the factor budget.")

traction = np.array([2000., -3000., 5000.])  # global Pa
loads = nodal_forces_from_traction(
    bundle, [(97, 2**63 + 1, tip)], traction_Pa=traction,
)
result = solve_static_tet4(bundle, forces_by_id=loads, fixed_dofs_by_id=held)
tip_row = next(i for i, node in enumerate(result.node_ids) if int(node) == tip)
area_m2 = np.sqrt(3.) * (3e-3)**2 / 2
np.testing.assert_allclose(loads[tip], area_m2 * traction / 3, rtol=1e-14)
tip_stiffness = np.array([2e5, 2e5, 6e5])  # analytical N/m for the free node
u_expected = (area_m2 * traction / 3) / tip_stiffness
np.testing.assert_allclose(result.displacement_m[tip_row], u_expected, rtol=1e-13)
print("Tip displacement (m):", result.displacement_m[tip_row])
```

The precheck reports the conservative factor-entry envelope; it is not
measured memory use or a guarantee of numerical solvability. It takes no
loads and neither assembles nor factors the stiffness matrix. The mesh
and factor limits have not changed.

The traction has three global components, in Pa; face order does not
define a normal. Face IDs must describe distinct exterior triangles.
The helper returns nodal forces in N. It does not convert a normal
pressure, distribute a specified total force, or select a surface for you.
See [surface traction](TET4_TRACTION.md) and the [Tet4 solver](TET4_STATIC.md).

The optional welded-junction VTU demo now reports a cyclic output-directory
symlink as a normal path error on affected Python versions. VTU output
still requires optional Meshio and refuses to overwrite existing files.
