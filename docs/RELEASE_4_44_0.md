# FiberNet 4.44.0

- **Fixed pressure drop.** Compare flow before and after a pore deforms while holding the end-to-end pressure difference constant. Each mesh uses its own axial period.
- **Faster matched cases.** On a 2,604-tetrahedron pore benchmark, median batch time fell from 1.80 to 1.52 seconds. Each geometry still solves and checks its own flow field.

On Python 3.10 or newer, run the bundled solid-to-pore example:

```sh
python -m pip install "fibernet[pore-flow]==4.44.0"
python -m fibernet.examples.tet4_solid_to_pore_flow --matched-force
```

To hold pressure drop in your own matched cases instead of the example's pressure gradient:

```python
from fibernet.sim.tet4_pore import solve_matched_pore_geometry_cases

cases = solve_matched_pore_geometry_cases(
    reference_pore, loaded_pores, viscosity_Pa_s=1e-3, pressure_drop_Pa=0.8
)
```

[Periodic pore-flow method and inputs](tet4_periodic_pore.rst)
