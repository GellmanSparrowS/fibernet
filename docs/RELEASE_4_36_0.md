# FiberNet 4.36.0

## Drive three-dimensional pore flow with a pressure drop

`solve_open_pressure_stokes` accepts a face-connected fluid `Tet4Bundle` and
computes steady flow between its two planar x-end caps. It returns integrated
cap flow, mass and power balance checks, and velocity and pressure samples at
the original mesh vertices. It is an experimental single-fluid fixed-geometry
solver; it does not change the periodic solver or couple back to mechanics.

Use CPython 3.12. The NGSolve/Netgen backend and the iterative solver are
optional; ordinary FiberNet installation is unchanged.

```sh
python -m pip install "fibernet[solid-iterative]==4.36.0" "ngsolve==6.2.2608"
```

To reproduce a square-duct check and solve a three-dimensional pore with
18,144 tetrahedra, use [the verification example](../scripts/smoke_tet4_open_stokes.py)
from this repository or from the 4.36.0 source archive. Once extracted, run:

```sh
python scripts/smoke_tet4_open_stokes.py
```

See the [open pressure-end guide](TET4_OPEN_STOKES.md) for cap geometry,
mesh limits, unit conventions, and the direct comparison solver.
