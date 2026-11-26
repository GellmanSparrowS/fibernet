# FiberNet 4.26.0

## DC conduction on larger three-dimensional solids

Pass `solver="cg"` to `solve_dc_tet4` to select the iterative solver instead
of the unchanged banded default. The iterative path accepts up to 60,000
nodes and 200,000 tetrahedra. It preserves stable IDs, electrode voltage,
SI fields, and current and power checks. No new dependency is needed.

Validated on two synthetic fused cylindrical network families, two grids
per family and two electrode directions. The largest tested mesh had
50,208 nodes and 186,388 tetrahedra. This option is for standalone
electrical conduction, not the solid mechanical or coupled thermal solver.
See [solid DC usage](TET4_DC_JOULE.md).

## Compare force and displacement controls for pore flow

The conforming [solid-to-pore-flow example](../examples/tet4_solid_to_pore_flow.py)
computes small elastic deformation, updates the matching pore mesh, and
computes stationary flow. Compare materials under the same displacement
or the same total grip force. The flow calculation does not feed loads
back into the solid.

Install the optional pore backend, then run the example from the repository
root or from an extracted source distribution:

~~~sh
python -m pip install "fibernet[pore-flow]==4.26.0"
python -m examples.tet4_solid_to_pore_flow --matched-force
~~~
