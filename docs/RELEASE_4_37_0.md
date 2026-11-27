# FiberNet 4.37.0

## Generate, export and open a three-dimensional flow field

This release adds an optional full quadratic Stokes velocity export to VTU,
and a local VTK/Trame browser for those saved fields. The solver stores
integrated cap flow and conservation checks; the browser displays steady
streamlines without running NGSolve again.

The example has three internal paths and two crosslinks in a voxelized duct.
It uses fixed geometry and fixed pressure, not coupled mechanics or a
round-fiber CAD mesh. Use CPython 3.12:

```sh
python -m pip install "fibernet[solid-iterative,browser-viz]==4.37.0" "ngsolve==6.2.2608" "meshio==5.3.5"
python -m fibernet.examples.tet4_stokes_browser_demo --output flow-demo
python -m fibernet.viz.stokes_browser flow-demo/velocity.vtu
```

Open `http://127.0.0.1:8765` on the computer running the viewer. For the
already generated FiberNet VTU, a supported CPython 3.12 environment only
needs FiberNet's `browser-viz` extra. No data is uploaded. The velocity file preserves original mesh IDs;
`flow-demo/summary.json` records flow and mass balance.

The default produces a 2,268-tetrahedron check. Add `--tier reference` to
the example for 18,144 tetrahedra, or `--reverse` to reverse the
pressure drop and drive flow from high x;
for the latter, seed the browser with `--from-max-x`. Choose a new output
directory with an existing parent. See the [open Stokes guide](TET4_OPEN_STOKES.md)
for the field format and the solver's physical scope.
