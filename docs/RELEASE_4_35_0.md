# FiberNet 4.35.0

## Inspect signed displacement in the browser

Switch from material colors to signed x displacement in the exterior or
reference cut. Both views use the same symmetric scale in micrometres. A
selection returns the original tetrahedron, material, and interpolated physical
displacement vector. Changing the display multiplier changes the rendered
geometry, not the vector. See the [viewer guide](TET4_BROWSER_VIEWER.md).

## Run a round-fiber network on the installed package

Generate three main paths joined by five fused crosslinks. Partition the
shared volume into two material regions, solve a fixed-grip displacement,
and recompute resistance on the deformed mesh for three conductivity
assignments. Write a VTU to use in the browser or ParaView.

```sh
python -m pip install "fibernet[solid-iterative,browser-viz]==4.35.0" "gmsh==4.13.1" "meshio==5.3.5"
python -m fibernet.examples.tet4_round_multipath --output round-demo --tier reference --write-vtu
python -m fibernet.viz.tet4_browser round-demo/deformation.vtu --section-z-mm 0.2605 --scale 20
```

Use CPython 3.12; Gmsh and the viewer are optional and not part of the core
install. On Linux Gmsh needs the system libGLU runtime. The fully synthetic
example computes the geometry effect at fixed conductivity, not intrinsic
piezoresistivity. Compare with the independent straight-rod control and
mesh checks in the [round-fiber guide](TET4_ROUND_MULTIPATH.md).
