# FiberNet 4.34.0

## Inspect three-dimensional solid results in a browser

The optional viewer opens the exterior and a material cut through a solved
FiberNet Tet4 volume. Each pick returns the original tetrahedron and material
tag. Color identifies the physical material region; changing the display
scale does not change the simulated result.

Try the installed two-material grip example on CPython 3.12:

```bash
python -m pip install "fibernet[browser-viz,solid-iterative]==4.34.0" "meshio==5.3.5"
python -m fibernet.examples.tet4_tagged_grip_iterative --output solid-demo --write-vtu
python -m fibernet.viz.tet4_browser solid-demo/deformation.vtu --section-z-mm 0.9265
```

Open `http://127.0.0.1:8765` in a WebGL-capable browser. The cut is defined
in the undeformed reference mesh and then warped by the computed displacement,
so the displayed section may not be planar. The viewer uses VTK, Trame and
vtk.js and needs no account or hosted service. The optional wheel set targets
Windows x64, Linux x64, Intel macOS and Apple Silicon macOS.

The viewer reads trusted local FiberNet files and does not solve or modify
them. For supported fields and mesh limits, see the [viewer guide](TET4_BROWSER_VIEWER.md).
