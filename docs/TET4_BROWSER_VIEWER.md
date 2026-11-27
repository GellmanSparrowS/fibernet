# Inspect a Tet4 result in a browser

Use the optional browser viewer to check the exterior and a cut through an existing FiberNet Tet4 calculation. Click a face to trace it back to the original tetrahedron and material tag. The viewer reads a file; it does not run the solver or change your results.

```sh
python -m pip install "fibernet[browser-viz]==4.34.0"
python -m fibernet.viz.tet4_browser result/deformation.vtu --section-z-mm 0.087
```

Use CPython 3.12. From a matching source checkout, use `python -m pip install ".[browser-viz]"` instead. The optional wheel set targets Windows x64, Linux x64, and macOS on Intel and Apple Silicon.

Open `http://127.0.0.1:8765` in a WebGL-capable browser. The app listens only on that loopback address and does not require hosting, an account, or a graphics card on the Python side. If the port is already in use, choose `--port 8766`.

- **Exterior** shows the boundary of the volume mesh. **Material section** cuts all tetrahedra at the given z-height in the **undeformed** reference configuration, then displays the section displaced by the solved vector field. At scale 1 the displaced section need not lie on a plane. Use `--scale 0` to see the reference geometry; `--scale 10` exaggerates the motion for display.
- Colors identify material **tags**, not an interpolated material value, and remain consistent between the two views. The pick panel shows the original tetrahedron ID as exact decimal text and its displayed location in millimetres. Near a shared edge, pick inside a triangle to avoid ambiguous selection. Section heights that pass through a mesh vertex are rejected to avoid ambiguous coincident faces; move the cut slightly.
- Switch from **Material** to **x displacement** for a signed field in micrometres. Exterior and section share the same symmetric range. The pick panel gives the interpolated physical displacement vector; changing the display multiplier does not rescale these values. A zero x displacement is shown as a constant. Material tags and element IDs are still available when picking in either color mode.
- Use a trusted local, linear-tetrahedron FiberNet VTU with cell fields `cell_id` and `material_id` and nodal `displacement_m`. This viewer preserves nonnegative 64-bit source cell IDs and accepts up to eight nonnegative 32-bit material tags, a 64 MiB input file and 250,000 volume tetrahedra. These are viewer input limits, not guarantees of interactive performance on all devices or memory limits for decoding a compressed file.
