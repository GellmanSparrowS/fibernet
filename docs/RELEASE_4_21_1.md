# FiberNet 4.21.1

## Generate and import a fused-cylinder CAD solid

This release packages a runnable example that fuses three perpendicular
cylinders into one homogeneous solid. It remeshes the same CAD geometry
independently at nominal sizes of 0.33 and 0.24 mm. Each mesh records
six named end caps and the ambient-side surface. The source distribution
also includes the original fitted-junction `.geo` alongside its unchanged
JSON fixture; that older CAD input is separate from the tagged example.

From a new directory, install the release and the optional tools, then
run the script from the matching source distribution:

```bash
python -m pip install "fibernet==4.21.1" "gmsh==4.13.1" "meshio==5.3.5"
python -m pip download --no-deps --no-binary=fibernet "fibernet==4.21.1"
python -m tarfile -e fibernet-4.21.1.tar.gz
python fibernet-4.21.1/examples/tet4_fused_cylinder_cad.py --output-dir fused-cad-output
```

Gmsh may need a system library such as `libGLU.so.1` on Linux. The output
directory must not exist. Use it only after the script creates the
`COMPLETE` marker. Each resolution provides an ASCII MSH 4.1 mesh, a
native Tet4Bundle with millimetre coordinates, and a JSON map from each
boundary name to the corresponding native bundle node IDs.

This is a mesh-generation and interchange example, not a material
comparison, simulation or mesh-convergence study. Installing FiberNet
alone does not install Gmsh or Meshio. See the
[tagged CAD and interchange guide](TET4_VTU_BUNDLE.md#generate-a-tagged-fused-cylinder-cad-mesh)
for file formats, units, and checks.
