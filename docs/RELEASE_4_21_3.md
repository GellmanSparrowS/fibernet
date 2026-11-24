# FiberNet 4.21.3

Install the release and download its matching source archive:

```sh
python -m pip install "fibernet==4.21.3"
python -m pip download --no-deps --no-binary=fibernet "fibernet==4.21.3"
python -m tarfile -e fibernet-4.21.3.tar.gz
cd fibernet-4.21.3
```

## Compare the response of a two-material fused junction

```sh
python -m pip install "gmsh==4.13.1" "meshio==5.3.5"
python examples/tet4_fused_cylinder_cad.py --junction-sphere --output-dir fused-cad-sphere
python examples/tet4_fused_cylinder_thermomechanics.py fused-cad-sphere/coarse.zip fused-cad-sphere/coarse.json --compare-junction-conductivity
```

The equal-power report includes the right end-face area and the
area-weighted mean axial displacement, while retaining the former node
mean. Use a matching bundle and boundary manifest after the CAD output
contains `COMPLETE`. Gmsh may require system libraries. The
[Tet4 thermoelastic guide](TET4_THERMOELASTIC.md) also gives a recipe
for varying only the junction stiffness at fixed temperature; the
matching 4.21.3 source archive includes the required example helper.
These are comparisons on discrete meshes with illustrative materials.

## Check signed material volumes

```sh
python examples/tet4_material_shell_check.py fused-cad-sphere/coarse.zip
```

With Meshio installed, the checker loads a trusted local Tet4 bundle
and compares signed boundary-shell volumes with summed cell volumes by
material ID. Closed cavity surfaces subtract rather than add. The
[Tet4 bundle guide](TET4_VTU_BUNDLE.md) shows how to inspect material IDs
in VTU and which companion files to retain. The checker does not test
global self-intersection or slicer compatibility.

This patch leaves the core solver APIs and default dependencies unchanged.
