# FiberNet 4.5.0

FiberNet can now round-trip a local bundle containing a Tet4 mesh and one
material assignment per cell. This is useful when preparing a three-dimensional
solid network for an external solver: geometry and connectivity stay with their
node and cell IDs, and the material table travels with the mesh.

The ZIP contains an ordinary `mesh.vtu` and a manifest. It preserves point and
cell ordering, tetrahedral corner order, and IDs through the full unsigned
64-bit range. The manifest declares the coordinate unit without converting the
geometry and stores each isotropic material's Young's modulus in Pa and
dimensionless Poisson's ratio. The bundle does not store loads or solution
fields. The existing `StructureGraph` and beam APIs are unchanged.

## Try the two-material example

```bash
python -m pip install "fibernet==4.5.0" "meshio==5.3.5"
```

Use `write_tet4_vtu_bundle` and `read_tet4_vtu_bundle` from
`fibernet.io.tet4_meshio`. The
[complete example and format contract](TET4_VTU_BUNDLE.md) show a mix of
materials and a material ID above the signed 64-bit range. Meshio is optional:
the ordinary FiberNet installation and line-mesh workflow are unchanged.
