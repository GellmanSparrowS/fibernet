# FiberNet 4.9.1

## Save graphs with NumPy segment counts

Native StructureGraph JSON now writes positive signed or unsigned
NumPy integer edge segment counts up to the signed 64-bit limit as
JSON integers. The live graph, stable IDs, parallel edges and ordinary
Python-integer JSON output are unchanged. This does not convert
arbitrary NumPy values in user metadata.

## Run the examples

The installed Python package includes the basic example:

```bash
python -m pip install "fibernet==4.9.1"
python -m fibernet.examples.basic_usage --output-dir fibernet_demo
```

The following standalone examples are in the **source distribution**,
not the installed wheel. In an empty directory, download and unpack it:

```bash
python -m pip download --no-deps --no-binary=fibernet "fibernet==4.9.1"
python -m tarfile -e fibernet-4.9.1.tar.gz
cd fibernet-4.9.1
python examples/tet4_static_fitted_placement.py --load-axis y --preflight
python examples/segmented_mask_topology.py
```

Remove `--preflight` to solve the three material cases. Choose `x` (the
default), `y` or `z` without changing the mesh or material placements.
Keep `examples/fixtures/fitted_axial_B.json` with the script. This is
a comparison on one mesh, not a mesh-convergence or strength claim.
The mask example uses synthetic data by default and accepts a Boolean
array in Python; it measures voxel connectivity, not individual
fibers or mechanical properties. Neither example requires Gmsh,
Meshio or FiberScope.

See the [fitted Tet4 guide](TET4_FITTED_PLACEMENT.md) and
[segmented-mask topology guide](SEGMENTED_MASK_TOPOLOGY.md).
