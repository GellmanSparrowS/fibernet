# Round fibers with fused junctions: a three-dimensional load-to-resistance example

Build three longitudinal fibers with four transverse crosslinks and one oblique
crosslink in Gmsh/OCCT. Fuse these into one connected solid, then partition the
solid at its midpoint into two conformal material volumes. The three round disks
on each end form the fixed and displaced grips. The synthetic specimen is not a
fit to experimental data.

The example uses the same stable-ID Tet4 bundle as the existing tagged-grip
workflow. It solves small-strain elasticity at 1.5 µm grip displacement and
recomputes DC under ±1.5 µm in transformed geometries. The two Young moduli
are fixed at 1 and 2 GPa (ν=0.3). Compare uniform (1,1), contrast (1,0.4), and
flipped contrast (0.4,1) S/m for the same structural solution. No intrinsic
piezoresistivity or feedback is included.

## Run from an installed FiberNet package

Use Python 3.12 and install the existing iterative mechanics extra plus the
standalone optional mesher. On Linux the Gmsh wheel also needs a system libGLU
runtime (for example, the Ubuntu package `libglu1-mesa`). The output folder must
not already exist.

```sh
python -m pip install "fibernet[solid-iterative]==4.35.0" "gmsh==4.13.1" "meshio==5.3.5"
python -m fibernet.examples.tet4_round_multipath --output round-demo --tier reference --write-vtu
```

To inspect this result in a browser, install the optional viewer and open a
reference cut at 0.2605 mm:

```sh
python -m pip install "fibernet[browser-viz]==4.35.0"
python -m fibernet.viz.tet4_browser round-demo/deformation.vtu --section-z-mm 0.2605 --scale 20
```

Open `http://127.0.0.1:8765`. A display scale of 20 exaggerates deformation
without changing the simulation. The `round-mechanoelectric.json` output records
actual forces, resistances, and the mesh-volume and resistance-derivative checks.

## Read the comparison

The independent straight-rod control already predicts **1.75×** for the
resistance contrast, and **4/3** for the change in force-normalized resistance
sensitivity if the conductivities are exchanged at fixed moduli. Similar
ratios on the fused network do not establish a new crosslink mechanism. Use
the finer tier for discretization comparisons, and do not interpret a solid
volume estimate as a resistance error bound. On this synthetic geometry,
linear meshes give approximately 4,725, 21,123 and 52,826 cells for the quick,
reference and finer tiers in the exercised Linux setup; exact element counts
need not be stable across platforms.
