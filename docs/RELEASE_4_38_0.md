# FiberNet 4.38.0

## Compare flow through separate openings in a saved 3D Stokes field

Use a native FiberNet quadratic velocity field to calculate the signed flow
through each inlet or outlet opening, without rerunning NGSolve or installing
VTK. Each report includes opening area, centroid, flux, and total cap imbalance.

```sh
python -m pip install "fibernet==4.38.0"
python -m fibernet.analysis.stokes_cap_flux flow-demo/velocity.vtu
```

Positive flux leaves the domain. The flux is integrated on the two planar
extreme-`x` caps of the saved affine tetrahedral field. It does not infer
pressure, validate wall leakage, or accept arbitrary VTU files.

To visualize the flow paths, install the optional browser dependencies in
a supported CPython 3.12 environment:

```sh
python -m pip install "fibernet[browser-viz]==4.38.0"
python -m fibernet.viz.stokes_browser flow-demo/velocity.vtu --cap-components
```

The browser allocates some seeds to each connected opening on the selected
inlet cap. The number of drawn streamlines is not a measurement of flux.
See the [Open Stokes guide](TET4_OPEN_STOKES.md) for how to create a native
velocity field and compare views and quantitative results.
