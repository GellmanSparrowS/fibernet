# FiberNet 4.32.0

## Compare materials on the same 3D mesh

Start with a generated two-material frame (28,512 tetrahedra):

```bash
python -m pip install "fibernet[solid-iterative]==4.32.0" "meshio==5.3.5"
python -m fibernet.examples.tet4_tagged_grip_iterative --output solid-demo --write-vtu --material-sensitivities
```

To use your own ASCII Gmsh 2.2 or 4.1 mesh, put the exact physical volume
names in a UTF-8 `materials.json` file. Each named volume gets Young modulus
in pascals and its own Poisson ratio:

```json
{
  "Soft phase": {"young_modulus_pa": 1000000000, "poisson_ratio": 0.24},
  "Junction phase": {"young_modulus_pa": 2000000000, "poisson_ratio": 0.32},
  "Stiff phase": {"young_modulus_pa": 5000000000, "poisson_ratio": 0.39}
}
```

```bash
python -m fibernet.examples.tet4_tagged_grip_iterative --input mesh.msh --length-unit mm --materials-json materials.json --output case-a --write-vtu --material-sensitivities
```

Use `--grip-groups "Fixed end" "Driven end"` if your mesh does not name
the exterior grip groups `hold_face` and `pull_face`. Fix the input,
units, axis and stroke across cases, and use a new output directory per
material configuration. Open `deformation.vtu` in ParaView or read
`summary.json` and `nodal-fields.npz` in Python.

`--material-sensitivities` reports how the signed grip reaction responds to
small changes in log Young modulus and Poisson ratio, keyed by physical
volume ID, without another solve. These are local derivatives at the current
material settings and fixed grip stroke, not predictions of a new displacement
field or large changes. The model is small-strain bonded isotropic linear
elasticity. See the [tagged mesh guide](TET4_VTU_BUNDLE.md#run-a-tagged-mesh-above-the-banded-solver-limits)
for mesh constraints and all outputs.
