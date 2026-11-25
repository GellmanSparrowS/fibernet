# FiberNet 4.25.0

## Compare materials at the same strain in a fused 3D network

The [three-path solid example](../examples/tet4_mechanoelectric_looped_network.py)
compares the resistance response of the same welded network with material
placed in different regions. It applies equal signed axial strain across
materials, reports the reaction force for the tensile state, and estimates
the resistance slope from the compressive and tensile states.

```bash
python -m pip install "fibernet==4.25.0"
```

Run `python -m examples.tet4_mechanoelectric_looped_network` from the matching
source checkout. The example transfers the elastic displacement to a
steady DC solve on the same solid mesh. It does not fit an intrinsic
strain-dependent conductivity law.

## Solve a small periodic pore-flow problem

```bash
python -m pip install "fibernet[pore-flow]==4.25.0"
```

With Python 3.10 or later, run `python -m examples.tet4_periodic_pore` from the
matching source checkout. The example compares flow through an empty duct
and a duct containing a synthetic three-path obstacle. The
[periodic pore guide](tet4_periodic_pore.rst) shows how to pass a Tet4 fluid
domain with matching axial boundary faces and how to distinguish fixed axial
pressure gradient from fixed total pressure drop. The paired-geometry
comparison does not compute solid deformation or fluid feedback on the solid.
