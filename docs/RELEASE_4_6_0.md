# FiberNet 4.6.0

FiberNet can now solve bounded small-strain, isotropic linear elasticity on
conforming four-node tetrahedral meshes. Pass an existing Tet4 bundle and
specify nodal forces and zero Cartesian restraints by stable node ID. The
result gives displacements in meters, reactions in newtons, and strain
energy and half external work in joules.

Import `solve_static_tet4` from `fibernet.sim.tet4`. It is separate from
the beam solver and does not change `StructureGraph`, `fibernet.simulate`,
or the Tet4 v1 exchange format. Meshio is only needed if you read or write
a VTU bundle. See the [Tet4 solver documentation](TET4_STATIC.md) for
admission and numerical limits.

## Run the welded-junction example

The exact release source distribution includes a standalone example. From
an empty directory:

```bash
python -m pip install "fibernet==4.6.0"
python -m pip download --no-deps --no-binary=fibernet "fibernet==4.6.0"
python -c "import pathlib, tarfile; p='examples/tet4_static_welded_junction.py'; t=tarfile.open('fibernet-4.6.0.tar.gz'); pathlib.Path(p.rsplit('/', 1)[-1]).write_bytes(t.extractfile('fibernet-4.6.0/' + p).read()); t.close()"
python tet4_static_welded_junction.py --tier quick
python tet4_static_welded_junction.py --tier reference
```

The script generates its own mesh of three perpendicular cylinders, clamps
the negative-x end cap, and applies a total of 1 mN at the positive-x cap.
Within each tier, the two cases use identical geometry and loads. The
second case changes the assigned core from 1 to 3 GPa. The output checks
force and moment balance and recomputes strain energy from element
displacement gradients.

The quick tier uses 2,064 tetrahedra; reference uses 5,424. The reference
tier is close to the factor-size limit on the tested Linux configuration.
If your solver refuses it at that limit, use quick. Here the core stiffens
both discrete models. Neither tier is a smooth-boundary or mesh-converged
prediction, and the result does not establish strength or failure behavior.
