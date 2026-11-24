# FiberNet 4.21.2

Install the release and download the matching source archive from
[PyPI](https://pypi.org/project/fibernet/4.21.2/#files). Then unpack it
and run the examples from the extracted directory:

```sh
python -m pip install "fibernet==4.21.2"
python -m tarfile -e fibernet-4.21.2.tar.gz
cd fibernet-4.21.2
```

## Compare conductivity in a fused-cylinder junction

```sh
python -m pip install "gmsh==4.13.1" "meshio==5.3.5"
python examples/tet4_fused_cylinder_cad.py --junction-sphere --output-dir fused-cad-sphere
python examples/tet4_fused_cylinder_thermomechanics.py fused-cad-sphere/coarse.zip fused-cad-sphere/coarse.json --compare-junction-conductivity
```

Use the matching bundle and manifest only after `COMPLETE` appears.
In this demonstration, the same conformal mesh carries both cases.
The lower junction conductivity is rerun at adjusted voltage to match
total Joule power. Heat is transferred one way to the thermoelastic
solve; temperatures and deformation do not feed back. The faceted
geometry and illustrative material values do not establish convergence
or real-material behavior. Gmsh may require native system libraries.
See the [Tet4 thermoelastic guide](TET4_THERMOELASTIC.md).

## Inspect backend inputs without an engine

From the same source archive directory:

```sh
python -B -m examples.cg_crosslinked_network --preflight
python -B examples/taichi_isolated_bcc.py --preflight
```

The LAMMPS preflight reports prepared SI deck identities and stable held
node IDs without writing a deck or starting an engine. The Taichi
preflight prepares a BCC lattice request without starting a worker.
Neither command qualifies nor executes a native provider. See the
[LAMMPS guide](integrations/LAMMPS_CG_CROSSLINKED_NETWORK.md) and
[Taichi guide](ISOLATED_TAICHI_CPU.md) for their respective runtime
requirements. The patch does not change core solver APIs or default
dependencies.
