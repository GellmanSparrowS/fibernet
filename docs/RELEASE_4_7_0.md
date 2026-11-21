# FiberNet 4.7.0

`fibernet.sim.tet4.solve_static_tet4` can return cell strain, Cauchy stress
and volume with `return_cell_fields=True`. The default still returns
displacement, reactions and energy without computing cell fields.
`fibernet.io.tet4_results.solve_static_tet4_to_vtu` optionally writes an
undeformed mesh with point displacement and cell arrays from the same
solve. It requires Meshio but neither the solver nor the JSON-only example
does. The VTU results are visualization data, not a replacement for the
strict v1 Tet4 input bundle. See the [Tet4 contract](TET4_STATIC.md).

## Run the welded-junction example

From an empty directory, install FiberNet and retrieve the exact standalone
script from its source distribution on the package index:

```bash
python -m pip install "fibernet==4.7.0"
python -m pip download --no-deps --no-binary=fibernet "fibernet==4.7.0"
python -c "import pathlib, tarfile; p='examples/tet4_static_welded_junction.py'; t=tarfile.open('fibernet-4.7.0.tar.gz'); pathlib.Path(p.rsplit('/', 1)[-1]).write_bytes(t.extractfile('fibernet-4.7.0/' + p).read()); t.close()"
python tet4_static_welded_junction.py --tier quick
```

Three perpendicular cylinders are clamped at the negative-x cap and
loaded with 1 mN at the positive-x cap. Compare a homogeneous 1 GPa
material with the same mesh and load and a 3 GPa core. The script checks
force and moment balance and independently integrates strain energy.
Use `--tier reference` for 5,424 tetrahedra instead of 2,064 in quick.
If your solver refuses the reference tier at its factor-size limit,
use quick. Neither tier establishes mesh convergence or strength.

## Write both VTU results

```bash
python -m pip install "meshio==5.3.5"
python -c "from pathlib import Path; Path('welded-vtu').mkdir()"
python tet4_static_welded_junction.py --tier quick --write-vtu-dir welded-vtu
```

The existing directory receives `welded-quick-homogeneous.vtu` and
`welded-quick-stiffer_core.vtu`. Substitute `reference` for `quick`
to change the tier and the filenames. Omitting the flag leaves the run
JSON-only; adding it leaves the JSON comparison unchanged.

The script checks both names before solving. Each file is published
atomically without overwrite, but there is no two-file transaction: if
the second case fails, the first file may remain. Use a new directory
for an unambiguous rerun. The directory must support hard links.

In ParaView, open the VTUs and click Apply. For direct comparison, use
the same Cell Data `stress_Pa` component (for example, component 0 is
xx) and the same color range. The geometry is undeformed and coordinates
are in metres. Apply Warp by Vector with point `displacement_m`
separately if you want to display deformation. See the
[VTU field order and units](TET4_STATIC.md#view-a-static-result-in-paraview).
