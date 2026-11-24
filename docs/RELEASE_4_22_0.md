# FiberNet 4.22.0

Use `fibernet.sim.tet4.solve_static_tet4_load_cases` to solve several
force cases on the same solid without repeating mesh validation, stiffness
assembly, or the checked factorization for each case.

Pass an ordered list or tuple of 1 to 8 force mappings. Each mapping binds
stable node IDs to global Cartesian force components in newtons. All cases
share the same mesh, elastic material assignments, and zero-displacement
restraints. Each tuple entry is a detached result for the corresponding
input case, with its own reaction, energy, and residual checks. A failure
returns no partial results. Mesh, factor, and numerical limits are unchanged;
there is no retained factor or cache.

Use separate calls when changing materials or restraints. Keep using the
single-case API for prescribed motion, cell strain and stress, and VTU
export. This release does not add an MCP tool or change dependencies.

## Run the two-material example

```sh
python -m pip install "fibernet==4.22.0"
python -m pip download --no-deps --no-binary=fibernet "fibernet==4.22.0"
python -m tarfile -e fibernet-4.22.0.tar.gz
cd fibernet-4.22.0
```

Run the inline Python recipe in the
[Tet4 guide](TET4_STATIC.md#solve-several-load-cases-on-the-same-solid)
from the extracted directory. The source distribution includes both the
guide and its two-material mesh helper. The example applies equal and
opposite 1 mN loads in separate cases on the same solid; it needs no Meshio
or Gmsh.
