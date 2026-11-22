# FiberNet 4.10.5

## Reject overwide prescribed displacements in 3D beam FEM

`BeamFrameFEM.solve_3d` now raises `ValueError` before stiffness assembly
if a prescribed displacement has more than six components; previously,
a seventh component could silently constrain the next node.

For each input node row, supply a prefix of `(ux, uy, uz, rx, ry, rz)`.
Valid prefixes and full six-component vectors keep their previous meaning.
The 2D solvers, material selection and unit handling are unchanged. See
[the beam FEM contract](METHODS_BEAM_FEM_CONTRACT.md) for model limits.
