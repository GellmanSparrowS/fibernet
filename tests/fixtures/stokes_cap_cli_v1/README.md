# Manufactured affine tetra10 prism
These two tiny synthetic fields were written once with FiberNet's native
`write_open_stokes_velocity_vtu` using the `prism`, `velocity`, and
`TET_EDGES` helpers in `tests/test_stokes_cap_comparison.py`. The candidate
reverses point and cell numbering (`prism(permuted=True)`) while keeping
the cap geometry. The reference x-speed is 1; candidate x-speed is
(1.02 + 0.06(y²-z²)). Both other components are zero. They are not outputs
of a fluid solve. Each cap has area 0.5; expected throughput ratio is 1.02 and
normalized profile residual is 0.02. The dependency-free read test uses
these exports but does not call the writer, which is covered separately.
