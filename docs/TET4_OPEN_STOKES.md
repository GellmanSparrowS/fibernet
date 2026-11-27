# Open pressure-end Stokes flow on a Tet4 pore

The experimental `solve_open_pressure_stokes` method accepts the existing
FiberNet `Tet4Bundle`, with one face-connected fluid domain and planar
x-min/x-max open faces. Remaining exterior surfaces are no-slip. Use SI
viscosity (Pa·s) and pressure difference (Pa); positive difference drives
positive x flow. The elastic placeholders in the fluid bundle are not used
by this calculation. Each cap must have an unconstrained pressure-work velocity
trace; refine triangular caps whose vertices and edges all touch no-slip walls.

This is a gradient-form Taylor–Hood P2/P1 Stokes solve with natural pressure
loads at fixed open caps, not a periodic body-force method, a prescribed
velocity inlet, or a fully coupled fluid–structure solution. The outlet
pressure load defines the reference, rather than enforcing its recovered
nodal pressure. Vertex arrays of the native P2 velocity are samples, not the
complete quadratic field. Use the integrated terminal flux for quantitative
flow comparisons.

Install the optional NGSolve 6.2.2608 native runtime separately on CPython
3.12, together with FiberNet's `solid-iterative` extra for the default
PyAMG-preconditioned solver. Use `solver="direct"` to compare with native
UMFPACK when available. The API is bounded by mesh counts and raw operator
size; the direct option admits at most 8,000 cells. These caps do not
guarantee a peak-memory limit. This does not change
existing periodic-flow or core dependency behavior.
