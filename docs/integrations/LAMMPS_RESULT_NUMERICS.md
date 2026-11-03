# LAMMPS result projection and numeric range

The dispatcher maps validated native trajectory samples to `SimResult` in graph
length units. Edge stretches use endpoint distance divided by the caller's
explicit harmonic equilibrium length; edge forces are signed harmonic bond
contributions, not general nodal forces or reactions. The maximum displacement
is the largest Euclidean displacement in graph length units.

Endpoint distance and displacement norms use scaled floating-point norm
algorithms rather than directly squaring components. This avoids avoidable
underflow and overflow for finite extremes. Stretch mean and standard
deviation are computed after scaling by the largest stretch. Results at ordinary scales can
change by a few last-place bits. If conversion, force, stretch, or aggregate
statistics produce nonfinite binary64 values, projection raises
`ValueError` instead of returning nonfinite values. A nonzero force or stretch
that underflows to zero is also rejected. Floating-point precision and range
limits still apply.

These checks do not authenticate manually constructed output, qualify an engine
or prove physical accuracy. The regression cases are synthetic protocol data,
not native LAMMPS runs. They do not expand the admitted runtime or solver scope.
