# LAMMPS result projection and numeric range

The dispatcher maps validated native trajectory samples to `SimResult` in graph
length units. Edge stretches use endpoint distance divided by the caller's
explicit harmonic equilibrium length; edge forces are signed harmonic bond
contributions, not general nodal forces or reactions. The maximum displacement
is the largest Euclidean displacement in graph length units.

Endpoint distance and displacement norms use scaled floating-point norm
algorithms rather than directly squaring components. This avoids avoidable
underflow and overflow for finite extremes. Stretch mean is computed after
scaling by the largest stretch. Standard deviation first subtracts the minimum
stretch, then scales by the remaining range so tightly clustered values retain
their small spread. Finite results can differ from the former formulas,
especially when the old norm or variance lost precision. These algorithms do
not guarantee correctly rounded statistics. If conversion, force, stretch, or aggregate
statistics produce nonfinite binary64 values, projection raises
`ValueError` instead of returning nonfinite values. A nonzero force or stretch
that underflows to zero is also rejected. Floating-point precision and range
limits still apply.

Subnormal support also depends on the active floating-point environment.
Taichi 1.7.4 enables flush-to-zero when its program is initialized; subsequent
Python/NumPy arithmetic on that thread can then turn a representable subnormal
force into zero. Projection rejects the detected loss with the same underflow
error. It does not change the caller's floating-point mode or guarantee gradual
underflow after another runtime changes it. Normal-range arithmetic and the
public unit conventions are unchanged. Tests keep a mandatory clean-interpreter
subnormal positive control and check fail-closed behavior in an active FTZ mode.
See [Taichi's initialization](https://github.com/taichi-dev/taichi/blob/v1.7.4/taichi/program/program.cpp).

These checks do not authenticate manually constructed output, qualify an engine
or prove physical accuracy. The regression cases are synthetic protocol data,
not native LAMMPS runs. They do not expand the admitted runtime or solver scope.
