# FiberNet 4.41.0

## Open-pressure Stokes flow

The default iterative solver can retry once when the discrete global liquid
balance needs tighter convergence. If the retry fails or does not improve the
balance, the first candidate still has to pass the existing algebraic,
cap-flux and energy checks. Mesh admission and dependencies are unchanged. Each pass is limited
to 600 iterations.

## Compare two saved cap fields

```sh
python -m pip install "fibernet==4.41.0"
python -m fibernet.analysis.stokes_cap_comparison reference.vtu candidate.vtu
```

The command emits JSON with the throughput ratio and the remaining
normal-velocity profile change on matched caps. Try the version-pinned inputs
in the [first-run guide](TET4_OPEN_STOKES.md#compare-throughput-and-velocity-redistribution-on-the-same-caps);
no solver is needed for this step.

The guide also includes [a fixed-geometry three-dimensional example](TET4_OPEN_STOKES.md#worked-example-when-a-narrower-passage-increases-flow-elsewhere)
with a common inlet and three branching passages. It uses the optional
provider stack and the existing bounded direct solver, separate from the
iterative retry.
