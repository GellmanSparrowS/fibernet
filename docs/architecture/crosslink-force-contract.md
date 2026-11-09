# Crosslink force endpoints

The standalone models in `fibernet.core.crosslinks` return
`force, next_state = model.compute_force(p1, p2, state)`. The returned vector
is the force on **p1**. Apply its negative to **p2**. The displacement is
`p2 - p1`, so a positive component in that direction attracts p1 toward p2.
These examples and regressions use three-component Cartesian positions.

For an active conservative radial interaction with `r = |p2 - p1|`,

```text
force_on_p1 = -gradient_p1 U = (dU/dr) * (p2 - p1) / r
force_on_p2 = -force_on_p1
```

Spring, Bonded (stretching), and Covalent crosslinks already use this convention.
Constraint, friction/slip, angular and broken-state behavior has additional
model-specific rules; the two-position energy is not a universal stateful
potential for every crosslink model.

## Screened ionic interaction

For `A = q1*q2 / (4*pi*epsilon_0*epsilon_r)` and Debye length `lambda_D`,

```text
U(r) = A * exp(-r/lambda_D) / r
dU/dr = -A * exp(-r/lambda_D) * (1 + r/lambda_D) / r**2
```

The corrected Ionic force uses this derivative. Opposite charges attract;
like charges repel. Swapping the endpoint positions reverses the force.
The gradient contract applies at nonzero separation outside the existing
`r < 1e-15 m` force guard and before a bond has broken.

Rupture semantics are unchanged: the threshold test is strictly
`norm(force) > breaking_force`. The threshold-crossing call returns its
nonzero force and a broken state. Subsequent calls with that broken state
return zero current force. The state can retain the earlier diagnostic
`state.force`, so consumers must use the returned vector for current force.
`energy(p1, p2)` has no state argument and still evaluates the unbroken
Yukawa potential. The zero-force coincidence guard does not regularize that
potential, which is singular at zero separation.

## Compatibility and migration

Before this correction, including in 4.4.1, Ionic returned the force on p2,
unlike Spring and Covalent. All nonzero Ionic force vectors, including the
force stored on a rupture transition, now have the opposite sign. The
constructor, factory type `create_crosslink('ionic', ...)`, return shape,
energy, force magnitudes, screening and rupture thresholds are unchanged.
This is an intentional numerical behavior change, not bitwise compatibility.

- Callers applying the result to p1 and its negative to p2 need no code change.
  Their Ionic trajectories change to the corrected direction.
- Callers applying the old Ionic result to p2 must instead apply the corrected
  result to p1, or negate it when computing p2's force.
- Remove an Ionic-only compensating negation if it was used to recover p1's
  force. Do not negate Spring or Covalent forces.
- Recompute sign-sensitive reference data or trajectories before comparing
  old and corrected results. Do not mix saved force vectors from the two
  conventions. There is no automatic migration of retained diagnostic states.

At this revision, repository consumers of `compute_force` are standalone
tests, not the network FEM, Taichi, fracture, transport or ML/RL solvers. The
factory exposes the model, but this correction does not add solver integration
or validate coupled simulation trajectories. The MCP adapter binds all native
Python source hashes, so its matching build needs refreshed binding and
compatibility validation even though it does not call these models.

## Regression evidence

`tests/test_crosslink_force_contract.py` checks both endpoint energy gradients
at several separations and orientations, attraction/repulsion for both charge
signs, endpoint reversal, analytic energy and magnitude, and the existing
Spring/Bonded/Covalent convention. It also checks the rupture threshold,
transition force, subsequent broken-state result, zero charge and the
near-coincidence guard. These are model-level numerical checks, not
experimental calibration or solver validation.
