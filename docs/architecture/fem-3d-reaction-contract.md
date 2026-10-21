# Linear 3D FEM reactions: physical stiffness recovery

## Contract and scope

`BeamFrameFEM.solve_3d` reports the full nodal physical residual
`reactions = (K @ u_full - f_global).reshape(n_nodes, 6)`, using the same physical
stiffness matrix returned as `K`. Each row is ordered
`[Fx, Fy, Fz, Mx, My, Mz]`. At constrained DOFs these are recovered support
forces/moments; at free DOFs they are residuals, not additional supports.
Applied nodal forces are subtracted, including loads applied on constrained
nodes. The 2D linear solver already uses this physical-stiffness convention.

This correction changes one recovery expression in `solve_3d`. The linear solve
still uses `K_damped = K + damping * I`, with the unchanged default
`damping=1e-6`. Assembly, displacement boundary conditions, solved displacement,
stress recovery, member forces, and returned physical `K` are unchanged.
No 2D or nonlinear algorithm, API default, result schema, or consumer is changed.

## Regularization is still present in the solution

Let `F` and `C` denote the free and constrained DOFs, and let `lambda` be the
numerical `damping` parameter. The existing solve satisfies, up to numerical
roundoff,

```text
(K_FF + lambda I_FF) u_F = f_F - K_FC u_C
R_physical = K u - f
R_physical,F = -lambda u_F
```

Thus physical free-row residuals can be nonzero. Returning physical reactions
does not remove regularization from displacement and does not certify exact
undamped equilibrium. The parameter is used as diagonal numerical
regularization here; this static calculation does not model dynamic damping.

Previously the 3D recovery used `(K + lambda I) u - f`, so its change is exactly
`R_new - R_old = -lambda u` algebraically, with floating-point roundoff in
numerical comparisons. A zero prescribed displacement has no correction at that
DOF. A nonzero prescribed translation or rotation previously contributed an
artificial `lambda u_C` term to its recovered force or moment. The old free-row
values instead reported the approximately zero *regularized* residual. At
`damping=0`, the old and new recovery expressions coincide.

## Independent axial example

For a length-1 m bar with `E=1000 Pa`, radius `.05 m`, zero left displacement and
right extension `.01 m`, the physical tension is
`E pi r^2 extension / L = .07853981633974484 N`. With an additional, independently
represented radius-`.10 m` parallel member, total tension is
`.3926990816987242 N`. Each member retains its own section and axial force even
if the second member reverses orientation.

With no external nodal loads, the left/right axial support reactions are the
negative/positive total tension. At unchanged default regularization, the old
3D right reaction was too large by `1e-8 N` in either example. The correction
removes this recovery term without changing displacements or member forces.
Nonzero external loads at either constrained endpoint must also be subtracted.
These fixtures have no nonzero free displacements; their exact force balance
must not be generalized to every regularized problem.

Default FEM parallel-member assembly already preserves both physical members.
This fix changes neither graph construction's `allow_parallel` choice nor the
explicit legacy `deduplicate=True` projection. It is independent of duplicate
handling.

## Adaptation, persistence, and identity

`BeamFrameFEM.to_sim_result` copies the raw array into `metadata['reactions']`.
Both summary and detailed JSON retain that metadata, and `SimResult.load`
preserves its values. Detailed result alignment through
`fibernet.viz.render._align_result` validates and reorders reaction rows by
`node_ids`, including sparse IDs and reversed insertion order. This does not
change the physical values. Summary JSON still omits canonical displacement
arrays and is not made into a full deformation result by retaining reactions.

The canonical `edge_forces` and `max_force` remain signed member axial forces
and their maximum absolute magnitude, respectively. The full beam resultants
remain separately available; `max_force` is not redefined as a support reaction.

Previously saved JSON is not rewritten or automatically reinterpreted. A
reaction array alone cannot establish which recovery expression, regularization,
or displacement field produced it. Recompute from known inputs and solver
provenance when corrected historical reactions are needed. In particular,
serialized translational displacement alone is insufficient to reconstruct all
six rotational/translation correction components.

## Verification and limitations

`tests/test_fem_3d_reaction_contract.py` covers:

- Single and unequal-radius, reversed-orientation parallel bars in 2D and 3D
- Default, zero, `1e-3`, and `.25` regularization without changing defaults
- Independent axial force/displacement oracles with nonzero applied nodal loads
- Nonzero prescribed translations and rotations, constrained and free DOFs,
  including the expected nonzero physical free-row residual
- Planar 2D/3D displacement, stress, and reaction agreement
- Raw adaptation, summary/detailed JSON, sparse/reversed node-ID alignment,
  and both public quick-simulation and stretch-test routes

The regression suite was frozen before changing production code: 29 failures
and 23 passes on the accepted source, then all 52 cases pass after the single
expression change. Failed assertions use unchanged `rtol=atol=1e-12` gates;
the original default `1e-8 N` discrepancy is not rounded away.

A separate frozen before/after capture covers 54 cases, including two 2D
nonlinear controls. Every non-reaction raw field and every non-reaction adapted
field is numerically identical, including `u`, all stress fields, `edge_forces`,
`node_stress`, selected rows, counts, and the CSR values/indices of `K`.
All 2D outputs, including reactions, are unchanged. All 3D reaction changes
match `-damping*u` within the frozen tolerance. A source comparison also verifies
that precisely the one recovery expression differs from the accepted baseline.

These are bounded linear analytical and recovery regressions. They do not
validate finite-rotation nonlinear equilibrium, arbitrary networks, separate
FEM implementations, historical research outputs, or a different regularization
strategy. Nonlinear final-state reactions, stiffness and node moments remain
unavailable where not recovered; unavailable values must not be used as zero
labels. Retained stress fields are linear-reference approximations, not verified
nonlinear equilibrium. See the [diagnostics availability contract](fem-linear-diagnostics-usage.md#nonlinear-finalization).

Run the focused suite from the repository root:

```sh
python -m pytest -q tests/test_fem_3d_reaction_contract.py
```
