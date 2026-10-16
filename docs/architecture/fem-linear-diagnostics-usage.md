# Linear FEM numerical diagnostics

`BeamFrameFEM.solve_2d` and `solve_3d` add `numerical_diagnostics` version 1.
Existing solution arrays, physical stiffness, reactions, stresses, member forces,
and defaults are unchanged. The same dictionary is copied into
`SimResult.metadata['numerical_diagnostics']` by `to_sim_result` when present.

```python
import numpy as np
from fibernet.ml.beam_frame_fem import BeamFrameFEM

fem = BeamFrameFEM(E=1000.)
raw = fem.solve_2d(
    np.array([[0], [1]]), np.array([[0., 0.], [1., 0.]]), np.array([.05]),
    forces=np.array([[0., 0.], [.1, 0.]]), fixed_nodes=[0], damping=.25)
diagnostic = raw['numerical_diagnostics']
assert diagnostic['regularization']['coefficient'] == .25
residual = diagnostic['physical_free_residual']
assert residual['status'] == 'available'
assert residual['node_rows'] == [1, 1, 1]
assert residual['dof_components'] == ['ux', 'uy', 'rz']
assert residual['values'] == raw['reactions'][1].tolist()
```

## Meaning and availability

`regularization` reports the actual scalar diagonal coefficient, including an
omitted/default coefficient (`1e-6`) or explicit zero. It is numerical
regularization, not physical support. Finite coefficients have status
`available`. Nonfinite coefficients use status `unavailable`, coefficient
`null`, and reason `nonfinite_regularization_coefficient`.

`physical_free_residual` contains **only free DOFs**, copied from the existing
physical residual `K_physical @ u - applied_load`. The 2D component order is
`ux, uy, rz`; 3D is `ux, uy, uz, rx, ry, rz`. Translation entries have force
units and rotation entries have moment units. The scalar coefficient acts on
mixed equations; it is not a universally dimensioned physical spring or damping.
Constrained reactions stay in `reactions` and are not replaced by synthetic zeros.
With every DOF constrained, the residual is available with three empty arrays.

A regularized solution generally has physical free residual approximately
`-coefficient * free_displacement`. A nonzero value is an observation, not a new
failure flag. `physical_equilibrium`, `load_compatibility`, and
`solution_uniqueness` always remain `not_evaluated`. Even a zero residual does
not certify stability, compatibility, uniqueness, or nonlinear convergence.

If any selected free residual is nonfinite, its status is `unavailable`, reason
is `nonfinite_recovered_residual`, and values are `null`; its DOF indices remain.
Only diagnostic metadata is made strict-JSON-safe. Legacy arrays are not repaired
or certified, and no broad JSON save behavior changes.

## Input rows and graph IDs

Raw diagnostics use `index_kind='input_node_row'`, with equal-length `node_rows`,
`dof_components`, and available `values` arrays. When adapting with a graph,
`to_sim_result` validates these records and replaces `node_rows` with `node_ids`,
using the **same resolved input-node map as displacement adaptation**. The new
index kind is `stable_node_id`. Without a graph, input-row semantics remain.

For direct-array input, provide `node_ids` and `edge_ids` for the exact input rows
along with the graph, following [the explicit identity contract](raw-array-identity-migration.md).
The legacy graph-input route still assumes insertion order; neither route can
prove an arbitrary caller's raw-array provenance. Malformed rows, lengths,
status/value combinations, or resolved mappings raise `ValueError` rather than
being truncated or silently relabeled. The adapter validates record shape, row
bounds, labels, and duplicate DOFs; without the full boundary conditions it
cannot independently verify that a caller supplied the complete true free set.

Both summary and detailed JSON preserve diagnostics through `SimResult.save`
and `load`. Summaries therefore include this residual vector as additive
metadata overhead. Renderer alignment deep-copies these self-contained stable-ID
records unchanged while reordering its known dense fields. Any new consumer must
inspect `index_kind`; row-indexed records do not become graph-indexed by rendering.
Historical raw results and JSON without diagnostics continue to omit them.

## Nonlinear finalization

The incremental nonlinear routine cannot recover final equilibrium residuals.
It replaces the last inner solve's diagnostics with an unavailable final record:
reason `final_nonlinear_equilibrium_residual_not_recovered`, `values=null`, and
no inherited row IDs, component indices, or index kind. Its model is
`incremental_updated_geometry_linear_solves`. The coefficient is retained only
with explicit regularization `scope='inner_linear_solves'`. Existing nonlinear
solver/field availability records keep their previous meaning.

See the [physical reaction definition](fem-3d-reaction-contract.md). This addition does
not implement a rank detector, compatibility algorithm, automatic restraints,
residual acceptance threshold, extra solve, or new export API.
