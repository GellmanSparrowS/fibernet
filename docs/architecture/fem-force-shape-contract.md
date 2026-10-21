# Linear FEM force-array shape contract

`BeamFrameFEM.solve_2d` and `solve_3d` accept translational nodal loads only.
For `N = node_pos.shape[0]`:

| Method | Exact `forces` shape | Columns |
| --- | --- | --- |
| `solve_2d` | `(N, 2)` | `Fx, Fy` |
| `solve_3d` | `(N, 3)` | `Fx, Fy, Fz` |

`forces=None` retains the existing zero-load behavior. Each row corresponds
to the same input row in `node_pos`; sparse graph node IDs are not row indices.
Use the existing graph-to-array ordering and result identity maps when needed.

## Rejected inputs and migration

Wrong row counts, missing columns, extra columns, and wrong ranks raise an
actionable `ValueError` before stiffness assembly or the linear solve. A single
row is rejected rather than broadcast to all nodes. The error states the expected
and received shapes and explains the translational-only contract.

In particular, `(N, 3)` loads in 2D and `(N, 6)` loads in 3D are rejected. Their
trailing columns were previously silently ignored; they were never applied
nodal moments. The solver's rotational displacement DOFs and reaction moments
do not imply support for rotational entries in `forces`.

Callers must supply the exact translational array. If extra columns genuinely
contain unused metadata, separate that metadata explicitly in the caller. If
they represent intended moments, do not silently slice them off: this API does
not implement that loading. This change does not introduce a moment API or
convert moments into equivalent force couples.

## Compatibility and scope

- Shape validation runs after the existing CPU Torch tensor `.numpy()`
  conversion and after replacing `None` with zeros. Supported NumPy numeric
  arrays, including read-only/noncontiguous arrays, retain their existing path.
- There is no new dtype, finite-value, masked-array, device, gradient, or
  array-like coercion policy. Torch tensors must still satisfy the existing
  `.numpy()` conversion requirements; GPU or gradient-enabled tensor support
  is not added here.
- Stiffness assembly, regularization/damping, displacement solutions, physical
  reactions (`K @ u - f`), stress recovery, output fields, and sparse-ID result
  adaptation are unchanged for valid shapes.
- `solve_2d_nonlinear` already forwards loads to `solve_2d`; that call inherits
  the shape rejection. Its own control validation and iteration algorithm are
  untouched. This does not broaden nonlinear validity or convergence claims.
- Regularization/residual diagnostics and disconnected-network stability remain
  separate work. A valid force shape is not evidence of physical equilibrium or
  a sufficiently constrained structure.

## Focused verification

`tests/test_fem_force_shape_contract.py` covers zero/nonzero extra columns,
missing components, wrong rows/ranks, rejection before assembly/solve, `None`
parity, supported NumPy storage/dtypes, optional CPU Torch conversion, sparse-ID
adaptation, and nonlinear forwarding. Adjacent reaction and beam-recovery tests
protect the corrected 3D reaction convention and existing numerical behavior.

The accompanying development evidence preserves the pre-change source and
failing tests. Frozen valid 2D/3D fixtures cover `None`, zero and nonzero loads,
prescribed displacement, independent parallel members, default/zero/nonzero
damping, and sparse-ID adaptation. Every raw and adapted returned field is
compared exactly, including the sparse stiffness matrix representation. These
focused checks do not constitute a full test-suite or general solver validation.
