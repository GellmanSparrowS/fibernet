# Result-processing source fixture

`accelerated_delta.json` records the exact `SimResult.to_dict` change for
eligible plain full-history exports. The optimized path reuses the detached
state returned by per-call validation. Other accepted representations retain
their existing validation and export path.

The delta composes after `reviewed_lammps_reference_source()`. Its inverse
removes only the result-processing change, preserving the preceding metadata,
nodal-control and original-SI displacement-reference contracts.

Source identities:

- Parent SHA-256: `e86c85d10d8e05057b1752df189ab8cb210c15a1269a2a1a21a70ae7a680f48f`
- Result SHA-256: `9a5ede0ec36d841b824702db4b605508e18166ca3b008938aa4df78044f003b0`
- Delta SHA-256: `f9af4731055f6a07c1d691cf562998b5c69a1eafe89a4498c1f55bf81a74cc98`

Byte-preserving Git attributes and exact forward/inverse checks bind this
fixture to those source identities. Public signatures and serialized result
fields remain unchanged.
