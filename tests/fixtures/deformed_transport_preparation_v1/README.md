# Retained deformation-preparation golden

`planar_golden.json` is a compact arithmetic/identity projection of a retained
synthetic 25-node/56-edge case. It contains the original
insertion-preserving graph record, first two SI displacement columns as explicit
caller-prescribed inputs, reference conductances, and retained preparation
outputs. No saved-mechanics reader is introduced.

The full graph/order hashes and historical lengths, conductances, and electrical
input identity are retained from the original evidence. They are not regenerated
by the new API. Historical integrity is checked separately from exact parity with
an independently spelled same-host arithmetic reference. Binary64 endpoint norms
can differ across Python versions; a current-host input digest is not substituted
for the saved physical input digest. The fixture includes the verified identity chain from the fixed
post-review hash, through its terminal hash and artifact manifest, to retained
`PREPARATION.json` and `DEFORMED_NETWORK.json` bytes.

Replaying the fixture runs preparation only. It does not recompute electrical,
thermal, mechanical, reference, or generator results and cannot enlarge the
original physical claim. The original artifacts remain unchanged.

The test pins the complete UTF-8 file bytes before decoding them. Git attributes
preserve those bytes in LF- and CRLF-oriented checkouts; checkout tests cover
both settings. Artifact labels in the provenance map are logical names, not
filesystem paths.

`arithmetic_reference.py.txt` pins the original preparation-law source identity
and the CPython 3.9 norm-source provenance. It uses only stdlib arithmetic, not
the new helper. Exact rational intervals propagate endpoint subtraction, norm
operations, product and division rounding; sqrt endpoints have exact rational
square certificates, with an additional 100/160-digit stability check. Bounds
are derived from this sealed normal-range case before observing candidate or
host-reference output. They are not a sample-fitted ULP threshold or a general
accuracy guarantee. The historical record and host reference must both satisfy
these bounds, and the candidate must still match the host reference exactly.

Candidate-only one-ULP changes, including a coordinated G/report/input-digest
change inside the bound, are rejected. On parity failure the test emits every
edge's compact hexadecimal/ULP diagnostics (at most 56 rows), with interpreter
identity. This does not rerun or recertify any historical physical result.
