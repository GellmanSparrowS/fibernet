# Batch CSV resume: exact input and run identity

`fibernet.batch_simulate_from_json` now checks what produced a saved CSV outcome
before skipping its filename. This is an additive batch-only persistence
contract. Its public signature, solver dispatch and graph fingerprint are
unchanged. It does not change `FiberNetDataset.generate`, `StructureSweeper`,
native Dataset save/load, training or other simulation caches.

## Normal use

```python
from fibernet import batch_simulate_from_json

csv_path = batch_simulate_from_json(
    "structures", "runs/stretch_1_5", target_stretch=1.5,
    stiffness=1e5, damping=.3, num_steps=1000, save_interval=200,
)
# Same input bytes, run settings and implementation: skip existing outcomes.
csv_path = batch_simulate_from_json(
    "structures", "runs/stretch_1_5", target_stretch=1.5,
    stiffness=1e5, damping=.3, num_steps=1000, save_interval=200,
)
```

The function still supports only `backend="spring"` and `mode="stretch"`.
Input files use the native StructureGraph JSON schema. `stiffness` is the
force-like axial rigidity k in F=k×strain, not an extension-based spring constant
in N/m; `damping` is the velocity damping coefficient. Stored per-edge material
modulus does not automatically set the spring coefficient.
Successful and failed outcomes are both completed attempts: an identical failed
case remains skipped. There is no new automatic retry/reset operation.

For changed inputs or settings, choose a fresh output directory. Reuse of the old
directory raises `ValueError` with the filename and mismatch category before
engine construction or output changes. Do not remove bindings or substitute
current settings into historical rows to make an old run appear compatible.

## What a row records

The reserved `run_binding` CSV column contains canonical JSON:

- Schema `fibernet-batch-run-1`
- SHA-256 of the exact input JSON bytes
- Effective mode/backend/target_stretch/stiffness/damping/num_steps/save_interval,
  plus the forced `auto_steps=false`, and their canonical SHA-256
- Implementation recipe `fibernet-batch-source-1`, four source-file SHA-256 values,
  Python/NumPy/Taichi distribution versions, and the implementation SHA-256

The source recipe covers `easy.py`, `sim/accelerated.py`,
`core/structure_graph.py` and `core/material.py`. It captures current parsing,
graph conversion, numerical dispatch and implicit solver defaults, including
the grip rule, x-direction loading, ramp fraction, time step and engine defaults.
It does not infer material properties or add user-defined boundary conditions.

Distribution metadata is read without constructing an engine or importing an
optional backend solely for identity. A missing NumPy or Taichi distribution
record is represented by JSON null, not a guessed version. Source files are
required; unavailable source bytes cause a verification error. Package version
and Git HEAD alone do not establish identity in modified source trees.

Canonical JSON uses sorted keys, compact separators and finite numbers. Physical
scalar arguments accept finite Python/NumPy real numbers, excluding booleans,
and are recorded as floats. Step counts and save intervals require positive
Python/NumPy integers, excluding booleans. Numeric strings, complex values,
nonfinite values and fractional counts are rejected before reuse. `num_steps=None`
and `num_steps=5000` have the same identity because forced `auto_steps=False`
gives that effective step count. The public default remains 1000. Supported
arguments are still forwarded to the solver as supplied; the binding does not
change their numerical dispatch.

## Resume and compatibility rules

1. Every retained nonempty CSV row must carry a complete supported binding with
   valid digests and matching run parameters/implementation. This includes old
   filenames absent from the current input directory.
2. Every currently overlapping filename must also have matching input bytes.
   All overlaps are checked before any new file is simulated, including a
   late-sorted overlap following an early new filename.
3. Nonempty legacy CSVs without bindings, mixed bound/unbound rows, malformed
   bindings and reserved-column collisions reject continuation. Their bytes are
   left unchanged. Use a fresh directory; there is no automatic migration.
   A valid empty legacy header has no old outcomes and can be populated.
4. New filenames under the same run append normally. Existing custom CSV columns,
   quoted/multiline values and old rows are retained. New rows have blank values
   for absent custom columns. The existing CSV structural validation still applies.
5. A matching all-hit call returns before engine construction and does not rewrite
   the CSV or result files. A removed historical input keeps its original bound
   row; its unavailable input bytes have not been reverified. With no current JSON
   files, the return value remains the historical empty string; existing rows
   are still checked for binding/config compatibility first.

Raw-byte identity deliberately rejects formatting-only changes, same-size edits
with unchanged timestamps and changes to stored material fields ignored by this
spring model. This conservative input identity does not claim every such change
alters the solved physics. The input directory path itself is not the identity:
identical bytes copied to another directory can match.

## Input capture and failure behavior

Preflight streams each file through SHA-256 and retains only its digest. It does
not keep the whole input directory in memory. Pending files are then captured
one at a time, checked against their preflight hash and parsed from those same
captured bytes. Parsing uses the original default text-decoding semantics of
`open(path)` plus `json.load`; UTF-16/BOM byte autodetection is not added.

A preflight read failure aborts before engine construction or output changes.
A pending-file read failure or detected mutation aborts instead of creating an
authenticated reusable row. If mutation occurs after earlier files have already
committed, those earlier outcomes remain committed. Inputs must remain unchanged
during a call; there is no directory snapshot, file-lock or hostile-concurrency
guarantee. A malformed JSON or invalid graph whose bytes were successfully
captured produces an ordinary bound failure row. Solver and result-save failures
likewise receive exactly one bound failure outcome.

The binding is published with its outcome in the existing atomic CSV replacement.
An ordinary CSV serialization, fsync or replacement failure leaves the old CSV
unchanged under the existing filesystem assumptions. It may leave an already
written result JSON, as before. This is not a transaction across result JSON and
CSV, crash durability, or a concurrent-writer protocol.

## Explicit limits

A matching hit reuses the CSV outcome. It does not prove that the associated
result JSON exists or is intact; missing or corrupted result JSON is not repaired
or detected by this check. These digests do not authenticate malicious edits.

Whole-file source hashing conservatively invalidates comments and unrelated source
edits. It assumes the ordinary immutable imported distribution and does not
certify monkeypatching, hot reloads, all transitive dependencies, native runtime
settings, hardware or bitwise reproducibility. It is not a scientific-calibration,
unit-consistency, material-model, specimen-independence or model-fitness claim.

See [material precedence](material-consumption-contract.md) and
[native Dataset persistence](dataset-persistence-usage.md) for separate contracts.
