# FiberNet 4.4.5

This patch makes batch continuation and VTU export detect ambiguous scientific
inputs before writing output. Public function signatures, solver models and
core dependencies remain unchanged.

## Installed library changes

- `batch_simulate_from_json` checks the output paths reserved by current
  inputs that already have CSV records as well as pending cases. The check
  runs before the no-pending return. A saved result in the input directory
  can therefore no longer silently enter the next batch as a new graph or
  overwrite an input. Completed bound cases still stay skipped. A legitimate
  independently supplied graph named `a_result.json` remains a valid input
  when its paths do not collide with another current input's output.
- VTU conversion and bundle export reject NumPy masked arrays/scalars in
  source geometry and boundary data, including nested and all-false masks.
  Node and edge record IDs must be nonnegative int64 values and match their
  graph dictionary keys. Validation runs before bundle publication. Valid
  sparse IDs, isolated nodes, parallel edges, field alignment and declared
  length units retain their existing behavior.

## Run the installed quick start

```bash
python -m pip install "fibernet==4.4.5"
python -m fibernet.examples.basic_usage --output-dir fibernet_demo
```

The quick start generates a graph, runs FEM, and writes the graph, response,
figure and JSON summary. VTU export uses the optional `meshio` dependency;
Taichi batch simulation uses the optional acceleration environment. Python
3.9 remains the core package minimum. Check optional backend platform support
in Getting Started before installing its extra.

## Migration

Keep batch input graphs and generated results in separate directories. If
an existing input folder contains earlier result files, identify them from
your run records and move them out before resuming; filenames alone do not
distinguish results from valid structure inputs. Preserve the existing CSV
and its source/parameter binding rather than editing rows to skip validation.

For VTU, resolve missing observations before export. If a masked array has
been checked and is known to contain complete values, explicitly convert it
to an ordinary array. Repair record IDs against the intended graph mapping;
do not substitute dictionary keys for an unknown original identity. The
exporter's length-unit declaration describes supplied coordinates and does
not rescale them.

Batch run bindings include implementation and runtime identity. A batch
recorded with an earlier package may therefore require a fresh output
directory when its binding differs after upgrade; retain its prior results.

## Versioned source example

`examples/matched_axial_backends.py` includes the deterministic 120-node,
573-bar 3D comparison and documented shared SI model, time grid and stable
IDs. The separately supplied LAMMPS and Taichi environments are required for
native runs. This example is included in the source distribution, distinct
from installed `fibernet.examples` commands. It compares the matched axial
model and does not establish equivalence of the backends' other models.

The publication process also waits for accepted distribution hashes to reach
the official PyPI Simple index before installation verification. This changes
release verification, not the installed library's simulation capabilities.
