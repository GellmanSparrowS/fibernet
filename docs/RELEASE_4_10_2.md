# FiberNet 4.10.2

## Keep saved batch results out of graph inputs

When resuming `batch_simulate_from_json`, a saved simulation result can
remain after its successful input graph was removed. If that JSON is present
among the inputs at the historical result path (or an alias of that file),
the batch now refuses it before initializing the solver or writing files.
A valid graph at that path is not filtered by filename alone.

Keep graph inputs and results in separate directories. Only checkpoints
already compatible with the current batch implementation, runtime and settings
can be resumed. This change does not read old incompatible checkpoints as
current ones or repair their records.

## Reject missing legacy fiber coordinates

The legacy `Fiber` constructor now rejects actively masked coordinates in
NumPy arrays and ordinary nested sequences. It does this before NumPy can
discard the mask and expose the hidden coordinate. Supply observed geometry
rather than filling values for export. Fully observed masked arrays and custom
array providers retain their prior behavior; resolve missing values upstream
when using a custom provider.

The legacy VTP file format and ordinary observed-geometry output are unchanged.
See the [legacy VTK/VTP guide](io/legacy_vtk_attributes.md).
