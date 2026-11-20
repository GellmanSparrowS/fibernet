# FiberNet 4.4.9

This release adds two input checks to 3D beam-frame simulations. It does
not add solid or tetrahedral finite elements.

The advanced `BeamFrameFEM.solve_3d` interface requires
`edge_index.shape == (2, M)`, with two endpoint rows for M beam members.
Older versions could use the first two rows of a tetrahedral connectivity
table and silently ignore the other nodes. The new shape check does not
identify the intent of every array: a `(2, 4)` array is a valid input for
four beams.

The `fibernet.simulate(..., backend="fem", mode="stretch")` shortcut loads
in x. In 3D, a nonunit ratio now requires nonzero x span. Before this check,
a graph with no extent in x could return an undeformed result for a
nonunit request. A ratio of 1.0 and strictly positive x spans are unchanged.

## Migration

Pass beam endpoint pairs with the documented orientation; do not pass
tetrahedral cells to the beam solver. For another loading direction, use
`BeamFrameFEM.solve_3d` with explicit loads or prescribed displacements.
Re-run earlier 3D jobs if either input was affected.

## Install

```bash
python -m pip install "fibernet==4.4.9"
python -m fibernet.examples.basic_usage --output-dir fibernet_demo
```

Package requirements and public function signatures are unchanged.
