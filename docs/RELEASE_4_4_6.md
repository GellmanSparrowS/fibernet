# FiberNet 4.4.6

This patch corrects which fibers connect when a custom cell is mapped onto
a 3D surface. `fibernet.gen.map_cells(..., base_graph=graph)` now handles
nonconsecutive StructureGraph node IDs, which can arise after node deletion.
Previous versions could silently connect fibers to the wrong corners or
raise an index error. Dense-ID built-in cells and public signatures are
unchanged.

## Migration

If you previously mapped an edited custom cell with gaps in its node IDs,
regenerate the mapped geometry from the original graph and surface. Review
the new fiber endpoints before reusing downstream simulation or fabrication
inputs. No manual renumbering of the graph is needed. The mapping returns
stitched fiber geometry, not a closed printer route or a watertight solid.

## Install

```bash
python -m pip install "fibernet==4.4.6"
python -m fibernet.examples.basic_usage --output-dir fibernet_demo
```

Python 3.9 remains the core package minimum. Solvers and core dependencies
are unchanged by this patch.
