# FiberNet 4.10.1

## Keep each graph paired with its ID

`graph_batch` now rejects sets, frozensets, and set-like views for the graph
collection, graph IDs, and selected edge features. These inputs are positional:
a set could pair an ID with the wrong graph or change feature column order
between Python processes. Supply corresponding ordered inputs:

```python
from fibernet.ml import graph_batch

batch = graph_batch(
    [graph_a, graph_b],
    graph_ids=["specimen-a", "specimen-b"],
    length_unit="m",
    coordinate_frame="lab",
    edge_features=["radius", "chord_length"],
)
```

The function does not sort these inputs. Ordered iterables, both
`GraphBatch` schemas and stable node and edge ordering are unchanged.
See the [graph data guide](architecture/graph-data-usage.md#identity-and-axes).

## Source examples

- The [fitted Tet4 local-refinement example](TET4_FITTED_PLACEMENT.md#check-local-refinement-without-changing-the-represented-geometry)
  compares two material layouts at the same axial stroke, before and after
  one fixed interior mesh refinement. That is a sensitivity check, not a
  mesh-convergence or strength result.
- The [saved-temperature mechanics example](SAVED_THERMAL_MECHANICS.md#command-line-example)
  leaves the requested output absent if a new solve or its JSON serialization
  fails, so you can correct the input and retry the same path. It still refuses
  overwrite and needs hard-link support to publish the file.

These scripts and the fitted mesh input are in the source distribution, not
installed by the wheel. To inspect the local-refinement plan without solving:

```bash
python -m pip install "fibernet==4.10.1"
python -m pip download --no-deps --no-binary=fibernet "fibernet==4.10.1"
python -m tarfile -e fibernet-4.10.1.tar.gz
cd fibernet-4.10.1
python examples/tet4_static_fitted_nested.py
```

Add `--solve` to run the comparison if admitted by the factor budget.
Neither Gmsh nor Meshio is required.
