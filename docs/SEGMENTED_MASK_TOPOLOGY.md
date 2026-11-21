# Compact topology of a segmented 3D mask

From a checkout with FiberNet installed, run
`python examples/segmented_mask_topology.py` for a small built-in example,
or pass your own already segmented Boolean array:

```python
from examples.segmented_mask_topology import mask_component_graph, component_summary

# mask_zyx is your Boolean numpy.ndarray; True means foreground.
graph = mask_component_graph(mask_zyx, voxel_size_um=1.3)
print(component_summary(graph))
graph.save_json("mask-topology.json")  # choose a new path
```

The array must be ordered as `mask[z, y, x]` in **caller-chosen** axes.
Transpose if your array uses a different order. These labels are not
inferred material directions. Voxel centers are at half-index offsets
from index origin. Omit `voxel_size_um` for voxel coordinates; supply a
positive isotropic spacing from 1e−6 to 1e6 µm for calibrated positions. The example
does not decode files, infer calibration, segment images or resample.

Each disjoint 10 × 10 × 10 block is split into components connected
through voxel faces. Each becomes a graph node weighted by its number
of foreground voxels, with one occupied voxel center as its position.
A graph edge aggregates occupied face contacts across adjacent blocks;
corner or edge contact does not count. Edge metadata records the
occupied face-pair counts as `face_witnesses_zyx`. The summary uses voxel counts,
not graph-node counts, and reports which components touch both
opposing array faces along each axis.

Each axis can contain 1 to 100 voxels. Trailing blocks use only the
available voxels; nothing is padded or trimmed. On a singleton axis,
any foreground voxel touches both index faces on that axis. The example
stops beyond 10,000 graph nodes or 50,000 contact edges. It is not an
image importer, fiber identification algorithm, or mechanical solver
input. Graph edges encode contacts, not beam properties. For a fixed
mask, contraction preserves this six-face component connectivity, not
shape, path lengths or quantitative transport or mechanics.
