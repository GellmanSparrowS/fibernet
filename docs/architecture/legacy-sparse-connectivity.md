# Sparse fiber connectivity

`FiberNetwork.connectivity_matrix()` returns a dense NumPy integer array by
default. Pass `sparse=True` to get a SciPy CSR matrix built directly from
crosslinks, with storage proportional to the fiber and crosslink counts.

```python
from fibernet.gen import ZigZagGenerator
from scipy.sparse.csgraph import connected_components

network = ZigZagGenerator(n_cols=30, n_rows=30).to_fiber_network()
adjacency = network.connectivity_matrix(sparse=True)
component_count, labels = connected_components(adjacency, directed=False)
assert adjacency.shape == (4500, 4500)
assert component_count == 1
```

Rows and columns follow fiber list order. Isolated fibers get zero rows and
columns. Each crosslink record adds one count in each direction. Repeated
records accumulate, and a self-link adds two to its diagonal entry. The CSR
path accepts Python and NumPy integer scalar indices. In-range negative
indices count from the end of the list. It rejects out-of-range indices and
advanced index values such as arrays and booleans.

For the network above, with 11,430 crosslinks, the CSR data, index and row
pointer buffers total about 292 KB on a 64-bit build with 64-bit integer
counts and 32-bit sparse indices. The dense array would require 162 MB with
the same counts. These sizes measure matrix buffers.
