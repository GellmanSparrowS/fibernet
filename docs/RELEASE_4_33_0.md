# FiberNet 4.33.0

## Optional faster topology analysis

FiberNet can use Rustworkx to find connected components, bridges and cut
vertices, while returning the same FiberNet result and preserving physical
fiber IDs, including parallel fibers. NetworkX is still the default.

```bash
python -m pip install "fibernet[graph-fast]==4.33.0"
```

```python
from fibernet.analysis.topology_metrics import analyze_topology

metrics = analyze_topology(graph, backend="rustworkx")
```

CPython 3.12 installed-wheel runs exercise the optional provider natively on
Windows x64, Linux x64, Intel macOS and Apple Silicon macOS. The speedup
depends on the graph size and conversion cost; the optional backend does not
change the default dependency set.

## Lower peak memory in prescribed-displacement Tet4 calculations

Release the factorization workspace and temporary reduced matrix after
computing the displacements, before recovering element forces. Solver options,
material inputs, residual checks and output fields are unchanged.

In a one-CPU reference run with a 132,000-tetrahedron solid and five DC
load states, peak process memory dropped from about 472 MiB to 418 MiB.
Reaction and resistance differences were at roundoff scale. This is a
reference benchmark, not a general runtime or memory guarantee.

## Install

```bash
python -m pip install "fibernet==4.33.0"
```
