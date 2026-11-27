# Optional Rustworkx topology analysis

FiberNet's existing NetworkX backend stays the default. To request Rustworkx
for topology on the same `StructureGraph`, install from the matching checkout:

```bash
python -m pip install ".[graph-fast]"
```

```python
from fibernet.analysis import analyze_topology

report = analyze_topology(graph, backend="rustworkx")
```

This is an optional CPU path for Python 3.10+. The supported binary test matrix
is Windows x64, Linux x64, and Apple Silicon and Intel macOS, using Python 3.12
for the release-independent wheel test. Python 3.9 and installations without
Rustworkx can still use the default. An explicit request for an unavailable
backend raises an installation error instead of silently changing backends.

Only the topology algorithm changes. The same result schema reports physical
edge IDs, parallel fibers, connected components, bridges, articulation nodes
and degrees. There is no geometry conversion, no new graph representation
for the user to maintain, and no change to beam or solid mechanics.
[Upstream package and license](https://pypi.org/project/rustworkx/0.18.1/).
