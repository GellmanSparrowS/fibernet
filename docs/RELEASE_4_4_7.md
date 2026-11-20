# FiberNet 4.4.7

This patch corrects connectivity in the legacy Gmsh geometry export
(`fibernet.io.to_gmsh` and `fibernet.api.export` for `.msh`). The old
exporter could join points that fell in the same eight-decimal rounding bin.
That could turn nearby rods into overlapping elements or shorten a rod to
zero length, depending on the scale and position of the coordinates.

The exporter now retains the original fiber endpoints, writes float64
coordinates with round-trip precision, and shares mesh nodes only for
exactly coincident samples. If resampling produces a zero-length line,
it raises before writing the destination.

## Migration

Re-export meshes from the original networks if they have small gaps or
short fibers; check their connectivity before using them in a simulation.
If a workflow relied on snapping nearby samples, make intended shared
coordinates explicit in the source geometry. This exporter does not find
crossings or transfer crosslinks, stable IDs, or material properties.
Package requirements and public function signatures are unchanged.

## Install

```bash
python -m pip install "fibernet==4.4.7"
python -m fibernet.examples.basic_usage --output-dir fibernet_demo
```
