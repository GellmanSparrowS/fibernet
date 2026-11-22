# FiberNet 4.10.4

## Prepare an isolated Taichi request from a tiled graph

Native tiled graphs store geometric boundary-face flags as Python or NumPy
booleans. The isolated Taichi request accepts both and writes the same six
JSON booleans in the existing format. These flags describe geometric faces,
not solver supports; set the motion constraints and forces separately.
Invalid values such as integer flags still fail before the worker starts.

The provider-free tests cover unmodified planar honeycomb and spatial cubic
request preparation. The spatial case checks the request, not a 3D native
simulation. The worker still needs the qualified Taichi 1.7.4 CPU runtime
on Linux x86-64 with CPython 3.12. The solver is unchanged.

Taichi is not required in the caller to read an existing result through
`SimResult.load` or the task's `result()`. See
[the worker setup and result-reading guide](ISOLATED_TAICHI_CPU.md).
