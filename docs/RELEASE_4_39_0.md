# FiberNet 4.39.0

## Prepare 3D beam-network deformation for electrical readout

`prepare_deformed_transport_3d` accepts explicit x/y/z translations tied to
the stable node IDs of an existing `StructureGraph`. It returns the deformed
graph, updated conductances, and a binding report. The DC solve remains a
separate operation with explicit electrode voltages.

```sh
python -m pip install "fibernet==4.39.0"
```

```python
from fibernet.sim.deformed_transport import prepare_deformed_transport_3d
from fibernet.sim.linear_transport import solve_linear_transport

deformed, conductance, binding = prepare_deformed_transport_3d(
    graph, translations_m_by_id=translations_xyz_m,
    reference_conductance_S=reference_G, length_unit="m",
)
result = solve_linear_transport(
    deformed, conductance, electrode_voltages_V,
    length_unit="m", model_metadata=binding,
)
```

The 3D preparation uses actual straight-member chord lengths, fixed effective
area, and the supplied reference conductances. The original planar API and
MCP preparation wire format are unchanged. See the [preparation guide](DEFORMED_TRANSPORT_PREPARATION.md)
for input, identity and output details.

## Reduce repeated numerical overhead

The shared electrical and thermal roundoff checks classify ordinary positive
normal values without unpacking their bit representation. Exceptional inputs,
subnormal handling, and conservative bounds work as before.
