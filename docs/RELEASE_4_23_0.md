# FiberNet 4.23.0

Assign different linear viscoelastic laws to members of one welded planar
frame. Keep the graph and supports fixed while changing which stable edge IDs
receive each material, then inspect edge forces, bending moments, and the
work and energy histories.

```bash
python -m pip install "fibernet==4.23.0"
```

## Compare materials on one welded network

Pass a complete `{edge_id: LinearViscoelasticMaterial(...)}` mapping as
`material` to `fibernet.simulate(graph, backend="fem", mode="history", ...)`.
See the [runnable three-node example](VISCOELASTIC_NETWORK.md) for the full
geometry, units, constraints, and material parameters. It includes a
fixed-displacement hold during which the free joint moves as the two
members relax at different rates.

The mapped mode is experimental and supports small-deformation, planar,
quasistatic frame histories. Limits are 256 edges, 513 time samples,
20,000 edge–sample pairs, and eight Maxwell branches per member material.
Use the detached `final_state` to continue an unchanged mapping in the
same Python process. Use the existing scalar material path for durable
v1 result files. The default dependencies and existing MCP tools are unchanged.

## Source examples

- [Cycle through three frequencies](VISCOELASTIC_NETWORK.md).
- [Export material-mismatch fields from a fused junction](TET4_THERMOELASTIC.md).
- [Reassign solid materials by stable cell ID](TET4_STATIC.md) when reusing the
  same element mesh after row reordering.
