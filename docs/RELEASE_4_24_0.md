# FiberNet 4.24.0

Measure how a three-dimensional network's resistance changes under load and
inspect how different materials contribute to a solid's elastic strain energy.
The 3D resistor example and solid analyses use the existing mechanics and
transport solvers; the release adds no optional or required dependencies.

```bash
python -m pip install "fibernet==4.24.0"
```

## Run the full-network 3D example

The [3D load-to-resistance guide](DEFORMED_TRANSPORT_PREPARATION.md#run-a-complete-3d-beam-network-comparison)
runs a 91-node, 360-member welded network at 0.5% axial strain and measures
the resistance along all three sample axes. It compares fixed cross-section
and fixed member volume on the same computed deformation, with stable
electrode IDs. This is a geometric forward calculation with fixed network
topology, not an intrinsic material piezoresistance model.

## Inspect material energy and compare load controls

Request `return_cell_fields=True` when solving a linear-static Tet4 model,
then call `static_tet4_strain_energy_by_material(result)` to sum the elastic
strain energy by material without re-solving. See the [Tet4 guide](TET4_STATIC.md)
for the API and a fused-junction example comparing force control and prescribed
displacement.

## Compare viscoelastic placements at equal material inventory

The [planar placement example](VISCOELASTIC_NETWORK.md) changes where the two
materials are placed while keeping material amount and member orientations
matched. The ordering of energy dissipation reverses between the two tested
driving frequencies in this example.
