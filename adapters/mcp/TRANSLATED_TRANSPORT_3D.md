# Steady electrical conduction after a prescribed 3D displacement

Start the matching Linux stdio adapter with `--enable-translated-transport-3d`
in addition to the regular `--manifest` and `--output-root` options. This
registers `fibernet_solve_translated_transport_3d`. Use the `graph_id` from
`fibernet_catalog`, for an operator-admitted 3D `StructureGraph` whose
geometry unit and coordinate frame are explicitly declared.

The tool takes:

- `frame_id` and `length_unit`, matching the catalog graph. Geometry units
  are `m`, `mm`, or `um`. Translations are always in metres.
- `translations_m`: one record `{"node_id": 1, "value_m": [0, 0, 0.0001]}`
  per stable node, with exactly three components in the graph frame.
- `reference_conductance_S`: one strictly positive Siemens value per stable
  edge, e.g. `{"edge_id": 4, "value_S": 0.25}`. Parallel edges stay separate.
- `dirichlet_voltage_V`: one or more fixed voltages, specified as
  `{"node_id": 1, "value_V": 0.0}`. Every electrically connected component,
  including isolated nodes, must have a voltage reference.

A result references one managed JSON artifact, retrievable with the standard
`fibernet_artifact_info` and `fibernet_artifact_read` tools. The artifact
contains the source and control digests, the native preparation binding and
the native DC fields. Oriented current is positive from the first stable
endpoint to the second. A collapsed member, an unanchored component, or a
numerical failure yields a structured error instead of a partial artifact.

The profile uses the current endpoint chord lengths, fixed effective area
and the supplied reference conductances. The caller supplies displacements:
this does not run a mechanical solver, update cross-sections, change contacts
or apply electrical feedback. Limits: 256 nodes, 1,024 edges, 4 MiB source,
256 KiB normalized request and 1 MiB result. The Linux worker shares the
existing timeout, cancellation and artifact lifecycle.
