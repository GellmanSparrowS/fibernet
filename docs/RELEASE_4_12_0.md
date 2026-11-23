# FiberNet 4.12.0

Install the core Python package:

```bash
python -m pip install "fibernet==4.12.0"
```

## Run a 3D mechanical snapshot from a saved temperature frame

Use the experimental
`fibernet.sim.saved_electrothermal_3d.solve_saved_electrothermal_frame_3d`
to select one frame in a complete local JSON archive made by
`export_saved_electrothermal`. Supply the original 3D graph, including its
materials, and the same geometry length unit used in the archive. Supply the
mechanical supports, per-member thermal expansion coefficients, stress-free
temperature and kinematic limits separately. The function returns the static
beam snapshot and a receipt identifying the graph, file and selected frame.

The archive must be a regular file of at most 4 MiB, with at most 64 frames.
The model is limited to 96 nodes, 192 straight circular members and 384 free
mechanical degrees of freedom. The optional `expected_file_sha256` checks
exact file bytes against a supplied digest; it does not authenticate who
created the file. Use trusted local inputs that are not being modified.

This is a one-way transfer to the existing small-deformation beam solver. It
does not restart a simulation, interpolate between frames, or feed deformation
back into the thermal or electrical model. The existing direct 3D and planar
workflows, MCP tools, and dependencies are unchanged.

See the [saved-frame guide](SAVED_ELECTROTHERMAL_MECHANICS_3D.md) for an example,
units and model limits.
