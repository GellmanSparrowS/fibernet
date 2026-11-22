# Experimental 3D mechanics from a saved electrothermal frame

Use the opt-in `fibernet.sim.saved_electrothermal_3d.solve_saved_electrothermal_frame_3d`
to run one [direct 3D thermoelastic snapshot](THERMOELASTIC_3D.md) at a selected
temperature frame. It does not restart the archive, interpolate temperatures,
solve for new temperatures, or feed deformation back into the source model.
The existing [planar saved workflow](SAVED_ELECTROTHERMAL_MECHANICS.md), root
exports, `simulate` dispatch and MCP tools are unchanged.

Supply the original ordinary 3D `StructureGraph` and a complete local JSON
envelope made with the existing `export_saved_electrothermal` writer. Use a
regular file, not a symlink. The prototype reads at most 4 MiB plus one byte
and rejects files larger than 4 MiB. It accepts up to 64 saved frames,
96 nodes and 192 physical edges; the direct mechanical limit of 384 free
degrees of freedom still applies. Incomplete trajectories are not accepted.
Use stable local files and graphs; concurrent mutation is unsupported.

For example, with your original graph, archive and direct mechanical controls:

```python
from fibernet.sim.saved_electrothermal_3d import solve_saved_electrothermal_frame_3d

run = solve_saved_electrothermal_frame_3d(
    graph,
    "saved_electrothermal.json",
    frame_index=2,                   # Zero-based index of an existing frame.
    length_unit="m",                 # Must match the archive exactly.
    stress_free_temperature_K=300.0, # Supply your own physical reference.
    expansion_alpha_per_K_by_edge_id=expansion_by_stable_edge_id,
    constraints_by_node_id=supports_by_stable_node_id,
    applicability_limits=your_six_kinematic_limits,
)
snapshot, receipt = run
```

Choose the expansion coefficients, reference temperature and restraints for
your mechanical problem; neither the electrical temperature law nor the
source electrical/thermal boundary conditions supply mechanical data. The
graph's explicit isotropic materials are used for the mechanical solve,
but their capture in the archive is not independent material validation.
The validator checks all recorded frames and the exact graph record
(including IDs, incidence, geometry, radii, materials and metadata) before
solving with a detached graph. See the direct API for units, constraint
components, outputs and the six applicability ceilings.

An optional `expected_file_sha256` from a trusted source checks the **file
bytes** before parsing. The receipt separately retains the byte digest, the
internal content and graph digests, the selected frame and exact temperatures
by stable node ID, and the source input identities. The reported native input
digest is not reconstructed. The result and receipt are immutable tuple-backed
records; they define no new persistent format. Hash checks establish consistency,
not author identity or authenticity, and the mechanical result does not
qualify stability, failure or weld stress.
