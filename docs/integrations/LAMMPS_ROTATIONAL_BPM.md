# Rotational BPM with the bounded LAMMPS CLI

Use `prepare_rotational_bpm` to prepare a `StructureGraph` with explicit sphere masses, diameters and bond coefficients. Preparation works without LAMMPS. An explicit `LAMMPSTask` uses the pinned one-rank provider. The harmonic backend continues to use `prepare_harmonic_beads`.

## Prepare a model

```python
from fibernet import StructureGraph
from fibernet.integrations import BPMParticle, RotationalBPMBond, RunSpec, prepare_rotational_bpm

g = StructureGraph(dimension=3)
a = g.add_node([0, 0, 0])
b = g.add_node([20, 0, 0])
edge = g.add_edge(a, b)

prepared = prepare_rotational_bpm(
    g, length_unit='um', maximum_bond_length_m=60e-6,
    particles={n: BPMParticle(mass_kg=1e-12, diameter_m=4e-6) for n in (a, b)},
    bonds={edge: RotationalBPMBond(
        radial_stiffness_N_per_m=2., shear_stiffness_N_per_m=.5,
        twist_stiffness_N_m_per_rad=1e-14, bend_stiffness_N_m_per_rad=4e-14,
        radial_limit_N=185e-9, shear_limit_N=1e-3,
        twist_limit_N_m=1e-9, bend_limit_N_m=1e-9,
    )},
    run=RunSpec(1e-8, 32, 1, ((-50e-6, 80e-6),) * 3),
    velocities_m_s={b: (.5, 0., 0.)},
    checkpoint_step=10,
)
print(prepared.manifest()['atom_to_node_id'])
```

`length_unit` applies to graph positions. The box, maximum bond length, coefficients, masses, diameters, timestep, initial linear velocities and optional `angular_velocities_rad_s` use SI. The compiler retains the graph's materials for provenance and uses the explicit records for the dynamics. All spheres start with identity orientations and the four damping coefficients are zero.

The pilot covers 2–32 spheres, 1–64 straight, unique physical bonds and 1–256 integration steps in a fixed nonperiodic box. Bond length while active must stay between 1 µm and the declared maximum. The provider uses `units micro`, `frame average`, `damping derivative`, `normalize no` and `smooth no`. Numerical shear resolution at very small angles remains subject to the native implementation. See the [pinned bond documentation](https://github.com/lammps/lammps/blob/stable_30Sep2026/doc/src/bond_bpm_rotational.rst).

In the pinned average-frame implementation, the breaking index sums the absolute radial force, shear force, twist torque and bend torque, each divided by its own limit. A bond breaks when the sum reaches one, including under sufficiently strong compression. The returned rupture step comes from the native event record. [Pinned implementation](https://github.com/lammps/lammps/blob/stable_30Sep2026/src/BPM/bond_bpm_rotational.cpp#L445-L453)

## Run with a pinned provider

The supported provider is the Linux x86-64 executable from the [official 30Sep2026 static release archive](https://github.com/lammps/lammps/releases/download/stable_30Sep2026/lammps-linux-x86_64-30Sep2026.tar.gz). Verify the archive SHA-256 `cb74093db9d621e3b379e55baa5562eec132342837175066eee806c780f3c263` and executable SHA-256 `599d046dde36bcf7666dacd45a0a6feb2265743058dea5c1e0d5970a57805784`. Use the [serial-static runtime inventory format](LAMMPS_HARMONIC_CLI.md#explicit-serial-static-cli-profile) with `expected_version=20260930`, `expected_banner='LAMMPS (30 Sep 2026)'`, canonical executable paths, and both interpreter and native executable hashes.

```python
from fibernet.integrations import LAMMPSTask, ResourceLimits, TaskState

task = LAMMPSTask(prepared, runtime, new_workdir, ResourceLimits())
try:
    task.start()
    task.wait(timeout=12)
finally:
    if task.state == TaskState.RUNNING:
        task.cancel()
        task.wait(timeout=2)
if task.state != TaskState.SUCCEEDED:
    raise RuntimeError(task.error)
result = task.result
```

`result.frames` contains one SI record per step, with positions, velocities, `quaternions_wxyz`, angular velocities, forces, torques and active edge IDs. Arrays follow `result.node_ids`. `result.breaks` identifies the original edge and endpoints and the native rupture step. When requested, `result.checkpoint` identifies the binary file, step, byte count and hash. The same task writes and consumes the restart at that step. The reported checkpoint-step frame comes from before the restart. At the duplicate native step, stored particle state and active bonds must agree exactly. Recomputed forces and torques can differ by floating-point rounding.

Run the [four-sphere example](../../examples/23_lammps_rotational_bpm.py) from the checkout to prepare its six-bond model. Add `--execute --runtime reviewed-runtime.json --workdir NEW_DIRECTORY --checkpoint-step 20` to run it through the explicit provider.

The portable tests run without an engine. To include the native regressions, set `FIBERNET_TEST_BPM_RUNTIME_JSON` to the reviewed runtime JSON when running `tests/test_lammps_rotational_bpm.py`.
