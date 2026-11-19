# FiberNet 4.4.4

This patch follows the published 4.4.3 core package. It improves native graph
fracture input, adds an explicit rotational BPM integration, and corrects model
input handling and animation output.

## Installed library changes

- `python -m fibernet.sim.fracture` accepts native StructureGraph 3.1 JSON with
  `--graph-format structure-graph-3.1`. Original IDs and initial missing-member
  masks are retained. The dedicated fracture schema remains the default.
  Material parameters, length units and grip IDs are explicit inputs.
- `fibernet.integrations.prepare_rotational_bpm` prepares sphere-and-bond models
  for the pinned official LAMMPS 30Sep2026 Linux x86-64 static provider. Typed
  SI output retains stable IDs, orientations, angular velocity, forces, torques,
  rupture steps and task-owned checkpoints. This opt-in pilot supports 2–32
  spheres, 1–64 unique physical bonds and 1–256 integration steps. The external
  executable is supplied separately. See the [provider and model contract](integrations/LAMMPS_ROTATIONAL_BPM.md).
- `PhysicalRegressor` rejects masked arrays and nested masked observations
  before conversion. Rejected refits preserve the current estimator and
  normalization. Feature and three-target label shape errors identify the
  expected and received dimensions before fitting.
- `visualize_deformation` uses each fiber's own cumulative point rows, fixes
  the view over the complete trajectory, and retains requested slow and
  non-divisor GIF frame durations. Playback milliseconds remain independent
  of the saved simulation timestep.

## Run the installed quick start

```bash
python -m pip install "fibernet==4.4.4"
python -m fibernet.examples.basic_usage --output-dir fibernet_demo
```

The quick start generates its own graph and writes the FEM response, figure and
JSON summary. The core dependencies and Python 3.9 minimum are unchanged.
Optional workflow requirements remain documented in their guides.

## Versioned source examples

The top-level `examples/` scripts are supplied with the repository and source
distribution; they are distinct from wheel commands under `fibernet.examples`.
Use a checkout or extracted source distribution matching this version:

- `fracture_scan.py --compare-notch` compares matched geometry and grip inputs
  with distinct initial notch masks and records per-case hashes and statuses.
- `fracture_transport.py` maps saved fracture masks to fixed original-rod
  conductances. It retains pre/post-break states and reports open circuits.
- `matched_axial_backends.py` compares explicit matched axial Taichi/LAMMPS
  models and reference trajectories under shared units and loading.
- `thermal_to_transport_export.py` passes temperature to thermoelastic
  deformation, then to a derived circuit and export. This is sequential
  one-way transfer. VTU retains graph IDs; STL contains geometry, with external
  JSON declarations for units and fiber/edge mapping. Capped cylinders are
  exported individually.
- `23_lammps_rotational_bpm.py` demonstrates the separately supplied pinned
  provider through the opt-in library API.

## Migration

Select complete observed rows before converting masked training data into
ordinary arrays. Keep feature rows, three physical target columns and group
labels aligned. Rejected training calls leave the prior model usable.

For animations, supply one displacement row per concatenated fiber centerline
point. Keep the physical time axis and display magnification with your results;
GIF playback duration is a presentation setting.

The fracture input addition retains the existing small-deformation solver and
the default dedicated schema. The fixed-reference fracture circuit and the
geometry-updated thermal example implement separate one-way workflows; neither
introduces electrical feedback into mechanics.
