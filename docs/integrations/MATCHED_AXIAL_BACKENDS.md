# Compare an axial network across LAMMPS and Taichi

Use `examples/matched_axial_backends.py` to run the same SI spring network through
FiberNet's native LAMMPS and Taichi entry points. The example saves unscaled
positions, velocities and energies, then compares their displacement histories
with an independent reference. The original cases retain every output step; the
optional 3D case retains every native LAMMPS step and compares both backends at
129 common physical times.

## Shared physical model

The braced planar network has 25 nodes, 56 straight members and five nodes fixed
along its left edge. Each right-edge node receives a constant x-directed force of
100 nN. Every node has mass 1 µg (1e-9 kg) and every member has physical axial stiffness
0.25 N/m. The simulation lasts 1 ms, starts at rest and uses zero damping.
Coordinates are metres, time seconds, mass kilograms, force newtons and energy
joules. The nodes have three spatial coordinates and the motion remains planar.

Both solvers use the potential `U = k (length - rest_length)^2 / 2`. LAMMPS takes
physical `stiffness_N_per_m=k`, compiling its harmonic coefficient as `k/2`.
Taichi takes the strain-based coefficient `spring_k=k*rest_length`, so its force
`spring_k*(length/rest_length-1)` is identical. This per-member conversion also
handles the longer diagonal members. Explicit nodal masses, whole-node supports,
constant forces and geometric rest lengths are shared. The graph's default
material and radius are provenance; these two explicit spring models obtain their
mechanical parameters from the supplied spring coefficients and nodal masses.

The second case is one supported spring with the same stiffness, mass and load.
Its analytical displacement is `F/k*(1-cos(sqrt(k/m)*t))`. The network reference
integrates the nonlinear central-force equations independently with SciPy DOP853.
The three integration grids use 256, 512 and 1024 steps over each fixed duration.

## Run

Choose a CPython/platform combination from the [Taichi installation requirements](../getting_started.rst). The Taichi pin applies to both `accel` and `full`; LAMMPS runtime qualification remains a separate step.

From a repository checkout, install FiberNet with its Taichi extra:

```bash
python -m pip install '.[accel]'
python examples/matched_axial_backends.py
```

The default command prepares and validates both LAMMPS models without launching
either simulation. Native execution requires an existing, explicitly qualified
LAMMPS runtime inventory as described in
[LAMMPS harmonic CLI](LAMMPS_HARMONIC_CLI.md). The example accepts the same JSON
fields as `RuntimeSpec`; provisioning and checking the engine remain separate.

```bash
python examples/matched_axial_backends.py --execute \
  --runtime reviewed_runtime.json --output new_comparison
```

`new_comparison` must not already exist. Each case retains the original graph,
native LAMMPS files and `states.npz`; `comparison.json` records model identities,
file hashes, displacement error and mechanical energy balance. The caller's
runtime inventory is retained alongside the results. Failed runs leave their
completed files available for inspection. A fresh directory starts a new run.

## Read the results

The solvers integrate the same force law with different time integrators:
LAMMPS uses velocity Verlet and Taichi uses kick-then-drift symplectic Euler.
Compare physical times and displacement errors as the time step decreases.
The normalization is the peak reference displacement, rather than absolute
coordinates containing the undeformed sample size.

For a constant dead load, external work is exactly `sum(F * (x - x0))`; with zero
initial kinetic and elastic energy, compare this with `kinetic + potential`.
This quantity measures the integration energy residual. Every saved state also
retains the raw native energy channels for independent recalculation.

In a Linux x86_64 CPU run using LAMMPS 22 Jul 2025 Update 4, Taichi 1.7.4 and
FiberNet 4.4.3, the 25-node network's peak reference displacement was 3.36 µm.
The maximum all-node, all-time position errors were:

| Steps | LAMMPS (nm) | Taichi (nm) | Between backends (nm) |
| ---: | ---: | ---: | ---: |
| 256 | 0.991 | 22.7 | 22.1 |
| 512 | 0.247 | 11.2 | 11.1 |
| 1024 | 0.0618 | 5.57 | 5.53 |

At 1024 steps, maximum energy residual divided by peak external work was
`3.75e-6` for LAMMPS and `6.04e-4` for Taichi; fixed-node displacement was zero
in both. The position errors converge at approximately second and first order,
respectively. Choose the time step against the quantities and accuracy needed
for a study. This example compares undamped axial spring dynamics; rotational
BPM and beam bending have their own model parameters and validation examples.


## Optional 120-node 3D case

The same script also builds a deterministic 6 × 5 × 4 grid with 120 nodes and
573 physical axial bonds: nearest neighbors, three face-diagonal families and
body diagonals. The 1 mm spacing gives a 5 × 4 × 3 mm sample. Twenty nodes on the
left face are fixed in all three directions; twenty right-face nodes each carry
100 nN in x. Mass, physical axial stiffness, zero initial velocity, zero damping,
SI units and the 1 ms physical duration are exactly those used above. The same
central-force reference equations and diagnostics are reused.

There are two ways to supply FiberNet and Taichi to the example interpreter:

- **Source environment:** install `'.[accel]'` from this checkout and use its
  `examples/matched_axial_backends.py`.
- **Installed environment:** install `'fibernet[accel]==4.4.4'` from PyPI and run
  the script from this checkout by its absolute path, from any results directory.
  Top-level repository examples are separate from the wheel's `fibernet.examples`
  package; the script adds no source directory to `sys.path`.

In both cases the native LAMMPS worker can use a separate, already-qualified
interpreter/runtime. Pass your own reviewed `RuntimeSpec` JSON using `--runtime`:
its explicit executable/package paths, exact banner, binary/library hashes and
permitted environment entries are checked unchanged. Follow
[the existing native qualification contract](LAMMPS_HARMONIC_CLI.md#explicit-runtime-and-lifecycle).
There are no example-specific automatic downloads, runtime discovery or fixed
machine paths. The demonstrated native route is Linux x86_64; see the Taichi
platform/Python requirements linked above before choosing the example environment.

```bash
# Preparation only: no provider or runtime inventory is needed.
python examples/matched_axial_backends.py --case braced3d

# One actual resolution, starting conservatively in a new output directory.
python examples/matched_axial_backends.py --case braced3d --steps 256 --execute \
  --runtime /absolute/path/to/reviewed_runtime.json --output new_3d_256

# Three resolutions over the same 1 ms, using the same caller-reviewed runtime.
python examples/matched_axial_backends.py --case braced3d --execute \
  --runtime /absolute/path/to/reviewed_runtime.json --output new_3d_convergence
```

When running outside the checkout, replace the script path by its absolute path.
`--steps` accepts distinct values from 256, 512 and 1024; without it, all three
run. Omitting `--case braced3d` preserves the original spring and planar cases.

The 3D case uses the existing `harmonic_sampled_v1` profile. Integration steps
are 3.90625, 1.953125 and 0.9765625 µs. Native trajectories contain 257, 513 and
1025 frames, respectively; strides 2, 4 and 8 select the same 129 times for both
backends. Every native frame's stable node IDs, step and time are checked before
selection. `states.npz` also retains node IDs, edge IDs, fixed rows and physical
inputs, so row indices are never mistaken for persistent identities. Energies
are selected from the corresponding native thermo steps. The existing profile
forecasts enforce the unchanged 32 MiB output/file and 2 GiB memory budgets
before launch. At 1024 steps, the conservative aggregate native forecast was
31,494,245 bytes; actual bytes are recorded separately in `comparison.json`.

A completed Linux x86_64 CPU validation on the 4.4.4 source tree, with the
previously qualified LAMMPS 22 Jul 2025 Update 4 and Taichi 1.7.4, produced:

| Steps | LAMMPS vs reference (nm) | Taichi vs reference (nm) | Between backends (nm) | Native trajectory bytes |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1.308 | 23.84 | 23.54 | 5,965,611 |
| 512 | 0.3264 | 11.84 | 11.77 | 11,948,018 |
| 1024 | 0.08157 | 5.902 | 5.883 | 23,900,880 |

The reference peak displacement was 4.54 µm. Cross-backend error at the finest
step was 0.130% of this peak. LAMMPS and Taichi position errors converged at
approximately second and first order; their finest normalized energy-work
residuals were 2.94e-6 and 5.33e-4, with zero fixed-node motion. Selected NPZ
files were about 1.39 MB each; their exact byte counts and SHA256 are recorded
per run. Timing fields distinguish the native task (startup, integration,
every-step I/O and parsing), the Taichi call (initialization/JIT, integration and
state extraction), and the independent reference. They are not a pure-engine
speed ranking, and fresh runs can differ in timings and serialized file sizes.

This is a bounded undamped axial-dynamics comparison, with geometric spring
extension. It does not test bending, contact, rotational BPM, fracture or
experimental material calibration, and is not a 2501-node dual-engine result.
Ordinary tests cover deterministic geometry, profile bounds, runtime arguments
and output alignment without launching either optional provider. Native runs
require the explicit `--execute` command; no new CI workflow is added.
