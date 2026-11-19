# Compare an axial network across LAMMPS and Taichi

Use `examples/matched_axial_backends.py` to run the same SI spring network through
FiberNet's native LAMMPS and Taichi entry points. The example saves unscaled
positions, velocities and energies from both solvers at every integration step,
then compares their displacement histories with an independent reference.

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
