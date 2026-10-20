# Synthetic crosslinked CG network

`examples.cg_crosslinked_network` builds an ordinary `StructureGraph` and calls
FiberNet's optional LAMMPS backend through `simulate`. It is not a separate solver.
The 4 × 4 × 3 scaffold has 48 beads, 104 harmonic bonds, 64 selected same-strand
cosine angles, 12 fixed beads and cycle rank 57. This is an intermediate network
example, not evidence of large-scale, GPU or MPI performance.

## Reference geometry and excitation

`build_case()` returns `(reference_graph, initial_graph, cg_request)`. Stable node,
edge and angle IDs are retained. Coordinates are in metres, with reference spacing
L = 1e-6 m; every bond explicitly retains rest length L even though the initial
coordinates are stretched and bent. Angles use `K*(1+cos(theta))`, whose minimum is
pi; no arbitrary rest-angle inference occurs. At lattice index `(i,j,k)`, the
loaded initial position is:

- x = 1.02 i L
- y = j L
- z = k L + 0.01 i L + 0.03 L [i(3-i)/2] [j(3-j)/2]

The i=0 face is held, initial velocities are zero, and remaining beads undergo
conservative free vibration. This is neither damped relaxation nor a quasistatic
loading test. Mass is 1e-15 kg per bead, harmonic stiffness 2e-6 N/m, and cosine
coefficient 3e-18 J. These are synthetic illustrative parameters, not calibrated
material properties. Graph radii are provenance, not a conversion to bond/angle
coefficients. Bonds preserve connectivity; angles connect only consecutive
collinear reference segments, not every pair of incident edges.

## Use the library backend

From the source root, provide an already configured and pinned `RuntimeSpec` JSON
for the official LAMMPS runtime. This example does not discover or install it.
Importing the module or calling `build_case()` does not launch native code.

```python
from examples.cg_crosslinked_network import run_example

# runtime is an explicit caller-provided RuntimeSpec.
# NEW_OUTPUT must not already exist.
result = run_example(runtime, "NEW_OUTPUT")
```

Both `run_example` and the CLI save both graphs and detailed/summary results,
check their public save/load round trips, and use the same bounded per-native-run
settings (10 CPU seconds, 20 wall seconds, 16 MiB output/file):

```sh
python -m examples.cg_crosslinked_network --runtime-json RUNTIME.json --output NEW_OUTPUT
```

Default: 500 steps at 1e-7 s, saving every step. A matching finer grid uses
`--steps 1000 --timestep-s 5e-8`. `--steps 0` performs a static force audit, not a
third temporal grid. Each invocation is an explicit new native run. Output
folders must be new. Native failures are surfaced by the existing backend/task
lifecycle, not converted into successful observations. These per-example settings
do not change the library's admission limits and are not intrinsic LAMMPS limits.

## Retained scientific observations

`tests/fixtures/lammps_cg_crosslinked_observations.json` contains compact
observations from one retained static audit and two temporal grids with the same
5e-5 s physical duration. Independent scalar energies/forces, all-step velocity-Verlet
consistency, fixed-state/domain checks and raw/public state comparisons passed
predeclared scientific gates. Initial total energy was 4.6435607640217203e-20 J.
Maximum total-energy drifts were 6.256473608178253e-25 J (coarse) and
1.564144265658713e-25 J (fine). Matched-state position difference was
2.3729284356053805e-7 L. This is observed two-grid refinement, not a convergence-
order theorem or validation against physical experiments.

The reference execution exceeded an orchestration CPU limit, so this case is not
a validated whole-workflow resource or performance benchmark. Short-process
sampling did not establish executable-only peak RSS or native cancellation
qualification. A larger scene requires new explicit admission based on measured
cost and output size; existing global caps remain unchanged.

## Result meaning and limits

Public displacement is measured from the loaded initial graph, not the unloaded
reference graph. Compare final positions with `reference_graph` explicitly for
reference-relative displacement. `max_force` is the axial bond contribution;
net nodal forces include bending. Internal force on held beads is opposite to the
support force under this convention, not a general reaction/stress/work API.
Summary exports omit trajectory arrays; use detailed exports for state analysis.

There are no pair forces, self-contact/excluded volume, fracture, BPM, atomistic
force fields, continuum beam EI/torsion calibration, or empirical material claims.
The reference/initial graphs can be passed to ordinary topology and visualization
analysis. Shared stable IDs make result-to-graph comparisons meaningful. No
manufacturing feasibility follows from these dynamics; run the applicable
manufacturing checks separately. MCP graph/topology/artifact interoperability is
separate from native physics execution and does not add an MCP simulation tool.
