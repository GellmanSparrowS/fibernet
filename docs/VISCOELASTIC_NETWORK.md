# Linear viscoelastic fiber networks

FiberNet's `LinearViscoelasticMaterial` and existing `BeamFrameFEM.history`
provide time-dependent **2-D, small-strain, straight-member welded-frame**
response, including axial deformation and bending. The unified entry point is
`simulate(graph, backend="fem", mode="history", ...)`.

```python
import numpy as np
import fibernet as fn

g = fn.StructureGraph(dimension=2)
left = g.add_node([0, 0])
right = g.add_node([10, 0])
g.add_edge(left, right, radius=0.1)
material = fn.LinearViscoelasticMaterial.standard_linear_solid(
    equilibrium_modulus=1e6, branch_modulus=2e6, relaxation_time=0.2)
t = np.linspace(0, 2, 201)
extension = 1e-5 * np.minimum(t / 0.2, 1)  # m; a finite ramp then hold
constraints = {
    left: [[0, 0, 0] for _ in t],
    right: [[x, None, None] for x in extension],
}
r = fn.simulate(g, backend="fem", mode="history", material=material,
                times=t, constraints=constraints, length_unit="mm")
r.save("relaxation.json")
restored = fn.ViscoelasticResult.load("relaxation.json")
```

The example uses synthetic coefficients, not a calibration for any named fiber
material. `examples/viscoelastic_network.py` provides relaxation, force-driven
creep/release and cyclic mixed-control workflows on the existing triangular
network generator, including non-contiguous IDs.

### Sweep cyclic dissipation on one network

From a source checkout or extracted source distribution, with FiberNet
installed, run [the frequency sweep](../examples/viscoelastic_frequency_sweep.py):

```sh
python examples/viscoelastic_frequency_sweep.py
```

The same welded triangular network runs ten zero-start cycles at each of
`omega_tau = 0.2, 1, 5` (`frequency_Hz = omega_tau / (2*pi*tau)`), with no
applied nodal forces. Left grips are clamped. Right grips move sinusoidally
in x, and the top right grip also moves in y at one fifth of the x
amplitude. The synthetic standard linear solid uses a 1 MPa equilibrium
spring and a 2 MPa branch. Use `--relaxation-time-s` to set the branch time
(default 0.2 s; accepted range 10⁻⁶ to 10⁶ s) and `--amplitude-um` for the x
amplitude (default 3 µm; accepted range 0.001 to 10 µm). On this millimetre
fixture, this spans nanometre to micrometre grip motion while avoiding
amplitudes whose work can underflow. These are example limits, not model limits.

The JSON report separates final-cycle `physical_loss_J` from numerical
loss. Check that `previous_physical_loss_J` is close before interpreting
the final cycle. Compare physical loss to `trapezoid_loop_work_J` from the
actual grip reactions. The solver's endpoint work includes the change in
stored energy and both losses. For this proportional drive, the single
relaxation time gives the largest loss of the three cases at `omega_tau = 1`.
Use `--quick` for 32 instead of 128 samples per cycle to see the trend with
more time-discretization error. This synthetic small-strain example does
not fit a material or test fracture.

### Change placement without changing the material inventory

From a source checkout or extracted source distribution with FiberNet installed, run [the placement example](../examples/viscoelastic_material_placement.py):

```sh
python examples/viscoelastic_material_placement.py
```

The 16-node welded network has 33 members. Both layouts assign the longer
relaxation time to six horizontal and five vertical 4 mm members of radius
0.04 mm, so they match in count, length, orientation and cylindrical section
volume. The rest use the shorter relaxation time. The two synthetic standard
linear solids share the same moduli (1 MPa equilibrium and 2 MPa branch); the
relaxation times are 0.20 and 0.035 s. Within each of the two eligible
orientations, the stable edge IDs are selected from opposite ends of a
uniform-elastic strain-energy ranking under the same prescribed stroke.

The example continues a state across eight cycles at each frequency, then
compares the final-cycle material loss. The independent check is a
steady-state complex-modulus solution for the same linear Euler–Bernoulli
frame. At the default 256 steps per cycle, the higher-reference-energy placement
dissipates about 29.18 vs 26.38 pJ at 0.80 Hz, and about 38.47 vs 41.35 pJ at
4.55 Hz. Compare the final-cycle loss across layouts at each frequency, and keep the
grip-loop work and backward-Euler numerical loss separate. Use
`--samples-per-cycle 512` to check refinement or 64 for a shorter run.
The example demonstrates a reversal for this synthetic fixture and loading;
the constitutive parameters are not a calibration.

## Installation and compatibility

The material/history API uses the base NumPy and SciPy dependencies. It does
not require Torch or launch Taichi, LAMMPS or another native dynamics engine.
Python 3.9 and newer remain the package target; supported-version runtime
coverage is provided by CI, not inferred from a single local interpreter.

## One material family

- `maxwell(modulus, viscosity)`: spring and dashpot in series
- `kelvin_voigt(modulus, viscosity)`: parallel spring/dashpot
- `standard_linear_solid(equilibrium_modulus, branch_modulus, relaxation_time)`
- `LinearViscoelasticMaterial(E_inf, ((E1, tau1), (E2, tau2)), eta_parallel)`:
  multiple Maxwell branches, equilibrium spring and optional parallel dashpot

Moduli are Pa, relaxation times seconds and viscosities Pa s. The assignment is
explicit and homogeneous across edges; radius-dependent section properties
are retained. The solver does not infer relaxation parameters from graph
`Material.youngs_modulus`, and does not change those stored static properties.
The instance's static FEM E is also superseded by this explicit law.

### Experimental: explicit laws for different members

For an in-memory FEM history, `material` can instead be a complete dictionary
from stable **edge IDs** to `LinearViscoelasticMaterial` instances. Each graph
edge, including each parallel member, needs an entry. Key order does not matter.
Use the same graph and assignment when continuing. This is a Python workflow,
not a saved format or an extension of other backends.

```python
import fibernet as fn

g = fn.StructureGraph(dimension=2)
left, joint, right = [g.add_node([x, 0]) for x in (0, 8, 20)]  # mm
a = g.add_edge(left, joint, radius=0.1)
b = g.add_edge(joint, right, radius=0.12)
laws = {
    a: fn.LinearViscoelasticMaterial.standard_linear_solid(1e6, 4e6, 0.08),
    b: fn.LinearViscoelasticMaterial.kelvin_voigt(2e6, 1e4),
}

def run(times, stroke, **kwargs):
    controls = {left: [[0, 0, 0]] * len(times),
                right: [[x, None, None] for x in stroke]}  # displacement in m
    return fn.simulate(g, backend="fem", mode="history", material=laws,
                       times=times, constraints=controls, length_unit="mm", **kwargs)

ramp = run([0, 0.1], [0, 2e-6])
hold = run([0.1, 0.25, 0.8], [2e-6] * 3, initial_state=ramp.final_state)
print(hold.displacements[:, hold.node_ids.index(joint), 0])
```

The joint can move during the hold even though the end displacement stays
fixed. The initial samples of the continuation repeat the final committed
time and controls. Mapped runs do not support `result.save`, `save_path` or
`ViscoelasticResult.load` as a way to save or continue that state. The existing
single-law save/load workflow is unchanged. Each mapped call accepts up to
256 edges, 513 time samples, 20,000 edge–sample combinations, and eight
Maxwell branches per law. These are experimental limits, not physical ones.

This is material memory, not numerical regularization, global drag, contact
dashpots, adhesion or fluid viscosity. A Kelvin-Voigt material does not have
Maxwell stress relaxation: after a finite ramp its dashpot contribution drops
when the strain rate becomes zero. An ideal strain jump would require an
impulse and is not silently approximated by a special initialization.

## Histories, units and initialization

`times` contains strictly increasing endpoint samples in seconds, at least two.
`times[0]` is an existing state, not the first advanced endpoint. Fresh runs
start at zero load/displacement and zero material memory. The first values of
every prescribed/load history must therefore be zero. Specify a finite ramp
explicitly. No automatic interpolation, load-step jump, or prestress is used.

`constraints` and `loads` map stable node IDs to `(len(times), 3)` samples.
Time, load and constraint samples must be complete. NumPy masked samples
raise `ValueError`; arrays with no masked samples retain their numeric meaning.
Constraints are `[ux, uy, rz]` in m/m/radians; `None` is free, `0` constrained.
The free/constrained pattern is constant in a run and its restart. Loads are
`[Fx, Fy, Mz]` in N/N/N m. Omitted loads are zero; omitted constraints are free.
Loads on prescribed DOFs are allowed and are included in reaction/work
accounting. Every disconnected component must have independent restraints for
its three rigid modes. Singular systems are not regularized into a result.

`length_unit` is required: `m`, `mm`, or `um`. It describes graph coordinates
and radii only. Outputs, controls and times are always SI. Very short members
below 1e-12 m, curved/backtracking members and nonplanar/3-D geometry fail.
Subdivide a curved fiber into explicit straight graph edges before this call.
Small-strain and small-rotation assumptions remain the caller's applicability
responsibility; the solver does not validate a finite-deformation model.

## Results and sign conventions

`node_ids` and `edge_ids` declare array order, independently of insertion order.
`displacements` has `[ux,uy,rz]`; `reactions` has `[Fx,Fy,Mz]`. Reactions are
internal minus externally applied loads; free-DOF values are residuals.
Moment residuals are divided by the maximum member length before the reported
force-scaled `free_residual` is calculated. This length is recorded in metadata.

`section_strain` contains axial strain plus curvature at the two Gauss points
xi=(1 +/- 1/sqrt(3))/2. Linear curvature variation is retained exactly for the
cubic Hermite beam. `axial_force` is positive in tension. `bending_moment`
contains Gauss-point M=-I*sigma_kappa, matching the existing BeamFrameFEM
curvature sign; it is not an endpoint maximum. Generalized curvature stress is
Pa/m, while curvature is 1/m. No extra graph or constitutive parallel-edge
deduplication is introduced.

## Energy and numerical interpretation

Backward Euler updates each branch's viscous strain/curvature and supplies a
consistent algorithmic tangent. The section metric is `(A L,I L/2,I L/2)`.
Results retain stored energy, cumulative physical viscous dissipation,
cumulative **numerical** backward-Euler dissipation, cumulative endpoint work,
and their balance residual, all in joules. Inertial energy is absent because
the history solves quasistatic equilibrium at each endpoint.

Endpoint work uses current external loads and constraint reactions times the
displacement increment. It is a declared discrete work convention, not an
exact continuous-time boundary-work integral. The identity is
`work = stored - initial_stored + physical_loss + numerical_loss`.
Material loss and backward-Euler loss must not be conflated. Time refinement
is necessary for physical accuracy; a balanced ledger alone is insufficient.
This strain-state implementation explicitly requires each branch factor
`tau/(tau+dt) >= 1e-6` to avoid cancellation of tiny retained branch extension.
Outside that numerical domain it reports the actual dt and asks for interval
subdivision. It does not silently discard a fast branch or change the law.

## Restart and persistence

Pass `initial_state=previous.final_state`, start new `times` at the exact last
time, and repeat the last control/load samples as the new initial samples.
The checkpoint records displacement, both bending histories, branch strains,
last stress and strain rate, units, controls, cumulative ledgers and exact
material/geometry/topology identity. Incompatible units, material, support mask
or graph are refused. Checkpoints are integrity-checked, not authenticated.

`final_state` and result attribute accesses return detached data. Failed solves
do not mutate the graph, input histories or supplied restart state. For a single law, result
`save` uses atomic replacement; `load` checks the result schema and numerical
shapes. JSON preserves float64 roundtrips without deliberate rounding.

## Applicability and reference grounding

No claim is made here for large rotation/strain, fracture, nonlinear fibers,
contact, temperature shifting, 3-D beams, or cross-backend equivalence.

Reference architecture: [MOOSE GeneralizedMaxwellModel, 2024-11-11](https://mooseframework.inl.gov/releases/moose/2024-11-11/source/materials/GeneralizedMaxwellModel.html).
Analytical benchmark background: [COMET-FEniCS linear viscoelasticity](https://comet-fenics.readthedocs.io/en/latest/demo/viscoelasticity/linear_viscoelasticity.html).
These equations are independently implemented; external engine code is not
copied or bundled. The included regression tests cover finite-ramp relaxation, creep, cyclic
loss, axial and bending response, energy accounting and serialized restart.
Those synthetic checks do not calibrate a specimen or validate behavior
outside the stated physical and numerical limits.
