# Stored materials versus effective solver parameters

Source and bounded-runtime audit, 2026-10-16. Native graph schema 3.1 preserving
Material is an input-persistence guarantee. It does not mean every simulator uses
those values or supports heterogeneous constitutive laws.

| Entry point / implementation | Effective constitutive source | Edge Material behavior |
|---|---|---|
| `easy.simulate(..., backend='fem')` → `ml/beam_frame_fem.py` | Uniform global E/nu by default; explicit `material_mode='graph'` selects per-edge isotropic E/nu/derived G | Graph mode requires `nonlinear=False`, `length_unit='m'`, and rejects explicit global E/nu |
| Direct linear `BeamFrameFEM` 2D/3D | Constructor E/nu by default; explicit `edge_materials` overrides raw rows | Per-edge linear elasticity only, validated before deduplication; no automatic selection |
| `BeamFrameFEM` incremental path | Constructor self.E/self.nu/self.G | Uniform only; no edge-material option |
| `ml/beam_frame_fem_sparse.py::SparseBeamFrameFEM` | Constructor global E/nu | Array-based API, no Material input |
| `sim/accelerated.py::TaichiEngine.dynamics` | Explicit spring_k if supplied, otherwise stiffness | No modulus/density inference; spring_k may be per-edge coefficients |
| `TaichiEngine.stretch_test` | Global stiffness forwarded to dynamics | No Material-derived stiffness |
| `sim/reduced_beam.py::ReducedBeamSolver` | Config stiffness, bending fraction, contact coefficient/radius, integration settings | No Material consumption; discrete model parameters |
| `sim/fracture::FractureSolver` | Explicit homogeneous FractureConfig E/nu and strengths | Checks every edge E/nu/G matches config and tensor is absent; rejects heterogeneous materials; strengths do not inherit Material.tensile_strength |
| `sim/linear_transport.py` | Explicit conductance_S keyed by edge ID | No material/geometry-derived electrical conductance |
| `sim/_fem_archive.py::BeamFEM` | Reads edge Material.youngs_modulus with fallback | Archived implementation, not current public quick FEM; must not be advertised as current heterogeneous support |

`sim/fem.py` no longer exports the archived BeamFEM. Some stale module text says
all simulation is Taichi; that is not a current capability inventory. The matrix
above follows implementations and public dispatch rather than those historical
claims. Separate legacy/archived implementations are outside current validation.

## Executed precedence checks

`tests/test_material_parameter_precedence.py` uses real solves where stated:

- Changing stored edge E from 1234 to 9876 does not alter quick FEM forces when
  explicit E=1000 is unchanged; changing explicit E to 2000 doubles axial force
- Changing stored E leaves reduced-model trajectories and force curves identical
- Real Taichi dynamics under the same explicit spring coefficient yields identical
  force/displacement outputs despite changing stored E
- Fracture initialization rejects a graph/config E mismatch rather than silently
  choosing either value
- A result-recovery invariant verifies force-like spring coefficients: equal strain
  with rest lengths 1 and 2 gives the same force, but energy scales with rest length

The first four cases passed in the optional CPU environment, including an actual
Taichi kernel. This does not validate all heterogeneous material laws, tensor
models or physical calibration. The explicit heterogeneous linear-frame mode
has separate checks described below; other quick mechanical backends retain
their existing material-consumption limits.

## Spring coefficient convention

Confirmed in `TaichiEngine.compute_forces`, dynamics kernel and
`SimResult.compute_detailed`:

- strain = (L − L0) / L0
- axial force F = k × strain
- elastic energy U = 0.5 × k × strain² × L0

Thus k is force-like axial rigidity. With consistent SI geometry/force units it
has units N (as EA does), not N/m. The equivalent conventional extension-based
spring constant is k/L0. For a linear axial bar, a caller can deliberately choose
k=EA after establishing consistent E, area and units; no current shortcut performs
that conversion automatically. Numerical coordinates and coefficient values alone
do not establish SI units, density-calibrated inertia or experimental validity.

Public accelerated-solver docstrings now state this convention. One actual
conflicting tutorial source label, `complete_tutorial_v4.ipynb` cell 20, printed
N/m; it now says model force units and F=k*strain. Only that source label changed;
saved historical notebook outputs were preserved rather than reinterpreted or
regenerated. The accelerated module's non-docstring AST is unchanged.

## Explicit linear-frame graph-material mode

The opt-in linear beam mode is documented in
[Per-edge materials in linear beam FEM](../heterogeneous_fem.rst). It uses the
same StructureGraph, circular sections and welded-joint topology. Effective
E/nu/derived G are captured at solve time and retained with selected input rows
and stable result edge IDs. Density, strengths, anisotropy and interface laws
are not consumed. Graph mode requires the caller to assert metre geometry and
Pa material properties; raw arrays retain caller-controlled consistent units.
Uniform defaults and the explicit uniform viscoelastic history are unchanged.

Independent numerical checks cover unequal series compliance, parallel-member
load partition, 2-D network energy-Hessian and dense-solution oracles, 3-D
bending and mixed-Poisson torsion, plus persistence and source ownership.
The existing axial energy proxy is divided by each captured effective E, not
the current graph material or converter instance. It still excludes bending
and torsion. These checks do not qualify nonlinear constitutive response or
material calibration.

## Gates for other future graph-material modes

Do not silently change precedence. A future opt-in material mode must specify:

1. graph_material versus global_override selection, including conflicts/errors
2. dimensional conversion for coordinates/radii and material SI properties
3. per-edge E, G, density and section support actually consumed by each backend
4. supported isotropic/anisotropic laws, heterogeneous properties and junction rules
5. analytical two-material bars/parallel fibers, unit equivalence and serialization
   round-trips that change the effective solved physics as intended
6. provenance recording effective values, input identities and chosen precedence

Until those gates pass for a particular backend and law, stored materials are
preserved data; global solver inputs remain authoritative where the matrix says
so. Graph JSON containing different E values does not activate a material mode.
