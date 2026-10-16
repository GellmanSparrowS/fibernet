# Steady electrical conductance on a fiber graph

Implemented in `fibernet/sim/linear_transport.py` with existing NumPy/SciPy only.
Direct module import; no base dependency or public top-level API changes.

## Exact model

An edge is an ideal resistor with caller-supplied constant conductance in S.
Nodes are ideal equipotential junctions. Prescribed node voltages are in V.
There are no distributed current sources, transient storage, reactions,
nonlinear constitutive laws, electrical contact resistance or mechanical
coupling. Geometry, radius and material properties do **not** determine
conductance. This is not a pore reconstruction or permeability model.

Positive current follows the stored edge `node_i → node_j` orientation:
`I = g * (V_i - V_j)`. Node injection is the signed sum of outward edge
currents, positive when current is supplied into the network. At free nodes
it must vanish; at prescribed nodes it is a valid boundary reaction.
Power per edge is `g * (V_i - V_j)^2` in W.

## Example

```python
from fibernet import StructureGraph
from fibernet.sim.linear_transport import solve_linear_transport

g = StructureGraph(dimension=3)
a = g.add_node([0, 0, 0])
b = g.add_node([1, 0, 0])
c = g.add_node([2, 0, 0])
e0 = g.add_edge(a, b)
e1 = g.add_edge(b, c)
result = solve_linear_transport(
    g, {e0: 2.0, e1: 6.0}, [(a, 2.0), (c, 0.0)],
    length_unit='m',
    model_metadata={'conductance_source': 'synthetic series circuit'},
)
# result.voltage_V = [2, 0.5, 0] V
# result.current_A = [3, 3] A; dissipated_power_W = 6 W
```

`conductance_S` must map every stable edge ID exactly once. Unknown/missing
IDs, booleans, fractional IDs, nonfinite values and negative conductance are
rejected. Zero conductance is allowed: that edge is inactive, with exactly
zero current/power, but retains its output row and identity.

Dirichlet inputs may be an iterable of `(node_id, voltage)` pairs or a mapping.
Pairs allow conflicting duplicates to be detected before loss of information;
a Python mapping has already collapsed duplicate keys before the call.
Same-valued duplicate pairs coalesce; conflicting values fail. Boundary values
are retained exactly, with no ramp or unit conversion.

## Connected components and no-path cases

Every component of the **positive-conductance** graph needs at least one
prescribed voltage. This includes isolated nodes and nodes joined only by
zero-conductance edges. Otherwise `FloatingComponentError` lists unanchored
node IDs. No implicit ground, arbitrary potential, hidden diagonal
regularization or component deletion is used.

Multiple disconnected but independently anchored components are valid. A
no-path network between two prescribed terminals can therefore return zero
current legitimately. One anchor in a connected component gives a constant
potential if no other driving boundary exists. A graph with no edges is valid
only when every node is explicitly anchored. An empty node set is rejected.

Parallel members remain separate current/power rows; their conductances add
in the nodal Laplacian. Coincident coordinates do not merge nodes. Internal
curves and periodic metadata are not remeshed or converted into new electrical
connections: only the explicit graph endpoints and supplied conductances are
used. The solver makes no electrical continuity inference from crossings.

## Result and provenance

`LinearTransportResult` is separate from mechanical SimResult. Arrays are
copied and read-only. Node rows follow sorted stable node IDs; edge rows follow
graph insertion order. It carries endpoint IDs, supplied conductance, voltage,
current, node injection, per-edge power, requested voltages and conservation
checks. `to_dict()` returns a detached JSON-ready representation.

Provenance states the model, units, assumptions, caller metadata, SciPy version,
solver selection, tolerances and a SHA-256 binding of IDs/connectivity,
conductance, requested voltages, coordinate values and declared length unit.
The hash is a reproducibility binding, not authentication, and does not claim
to hash the full mechanical project. Result dictionaries are detached but are
not recursively immutable.

Geometric units must be explicitly declared as m/cm/mm/um/nm. Coordinates are
not rescaled and do not enter the conductance law. Changing geometry units
while supplying the same electrical graph/conductances leaves its electrical
solution unchanged. No effective bulk conductivity is inferred from a graph
without an explicit area/length convention.

## Numerical checks and limits

The backend assembles the sparse weighted Laplacian and eliminates prescribed
voltages. `spsolve(use_umfpack=False)` selects the SciPy path without optional
UMFPACK dispatch. Singular-matrix warnings and nonfinite outputs fail.
All-prescribed systems do not need a linear solve.

Default checks are local, so an unrelated high-current branch/component cannot
supply the acceptance scale for a low-current free node:

- Each free-node current residual is checked against its own incident absolute current sum, multiplied by rtol
- A local floating-point allowance adds 32 times float64 machine epsilon times the incident conductance-weighted absolute node/neighbor voltages relative to that component's reference potential
- Each positive-conductance component independently checks signed current balance against its boundary absolute injection sum, and power balance against its own dissipated/supplied power
- Component boundary power is computed after shifting voltage by a component boundary reference, reducing arbitrary-offset cancellation; returned boundary_power_W is the stable sum of these component-local values, not a cancellation-prone global raw voltage/injection dot product
- Defaults: rtol 1e-10; current_atol_A = 0 A; power_atol_W = 0 W

The local floating-point term is a backward-error allowance, not a voltage
error bound. It uses only conductances incident on that node and component-
relative voltages, never a global maximum current. Components with identical
prescribed voltages use their exact constant solution, avoiding spurious
roundoff currents for zero-driving cases.

Absolute tolerances are optional physical quantities. A caller who explicitly
sets a nonzero current/power floor can permit relative errors to dominate very
small signals; these floors must be chosen intentionally. Earlier universal
1e-12 A/W defaults were removed after adversarial review showed that they could
accept completely wrong tiny-scale solutions.

Positive conductance with nonzero computed voltage drop must produce normally
representable, finite float64 current and dissipation. Zero/subnormal driven
current or power raises TransportSolveError with an underflow message; no
invented zero physics or silent unit rescaling is returned. Exact zero driving
and explicitly zero conductance remain valid zero-current/power cases. This
conservative policy excludes subnormal results even when a nonzero float can
be stored, because its relative precision is degraded.

Residual checks do not provide an error bound on voltage for an ill-conditioned
graph. There is no condition-number estimate, performance guarantee,
preconditioner selection or transient solver. Assembly/right-hand-side/scale
overflow fails explicitly; no numerical regularization is hidden. Component-
local acceptance does not guarantee that summing extremely different component
powers into one float preserves the small contribution; per-component values
are retained in provenance for inspection.

## Verified scope

Synthetic series/parallel cases, the saved four-node analytic fixture,
conservation and power balance, sparse IDs, orientation, zero-conductance and
no-path cases, disconnected anchored/floating components, validation failures,
repeatability, nonmutation, unit nonconversion and a generated square network.
Injected nonfinite, singular-warning and nonconserving solver outputs are
rejected. This is a software/analytical validation of the defined linear model,
not calibration or experimental validation of any material.
