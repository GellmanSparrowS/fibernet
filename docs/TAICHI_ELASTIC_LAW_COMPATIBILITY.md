# Native axial dynamics elastic-law compatibility correction

The dynamics elastic kick now uses the existing compute_forces ideal axial law:
F_i = k (r-L0)/L0 (x_j-x_i)/r, with its existing small-distance branch.
For separated endpoints this equals minus the position gradient of
U = sum k (r-L0)^2/(2 L0). Parallel physical members contribute separately.
k is a force-like coefficient. Unit nodal mass is the default; explicit masses
are described in the [mass/state contract](NATIVE_AXIAL_MASS_STATE.md).

Previously dynamics used r+1e-6 in its elastic direction and strain. That central
force can have its own radial potential: it is not inherently nonconservative.
It instead disagreed with the documented/reported ideal potential and shifted
the unloaded equilibrium length. Removing that shift is an intentional
trajectory compatibility change. Old trajectories are not promised identical.

The production step is a velocity kick, exponential drag, position drift, then
constraint clamping, not Verlet. It does not provide contact, adhesion,
tension-only behavior or prestress. The dashpot softened direction is preserved
separately and uses a common old-velocity snapshot; see [dashpot staging](TAICHI_DASHPOT_STAGING.md).
The legacy damping argument remains metadata-only; dashpot and drag are active.

The small-distance branch is copied from the actual existing force kernel. The
nominal threshold is 1e-12; Taichi literal precision can place a distance exactly
1e-12 on the nonzero side. Tests compare both native kernels there and use
well-separated values below/above for zero/nonzero controls. No global smooth
potential claim includes coincidence or the cutoff. _element_data also floors
reference lengths at 1e-12; degenerate reference members are outside the
separated, positive-length potential checks.

Verification should compare the actual velocity kick with compute_forces under
zero dashpot, drag, external load and constraints. Relevant cases include rest,
deformation/compression, member orientation, parallel members, coordinate
rescaling, energy gradients, rigid rotation, force/torque balance and the cutoff
branch. A force-law check alone does not establish integration accuracy, physical
energy accounting, contact/capture behavior or large-network performance.

Source identity: accelerated.py is automatically covered by batch and dataset
checkpoint source recipes; its changed SHA256 invalidates those exact-source
bindings. In-process Taichi kernels require a
fresh interpreter after a source upgrade; hot-swapping already compiled code is unsupported. Follow the
[lifecycle contract](architecture/taichi-lifecycle-audit.md) for ownership and
cleanup requirements.

## Process completion is part of reliability

Shutdown crashes have occurred in standalone output probes even after arrays
were written. Saved numerical output alone does not establish successful process
completion. Require an ordinary clean exit and bounded cleanup, and do not treat
force-law verification as whole-process reliability. Process reuse and teardown
must be qualified for the runtime being used; no cause or universal workaround
is established by numerical parity alone.
