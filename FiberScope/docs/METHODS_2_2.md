# FiberScope 2.2 — implementation methods

This document describes the implemented computational methods. It does not
claim equivalence to a finite-element solver or to every algorithm in the
supplied research manuscript. Force and length values use the application's
model units unless separately calibrated. The existing mechanical solver was
retained; the release validation records its numerical regression evidence.

## Cell representation and deformation

A cell is an undirected graph whose stored endpoint order defines local edge
tangents and normals. A shared set of dimensionless tangent/normal offsets is
interpolated along every cell edge, with displacement proportional to that
edge's reference length. Consequently a common parameter vector deforms
different topology families without silently replacing their graphs. The
authoring canvas accepts direct path drawing, node movement and edge editing.
Its preview uses the same factory as subsequent computation.

Periodic orientation rules act about the cell center before translating the
cell by integer multiples of the reference cell size L=10. In addition to
translation, the application supports alternating 180-degree rotations,
alternating column or row reflections, successive quarter turns, checkerboard
reflections and a user-defined orientation table. For the latter, a table entry
specifies a local x reflection followed by 0, 1, 2 or 3 counterclockwise quarter
turns. A table has at most four rows and four columns and repeats periodically.
Coincident graph vertices and duplicate edges follow the existing graph
assembly's deduplication semantics. Connectivity and Eulerian degree checks
are reported as graph properties, not as complete fabrication certification.

## Descriptors and labels

The default feature-analysis interface reports structure and pore descriptors.
Geometric contact descriptors are excluded from this interface, default
exports and the learning schema. Dynamic contact forces remain part of the
mechanical solver, independently of this display/schema change.

The learning input comprises 14 graph descriptors: segment-length mean,
standard deviation and coefficient of variation; axial orientation entropy in
12 equal bins over [0, pi); fractions of degree-2, degree-3, degree-4 and other
nodes; radius of gyration divided by bounding-box diagonal; node and segment
counts divided by 1,000; segments per node; bounding-box aspect ratio; and mean
segment length divided by the bounding-box diagonal. Schema identifier:
`structure14-v2-no-contact`.

Physical learning labels use the existing reduced-order Engine2 with target
stretch 1.8, 1,500 steps and 24 increments. The outputs are peak recorded force,
an early-curve least-squares force–strain slope over 5–40% of the recorded
strain span, and trapezoidal force–strain area. These are computational labels,
not experimental measurements. The inverse-design solver's existing
"stiffness" objective is an early-force increment, not this fitted slope;
they must not be interchanged without an explicit evaluator adapter.

Structure-only generation stores missing labels rather than running a hidden
simulation. Model-predicted datasets are separately marked and cannot enter
the UI's supervised training path. Each dataset contains its exact generation
configuration, next candidate seed, accepted factory specifications and label
provenance. Same-directory atomic replacement prevents a partially written
checkpoint from replacing the last complete dataset.

## Regression and active sampling

Six estimator configurations are available: Ridge, distance-weighted nearest
neighbors, random forest, extra trees, a one-hidden-layer MLP and an MLP with
two to five hidden layers. The implementations use scikit-learn. The MLPs use
the library's default ReLU activation and Adam optimizer. Neural fits expose
learning rate, width, depth where applicable, epoch limit and early-stopping
parameters. Tree models expose tree count and maximum depth; Ridge exposes
regularization strength and nearest-neighbor regression exposes neighbor count.

Rows are split using a seeded permutation, with approximately 20% held out
(at least two rows). Input/output normalization statistics are computed only
from training rows. Neural fits retain the weights with best validation MSE;
the interface reports held-out R-squared and corresponding scatter points.
This validation set is also used for early stopping and is therefore not an
independent final test set. Generalization claims require a separate test set,
including held-out topology families when transfer is claimed.

Active sampling selects one structure from a bounded candidate pool at each
step. Diversity maximizes minimum squared distance from previously selected
features after scaling by pooled feature standard deviations. Committee
sampling fits five bootstrap Ridge regressors to finite physical labels and
selects maximum mean prediction variance across the three normalized targets.
The hybrid strategy adds separately max-normalized disagreement and diversity
scores with equal weight. Before eight finite labels exist, committee-based
strategies use diversity. Selection is followed by physical labeling only when
the user chooses that mode. No benchmark superiority is implied by providing
multiple algorithms.

## Search and reinforcement learning

CEM remains the default. Its initial evaluation uses the current topology,
grid, periodic rule and deformation spectrum. Subsequent candidates include
local perturbations and global exploration. Elites update the distribution
with bounded variance. The user-visible budget bounds mechanical evaluations;
cancelled empty batches do not update a distribution from missing samples.

The optional reinforcement-learning environment represents geometry using the
bounded spectrum/perturbation vector. Continuous actions add bounded changes;
DQN actions increment or decrement one coordinate. The reward is the hyperbolic
tangent of improvement over the previous best objective, normalized by the
larger of its absolute value and 0.001. Episodes reset to the best known state
after 16 transitions. A mechanical objective, not a pretrained prediction,
supplies the current built-in reward. PPO, A2C, DQN, SAC, TD3 and DDPG use
Stable-Baselines3 in a separate CPU process. Training and policy/replay
checkpoint restoration are tested separately from scientific performance.

## Generalized surface mapping

For each quad patch, the actual chosen unit graph is constructed with its
deformation spectrum. Its x and y bounding intervals are normalized to [0,1],
then uniformly expanded about (0.5,0.5) by a default factor 1.06. The resulting
coordinates are mapped through

P(u,v) = (1-u)(1-v)Q0 + u(1-v)Q1 + uvQ2 + (1-u)vQ3.

Coordinates outside [0,1] intentionally extrapolate beyond the patch. The
fiber graph does not include the original support-mesh edges. For each shared
patch edge, a common midpoint anchor is connected to the closest point on a
fiber in each neighboring cell. The closest point is computed in cell
coordinates, its corresponding rendered segment is split, and an explicit
stitch segment joins the split point to the shared anchor. Numerical
coincidences are welded with tolerance proportional to mesh extent. These
stitches change connectivity deliberately; projected crossings alone do not
establish a mechanical junction. Disconnected surface islands or disconnected
custom cell graphs require separate assessment.

Non-quad OBJ polygons are converted to corner quads using shared edge
midpoints and face centers. Source mesh coordinates are retained. A principal
front plane is computed from the smallest-eigenvalue direction of the centered
vertex covariance; a projected world-up direction fixes the displayed basis.
This is a deterministic viewing convention, not semantic recognition of a
garment's front. The user can choose other views without modifying the asset.

The four new garment-like assets are original procedural quad demonstration
surfaces generated by `scripts/generate_demo_meshes.py`. They are not
anatomically fitted patterns or validated wearable products.

## Reproducibility and limitations

See `docs/open_source/API.md` for interfaces and storage, and the versioned
release record for executed checks. Random seeds and configuration are retained.
Long RL jobs preserve policies, best states and off-policy buffers; warm
restarts need not reproduce an uninterrupted stochastic trajectory exactly.
Numerical accuracy, material calibration, true fabrication constraints,
surrogate transfer accuracy and RL sample efficiency require separate studies.
