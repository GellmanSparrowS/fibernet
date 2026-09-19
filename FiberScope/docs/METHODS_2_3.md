# FiberScope 2.3 — implementation and validation methods

This document supersedes the learning, inverse-design and surface-demo sections
of METHODS_2_2.md. Mechanical forces, integration, loading, feature definitions,
unit construction and periodic transforms retain their 2.2 implementations.

## Physically labelled supervised learning

The learning workflow generates a structure and immediately computes its three
labels using Engine2: peak force, an early-loading linear slope, and integrated
force–strain area. The default dataset contains 60 samples, deformation amplitude
0.20, node perturbation 0.10 and a deterministic candidate seed stream. Four
acquisition rules are supported: random, maximum minimum feature distance,
five-member bootstrap ridge committee variance, and equally weighted normalized
committee variance plus diversity. Before eight physical observations, committee
rules fall back to diversity. Predicted values select candidates but never
replace physical training labels. The desktop and assistant expose only this
physical-label dataset workflow. Historical low-level unlabelled/prediction
dataset formats remain readable by the storage API.

Fourteen contact-free features are standardized using training rows only. Labels
are likewise standardized from training rows. The default estimator is the
original NumPy network: one 32-unit tanh layer, linear three-output layer, Adam
optimizer. The desktop defaults remain learning rate 0.003, at most 200 epochs,
patience 15, and minimum validation improvement 0.0001. Best validation weights
are restored. Its seeded split follows the original initialization/split order.
Other estimators are a three-layer 64-unit sklearn MLP by default, ridge regression,
distance-weighted nearest neighbors, random forest, and extra trees. The last
two grow in increments of ten trees to the requested total (default 80, depth 8);
their final estimator retains all requested trees. Ridge and nearest neighbors
perform one fit. Progress axes distinguish epochs, trees and a completed fit.

Twenty percent of rows form the held-out set (12 at the default size); the
sklearn adapters reserve at least two. Prediction scatter, R² and identity line
use these rows only. Training and validation MSE curves use standardized units;
scatter plots use physical label units and reset their viewport on completion
and target change. A constant target has undefined R². These numerical tests do
not establish out-of-distribution predictive accuracy.

## Shared deformation search and reinforcement learning

The inverse page independently selects any registered topology, initially square.
Grid, periodic rule and initial spectrum are copied from the structure page;
changing that page does not overwrite the inverse page's topology selection.
The job captures its factory and seed before starting. Both CEM and RL use the
same tangent/normal reference-line offsets and a node-perturbation coordinate.
The maximum offset is user-adjustable from 0.05 to 1.00 times the reference edge
length, initially 0.60. Existing initial geometry is evaluated first.

CEM initially draws 80% of its first population uniformly from the normalized
parameter box and 20% near the initial spectrum with standard deviation 0.85.
Later populations retain at least one global draw (approximately 25%); remaining
draws use the shrinking elite distribution. Candidate offsets are bounded; node
perturbation is clipped to [0, 0.5]. The default budget is 40 actual mechanical
evaluations using the pre-existing fast inverse configuration: 6,000 steps,
60 increments, target stretch 2.0. Curve-shape objectives and signed scalar
mechanical objectives share the same evaluator.

PPO, A2C, DQN, SAC, TD3 and DDPG use the actual Stable-Baselines3 implementations
in an optional external Python process. Default deformation step is 0.65 in
normalized parameter coordinates. DQN changes one coordinate; the others use
continuous actions. Policies retain two 32-unit layers, one CPU thread and
bounded replay buffers. Every mechanical evaluation writes an atomic candidate
array snapshot before publishing its JSON progress record. The parent loads
that snapshot, forwards it to the Qt thread, then deletes the transient file.
Candidate geometry is displayed on every evaluation; the objective curve tracks
the best candidate and the final view restores the best structure.

Ordinary starts create new job folders. Explicit resume loads the most recent
matching factory/algorithm/parameter/seed/amplitude/protocol checkpoint. Increasing
the total budget resumes additional evaluations. Atomic arrays, policy/replay
files and job metadata are retained. This is a warm restart, not a bit-identical
continuation of random trajectories. Demonstration budgets are not evidence of
policy convergence or superiority over CEM.

## Real demo geometry and OBJ conversion

Original lung, heart and shoe assets are preserved. New default demos derive
from Khronos glTF Sample Assets: FlightHelmet, SheenChair and WaterBottle, each
marked CC0 by its upstream README. Original geometry buffers and source metadata
are downloaded by a resumable preparation script; no texture download is needed.
A 64-cell-resolution voxel union envelope retains a continuous display shell
before quadric edge-collapse reduction to 700 triangles. Shared edge midpoints
and face centers then split each triangle into three conforming quadrilaterals,
yielding 2,100 quads per demo. These are approximated display surfaces, not exact
CAD reconstructions. Source URLs, source hashes, output hashes, counts and
modification method are recorded in assets/obj/real_models.json; upstream license
records are distributed alongside the executable.

OBJ import accepts triangles, quadrilaterals and bounded polygon arity, including
negative indices and inline comments. Zero, repeated and out-of-range face
indices are rejected. Non-quad meshes become conforming corner quads with shared
midpoints. Dense all-triangle imports are reduced toward a 1,500-quad display
budget; original files are never overwritten. Reduction uses bounded spatial
seam welding when disconnected fine details block edge collapse. Import limits
are 64 MB, 100,000 vertices and 200,000 faces in budgeted mode.

Each real cell is mapped to a bilinear patch at 106% coverage and connected to
neighbors through explicit seam anchors. The supporting square scaffold is not
part of the output. Mapping is capped at 150,000 points and 200,000 segments.
Complex cells trigger a coarser display patch mesh; if tiny disconnected parts
prevent reduction, components covering at least 95% of surface area are retained.
The UI reports this adaptive reduction. Unfit meshes raise a bounded error.
This geometry pipeline is not a surface mechanics or garment-fit solver.

## Reproducible verification

`scripts/validate_orthogonal.py` executes a full factorial regression rather than
sampling a few favorable model runs. Supervised coverage is 6 estimators ×
4 acquisitions × 3 topologies (square, triangle, reentrant), with 60 physical
samples, default model parameters, all three target scatters, held-out row
separation, explicit viewport disturbance/recovery and forest-size checks.
Inverse coverage is 7 searches × 3 topologies × 2 objectives (J, max_peak),
40 actual physical evaluations each, verifying 40 geometry updates, more than
five distinct candidates, final best objective and unlocked GUI controls.

`tests/selftest_regression23.py` additionally checks every declared model/search
parameter endpoint, all eleven topologies with actual mechanics, fresh-versus-
resumed external jobs and invalid imports. RL endpoint tests use a bounded
analytic objective to isolate parameter API behavior; physical RL evaluation is
covered by the factorial matrix. `scripts/validate_surface23.py` checks all eleven
topologies on all three new assets plus triangular imports and actual widgets.
Atomic per-case JSON reports permit continuation and identify relevant source
hashes. These checks establish software behavior within the stated coverage;
they do not exhaust all continuous parameter combinations or validate scientific
generalization.
