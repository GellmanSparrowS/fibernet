# BeamFrameFEM recovery correction: dependency and impact boundary

Source audit: 2026-10-16, current working tree. This is a call-path audit and one
bounded downstream smoke, not a regeneration or validation of historical results.
All existing result files and checkpoints were preserved.

## Corrected producer and directly affected fields

`fibernet/ml/beam_frame_fem.py::BeamFrameFEM` owns the corrected Hermite endpoint
curvature recovery. `solve_2d` calls `_compute_element_stress_2d` after solving;
`solve_2d_nonlinear` calls `solve_2d` at increments and recovers final fields from
accumulated displacement on original geometry. `solve_3d` uses
`_hermite_endpoint_curvatures` in its two local bending planes, with the local
slope convention `v'=theta_z`, `w'=-theta_y` and endpoint moment-vector norms.

The recovery correction directly concerns 2D endpoint moments, recovered shear,
bending/total stress, and derived node stress/moment summaries; 3D bending/total
stress and derived node stress also depend on the corrected recovery. It is a
post-solve correction: do not infer changed stiffness, solved displacement,
axial force, or clamp reaction merely from changing these recovery expressions.
Nonlinear recovery remains subject to its separate availability/approximation
contract; this audit does not establish finite-rotation objectivity.

## Implemented routes into this producer

- `fibernet/ml/__init__.py` lazily exports this `BeamFrameFEM`.
  `fibernet/ml/beam_frame_fem_v6.py::BeamFrameFEM_v6` is an alias to the same class.
- `fibernet/__init__.py` exports `easy.simulate`.
  `easy.simulate(backend='fem')` → `stretch_test` → 2D linear/nonlinear or
  `solve_3d` → `to_sim_result`. The adapter preserves stress fields in metadata
  and the recovered resultant array as `beam_resultants`. Its public
  `edge_forces`/`max_force` use axial force only; its energy is an axial-stress
  proxy. Those scalar labels are not automatically changed by bending recovery.
- `easy.batch_simulate` explicitly selects the FEM backend, exporting force,
  displacement, and energy to CSV; `easy.train_model` can consume that CSV.
  This is a real indirect learned-label route, but the current exporter does
  not export bending/total stress labels. Training itself reads CSV rather than
  invoking FEM. A model trained on separately exported stress labels would need
  label-provenance review; no such generic claim is justified for every model.
- `scripts/fem_showcase.py::run_fem_stretch` → `stretch_test` and consumes
  `sigma_bending`/`sigma_total` for reported stress and plots.
- `scripts/deformation_test/run_large_deformation_fem.py::run_fem_2d` calls
  `solve_2d_nonlinear` or `solve_2d`; `run_fem_3d` calls `solve_3d`. Stress
  exports/plots from these paths should be treated as recovery-version-dependent.
- Direct source callers also exist in `benchmarks/phase6_v6_validation.py`,
  `phase6_fix_and_visualize.py`, `phase6_comprehensive_fem_test.py`, and
  `phase6b_proper_fem_test.py`. These were inspected, not rerun. Historical
  `phase5_beam_fem.py` references the class but also imports a missing module-level
  `deduplicate_edges` and uses an unsupported constructor `dim` argument; a
  historical import reference is not evidence of a presently executable path.

## Pruning, recruitment, fracture, and learning distinctions

### Real FEM connection, narrower than stress-driven pruning

`benchmarks/independent_fem_cycle_intervention.py::IndependentFemCycleCheck._measure`
executes `BeamFrameFEM.solve_2d` with grips matched to a reduced-solver run. Its
`run` method reuses cycle deletions chosen in a constrained-cycle reference.
The returned FEM metrics are right/left clamp reactions and balance, not recovered
stress or a stress threshold. Thus recovery executes here, but the correction
alone is not evidence that stored reaction rankings or chosen deletions changed.

`benchmarks/recruitment_fem_validation.py::FemRecruitmentValidation.one` calls
`easy.simulate(backend='fem', nonlinear=True)` and then computes recruitment from
final edge-length strain. It reports axial `max_force`; it does not threshold
recovered bending stress. This is a source-confirmed FEM path, not runtime-covered
by this audit.

### Separate reduced-model paths

- `benchmarks/percolation_pruning_pilot.py::PruningPilot.simulate` uses
  `fslab.engine2.Engine2`. Matched-pruning and percolation-cycle studies inherit
  that path. `ConstrainedCycleIntervention._simulate` explicitly uses
  `ReducedBeamSolver`.
- `benchmarks/ai_cycle_selector.py::AICycleSelector` uses
  `ConstrainedCycleIntervention` for baseline/candidate simulation and labels.
  Its learned cycle selection is a reduced-model workflow, not a
  `BeamFrameFEM` stress-label workflow.
- `fibernet/ml/physical_dataset.py` generates physical labels in `_sample` with
  `ReducedBeamSolver`; `physical_learning.py` trains/selects from supplied labels.
- APP compatibility `FiberScope/fslab/engine2.py` imports the shared
  `fibernet.sim.reduced_beam` engine (where `ReducedBeamSolver = Engine2`).
  `fslab/simcache.run_stretch` executes it; `fslab/mlmodel.make_sample` does too.
  APP `dataset_stream` adapts the shared dataset engine to that sample function.
  `web_demo/app.py` uses these APP simulation/learning adapters. No corrected
  BeamFrameFEM recovery dependency was found along these routes.
- The separate `fibernet/ml/dataset.py::FiberNetDataset.generate` uses
  `TaichiEngine.stretch_test`, not this beam solver. Generic dataset containers,
  regressors, or arbitrary externally supplied labels cannot establish solver
  provenance by themselves.

### Separate Timoshenko fracture implementation

`fibernet/sim/fracture/_engine.py::equilibrium` →
`_sparse.assemble_sparse(model='timoshenko')` →
`_beam.element_stiffness`/`local_stiffness`. `_sparse.numeric_modules` imports the
local `_beam`, not `ml.beam_frame_fem`. `equilibrium` obtains twelve local end
resultants from each element stiffness times displacement, then passes them to
`_engine.stress_envelope`; `advance` uses that criterion for irreversible removal.
The envelope combines axial/bending normal and transverse/torsional shear terms.
It does not call `_compute_element_stress_2d` or `BeamFrameFEM.solve_3d`.
`fracture/PROVENANCE.json` records its extracted Timoshenko source lineage.

Therefore this recovery correction does **not** establish an error in all
historical fracture results, fracture event order, or fracture-derived labels.
Those require their own solver/provenance audit. Separate does not mean validated.
The fracture package itself declares an experimental, uncalibrated linear-static
stress-removal model, not finite-contact fragment dynamics.

`fibernet/ml/beam_frame_fem_sparse.py::SparseBeamFrameFEM` is also a separate
implementation, not a subclass/alias. Its benchmarks are not automatically fixed
by changing `BeamFrameFEM`. Likewise `api.simulate_mechanics` names the separate
`FiberFEM`/`NonlinearFEM` routes; a generic “FEM” label is insufficient attribution.

## Runtime evidence and limits

Executed exactly one successful downstream numerical case with
`../fibernet-test-venv/bin/python`, `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, and `PYTHONDONTWRITEBYTECODE=1`:

- `IndependentFemCycleCheck(stretch=1.01, max_nodes=100, max_edges=150)`
- `pattern_2d(unit='square', box=(1.,1.), grid=(1,1))`
- `case._measure(graph)`: 4 nodes, 4 edges; in-memory tracing recorded exactly
  one `_compute_element_stress_2d` call over all 4 elements
- FEM right reaction: `628318.5307179593`; left: `-628318.5307179592`
- Relative reaction imbalance: `1.8528073920390474e-16`
- Reduced raw reaction: `2356.19189453125`
- Maximum absolute bending stress: `0.0` for this axial square case

This confirms the real pruning-comparison call path and its current equilibrium
check, not nonzero bending accuracy, agreement between constitutive models,
3D correctness, fracture events, learned-model quality, or a full deletion study.
A first harness invocation used a positional `pattern_2d` argument and failed
before simulation; retry used the keyword-only API. No production/test code,
dependency, stored scientific output, or checkpoint was modified. All other
routes above are source-only evidence.

## Not implemented or established by this audit

No unified fracture/pruning/learning backend was introduced. No automatic
relabeling, stress-driven BeamFrameFEM removal loop, historical output migration,
or model retraining was performed. Before reusing an existing stress-based
artifact, identify the actual producer and source revision, preserve the original,
and regenerate a separately named result only for the affected call path.
