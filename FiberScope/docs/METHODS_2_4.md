# FiberScope 2.4 implementation methods

This document describes changes relative to [2.3](METHODS_2_3.md). The mechanical integrator, feature definitions and physical labelling protocol are unchanged.

## Periodic cell authoring

A cell is a connected finite graph whose vertex coordinates are expressed relative to a unit repeat period. The interval [0,1] on each axis defines the period, rather than a clipping boundary. Vertices outside that square are retained during persistence and construction. Existing graph validation (2–48 vertices, finite coordinates, legal edges and connectivity) remains active; coordinates above an absolute magnitude of 10^6 are rejected as numerically unreasonable. Translated copies use the existing geometric merge implementation. Drawing outside the period alone does not guarantee connections: corresponding geometry must meet. The workbench displays the resulting repeated structure before saving.

The canvas supports continuous drawing, wheel zoom, middle-button pan and fit-to-content. Built-in expansion rules and the periodic orientation editor are accessed through a compact dialog inside the cell workbench. Rule editing uses a copied configuration and applies on acceptance.

## Physical and surrogate inverse evaluation

Physical evaluation remains the default. The initial UI selects the square topology, normalized C-shaped force curve, CEM search and 100 objective evaluations. The candidate topology remains independently selectable. The existing broad initial spectrum exploration and amplitude control are retained.

An optional model button snapshots the currently trained supervised model from the learning tab. The initial candidate topology is set to that model's dataset topology. The snapshot predicts the same three physical-label quantities used during training: peak force, initial stiffness and toughness. Given a graph G, its existing 14-component descriptor x(G) is supplied to the trained predictor. The scalar minimization objective is s times prediction component k, where s is -1 for maximization and +1 for minimization. C-shaped or other full-curve objectives are rejected in this mode because the model does not predict a force trajectory.

Model evaluation returns a `PredictedStructure` containing positions, edges and three predictions. It never returns synthetic simulation frames or strain fields. The UI displays candidate geometry, the best candidate and predicted scalar values. These values are model estimates, not newly obtained physical labels. The model is not trained further during inverse search.

CEM and the six existing SB3 methods share this evaluator option. External workers receive an atomic, SHA-256-checked snapshot of the application's own model. Checkpoint identity includes the model hash, target, topology, deformation settings and algorithm settings, so model and physical jobs cannot resume one another. The private pickle snapshot is an internal process transport, not a public untrusted-model import interface.

## Long-run resource control

Positive iteration counts have no application-level maximum. Stop requests remain supported. Numerical and geometry budgets remain active independently of the requested evaluation count. Optimizer records and live GUI traces retain the most recent 2,000 evaluations; the cumulative evaluation counter and best structure are preserved. CSV exports therefore contain the retained window. RL result snapshots are atomically replaced and include cumulative evaluation counts. Removing the input cap does not imply infinite resources or guarantee convergence.

## Surface display and attribution

The default OBJ selector contains the original lung, heart and shoe. The three newly added 2.3 assets are removed from the distribution. Triangle-to-quad import and geometry reduction remain available without rewriting the input file. The spectrum amplitude control now spans 0–500%; the same mapping and explicit seam connections are used. Large amplitudes can produce self-intersections and are intended for exploration, not a claim of physical manufacturability.

Author attribution is stored as `AUTHOR_SIGNATURE` and in the Windows executable copyright resource. It is accessible by right-clicking the version chip or pressing Ctrl+Shift+I, and is absent from the main page content.

## Verification

`python scripts/run_tests.py` includes `gui_refinement24`: physical-labelled hexagonal samples, all six supervised models driving CEM, actual updates through all six RL methods in prediction mode, asynchronous UI execution and mode switching, drawing outside the unit square, saved custom geometry, rule dialog, hidden attribution, sample navigation and large-amplitude mapping of the three original meshes. A lightweight deterministic evaluator runs 10,005 times to check removal of the old cap and retention of 2,000 records. This is a functional regression test, not a benchmark of model accuracy or optimization convergence. Frozen smoke additionally verifies learned inverse evaluation using bundled supervised dependencies.
