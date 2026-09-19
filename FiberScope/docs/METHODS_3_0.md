# FiberScope 3.0 methods and validation

## Shared parameter perturbation
For n interpolation positions, one seed produces a shared n-by-2 basis B sampled uniformly from [-0.25, 0.25]. The amplitude p gives D = round(clip(p B, -1, 1), 3). The default n=5 exposes ten displacement values. Every repeated unit receives the same values. Manual edits replace the basis after division by the current nonzero amplitude; subsequent slider movements scale that edited basis. A new seed selects a new basis. Randomization at zero amplitude activates 25% so the change is visible. The resolved-spectrum flag prevents these visible values from being perturbed a second time during geometry construction.

## Ring discretization
A ring uses four quarter arcs and four radial connectors. With n interior samples, quarter-arc coordinates are c + cos(t pi/2)(a-c) + sin(t pi/2)(b-c), t=k/(n+1). Endpoint displacement interpolation and tangential/normal displacement are added to this reference. At n=5, a single undeformed cell contains 48 unique geometric coordinates; independently identified return fibers retain 88 logical nodes and 96 logical edges. Coincident coordinates are not evidence that logical path edges may be deleted. Euler route identity and closure remain checked.

## Manufacturing and interface
The selected planar source uses a 2D preview; a curved source uses a 3D preview. Source selection and route playback share one row, and export actions share one row. Mesh generation remains an isolated, cancellable worker. The ring change reduces redundant centerline samples without changing the tube cross-section chord-error setting (0.0192147 mm at the tested diameter). This geometric tolerance is not a printer accuracy guarantee.

## Recording and provenance
The explicit finals-recording command is parsed for supported unit aliases, sample count, inverse evaluation budget, target, stretch ratio, fiber diameter and print size. This deterministic entry is not a general natural-language planner. Each actual workflow stage produces an expanded tool card with arguments and the observed application state/result. Input events are blocked during execution without disabling pages, allowing actual programmatic navigation. Solid generation, export and slicer handoff remain on the manufacturing page. Errors stop execution and preserve completed artifacts. Dataset configuration includes the numerical engine source hash to prevent stale geometry labels from being reused.

## Validation
22 regression suites passed. A separate real square-network run generated 300 physical labels and performed 200 J-target inverse evaluations; seven pages and eleven tool results were checked. Its STL was independently imported by the installed Bambu Studio. No physical print was started. Ring pyramid and shirt cases completed in 25.19 and 32.98 seconds on this machine, compared with 43.91 and 57.34 seconds in the previous release; these are observed runs, not a controlled performance guarantee. Their meshes contained 1,430,490 and 1,707,532 triangles, passed independent watertightness/winding checks, and fit the 250 mm envelope. Evidence is in docs/validation.
