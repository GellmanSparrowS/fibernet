# Legacy flow boundaries and orientation plotting

## Endpoint orientation plot (available)

`fibernet.viz.plot2d.plot_orientation_distribution(network, save_path=None,
figsize=(6, 6))` accepts `FiberNetwork`, not `StructureGraph`, and returns a
Matplotlib Figure. It plots the **XY projection of the endpoint chord axis**:

- 2D and 3D inputs both use XY; this is not a spherical 3D orientation plot.
- A curved fiber contributes one endpoint direction, not its local tangents.
- Every fiber has equal weight; length and centerline sampling do not add weight.
- The 36 bins span `[0, pi)` radians. Reversing endpoints preserves the axis.
  The XY vector is put into a canonical hemisphere before computing its angle,
  preventing opposite directions from rounding onto different sides of a bin edge.
- Bars are centered on the bin centers and have width `pi/36`.
- Heights are probability density in inverse radians: height times width sums
  to one. Heights alone do not sum to one.

The plot rejects malformed/nonfinite centerlines, nonfinite computed chords or
norms, and undefined endpoint or XY-projected directions. In particular, a
closed curve or vertical Z-directed fiber is not fabricated as x-aligned.
The `1e-12` threshold follows `Fiber.direction` in the supplied coordinate
units; this function does not infer or convert units. A rejected fiber is
identified by its list index; it is not silently omitted from normalization.
An empty network returns a polar figure titled `No fibers`, and `save_path`
works for that figure too. Matplotlib is required.

```python
from fibernet import Fiber, FiberNetwork
from fibernet.viz.plot2d import plot_orientation_distribution
import matplotlib.pyplot as plt

network = FiberNetwork([
    Fiber([[0, 0, 0], [1, 0, 0]]),
    Fiber([[0, 0, 0], [0, 1, 0]]),
], dimension=2)
figure = plot_orientation_distribution(network, save_path="orientation.png")
plt.close(figure)
```

## Legacy configuration execution (not restored)

`run_from_config` rejects these requests before network generation, material
lookup, solver imports, or output:

| Configuration request | Status and migration boundary |
| --- | --- |
| `simulation.type: mechanical` | The old `FiberFEM` path is unavailable. `examples/open_source_workflow.py` demonstrates explicit `StructureGraph` small-deformation linear FEM; it is not an equivalent model or an automatic configuration conversion. |
| `analysis.morphology: true` | The old report is unavailable. `FiberNetwork.total_length` and `mean_fiber_length` provide individual arc-length quantities only. `analyze_topology` and `analyze_geometry_paths` require `StructureGraph`; neither accepts a `FiberNetwork` as an interchangeable object or reproduces the old report. |
| `analysis.stress_strain: true` | The old implicit extraction path is unavailable. A replacement requires an explicit solver, load and result definition. |

All requested obsolete mechanical/analysis fields are reported together.
False/absent analysis flags are not rejected. Existing thermal rejection keeps
its prior priority and message; the explicit transient thermal example is not
an effective-conductivity or CTE replacement. Configuration parsing, schema
validation, serialization and hashing are unchanged. Passing schema validation
does not mean every historical execution mode is supported.

This preflight replaces late import failures with actionable errors; it does
**not** restore the retired models. It does not silently drop requested analysis,
substitute a solver, convert representations, or alias legacy feature vectors
onto the unrelated 94-feature extractor. `api.analyze` remains migration work.
For the separate counts-only `ml.FeatureExtractor` boundary, unsupported legacy
groups and the explicit GNN projection, see
[Legacy ML feature boundary](LEGACY_ML_FEATURE_BOUNDARY.md). This does not restore
the configuration analysis modes above.
