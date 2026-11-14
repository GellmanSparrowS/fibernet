# Replay the saved reduced recruitment example

From a FiberNet checkout, run:

```bash
python -m examples.reduced_recruitment_workflow --output-dir demo_stretch
```

The example writes `trajectory.npz` and the existing `summary.json`. New
archives have scalar `trajectory_schema_version=1`. All seven original keys
(`frames_xy`, `edges`, `edge_strain`, `force_curve`, `stretch_levels`,
`active_edges`, `spanning_edges`) retain their meaning, shapes and dtypes.
Existing readers selecting those fields continue to work. Older archives
without the version field do not contain enough information for the replay
below; regenerate them rather than guessing grip sets or analysis defaults.

## Recompute recruitment from saved data

This uses the public analysis API without regenerating or rerunning the graph.
All analysis settings are explicit, including hysteresis and grip pruning.
The JSON fields are Unicode scalars, not Python object arrays. Keep
`allow_pickle=False` when reading them.

```python
import json
import numpy as np
from fibernet.analysis import analyze_tensile_recruitment

with np.load("demo_stretch/trajectory.npz", allow_pickle=False) as data:
    if data["trajectory_schema_version"].item() != 1:
        raise ValueError("unsupported reduced recruitment trajectory schema")
    settings = json.loads(data["recruitment_config_json"].item())
    recruitment = analyze_tensile_recruitment(
        data["edge_strain"], data["edges"],
        data["left_nodes"], data["right_nodes"],
        n_nodes=len(data["node_ids"]), **settings,
    )
    np.testing.assert_array_equal(recruitment.active_edges, data["active_edges"])
    np.testing.assert_array_equal(recruitment.edge_in_spanning, data["spanning_edges"])
    np.testing.assert_array_equal(recruitment.threshold_curve, data["threshold_curve"])
    assert recruitment.perc_frame == data["first_spanning_frame"].item()
```

`edges`, `left_nodes` and `right_nodes` contain dense node **row indices**,
not graph IDs. `node_ids[row]` gives the original graph node ID;
`edge_ids[column]` gives the original edge ID for the strain and recruitment
columns. Parallel fibers remain separate columns. The archive also stores
`spanning_frac`, `backbone_frac` and `edge_depth_norm`, so the complete
recruitment result can be compared without hidden state.

## Rerun the native solver from the original graph

`graph_json` is the native `StructureGraph.to_dict()` schema, with its edge
records placed in saved solver order. This preserves noncontiguous IDs,
parallel fibers, edge attributes, weld metadata and the exact original
coordinates. It avoids using the float32 first output frame as a substitute
for the reference geometry. The native loader preserves that record order.

```python
import json
import numpy as np
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim import ReducedBeamConfig, ReducedBeamSolver

with np.load("demo_stretch/trajectory.npz", allow_pickle=False) as data:
    if data["trajectory_schema_version"].item() != 1:
        raise ValueError("unsupported reduced recruitment trajectory schema")
    graph = StructureGraph.from_dict(json.loads(data["graph_json"].item()))
    config = ReducedBeamConfig(**json.loads(data["solver_config_json"].item()))
    rerun = ReducedBeamSolver(graph, config).run()
    np.testing.assert_array_equal(rerun.metadata["node_ids"], data["node_ids"])
    np.testing.assert_array_equal(rerun.metadata["edge_ids"], data["edge_ids"])
    np.testing.assert_allclose(rerun.force_curve, data["force_curve"])
```

`solver_config_json` and `graph_config_json` record every configuration field
used by this example. The stored native graph is the authority for a rerun;
the generator configuration records how it was made. `result_metadata_json`
preserves the solver's sampling/work definitions and diagnostic metadata.
`edge_rest` preserves the recorded reference lengths. `energy_axial`,
`energy_bend`, `energy_contact`, `energy_kinetic` and `energy_work` retain the
solver's framewise energy arrays and their original float64 precision.

The workflow disables contact and does not export velocities or contact
history. This is a post-processing archive and an input for a **fresh run**,
not a checkpoint for resuming integration. Each file is replaced atomically;
the NPZ and JSON summary are not a two-file transaction. Keep the same
FiberNet source and numerical environment for reproducibility checks; this
format does not guarantee identical numerical results across solver versions
or platforms. It introduces no new root-level or MCP API.

The model remains a reduced spring/bending approximation. Its reaction and
energy values are model quantities, not calibrated experimental predictions,
and a saved trajectory alone does not establish equilibrium or convergence.
Recruitment measures thresholded positive axial strain, not complete force
flow. See [the reduced solver method](METHODS_REDUCED_SOLVER_UNIFIED.md) and
[the recruitment method](METHODS_TENSILE_RECRUITMENT.md).
