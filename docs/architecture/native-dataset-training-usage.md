# Train directly from a FiberNetDataset

`fibernet.train_model` (the easy sklearn API) accepts either an existing CSV path
or an explicit `FiberNetDataset`. Native calls preserve raw feature and target
names; they do not export/reparse a temporary CSV. The separate Torch training
loop at `fibernet.ml.training.train_model` is unchanged.

## Minimal native call

```python
from fibernet import train_model
from fibernet.ml.dataset import FiberNetDataset

dataset = FiberNetDataset.load("my-dataset")
metrics = train_model(
    dataset,
    target="max_axial_force_N",  # Exact existing raw target name
    model_type="lr",            # lr, rf, or gb
    test_size=.3,
    split="row",               # Explicit row holdout, not origin validation
    save_path="new-model.pkl",
)
```

The first argument retains the historical keyword `csv_path`, so existing
`train_model(csv_path="results.csv", ...)` calls still work. Native calls may
also use `csv_path=dataset`; there is no `dataset=` alias. A directory string is
not automatically loaded as a Dataset. Load it explicitly first.

`features=None` uses `dataset.feature_names` in its stored order. An explicit
nonempty list selects raw names in the requested order. Do not add `feat_` or
`label_`. Target defaults to literal `max_force`; if that name is absent, pass
the real target explicitly. The function never guesses the first target.

Only selected features and the selected target are validated. Missing values,
text, complex numbers, masked values and nonfinite selected numbers fail before
splitting/fitting. Unselected unavailable labels and opaque metadata/params do
not become numerical inputs. Selected features cannot have any declared target
name. Native namespaces, rather than CSV prefixes, identify features/labels.
Declared feature/target lists must contain unique nonempty string names. Sample
IDs must be nonempty strings at this training boundary; duplicate IDs are kept
with their distinct original row indices. Dataset persistence is unchanged.

## Save one sample from a detailed 3D beam solve

Run the [mixed-material example](../../examples/heterogeneous_3d_graph_workflow.py)
from the source archive or checkout:

```sh
python examples/heterogeneous_3d_graph_workflow.py --output solve
```

Use the unmodified graph and detailed result saved by this same run.
The feature rows come from the undeformed graph, not the solved shape.
The four x-aligned beams leaving the fixed end carry signed axial
forces; summing those values gives this example's tensile cut force.
This is not a general reaction or cut-resultant calculation for oblique
or bending-dominated beams.

```python
from hashlib import sha256
from math import fsum
from pathlib import Path

from fibernet.api import load
from fibernet.ml import graph_batch
from fibernet.ml.dataset import DatasetSample, FiberNetDataset
from fibernet.sim import SimResult

g = load("solve/graph.json")
r = SimResult.load("solve/result.json")
b = graph_batch([g], graph_ids=["four-path"], length_unit="m",
                coordinate_frame="specimen_xyz", edge_features=("radius", "chord_length"))
ids, force = r.metadata["edge_ids"], r.edge_forces
if (len(ids) != len(force) or len(ids) != len(set(ids))
        or set(ids) != set(b.edge_ids) or not b.edge_feature_available.all()):
    raise ValueError("expected a complete detailed result for the solve graph")
by_id = dict(zip(ids, force))
cut = [e for e, beam in g.edges.items()
       if (g.nodes[beam.node_i].position[0] == 0) != (g.nodes[beam.node_j].position[0] == 0)]
energy = r.metadata["beam_energy"]
if len(cut) != 4 or energy["status"] != "available" or energy["unit_system"] != "SI":
    raise ValueError("expected four cut beams and available SI beam energy")
features = {"total_chord_m": fsum(b.edge_features[:, 1]),
            "mean_radius_m": float(b.edge_features[:, 0].mean())}
labels = {"cut_force_N": fsum(by_id[e] for e in cut), "strain_energy_J": energy["strain_energy"]}
lineage = {name + "_sha256": sha256(Path(f"solve/{name}.json").read_bytes()).hexdigest()
           for name in ("graph", "result")}
sample = DatasetSample("four-path", "cubic", (2, 1),
                       {"grid_xyz": [2, 1, 1], "length_unit": "m", "x_stretch_ratio": 1.001},
                       features, labels, lineage)
data = FiberNetDataset([sample], feature_names=list(features), target_names=list(labels))
data.save("four-path-dataset")
```

Use a new output path. The digests record which files were used; they
do not prove both files came from one solve. Reloading with
`FiberNetDataset.load("four-path-dataset")` should preserve both feature
and target names. Saving a missing declared target raises an error
rather than assigning zero. This single row checks the data handoff; it
is not a training or holdout set.

## Choose the dependency unit deliberately

Native calls require `split="row"` or `split="group"`. There is no implicit
native row-split default.

- Row mode shuffles source rows with seed 42. Related runs of the same network
  can land in both partitions; unique sample IDs do not prevent that leakage
- Group mode holds out entire caller-declared groups with seed 42. It checks
  labels, row alignment and train/test disjointness; it cannot verify that the
  supplied labels describe genuine independent networks

```python
# origin_ids must come from your reviewed lineage and match dataset.samples.
metrics = train_model(
    dataset, target="max_axial_force_N", model_type="rf",
    split="group", groups=origin_ids, test_size=.25,
)
```

`groups` must be a list/tuple of nonempty strings, one per row, with at least two
distinct labels. Row mode rejects supplied groups rather than ignoring them.
In group mode `test_size` is a fraction of groups, not rows; unequal-size groups
produce a different held-out row fraction. Both native modes require a finite
fraction strictly between 0 and 1, at least one training row and at least two
test rows because the result includes R². Unsplittable cases fail before fit.

Grouping is never inferred from `sample_id`, unit/grid, a params hash, exact
feature equality or a graph fingerprint. In the retained ten-square reference,
unit/grid gives only one group, while IDs and parameter pairs are all distinct.
None provides an automatic independent-origin holdout. Its example deliberately
uses a row split for an API check, not a scientific generalization claim.

Explicit groups do not prevent every form of leakage: target-derived features,
repeated tuning on the same holdout, or incorrectly declared origins remain the
caller's responsibility. Keep the original Dataset and its generation lineage.

### Run a three-role grouped beam example

From a checkout containing the
[grouped beam example](../../examples/heterogeneous_3d_grouped_dataset.py),
use a fresh output directory:

```sh
python -m pip install scikit-learn
python examples/heterogeneous_3d_grouped_dataset.py --output beam-groups
```

The example solves five synthetic cubic networks from three rest-geometry
groups. Each material variant stays with its geometry group. It checks
the axial force and strain energy against a series-spring calculation,
then calls `FiberNetDataset.split(method="group")` and checks that every
row appears once and no source group appears in two roles. The original
graphs and detailed results are saved beside the split datasets. This is
a split and data-lineage check, not a fit or a test of generalization.

## Standardize on training groups

When features have different scales, fit the scaler only on the training rows.
Pass your loaded Dataset, selected raw feature names, target name, and
caller-declared origin labels in sample order. This recipe uses the native
group split's seed 42 and holds out 25% of the groups. It writes into a new
directory and records the fitted rows and column mapping with the model.

```python
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from fibernet import train_model
from fibernet.ml.dataset import FiberNetDataset


def fit_scaled_groups(dataset, raw_names, target, groups, directory):
    raw_names = list(raw_names)
    if set(raw_names) & set(dataset.target_names):
        raise ValueError("Selected inputs must exclude every declared target")
    if not raw_names or not set(raw_names).issubset(dataset.feature_names):
        raise ValueError("Select declared raw Dataset feature names")

    raw = FiberNetDataset(dataset.samples, feature_names=raw_names).features
    test_size = .25
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train, test = next(splitter.split(raw, groups=groups))
    scaler = StandardScaler().fit(raw[train])
    z_names = [f"z_{name}" for name in raw_names]
    samples = [replace(sample, features=dict(zip(z_names, row.tolist())))
               for sample, row in zip(dataset.samples, scaler.transform(raw))]
    scaled = FiberNetDataset(samples, feature_names=z_names,
                             target_names=dataset.target_names)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / "model.pkl"
    metrics = train_model(scaled, target=target, model_type="lr", split="group",
                          groups=groups, test_size=test_size, save_path=model_path)
    split = metrics["training_data"]["split"]
    if (split["train_rows"], split["test_rows"]) != (train.tolist(), test.tolist()):
        raise RuntimeError("Training split differs from the scaler's split")

    state = {
        "raw_names": raw_names, "z_names": z_names,
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "fit_rows": train.tolist(),
        "fit_sample_ids": [dataset.samples[i].sample_id for i in train],
        "fit_groups": [groups[i] for i in train],
        "model_sha256": sha256(model_path.read_bytes()).hexdigest(),
    }
    (directory / "preprocessing.json").write_text(json.dumps(state, indent=2, allow_nan=False),
                                                  encoding="utf-8")
    return metrics
```

Call `fit_scaled_groups(dataset, raw_names, target, groups, "scaled-model")`
with your inputs. For prediction, pass raw feature mappings in the original
units and keep the sample IDs in the same order. The digest matches the model
bytes to the recorded scaler state. Reload only the trusted local model you
created, using `predict_scaled_groups("scaled-model", raw_rows)`.
The recipe converts finite real inputs to float64 before doing the same
subtraction and division used in training.

```python
from hashlib import sha256
import json
from numbers import Real
from pathlib import Path
import pickle

import numpy as np

from fibernet import predict_named


def _raw_float(value):
    message = "Raw features must be finite real numbers"
    if np.ma.is_masked(value) or not isinstance(value, (Real, np.bool_)):
        raise ValueError(message)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(number):
        raise ValueError(message)
    return number


def predict_scaled_groups(directory, raw_rows):
    directory = Path(directory)
    state = json.loads((directory / "preprocessing.json").read_text(encoding="utf-8"))
    payload = (directory / "model.pkl").read_bytes()
    if sha256(payload).hexdigest() != state["model_sha256"]:
        raise ValueError("Model does not match the saved scaler")
    artifact = pickle.loads(payload)
    if artifact["features"] != state["z_names"]:
        raise ValueError("Model features do not match the scaler output columns")
    columns = list(zip(state["raw_names"], state["z_names"], state["mean"], state["scale"]))
    rows = [{z: (_raw_float(row[raw]) - mean) / scale for raw, z, mean, scale in columns}
            for row in raw_rows]
    return predict_named(artifact, rows)
```

## Preserve selected source assertions

Native results always retain source row indices, sample IDs, actual train/test
indices, method and seed. Group results also retain the aligned group vector.
They do not automatically copy every parameter or metadata object.

```python
metrics = train_model(
    dataset, target="max_axial_force_N", model_type="lr",
    split="row", test_size=.3,
    metadata_keys=["length_unit", "target_unit", "target_meaning"],
)
provenance = metrics["training_data"]
```

Every selected metadata key must exist in every row. Selected values are
snapshotted recursively without coercion and must contain only JSON-native
null, bool, string, int, finite float, list, or string-keyed dict values.
NumPy scalars/arrays, tuples, sets, custom objects, nonfinite floats and cycles
are rejected before fitting or saving. Shared noncyclic structures are allowed
and copied. Later caller mutation does not change the returned snapshot.
Unselected metadata is not traversed. `None` or `[]` selects no metadata.

Null is valid in retained availability metadata, but is not a valid selected
numerical feature/target. For the reference, the three keys above preserve “m”,
“N” and the maximum-member-force interpretation. Optional mechanics declarations
such as E, nu or loading conditions must be selected explicitly when needed.

These are preserved source assertions, not verified units or physics. There is
no inferred feature-unit schema, metadata normalization, or `verified` flag.
A Dataset has no durable source locator; the artifact does not invent one.
Unselected metadata, params and full numerical data remain only in the source
Dataset. Saving only a model does not archive those omitted values. Relative
result paths still need their original external base directory.

## Result and trusted saved artifact

Native metrics contain the existing `r2`, `rmse`, `n_samples`, `features`, plus
raw `target` and `training_data`:

```python
{
    "source_kind": "FiberNetDataset",
    "rows": [
        {"row_index": 0, "sample_id": "square-00",
         "metadata": {"length_unit": "m", "target_unit": "N",
                      "target_meaning": "maximum absolute member axial force, not summed grip reaction"}},
        # Remaining source rows, in their original order
    ],
    "metadata_keys": ["length_unit", "target_unit", "target_meaning"],
    "split": {
        "method": "row", "random_state": 42,
        "train_rows": [0, 7, 2, 9, 4, 3, 6], "test_rows": [8, 1, 5],
    },
}
```

The `metadata` row key is absent when no metadata is selected. Group mode adds
`split["groups"]`. This record has O(rows) size; it is not a complete archived
dataset or proof of identity after the original data changes.

A saved pickle keeps the existing v1 envelope:
`model`, `features`, `metrics`, `artifact_version`, `target`. Its `metrics` equals
the returned dictionary, including native provenance. The model remains the
bare sklearn estimator. No automatic serialization security or physical-schema
checking is introduced. Load only a trusted artifact you created or otherwise
have reason to trust; never load an untrusted pickle.

Both native and CSV training now write and close a complete sibling pickle
before replacing the destination. Ordinary serialization, write, or replacement
failures preserve the previous file bytes or its absence. Existing targets must
be regular files without symlinks or special permission bits, and the parent
directory must already exist. This single-file replacement does not guarantee
crash durability, concurrent-writer safety, or preservation of inode ownership
and ACLs. See [model artifact persistence](model-artifact-persistence-usage.md)
for ordinary permissions, cleanup warnings, and the complete limits.

```python
# artifact is an already-loaded, trusted dictionary.
from fibernet import predict_named
rows = [{name: sample.features[name] for name in artifact["features"]}
        for sample in dataset.samples]
result = predict_named(artifact, rows)
identified_predictions = [
    {"row_index": i, "sample_id": sample.sample_id, "prediction": prediction}
    for i, (sample, prediction) in enumerate(zip(dataset.samples, result["predictions"]))
]
```

`predict_named` is unchanged: validation is `named_columns_only`; it does not
consume a Dataset, validate units or propagate IDs. Preserve the same row order
when joining predictions to samples. Predictions on training-source rows are
not an independent evaluation. Bare `.predict(ndarray)` still cannot detect
semantically reordered equal-width columns.

## CSV and optional dependencies

CSV calls keep their original default split, four metrics keys, v1 envelope,
model settings and fitted predictions. Complete Dataset CSVs still use prefixed
`feat_`/`label_` names. No prefix translation or native provenance is retrofitted
into old artifacts. Non-default native-only `split`, `groups` or
`metadata_keys` options on CSV calls are rejected.

Native fitting needs scikit-learn, with pandas confined to the CSV path. Native
mapping-based inference needs no pandas. Torch is neither imported nor needed
for this API. The existing `ml` installation extra still includes pandas and
sklearn; packaging has not changed. Fresh-process tests block pandas/Torch
imports across LR/RF/GB native fitting and named inference.

See the [reference research journey](reference-research-journey.md) for the
native end-to-end example. Row splitting tests held-out rows; meaningful origin
generalization requires valid caller-supplied independent-origin groups.
Historical saved journeys and their prefixed model artifacts are unchanged.
