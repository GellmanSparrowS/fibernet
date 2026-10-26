# Checked availability for the legacy 94-feature extractor

Use `GraphFeatureExtractor.extract_with_status()` when a feature's meaning depends
on optional image dependencies or on whether its measurement domain exists.
Then use `FiberNetDataset.feature_matrix_checked()` to select compatible, finite
columns explicitly. A missing OpenCV dependency is not a measured zero.

These methods are opt-in. Existing `extract()`, `extract_vector()`, batch methods,
`dataset.features`, training defaults, and the separate 14-feature schema retain
their existing behavior.

## Extract, retain provenance, select columns

```python
from fibernet.analysis.graph_features import GraphFeatureExtractor
from fibernet.ml.dataset import DatasetSample, FiberNetDataset

# graph is an existing simple undirected NetworkX graph with node "pos"
# attributes, or an existing StructureGraph without parallel physical edges.
extractor = GraphFeatureExtractor(
    canvas_size=256, thick=5, edge_margin=0.12,
    top_k=3, area_thresh=0.005, connectivity=8,
)
result = extractor.extract_with_status(graph)
contract = {key: value for key, value in result.items() if key != "values"}

sample = DatasetSample(
    sample_id="sample-1",
    unit="measured",             # your sample's existing source description
    grid=(1, 1),                  # your sample's existing grid metadata
    params={},                   # your sample's existing design parameters
    features=result["values"],
    labels={},
    metadata={"feature_contract": contract},
)
selected = ["total_pore_count", "total_pore_ratio"]
dataset = FiberNetDataset([sample], feature_names=selected, target_names=[])
X = dataset.feature_matrix_checked()  # shape (1, 2), in selected order
```

The selected pore columns are accepted when their recorded states and finite
values qualify; the call raises if OpenCV was unavailable or another required
condition failed. It never substitutes zeros. Choose columns for the question
you are studying, rather than automatically dropping rejected columns. To inspect
why a column was rejected, read `result["status"][name]`.

`feature_matrix_checked(names=None, *, allow_reasons=())` uses the dataset's
`feature_names` when `names` is omitted. Passing an explicit list selects that
order without modifying the dataset's selection. Names must be unique members
of `GraphFeatureExtractor.FEATURE_COLS`. Every requested value must exist and be
a finite real scalar. Metadata validation applies to every included row, even
when only a small subset of columns is requested. Empty datasets retain the
usual empty-matrix shape; they do not establish provenance for unseen rows.

## What the record describes

`extract_with_status()` returns the original numeric `values` plus:

| Field | Meaning |
|---|---|
| `schema` | `fibernet.graph_features94_result/1` |
| `feature_schema` | `fibernet.graph_features94_legacy/1` |
| `definition_id` | `graph94-image-availability-v1` |
| `geometry` | `simple_undirected_xy_endpoint_chords_legacy_v1` |
| `feature_names` | All 94 names in the original vector order |
| `audited_names` | The 63 names covered by this availability contract |
| `parameters` | The six actual extraction settings |
| `dependencies` | Availability and reported versions of NumPy, NetworkX, OpenCV and SciPy |
| `status` | One `{state, reason}` record for each of the 94 names |

The 63 covered names comprise 59 image measurements and four configuration/count
entries. The remaining 31 structural features stay
`unknown/outside_audited_scope`, even if their values are finite. Consequently,
**checked selection of all 94 features intentionally fails**. Selecting all 63
covered features can also fail where a statistic has no measurement domain.
Availability coverage is not a scientific accuracy certification.

The geometry identifier describes **XY projection and straight endpoint chords**.
Curve interiors do not enter these measurements, and 3-D coordinates are not a
lossless 3-D representation. The identifier does not certify that a supplied
network is physically planar or straight. Raster normalization and the extraction
settings also affect comparability.

The record is declared provenance, not authentication. The checked method does
not prove that metadata were not edited, that dependency versions yield identical
numbers, or that different physical input scales are equivalent. Dependency
versions remain visible in the record; cross-row equality of those versions is
not enforced.

## Compatible extraction settings

Every included row must have the supported schemas, definition and geometry,
exact 94-name status coverage, and exactly the same values for **all six** settings,
including settings irrelevant to a chosen subset. The checked regime is:

| Parameter | Accepted value |
|---|---|
| `canvas_size` | Integer greater than 10 |
| `thick` | Positive integer |
| `edge_margin` | Finite real in `[0, 0.5)` |
| `top_k` | Integer greater than or equal to 0 |
| `area_thresh` | Finite real in `[0, 1]` |
| `connectivity` | Integer 4 or 8 |

Booleans are not accepted as these numeric settings. Settings are compared exactly,
not approximately. These checks do not change the legacy constructor's acceptance
rules. `top_k=0` records a disabled selection, not a measured selected-pore shape.

## States and explicit convention policies

The default policy accepts `computed` and `defined_empty` states, provided the
record is valid and the selected value is finite. The state/reason pair must also
be valid for that particular feature; a caller cannot relabel an unknown or
fallback result as computed.

| State | Interpretation |
|---|---|
| `computed` | Evaluated measurement, recorded input, or a defined no-pore ratio |
| `defined_empty` | Defined count or sum for an empty population |
| `not_applicable` | No applicable measurement domain, denominator or selection |
| `fallback` | A legacy convention, raster failure, or nonfinite output |
| `unavailable` | Required dependency, positions or render unavailable |
| `unknown` | Outside the 63-name audited scope |

If a finite legacy convention is appropriate for your analysis, request its
registered reason explicitly, for example:

```python
X = dataset.feature_matrix_checked(
    ["pore_area_cv"],
    allow_reasons=["stat.single_pore_default"],
)
```

This admits the registered `fallback/stat.single_pore_default` pair where it is
valid; it does not admit a missing dependency or a nonfinite value. `allow_reasons`
must be a sequence of known reason strings, not a bare string. The complete set
of eligible reasons is:

- `not_applicable`: `domain.no_pores`, `domain.zero_mean_area`,
  `domain.insufficient_pores`, `domain.no_selected_pores`,
  `selection.disabled_top_k`, `domain.no_edges`, `domain.no_overlap_pairs`,
  `domain.no_overlap_components`, `domain.zero_pixel_union`,
  `domain.zero_centerline_length`, `domain.zero_canvas_area`.
- `fallback`: `stat.single_pore_default`, `stat.single_occupied_bin_default`,
  `shape.no_contour`, `shape.zero_hull_area`, `shape.zero_perimeter`.

The registry also distinguishes ordinary `computed/evaluated`,
`computed/input_recorded` and `computed/domain.no_pores` results;
`defined_empty` uses `domain.no_pores`, `domain.no_selected_pores`,
`selection.disabled_top_k`, `domain.no_overlap_pairs` or
`domain.no_overlap_components` where defined for a feature.

Policies cannot admit `unknown/outside_audited_scope`,
`unavailable/dependency_missing.cv2`, `unavailable/dependency_missing.scipy`,
`unavailable/input.no_positions`, `unavailable/render_missing`,
`fallback/render_empty_edge` or `fallback/stat.nonfinite`. They also cannot
bypass missing values, nonnumeric or nonfinite values, malformed records,
unsupported schemas or incompatible settings.

Legacy samples without `metadata["feature_contract"]` remain usable through the
old APIs; checked access raises `legacy.status_absent`. Invalid record shapes,
name coverage or state/reason pairs raise `contract.malformed`; unsupported
schema/definition/geometry identifiers raise `schema.unsupported`. Errors identify
the row/sample and, where relevant, the selected feature. No inference of status
is made from a finite value, a zero, a filename, or currently installed OpenCV.

## Serialization and evidence boundary

Keep the numeric values and metadata together. The full extraction result is a
Python object, **not an unconditional strict-JSON payload**: legacy calculations
can return NaN or infinity. Equal-area SciPy skewness/kurtosis is one such path.
Those values are preserved, not converted to zero or null. Strict encoding with
`json.dumps(result, allow_nan=False)` therefore rejects nonfinite results.

The metadata-only contract, once accepted by the checked method, is compatible
with strict JSON. Existing dataset persistence still enforces its selected-value
checks and retains its existing serialization behavior; it does not acquire a
new blanket strict-JSON guarantee. A deliberately selected finite subset can be
saved as such while retaining the full contract, without claiming that omitted
numeric values were preserved. Dataset splits/subsets retain metadata.

The regression fixtures cover missing-OpenCV behavior, controlled mock branch
routing and checked selection. These tests do not establish native OpenCV raster
accuracy or numerical parity. Controlled spectral starts do not make the legacy
public API deterministic.
