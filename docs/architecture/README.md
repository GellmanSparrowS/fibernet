# Public API and scientific contracts

This index describes implemented interfaces and their boundaries. The examples
are synthetic software demonstrations; they do not establish experimental
calibration, realistic-network accuracy or independent-origin generalization.

## Start here

- [Linear FEM quickstart](linear-fem-quickstart.md)
- [Reference research journey](reference-research-journey.md)
- [Material consumption](material-consumption-contract.md)
- [Result identity](result-identity-contract.md)
- [Direct FEM array identity](raw-array-identity-migration.md)
- [Snapshot graph identity](snapshot-identity-contract.md)
- [Linear FEM diagnostics](fem-linear-diagnostics-usage.md)
- [3D reaction definition](fem-3d-reaction-contract.md)
- [Dataset persistence](dataset-persistence-usage.md)
- [Dataset loading](dataset-load-validation-usage.md)
- [Native Dataset training](native-dataset-training-usage.md)
- [Named inference](named-inference-usage.md)
- [Model persistence](model-artifact-persistence-usage.md)
- [Taichi lifecycle](taichi-lifecycle-audit.md)
- [Meshio export](../integrations/MESHIO_USAGE.md)
- [Linear transport](../integrations/LINEAR_TRANSPORT_USAGE.md)
- [Explicit thermal API](../integrations/TRANSIENT_THERMAL_USAGE.md)

## Scientific and compatibility boundaries

Mechanical shortcuts consume global E or explicit spring k as documented;
persisted per-edge materials do not imply heterogeneous constitutive support.
Parallel-member handling is backend-specific; inspect the selected backend's
contract rather than inferring it from graph serialization.

Nonlinear final-state stiffness, reactions and node moments are unavailable
where they are not recovered. Linear-reference stress approximations do not
establish nonlinear equilibrium. Numerical regularization is not a physical
support or an automatic mechanism detector.

Taichi runtime support is pinned to 1.7.4. The lifecycle checks have CPU coverage;
GPU execution and a cross-platform runtime matrix are not established.
External physics integrations beyond the explicit implemented APIs are not
advertised as supported or validated solvers.

Regression tests live in tests/.
