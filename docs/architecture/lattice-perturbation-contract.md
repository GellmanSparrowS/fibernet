# Lattice perturbation and Kagome migration

`fibernet.gen.unified.lattice_2d` and
`fibernet.api.create("lattice_2d", ...)` use the same displacement contract for
square, triangular, honeycomb and Kagome topology names (case-insensitive).

```python
from fibernet import api as fn

network = fn.create(
    "lattice_2d", topology="kagome", cell_size=2.5, grid_size=(2, 2),
    perturbation=0.1, seed=17, rotation=0.3,
)
```

For a positive perturbation, each stored node identity receives independent
uniform x/y offsets between `-perturbation * cell_size` and
`perturbation * cell_size`. Both coordinates use `cell_size`, not separate tile
width and height. Displacement follows connectivity construction and precedes
global rotation about the origin. The offset bound is in the unrotated frame.
The seed initializes a local NumPy `RandomState`; it does not consume the
global NumPy random stream. A fixed seed repeats the geometry for the same
inputs and implementation. `seed=None` does not promise repeatability.

The existing Kagome tiler rewrites shared edge endpoint references before
displacement. Those references remain shared, so their crosslink positions
move with their fiber endpoints. Node keys, iteration order and edge order
are retained, including unused node aliases. No new welding or topology
reconstruction is performed. Some distinct referenced node identities already
occupy the same position in the base geometry and can separate under
perturbation. This preserves the legacy graph construction, not every
geometric coincidence. This change does not redefine the legacy Kagome unit
cell or certify it as a standard scientific lattice.

The signature, return type (`FiberNetwork`), radius and material forwarding
are unchanged. As before, conversion omits edges shorter than `1e-10` in
caller length units; extreme parameters or collapsed edges can therefore
change retained fiber counts. Perturbation is not a guarantee against
self-intersection, small edges or ill-conditioned simulation inputs.

## Migrating historical Kagome calls

Earlier versions silently ignored positive perturbation and seed for this
Kagome entry point. Positive perturbation now intentionally changes geometry,
including edge lengths and quantities subsequently derived from it. To
reproduce the previously unperturbed geometry, pass `perturbation=0` with the
same other inputs. Zero/default perturbation, the historical nonpositive
no-displacement branch, and other lattice topologies are unchanged. Invalid
seeds on the positive branch now reach NumPy's usual seed validation, as they
already did for the other topologies.

Saved network files are not modified. Record the generating implementation
alongside parameters when comparing regenerated datasets; do not relabel old
unperturbed outputs as newly perturbed samples. The separate
`kagome_lattice_2d` generators and FiberScope providers are unchanged.

Regression coverage is in `tests/test_kagome_perturbation_contract.py`.
