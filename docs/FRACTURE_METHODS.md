# Experimental linear-static fracture API

From the checkout containing this guide, install the fracture dependencies with
`python -m pip install '.[fracture]'`. Run
`python -m examples.fracture_quickstart` there for a synthetic example.
The explicit API is `fibernet.sim.fracture.{FractureConfig,FractureSolver,FractureResult}`.

## Accepted geometry and units

Pass a `StructureGraph`, explicit `fixed_nodes`, `loaded_nodes`, and `length_unit` (`m`, `mm`, `um`). Original integer IDs, including gaps, are retained. Radii and node/internal-point coordinates share the declared unit. Positions are converted to metres; E and strengths are Pa, reactions N, displacements m, rotations rad, and energies J. `maximum_displacement_m` is displacement, not stretch ratio. E and nu must match every edge's isotropic Material, and G must equal E/(2(1+nu)). Strengths are explicit config values; no implicit material-strength lookup occurs.

The first version supports straight, homogeneous, uniform-radius circular welded beams. Collinear ordered interior points are acceptable; curved edges, duplicate endpoint pairs/parallel fibers, mixed radii or E/nu, anisotropic stiffness, non-welded joints, initially disconnected grips, and malformed IDs are rejected. No silent geometry simplification is performed. Initial notches use `initially_broken_edges` and remain irreversible. Both grips fix all six nodal degrees of freedom; loaded nodes receive an axial translation. Grip IDs must be chosen explicitly. This differs from automatic grip-band discovery in the private study tool.

## Numerical model

The operator is the exact static two-node circular Timoshenko stiffness matrix with six degrees of freedom per node and Cowper shear factor 6(1+nu)/(7+6nu). Sparse assembly uses symmetric diagonal scaling and SuperLU. No weak springs or pseudoinverse are introduced. Free-equilibrium residual, global force/moment balance, and half-constraint-work consistency checks use 1e-8 relative gates. A software test does not certify positive definiteness for every possible network.

At each beam end, the conservative envelopes are normal = |N|/A + r|M|/I and shear = 4|V|/(3A) + r|T|/J. The failure index is sqrt((normal/sigma)^2 + (shear/tau)^2); the larger end value is used. These component maxima need not occur at one material point. Declared brittle strengths are not fitted PLA parameters.

Linear response at a reference displacement predicts the next event. At each imposed displacement, beams with index >= 1 - event_tol are removed together; equilibrium is solved again until the avalanche is exhausted. The frozen numerical schema is `lightweight-near-tie-v1`; the default event_tol is 1e-10, with an explicit 1e-8 sensitivity option. `event_pre` observations retain pending failure IDs, end resultants, actual indices and below-one near-tie additions. Reference-displacement probes are excluded from the observed loading trace.

A material path loss means no connected active-edge component touches both grips. It does not mean pieces have physically separated. Detached free components/isolate poses and rotations are null. This is a small-displacement static model, with no automatic contact, finite kinematics, recoil, collision or validated large-fragment motion. Strain energy and half constraint work are not fracture toughness Gc; the API intentionally has no toughness field.

## Checkpoints and limits

`solver.run(checkpoint="case.checkpoint.json")` writes only complete equilibrated epochs using a temporary-file/atomic-replacement sequence. Use `resume=True` explicitly to continue the same file. Config, geometry, numerical parameters, source files and payload hashes are checked; the saved equilibrium is re-solved and event indices/masks checked. Source edits or a changed solver version require a new checkpoint namespace. These integrity hashes do not authenticate data against a malicious editor. A failed/incomplete avalanche does not replace the last complete checkpoint.

`stop_after_epochs` permits deliberate bounded pauses. `paused`, `epoch_bound`, and `numerical_stall` are incomplete outcomes. The solver caps graph sizes, avalanche/epoch counts, history bytes and cooperative runtime/RSS use. Its default memory budget is 768 MiB above baseline with 512 MiB free-memory reserve. A sparse factorization forecast can refuse a graph below the count limit. Memory/time checks are cooperative; they are not OS hard isolation. Numerical thread pools are limited to one during execution, so do not run concurrent calls in threads in the same process; use independently budgeted processes.

Results expose stable original node/edge IDs, failure events, pre/post event observations, final state and source/config/geometry bindings. `FractureResult.save` creates a new JSON file and refuses overwrite. JSON null fragment poses must not be interpolated as zero motion. The existing physical-learning three-target schema and old archived fracture/toughness fields are unchanged; fracture labels require a separate schema and must never silently be mixed with reduced-model labels.

## Command line

```bash
python -m fibernet.sim.fracture --graph graph.json --config fracture_config.json --fixed 0 --loaded 2 --length-unit m --output fracture.json
```

The default `--graph-format fibernet-fracture-geometry-1` reads the dedicated
geometry schema. Supply `nodes: [{id, position}]` and
`edges: [{id, node_i, node_j, radius, material, internal_points?}]`.
Each edge material must specify `youngs_modulus`, `poissons_ratio` and
`shear_modulus`.

For a native graph written with `StructureGraph.save_json` or `api.export`, add
`--graph-format structure-graph-3.1`. Supply explicit node and edge IDs, XYZ
positions, endpoints, radii and all three elastic constants on each edge.
This mode retains graph metadata, internal points and parallel edges and applies
the same fracture physics checks as `FractureSolver`. Select grips and any
initially broken edges using the stored IDs. Supply `--length-unit` for the
file's coordinates and radii. The material moduli and config remain in SI.

The config JSON contains the `FractureConfig` fields. Optional controls are
`--initially-broken`, `--checkpoint` and `--resume`.

## Reproducible notched-network scan

From a source checkout, install the fracture dependencies and run:

```bash
python -m pip install -e ".[fracture]"
python -m examples.fracture_scan --output-dir fracture_scan
```

The example generates three seeded, notched triangulated networks, each with
63 nodes and 158 original members. Each geometry is loaded to both 10 and
400 micrometres under the same material, notch and grip conditions. The short
and full runs retain the same geometry for each seed. An existing output
directory is refused.

Each case saves its complete input and native result. `scan.csv` and
`summary.json` collect peak reaction, first failure, observed material-path
loss, external work, final elastic strain energy and released elastic energy.
The summary retains completion and termination states. A path-loss displacement
is recorded only when loss is observed; a run that retains a path at its loading
limit has a null threshold rather than a zero-valued failure label.

External work is integrated along the recorded loading sequence, retaining
force drops at repeated displacement. Released elastic energy is external work
minus final stored energy and is checked against the energy drops at fixed
displacement. It measures energy released by this member-removal model, rather
than a calibrated material fracture energy. Seeds, geometry, explicit SI
parameters and source bindings accompany the results for reproduction.

The reported peak uses the **first** observation at the maximum recorded
reaction force. `peak_observation_index` is its zero-based position in the
native result's observation list, and `peak_displacement_m` is its recorded
displacement. `pre_peak_external_work_J` includes the interval ending at the
peak. `post_peak_external_work_J` sums subsequent recorded intervals only.
Both use the same force–displacement trapezoids as total external work.

Check `post_peak_observation_status` and `post_peak_displacement_span_m`.
`no_later_observation` means the peak was terminal and post-peak work is
null (blank in CSV). `constant_displacement_only` means later observations
exist but do zero external work, even if elastic energy is released.
`positive_displacement_span` means the trace includes further loading;
the reported work ends at the last recorded point. These describe the
observations separately from path loss, right-censoring or incomplete
termination. They neither extrapolate to failure nor supply a shared
integration window or fracture toughness for comparing cases.

The paired mode compares an intact network with its notched counterpart.

```bash
python -m examples.fracture_scan --output-dir fracture_paired --compare-notch
```

Each seed and loading bound produces one `unnotched` and one `notched` case.
Cases in a pair share the same generated graph, material, grips and displacement
settings. Only the list of initially absent edges changes. Join cases by
`pair_id` in the CSV or JSON. The shared `source_graph_sha256` hashes the
pre-notch graph as compact UTF-8 JSON with sorted keys. The solver's geometry
binding also includes the initial active mask, so the two bindings differ.

Read peak reaction, external work and released elastic energy alongside each
case's `final_displacement_m` and status. Full runs can stop at different
displacements when a load path is lost. A completed run that reaches its
loading limit with a path is right-censored. Incomplete runs and early stops
with a remaining path have separate statuses and no observed path-loss displacement.

## Distribution and provenance

`PROVENANCE.json` records the original hashes and verbatim extracted function names for beam, sparse and event kernels; the public adapter, guard, driver and checkpoint envelope are new. MIT repository attribution is preserved. NumPy/SciPy/psutil/threadpoolctl retain their separate licenses. No LAMMPS source, DLL, executable, private study request/result, research image or unpublished manuscript is part of this addition. The separate LAMMPS route is a different model with separate distribution obligations and platform validation; installing stock LAMMPS cannot replace the patched research engine.

## Track electrical conductance during fracture

From a checkout installed with the fracture extra, run the seed 11 notched
case with fixed electrodes on the original grip node IDs.

```bash
python -m examples.fracture_transport --output-dir fracture_transport
```

The one-way model assigns each intact member a reference conductance
`G = σπr²/L₀` in S with conductivity `σ = 1 S/m`, the original radius `r`
and original length `L₀` in m. Failed or initially absent members keep their
stable edge IDs and carry zero current. Components without electrodes receive
a zero-voltage reference with zero injection. An open circuit reports zero
effective conductance and a null resistance.

`electrical-history.csv` records every pre- and post-break observation,
including consecutive changes at equal displacement. Each row pairs the
mechanical reaction and strain energy with the electrical values. The JSON
retains stable ID-ordered fields and hashes for the source graph and fracture result.
The mechanical observation history does not supply per-observation nodal
poses, so the calculation holds geometry and area at their reference values.
It uses no electrical feedback into fracture.

## Manuscript wording draft

We implemented an experimental progressive member-removal backend using linear, small-displacement Timoshenko frame elements with six degrees of freedom per node. Circular beam sections shared isotropic elastic constants and prescribed brittle strength parameters. A conservative combined normal/shear stress envelope determined member removal. At each event displacement, near-tied members were grouped by a versioned numerical tolerance and equilibrium was recomputed at the same displacement until no further member met the removal condition. Observations immediately before removal preserve the failure criterion and end resultants. Source, geometry and parameter hashes bind resumable equilibrated checkpoints. Loss of a grip-spanning material path was recorded as a topological endpoint; absolute poses of unrestrained fragments were left undefined. The model excludes automatic contact and finite-kinematic separation, and no experimental calibration or fracture-toughness identification is asserted. This software description must be paired with the actual release commit and the independently reported validation scope; it does not establish a new empirical result.
