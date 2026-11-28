# Open pressure-end Stokes flow on a Tet4 pore

The experimental `solve_open_pressure_stokes` method accepts the existing
FiberNet `Tet4Bundle`, with one face-connected fluid domain and planar
x-min/x-max open faces. Remaining exterior surfaces are no-slip. Use SI
viscosity (Pa·s) and pressure difference (Pa); positive difference drives
positive x flow. The elastic placeholders in the fluid bundle are not used
by this calculation. Each cap must have an unconstrained pressure-work velocity
trace; refine triangular caps whose vertices and edges all touch no-slip walls.

This is a gradient-form Taylor–Hood P2/P1 Stokes solve with natural pressure
loads at fixed open caps, not a periodic body-force method, a prescribed
velocity inlet, or a fully coupled fluid–structure solution. The outlet
pressure load defines the reference, rather than enforcing its recovered
nodal pressure. Vertex arrays of the native P2 velocity are samples, not the
complete quadratic field. Use the integrated terminal flux for quantitative
flow comparisons.

Install the optional NGSolve 6.2.2608 native runtime separately on CPython
3.12, together with FiberNet's `solid-iterative` extra for the default
PyAMG-preconditioned solver. Use `solver="direct"` to compare with native
UMFPACK when available. The API is bounded by mesh counts and raw operator
size; the direct option admits at most 8,000 cells. These caps do not
guarantee a peak-memory limit. This does not change
existing periodic-flow or core dependency behavior.

## Export the complete quadratic velocity to VTK

Opt in at solve time if the full P2 velocity is needed for visualization.
The default still returns only velocity samples at the original vertices,
and does not build the P2 field or import Meshio.

```python
from fibernet.sim.tet4_open_stokes import (
    solve_open_pressure_stokes, write_open_stokes_velocity_vtu,
)

result = solve_open_pressure_stokes(
    pore_bundle, viscosity_Pa_s=0.001, pressure_drop_Pa=0.0035,
    include_quadratic_velocity=True,
)
write_open_stokes_velocity_vtu("pressure-flow.vtu", result.quadratic_velocity)
```

The writer needs Meshio 5.3.5 and an existing destination directory that
supports hard links. It publishes a complete new VTU without overwriting
an existing path and rejects files larger than 16 MiB. The detached field stores SI coordinates
and velocity on corners and shared edge midpoints of VTK tetra10 cells.
Coordinates are metres and velocity is metres per second, regardless of
the input length unit. This is a full quadratic velocity representation on
affine cells, not streamlines or time-resolved trajectories. The two unsigned
integer components of `source_node_ids` repeat a vertex's original ID
or list the two original endpoint IDs (ascending) of a midside point.
`source_cell_id` stores the original tetrahedron IDs. Do not coerce these
identifiers to floating point. Use integrated cap flow, not visual line
counts, to compare throughput.

## View a saved velocity field without rerunning the solve

Install the optional `browser-viz` extra for VTK, Trame and vtk.js. To plot
a quadratic velocity file that you created with the writer above:

```sh
python -m fibernet.viz.stokes_browser pressure-flow.vtu
```

Open `http://127.0.0.1:8765` on the computer running the viewer. This is
a loopback-only postprocessor, not a remote service or a file upload.
It accepts only regular, locally generated files in FiberNet's single-piece
tetra10 format, not arbitrary VTK datasets. It does not need the NGSolve solver to replay a saved file.
Use `--from-max-x` to seed from the high-x cap, the inlet for negative
pressure drop. It does not reverse
the stored velocity. The boundary button shows the entire fluid-domain
boundary, including the outer duct. Lines are computed steady streamlines;
use integrated cap flow from the solver for quantitative comparisons.
The speed colors clip above the 99.5th percentile of the displayed lines.

For a planar extreme-x boundary with several separate openings, add
`--cap-components`. The 49-seed budget reserves at least one seed for each
edge-connected opening and distributes the rest by area. The viewer reports
how many openings produced a streamline. This setting is for inspection, not
flow weighting, and does not change the saved field.

### Quantify the flow through separate openings

Once a solve has exported the native quadratic velocity VTU, integrate the
saved solution through each opening without rerunning the solver:

```bash
python -m fibernet.analysis.stokes_cap_flux results/velocity.vtu
```

The JSON report groups triangles sharing a complete edge on each extreme-`x`
cap. Each opening includes its area, centroid and signed volumetric flow in SI
units. Positive flow leaves the domain, independent of the pressure-gradient
direction. Use the total for each cap and the relative cap imbalance as a
cap-balance check, not proof of zero interior divergence or wall leakage.
Two zero cap totals yield zero imbalance. A count of displayed streamlines is
not a flow fraction. Use geometric location, not array position, to compare
openings against the viewer, which orders components differently.

This command integrates the affine quadratic velocity basis and accepts the
native FiberNet quadratic VTU export, not curved cells or arbitrary external
VTU files. NumPy and SciPy suffice to inspect the saved data. For Python use
`from fibernet.analysis.stokes_cap_flux import audit_saved_stokes`.


### Compare throughput and velocity redistribution on the same caps

Use two native velocity exports to separate the change in total flow from
a change in the distribution of normal velocity across the ends.

To try the comparison without running the solver, save
[`reference.vtu`][stokes-reference] and [`candidate.vtu`][stokes-candidate]
in the same directory. They contain two manufactured velocity fields on a
triangular prism, not simulated or measured flows.

With FiberNet 4.40.0 or later, run this in Python from that directory:

```python
from fibernet.analysis.stokes_cap_comparison import compare_saved_stokes_caps

report = compare_saved_stokes_caps("reference.vtu", "candidate.vtu")
outlet = report["x_max"]
print(f"{outlet['throughput_ratio']:.2f}")                 # 1.02
print(f"{outlet['normalized_profile_residual_L2']:.2f}")    # 0.02
```

The signed outlet flux goes from 0.50 to 0.51 m³/s. The first result is the
ratio of those fluxes; the second is the remaining velocity-profile change
relative to the reference, after removing that uniform scaling. The files
are ordered: swapping them gives a throughput ratio of about 0.980392 and
a profile residual of about 0.019220.

If you installed from repository main after the 4.40.0 release, the shell
command below writes the same report as JSON. The PyPI 4.40.0 release
supports the Python API above, not the shell command.

```sh
python -m fibernet.analysis.stokes_cap_comparison reference.vtu candidate.vtu
```

[stokes-reference]: https://raw.githubusercontent.com/GellmanSparrowS/fibernet/946f73a7a17b7f7ab8820b98dfa8d9a323d4209d/tests/fixtures/stokes_cap_cli_v1/reference.vtu
[stokes-candidate]: https://raw.githubusercontent.com/GellmanSparrowS/fibernet/946f73a7a17b7f7ab8820b98dfa8d9a323d4209d/tests/fixtures/stokes_cap_cli_v1/candidate.vtu

The extreme-`x` cap planes and their physical triangles and edge midpoints
must match exactly, independently of point and cell numbering. Both fields
must have nonzero flux of the same direction on each matched cap. The two
values express the candidate-to-reference signed flux ratio and the
relative L2 norm of the remaining normal velocity after that uniform scaling
is removed. For example, a proportional speed-up changes the first value
but leaves the second at zero. The routine treats all triangles on each cap
together. It does not reconstruct load conditions, infer permeability, or
trace fluid paths through the interior. Check that the cases use comparable
loads and boundary conditions before interpreting the difference.
