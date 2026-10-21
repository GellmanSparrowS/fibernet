# Archived transient thermal network

The animation replays all 81 saved states of a synthetic 4,340-node,
5,073-edge network from 0 to 20 s, sampled every 0.25 s. The archived model
uses constant lumped thermal capacities and conductances, backward Euler,
and one-way Joule heat input. Geometry stays fixed. This is not a mechanical
deformation or two-way electrothermal demonstration.

`thermal.npz` contains the complete archived arrays, losslessly recompressed.
`archive_provenance.json` records the original member and archive checksums.
Stable node and edge identifiers, endpoint mappings, SI positions, temperatures,
heat rates, boundary values and residual diagnostics are retained. The initial
interval is explicitly unavailable: residual diagnostics contain NaN while the
initial stored heat change is zero. These values must not be silently treated
as an accepted solved interval.

`summary.json`, `energy_ledger_80.csv` and `campaign_analysis.json` preserve
archived validation results, including an initial relative heat-rate permutation
gate failure. The saved 40-to-80 step comparison has a maximum common-time
nodal temperature difference of 0.0021986461260325996 K and a final maximum
nodal difference of 0.0017796851928437718 K. This is a bounded synthetic
scenario comparison, not general time-accuracy or experimental validation.

## Reproduce the visual

From the repository root, with NumPy and Pillow available:

    python docs/media/new/thermal/render.py

The renderer performs no physical solve. It checks saved data consistency,
then draws each recorded temperature state once, in the original order. A
fixed scale covers 300 to 300.4202906813709 K. Edge colors are means of saved
endpoint temperatures for display; they are not independent edge solutions.
Recorded positions do not move. The complete sequence plays at 2.5 times
physical time and then loops; there are no added interpolated states.

The GIF and fully editable vector SVG poster use an A4 portrait aspect ratio.
The synchronized energy plot displays input, accumulated stored energy, and
energy transferred to reservoirs. Display conversion uses microjoules while
all retained source arrays remain SI. Matched public checksums are recorded
in `docs/media/new/thermal/transient_thermal_network.json`.
