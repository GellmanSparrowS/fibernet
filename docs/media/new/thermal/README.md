# Recorded transient Joule heating

The animation shows 81 saved temperature states over 20 seconds on a 4,340-node, 5,073-edge synthetic network. Geometry remains fixed; this is one-way electrical heating, not mechanical deformation. Temperature and the energy ledger evolve together.

See [scenario and numerical evidence](data/README.md), [campaign analysis](data/campaign_analysis.json), and [display provenance](transient_thermal_network.json).

From this directory, run `python render.py` to redraw with NumPy and Pillow, or `python check.py` to validate the saved data and media without a physical solve.
