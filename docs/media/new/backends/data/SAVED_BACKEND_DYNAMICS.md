# Saved backend dynamics

Two reproducible views of previously computed synthetic network states:

- **LAMMPS:** 64 sites and 441 harmonic bonds. The animation samples every tenth state from the saved 401-state trajectory, from 0 to 41.667 microseconds. Coordinates and energies retain SI units. It depicts short-time bonded-network motion, not a calibrated atomistic material or universal BPM capability.
- **Taichi:** 4,340 nodes and 5,073 axial edges. All nine saved states from the 1,600-step run are shown. Coordinates, time, mass and energies are nondimensional. The source recorded sampled cutoff checks but no work ledger; the plotted energies alone cannot establish a forced-system energy balance.

Both cameras are fixed. Colored geometry is the initial saved geometry plus **1,000 times the actual saved displacement**. Muted gray geometry is the initial state. Colors and numeric displacement readouts use the unamplified displacement. Amplification does not alter the saved data or energy values. Frames are not interpolated, and the loop restart is a playback reset rather than reverse dynamics.

Run `python docs/media/new/backends/render.py` from the repository to regenerate both GIFs, PNGs and editable vector SVG posters. The renderer uses only NumPy and Pillow; it does not import or run either solver. Provenance and content checksums are in `saved_dynamics_provenance.json`. The NPZ files preserve the displayed states, stable node/edge IDs, endpoint indices, time, step and saved energies.

The 800 × 1132 portrait master follows the A4 aspect ratio. Geometry is uniformly scaled, with metrics and energy traces below the network.
