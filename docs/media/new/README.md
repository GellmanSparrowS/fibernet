# Recorded simulation and learning media

Each directory contains saved data, a render-only recipe and methods. Render scripts use Python with NumPy and Pillow; the learning figures additionally use Matplotlib. They do not run physics, fit models, or perform inference. Parameters are synthetic.

- [LAMMPS and Taichi dynamics](backends/README.md): actual saved positions and energies; displacement display ×1,000.
- [Viscoelastic dynamics](viscoelastic/README.md): stress, motion and distinct material/numerical losses; displacement display ×3,000.
- [Quasistatic fracture](fracture/README.md): actual member-removal observations on reference geometry, reaction and elastic energy.
- [Transient thermal response](thermal/README.md): one-way Joule heating, fixed geometry, all 81 saved states.
- [GINE study and RL environment records](learning/README.md): historical learning research and prescribed environment actions, separated from current integrated training capabilities.

The original twelve homepage animations remain available. These additions distinguish recorded physical evolution, saved learning history and discrete environment interaction from camera rotation and interface tours. Technical provenance retains numerical limits and adverse outcomes beside each example. `MANIFEST.json` records publication payload checksums and animation dimensions.
