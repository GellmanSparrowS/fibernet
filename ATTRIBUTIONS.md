# Open Source Attributions

FiberNet integrates with several open-source libraries. We gratefully acknowledge their contributions.

## Core Dependencies

### NumPy
- **License**: BSD 3-Clause
- **URL**: https://numpy.org/
- **Usage**: Core numerical operations, array manipulation

### SciPy
- **License**: BSD 3-Clause
- **URL**: https://scipy.org/
- **Usage**: Sparse matrices, optimization, interpolation, signal processing

### Matplotlib
- **License**: PSF (Python Software Foundation)
- **URL**: https://matplotlib.org/
- **Usage**: Visualization and plotting

## Optional Integrations

### NetworkX
- **License**: BSD 3-Clause
- **URL**: https://networkx.org/
- **Usage**: Advanced graph analysis, community detection, centrality measures
- **Integration**: `fibernet.integrations.networkx_integration`

### MDAnalysis
- **License**: GPL v2
- **URL**: https://www.mdanalysis.org/
- **Usage**: Molecular dynamics trajectory analysis, RMSD, radius of gyration
- **Integration**: `fibernet.integrations.mdanalysis_integration`
- **Note**: GPL v2 requires derivative works to be GPL-compatible

### LAMMPS
- **License**: GPL v2
- **URL**: https://www.lammps.org/
- **Usage**: Molecular dynamics simulations with various force fields
- **Integration**: `fibernet.integrations.lammps_integration`
- **Note**: GPL v2 license applies when using LAMMPS integration

### OVITO
- **License**: GPL v3
- **URL**: https://www.ovito.org/
- **Usage**: Advanced visualization, common neighbor analysis
- **Integration**: `fibernet.integrations.ovito_integration`
- **Note**: GPL v3 license applies when using OVITO integration

### Taichi
- **License**: Apache 2.0
- **URL**: https://taichi-lang.org/
- **Usage**: GPU-accelerated simulations
- **Integration**: `fibernet.sim.accelerated`

## License Compatibility

FiberNet is released under the MIT License. When using optional integrations:

- **BSD/MIT integrations** (NetworkX, NumPy, SciPy): No additional restrictions
- **Apache 2.0 integrations** (Taichi): Compatible with MIT
- **GPL integrations** (MDAnalysis, LAMMPS, OVITO): If you distribute FiberNet with these integrations enabled, the combined work may need to be GPL-licensed

## Installation

Install optional dependencies as needed:

```bash
# Core (always required)
pip install numpy scipy matplotlib

# Optional integrations
pip install networkx           # Graph analysis
pip install MDAnalysis         # MD trajectory analysis
pip install lammps             # MD simulations
pip install ovito              # Visualization
pip install taichi             # GPU acceleration
```

## Citation

If you use FiberNet in your research, please cite:

```bibtex
@software{fibernet2024,
  title = {FiberNet: A Comprehensive Fiber Network Simulation Library},
  author = {ML-BioMat Lab},
  year = {2024},
  url = {https://github.com/GellmanSparrowS/fibernet}
}
```

And cite the integrated libraries you use:

- **NetworkX**: Hagberg, A., Swart, P., & Schult, D. (2008). Exploring network structure, dynamics, and function using NetworkX.
- **MDAnalysis**: Michaud-Agrawal, N., et al. (2011). MDAnalysis: A toolkit for the analysis of molecular dynamics simulations.
- **LAMMPS**: Thompson, A. P., et al. (2022). LAMMPS-a flexible simulation tool for particle-based materials modeling.
- **OVITO**: Stukowski, A. (2010). Visualization and analysis of atomistic simulation data with OVITO.
