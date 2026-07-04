# Changelog

All notable changes to FiberNet will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-05

### 🎉 First Stable Release

FiberNet has reached 1.0.0! This marks the first stable release of our comprehensive fiber network simulation framework.

### Added

#### Core Features
- **50+ Network Generators**: Complete suite of 2D/3D fiber network generators
  - Disordered: random straight/curved, oriented, random walk
  - Ordered: square/hexagonal/triangular lattices, cubic/diamond/octet 3D structures
  - Chiral: helices, twisted bundles, braided ropes
  - Woven: plain/twill/satin 2D weaves, 3D woven structures
  - Hierarchical: gradient density, core-shell, fractal networks
  - Advanced: Voronoi, electrospun, meltblown, biomimetic (collagen, fibrin)
  - Special: auxetic, kirigami, foam-like, gyroid infill

- **Comprehensive Simulation Modules**
  - **Mechanical**: Linear FEM, nonlinear FEM with large deformation, fracture mechanics
  - **Viscoelastic**: Maxwell, Kelvin-Voigt, Standard Linear Solid, Generalized Maxwell models
  - **Thermal**: Steady-state and transient heat conduction
  - **Fluid/Acoustic**: Darcy flow, pore network models, acoustic wave propagation
  - **Electromagnetic**: Static electric/magnetic fields, wave propagation
  - **DMA**: Dynamic Mechanical Analysis for frequency/temperature sweeps

- **Advanced Crosslink Models**
  - Rigid constraints
  - Spring connections
  - Breakable bonds with failure criteria
  - Frictional contacts
  - Covalent bonds (Morse potential)
  - Hydrogen bonds (angle-dependent, reformable)
  - Ionic bonds (screened Coulomb)
  - Physical entanglements (sliding contacts)

- **Machine Learning Integration**
  - Feature extraction (150+ structural, topological, mechanical features)
  - Property prediction with scikit-learn
  - GNN support (PyTorch Geometric and DGL formats)
  - Dataset generation utilities
  - Property predictor with train/evaluate interface

- **GPU Acceleration with Taichi**
  - Parallel force computation
  - Parallel dynamics simulation
  - Random network generation
  - 10-100x speedup for large networks

- **Periodic Boundary Conditions**
  - Image-based minimum distance calculations
  - Periodic FEM simulations
  - Bulk property computation
  - Ensemble averaging

- **Comprehensive I/O Support**
  - JSON/YAML for configuration and data
  - LAMMPS for molecular dynamics
  - VTK/VTU for ParaView visualization
  - GMSH for finite element meshing
  - PDB/XYZ for molecular structures
  - Pandas DataFrames for data analysis

- **Unit System Support**
  - SI (meters, kg, seconds, Newtons, Pascals)
  - CGS (centimeters, grams, dynes, barye)
  - Micro (micrometers, micrograms, micronewtons, MPa)
  - Nano (nanometers, nanograms, nanonewtons, GPa)
  - Molecular (Angstroms, Daltons, kJ/mol)

- **Analysis Tools**
  - Morphology: fiber length distribution, orientation, connectivity
  - Topology: degree distribution, clustering, shortest paths
  - Stress-strain curves with yield/ultimate strength detection
  - Nematic order parameter
  - Porosity calculations
  - Spectral analysis
  - Pore structure analysis
  - Anisotropy quantification
  - Structural fingerprints

- **Visualization**
  - Matplotlib 2D/3D plotting
  - PyVista interactive 3D rendering
  - Stress field visualization
  - Deformation animations
  - Damage evolution plots

- **Configuration Management**
  - YAML-based experiment configuration
  - Reproducible workflows
  - Template configurations
  - Configuration validation

- **Ensemble Generation**
  - Statistical sampling of networks
  - Convergence analysis
  - Ensemble statistics (mean, std, min, max)
  - Parallel generation support

#### Infrastructure
- **Comprehensive Test Suite**: 375 tests (369 passing, 6 skipped for optional dependencies)
- **Documentation**: README with extensive examples, API documentation
- **Examples**: 8 end-to-end integration examples covering all major workflows
- **Type Hints**: Full type annotations for better IDE support
- **Code Quality**: Pre-commit hooks, black formatting, flake8 linting

### Changed
- Updated version from 0.9.0 to 1.0.0
- Improved README with comprehensive quick start guide
- Enhanced error messages and validation throughout
- Optimized performance for large networks

### Fixed
- Crosslink attribute naming consistency (fiber_i/fiber_j)
- Periodic boundary condition dimension handling
- GMSH import compatibility
- Various edge cases in generators and simulations

### Performance
- Taichi GPU acceleration provides 10-100x speedup
- Optimized spatial hashing for contact detection
- Efficient sparse matrix operations in FEM solvers
- Parallel ensemble generation

### Documentation
- Comprehensive README with examples for all major features
- 8 integration examples demonstrating complete workflows
- API documentation for all modules
- Inline code documentation

## [0.9.0] - 2026-07-05

### Added
- GPU acceleration with Taichi
- Fiber-fiber contact detection
- Progressive damage simulation
- 3D visualization with matplotlib and pyvista

## [0.8.0] - 2026-07-05

### Added
- Dynamic Mechanical Analysis (DMA)
- Configuration system for reproducibility
- Advanced crosslink models
- Statistical ensemble generation

## [0.7.0] - 2026-07-05

### Added
- Nonlinear FEM with multiple constitutive models
- Fluid flow and acoustic simulation
- Enhanced I/O formats

## [0.6.0] - 2026-07-05

### Added
- Machine learning feature extraction
- Property prediction
- Dataset generation utilities

## [0.5.0] - 2026-07-05

### Added
- Thermal simulation
- Electromagnetic simulation
- Unit system support

## [0.4.0] - 2026-07-05

### Added
- Advanced generators (Voronoi, electrospun, biomimetic)
- Advanced analysis tools
- I/O format extensions

## [0.3.0] - 2026-07-05

### Added
- Chiral structure generators
- Woven structure generators
- Hierarchical network generators

## [0.2.0] - 2026-07-05

### Added
- Ordered lattice generators
- Basic FEM simulation
- Simple visualization

## [0.1.0] - 2026-07-05

### Added
- Initial release
- Basic random network generation
- Simple analysis tools
- JSON I/O

[1.0.0]: https://github.com/GellmanSparrowS/fibernet/releases/tag/v1.0.0
[0.9.0]: https://github.com/GellmanSparrowS/fibernet/releases/tag/v0.9.0
[0.8.0]: https://github.com/GellmanSparrowS/fibernet/releases/tag/v0.8.0
[0.7.0]: https://github.com/GellmanSparrowS/fibernet/releases/tag/v0.7.0
[0.6.0]: https://github.com/GellmanSparrowS/fibernet/releases/tag/v0.6.0
[0.5.0]: https://github.com/GellmanSparrowS/fibernet/releases/tag/v0.5.0
[0.4.0]: https://github.com/GellmanSparrowS/fibernet/releases/tag/v0.4.0
[0.3.0]: https://github.com/GellmanSparrowS/fibernet/releases/tag/v0.3.0
[0.2.0]: https://github.com/GellmanSparrowS/fibernet/releases/tag/v0.2.0
[0.1.0]: https://github.com/GellmanSparrowS/fibernet/releases/tag/v0.1.0

## [1.1.0] - 2026-07-05

### Added
- **Coupled Multi-Physics Simulation** (`fibernet/sim/coupled.py`)
  - Thermo-mechanical solver with steady-state and transient modes
  - Electro-mechanical solver for piezoelectric materials
  - Temperature-dependent mechanical properties
  - Mechanical heating from plastic dissipation
  - Convenience function `run_thermo_mechanical_analysis()`

- **Fracture Mechanics Module** (`fibernet/sim/fracture_mechanics.py`)
  - Crack propagation simulation with energy-based criterion
  - J-integral calculation
  - Stress intensity factors (Mode I and Mode II)
  - Maximum circumferential stress criterion for crack growth direction
  - Energy release rate computation
  - Fracture toughness characterization

- **CI/CD Pipeline** (`.github/workflows/ci.yml`)
  - Automated testing on push and pull requests
  - Multi-platform testing (Linux, macOS, Windows)
  - Multi-version Python testing (3.9-3.12)
  - Code quality checks (black, flake8)
  - Automatic PyPI publishing on release

- **Enhanced PyPI Packaging**
  - Added metadata (authors, keywords, classifiers)
  - Added project URLs (homepage, documentation, repository, issues)
  - Added setuptools package discovery configuration

- **Tests**
  - 16 new tests for coupled multi-physics and fracture mechanics
  - Total: 381 tests (all passing)

### Changed
- Version bump to 1.1.0
- Updated README version reference

### Documentation
- Added coupled multi-physics examples to README
- Added fracture mechanics usage examples

## [1.2.0] - 2026-07-05

### Added
- **Percolation Analysis Module** (`fibernet/analysis/percolation.py`)
  - PercolationAnalyzer for connectivity analysis
  - Cluster size distribution and statistics
  - Percolation threshold detection
  - Correlation length computation
  - Backbone identification
  - Effective conductivity estimation using percolation theory
  - Percolating path finding algorithms
  - `estimate_percolation_threshold()` utility function

- **Jupyter Notebook Tutorials**
  - `01_getting_started.ipynb` - Introduction to FiberNet basics
  - `02_mechanical_simulation.ipynb` - FEM analysis and stress-strain curves
  - Interactive examples for network generation, visualization, and analysis

- **Tests**
  - 8 new tests for percolation analysis
  - Total: 389 tests (all passing)

### Changed
- Version bump to 1.2.0

### Documentation
- Added percolation analysis documentation
- Enhanced README with percolation examples

## [1.3.0] - 2026-07-05

### Added
- **Rheology Module** (`fibernet/sim/rheology.py`)
  - FiberSuspensionRheology for suspension rheology
  - Effective viscosity computation (Batchelor theory)
  - Normal stress differences (Dinh-Armstrong model)
  - Jeffery orbit computation for single fibers
  - Orientation tensor evolution (Folgar-Tucker equation)
  - Shear flow sweep and transient shear simulations
  - Intrinsic viscosity and dilute limit utilities

- **Interactive Plotly Visualization** (`fibernet/viz/plotly_viz.py`)
  - Interactive 3D visualization with rotation, zoom, hover
  - Stress field visualization with colorbars
  - Network comparison side-by-side
  - HTML export for web sharing
  - Configurable colors, opacity, and sizes

- **ML Tutorial Notebook**
  - `03_machine_learning.ipynb` - ML workflows with FiberNet
  - Feature extraction and property prediction
  - GNN integration examples
  - Structure-property visualization

### Changed
- Version bump to 1.3.0
- Updated README with rheology and plotly examples

### Documentation
- Added ML tutorial notebook
- Added rheology module documentation
- Added plotly visualization guide

## [1.4.0] - 2026-07-05

### Added
- **Damage Mechanics and Fatigue Module** (`fibernet/sim/damage.py`)
  - DamageMechanicsSolver for continuum damage mechanics
  - Progressive failure simulation under monotonic loading
  - Fiber breakage tracking with damage evolution laws
  - Residual stiffness and strength computation
  - FatigueSolver for fatigue life prediction
  - S-N curve generation (Basquin's law)
  - Miner's rule for variable amplitude loading
  - Damage tolerance analysis utility
  - FatigueResult and ProgressiveFailureResult data classes

- **Tests**
  - 12 new tests for damage mechanics and fatigue
  - Total: 401 tests (all passing)

### Changed
- Version bump to 1.4.0
- Fixed numpy 2.0 compatibility (np.trapz → np.trapezoid)

### Documentation
- Added damage mechanics module documentation
- Updated README with fatigue analysis examples

## [1.5.0] - 2026-07-05

### Added
- **Multi-scale Modeling Framework** (`fibernet/sim/multiscale.py`)
  - HomogenizationSolver for effective property computation
  - RVEAnalyzer for representative volume element analysis
  - Elastic property homogenization (orientation averaging)
  - Thermal property homogenization
  - Periodic boundary condition application
  - Effective stiffness tensor computation
  - RVE size convergence study utilities
  - `compute_effective_properties()` convenience function
  - `estimate_rve_size()` for convergence analysis
  - HomogenizedProperties and RVEResult data classes

- **Tests**
  - 10 new tests for multi-scale modeling
  - Total: 411 tests (all passing)

### Changed
- Version bump to 1.5.0

### Documentation
- Added multi-scale modeling documentation
- Updated README with homogenization examples
