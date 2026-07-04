# Changelog

All notable changes to FiberNet are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.1] - 2026-07-05

### Fixed
- Made optional dependencies truly optional (networkx, tqdm, matplotlib, pyvista, etc.)
- Fixed `api.analyze()` to gracefully handle missing topology analysis
- Added proper error messages when optional dependencies are not installed
- Fixed pyproject.toml license format to modern SPDX expression
- Added comprehensive integration tests

### Added
- Comprehensive demo script (`examples/comprehensive_demo.py`)
- ML integration example (`examples/ml_example.py`)
- Comprehensive Sphinx documentation with autodoc
- 10 acoustic simulation tests

## [1.5.0] - 2026-07-04

### Added
- **Multi-scale modeling framework**: RVE-based homogenization
- **Multi-scale RVE generator**: Create representative volume elements
- **Homogenization solver**: Compute effective mechanical properties
- Electromagnetic simulation tests
- Comprehensive integration test suite
- Improved README with high-level API examples

## [1.4.0] - 2026-07-04

### Added
- **Damage mechanics module**: Progressive damage under loading
- **Fatigue simulation**: Cycle-by-cycle damage accumulation
- **Residual stiffness tracking**: Monitor degradation

## [1.3.0] - 2026-07-03

### Added
- **Rheology module**: Fiber suspension rheology
- **Jeffery orbit solver**: Single fiber dynamics in shear flow
- **Folgar-Tucker model**: Orientation evolution
- Plotly interactive visualization
- ML integration tutorial

## [1.2.0] - 2026-07-03

### Added
- **Percolation analysis**: Cluster detection and connectivity
- **Multi-point constraint (MPC)**: Advanced boundary conditions
- Jupyter notebook tutorials
- Enhanced documentation

## [1.1.0] - 2026-07-02

### Added
- **Coupled multi-physics**: Thermo-mechanical coupling
- **Fracture mechanics**: LEFM crack propagation
- **CI/CD pipeline**: Automated testing on GitHub Actions
- Expanded generator library

## [1.0.0] - 2026-07-02

### Added
- **First stable release**
- Core fiber network data structures (Fiber, FiberNetwork, Material, Crosslinks)
- 50+ network generators (random, ordered, chiral, woven, hierarchical, biomimetic)
- FEM simulation (linear and nonlinear beam elements)
- Dynamic simulation (explicit time integration)
- Thermal and electromagnetic solvers
- Acoustic wave propagation
- Fluid flow (Darcy, pore network)
- Morphology and topology analysis
- Machine learning integration (feature extraction, GNN)
- I/O support for JSON, LAMMPS, VTK, GMSH, PDB, XYZ, HDF5
- Visualization with matplotlib and pyvista
- GPU acceleration with Taichi
- Reproducible experiment framework (YAML config, ensemble generation)
- Unit system support (SI, CGS, micro, nano, molecular)
- Comprehensive test suite (430+ tests)
- Full API documentation with Sphinx

## [Unreleased]

### Planned
- Advanced contact mechanics with friction
- Multi-scale domain decomposition
- Real-time visualization
- Web-based interactive interface
- Performance profiling and optimization
