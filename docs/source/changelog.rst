Changelog
=========

v0.6.0 (2026-07-04)
-------------------

**New Features:**

- **Batch Simulation** (``utils/batch.py``): Run simulations on multiple networks in parallel or sequentially
- **NetworkX Integration**: Convert networks to graphs for advanced topological analysis
- **Network Validation**: ``net.validate()`` for integrity checks
- **Convenient Plotting**: ``net.plot()`` and ``net.plot_statistics()`` with smart kwargs filtering
- **Pandas I/O**: Convert networks to DataFrames for analysis
- **3D Generators**: ``oriented_random_3d``, ``random_curved_fibers_3d``
- **Multi-Physics Coupling**: Thermo-mechanical and electro-mechanical solvers
- **Parametric Studies**: Systematic parameter sweeps with Monte Carlo analysis

**Improvements:**

- Deep copy utilities for safe object duplication
- Robust error handling throughout simulation modules
- Custom exception hierarchy for clear error messages
- Fixed singular matrix warnings in mechanical/EM solvers
- Fixed PBC handling for 2D networks

**Testing:** 248 tests passing

v0.5.0 (2026-07-04)
-------------------

- Copy utilities, parametric study tools, multi-physics coupling
- Error handling module, Jupyter tutorial
- 234 tests passing

v0.4.0 (2026-07-04)
-------------------

- Specialized generators, validation, benchmarks, packaging
- Integration tests, comprehensive documentation
- 202 tests passing

v0.3.0 (2026-07-04)
-------------------

- Initial generators, mechanical simulation, basic analysis
- 100+ tests passing
