FiberNet Documentation
======================

**FiberNet** is a comprehensive Python toolkit for fiber network structure generation, simulation, and analysis.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   tutorials/index
   api/index
   examples/index
   contributing

Overview
--------

FiberNet provides tools for:

- **Generation**: 40+ generators for 2D/3D fiber networks (ordered, disordered, chiral, woven, hierarchical, biomimetic)
- **Transformation**: Mirror, rotate, scale, merge, tile, and pattern operations
- **Simulation**: Mechanical (linear/nonlinear), dynamics, fracture, thermal, electromagnetic, and multi-physics coupling
- **Analysis**: Topology, morphology, spectral analysis, pore distribution, anisotropy
- **I/O**: Interoperability with LAMMPS, VTK, GMSH, PDB, XYZ formats
- **Acceleration**: Taichi CPU/GPU parallel computing

Key Features
------------

- **Nonlinear Mechanics**: Hyperelastic (Neo-Hookean, Mooney-Rivlin, Arruda-Boyce), plasticity, viscoelasticity
- **Periodic Boundary Conditions**: Minimum image convention, RDF computation
- **Advanced Crosslinks**: Rigid, spring, breakable, friction, bonded models
- **Unit Systems**: SI, CGS, micro, nano, molecular units with automatic conversion
- **Visualization**: Stress/temperature/displacement fields, cross-sections, animations

Installation
------------

.. code-block:: bash

   pip install fibernet
   
   # With all optional dependencies
   pip install fibernet[full]
   
   # For development
   pip install fibernet[dev]

Quick Start
-----------

.. code-block:: python

   import fibernet as fn
   from fibernet import gen
   
   # Generate a random 2D fiber network
   net = gen.random_straight_2d(num_fibers=100, fiber_length=15, box_size=(50, 50))
   
   # Analyze structure
   results = fn.analyze(net)
   print(f"Nematic order: {results['nematic_order']:.3f}")
   
   # Run mechanical simulation
   results = fn.simulate_mechanics(net, strain=0.01, axis=0)
   print(f"Effective modulus: {results['modulus']:.2e} Pa")
   
   # Export to VTK for visualization
   fn.export(net, "network.vtk")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
