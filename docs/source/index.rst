Welcome to FiberNet's documentation!
=====================================

**FiberNet** is a comprehensive Python library for generating, analyzing, and simulating fiber network structures. It provides tools for studying mechanical, thermal, electromagnetic, and multi-physics properties of fiber-based materials.

.. note::
   This documentation is for FiberNet version |version|.

Key Features
------------

- **50+ Network Generators**: From simple random networks to complex biomimetic structures
- **Multi-Physics Simulation**: Mechanical, thermal, electromagnetic, fluid flow, acoustic
- **Advanced Analysis**: Topology, morphology, spectral analysis, statistical tools
- **Machine Learning Integration**: Feature extraction and property prediction
- **High-Performance Computing**: Taichi GPU acceleration support
- **Professional I/O**: LAMMPS, VTK, GMSH, PDB, XYZ format support

Quick Start
-----------

Install FiberNet:

.. code-block:: bash

   pip install fibernet

Generate and analyze a fiber network:

.. code-block:: python

   import fibernet as fn
   from fibernet import gen

   # Generate a random 2D network
   net = gen.random_straight_2d(
       num_fibers=100,
       fiber_length=10.0,
       box_size=(50, 50),
       radius=0.1,
       seed=42
   )

   # Analyze structure
   from fibernet.analysis import MorphologyAnalyzer
   analyzer = MorphologyAnalyzer(net)
   print(f"Nematic order: {analyzer.nematic_order():.3f}")

   # Run mechanical simulation
   from fibernet.sim import MechanicalSolver
   solver = MechanicalSolver(net)
   result = solver.uniaxial_tension(strain=0.01, axis=0)
   print(f"Effective modulus: {result.modulus:.2e} Pa")

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/generators
   api/simulation
   api/analysis
   api/io

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
