.. FiberNet documentation master file

Welcome to FiberNet's Documentation
====================================

**FiberNet** is a comprehensive Python toolkit for fiber network structure
generation, simulation, and analysis. It provides tools for researchers in
materials science, biomechanics, polymer physics, and composites engineering.

.. note::
   This documentation covers version |version| (release |release|).

Quick Start
-----------

.. code-block:: python

   import fibernet as fn

   # Create a random 2D fiber network
   net = fn.create('random_2d', num_fibers=100, fiber_length=10.0, 
                   box_size=(30, 30), seed=42)

   # Analyze the network
   stats = fn.analyze(net)
   print(f"Order: {stats['nematic_order']:.3f}")

   # Run a mechanical simulation
   result = fn.simulate_mechanics(net, strain=0.01)

   # Visualize
   fn.plot(net)

Installation
------------

.. code-block:: bash

   pip install fibernet                    # Basic (numpy + scipy only)
   pip install fibernet[full]              # All optional features
   pip install fibernet[viz]               # + matplotlib/pyvista
   pip install fibernet[graph]             # + networkx
   pip install fibernet[ml]                # + scikit-learn/tqdm
   pip install fibernet[accel]             # + taichi (GPU)

Features Overview
-----------------

.. list-table::
   :header-rows: 1

   * - Category
     - Capabilities
   * - Generation
     - 50+ generators: random, ordered, chiral, woven, hierarchical, biomimetic
   * - Simulation
     - FEM, dynamics, fracture, damage, thermal, electromagnetic, acoustic, fluid
   * - Analysis
     - Morphology, topology, percolation, multi-scale homogenization
   * - ML
     - Feature extraction, GNN, property prediction
   * - I/O
     - JSON, LAMMPS, VTK, GMSH, PDB, XYZ, HDF5

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api/index
   tutorials
   changelog
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
