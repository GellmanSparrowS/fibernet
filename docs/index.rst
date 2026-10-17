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

   graph = fn.pattern_2d(unit="honeycomb", box=(10, 10), grid=(2, 2),
                         radius=0.05, seed=23)
   result = fn.simulate(graph, backend="fem", mode="stretch", strain=1.01)
   print(graph.num_nodes, graph.num_edges, result.max_displacement)

Here ``strain=1.01`` means a stretch ratio of 1.01 (1% extension).
Follow :doc:`quickstart` to save a figure and reload the simulation result.
Read :doc:`getting_started` before choosing a different mechanical model.

Installation
------------

.. code-block:: bash

   python -m pip install fibernet            # NumPy, SciPy, NetworkX, Matplotlib
   python -m pip install "fibernet[viz]"      # Optional PyVista rendering
   python -m pip install "fibernet[ml]"       # Optional ML dependencies

The Python library and the FiberScope desktop application have separate
installation steps. See :doc:`getting_started` for source installation,
optional extras, and the Windows desktop download.

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
   quickstart
   api/index
   tutorials
   changelog
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
