FiberNet Documentation
=====================

.. image:: https://img.shields.io/badge/version-0.6.0-blue
   :target: https://github.com/GellmanSparrowS/fibernet

FiberNet is a comprehensive toolkit for fiber network structure research.
It provides tools for generating, analyzing, and simulating 2D/3D fiber networks.

**Features:**

- **Generation**: 26+ generators (disordered, ordered, chiral, woven, hierarchical, specialized)
- **Simulation**: Mechanical, dynamics, fracture, thermal, electromagnetic, nonlinear
- **Analysis**: Topology, morphology, network statistics
- **Visualization**: 2D/3D rendering, animations, statistical plots
- **I/O**: LAMMPS, VTK, PDB, GMSH, XYZ, Pandas DataFrame
- **Integration**: NetworkX, Pandas, Matplotlib, PyVista

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/basic_workflow
   tutorials/mechanical_simulation
   tutorials/machine_learning

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/generators
   api/simulation
   api/analysis
   api/io

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
