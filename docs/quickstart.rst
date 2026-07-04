Quick Start Guide
=================

This guide will help you get started with FiberNet in just a few minutes.

Installation
------------

.. code-block:: bash

    pip install fibernet

Or install from source:

.. code-block:: bash

    git clone https://github.com/yourusername/fibernet.git
    cd fibernet
    pip install -e .

Basic Usage
-----------

1. Generate a Network
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fibernet import gen
    
    # Random 2D network
    net = gen.random_straight_2d(num_fibers=100, fiber_length=10.0, seed=42)
    
    # Square lattice
    net = gen.square_lattice_2d(spacing=2.0, grid_size=(10, 10))
    
    # Honeycomb
    net = gen.honeycomb_lattice_2d(cell_size=2.0, grid_size=(8, 8))
    
    # 3D random network
    net = gen.random_straight_3d(num_fibers=100, fiber_length=8.0, seed=42)

2. Analyze the Network
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fibernet.analysis import (
        TopologyAnalyzer, MorphologyAnalyzer
    )
    from fibernet.analysis.spatial import compute_spatial_statistics
    
    # Topology
    topo = TopologyAnalyzer(net)
    topo.analyze()
    
    # Morphology
    morph = MorphologyAnalyzer(net)
    S = morph.nematic_order_parameter()
    
    # Comprehensive statistics
    stats = compute_spatial_statistics(net)
    print(f"Nematic order: {stats['nematic_order']:.3f}")

3. Run Simulations
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fibernet.sim import FiberFEM
    
    # Mechanical simulation
    fem = FiberFEM(net)
    E = fem.effective_modulus()
    print(f"Effective modulus: {E:.2e} Pa")

4. Use Pre-defined Materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fibernet.materials import get_material
    
    # Get material properties
    steel = get_material('steel')
    print(f"Steel E = {steel.youngs_modulus/1e9:.1f} GPa")
    
    # Available materials
    from fibernet.materials import list_materials
    print(list_materials())

5. Compare Networks
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fibernet.analysis.comparison import network_similarity
    
    net1 = gen.random_straight_2d(num_fibers=80, seed=42)
    net2 = gen.square_lattice_2d(spacing=2.0, grid_size=(8, 8))
    
    sim = network_similarity(net1, net2)
    print(f"Similarity: {sim:.3f}")

6. Export Results
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fibernet.io import export_vtk, export_json
    from fibernet.io.mesh_export import export_stl, export_obj
    
    # Export for visualization
    export_vtk(net, 'network.vtk')
    
    # Export for external tools
    export_json(net, 'network.json')
    
    # Export as mesh for FEM solvers
    export_stl(net, 'network.stl')
    export_obj(net, 'network.obj')

7. Visualize
~~~~~~~~~~~~

.. code-block:: python

    from fibernet.visualization import NetworkVisualizer
    
    viz = NetworkVisualizer(net)
    viz.plot_2d()
    viz.save('network.png')

Advanced Features
-----------------

Parameter Sweeps
~~~~~~~~~~~~~~~~

.. code-block:: python

    from fibernet.doe import DesignOfExperiments
    
    def compute_modulus(net):
        fem = FiberFEM(net)
        return {'modulus': fem.effective_modulus()}
    
    params = {
        'num_fibers': [50, 80, 110],
        'fiber_length': [6.0, 10.0, 14.0],
    }
    
    doe = DesignOfExperiments(gen.random_straight_2d, {'seed': 42}, compute_modulus)
    results = doe.grid_search(params)

Effective Properties
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fibernet.analysis.homogenization import compute_effective_properties
    
    props = compute_effective_properties(net)
    print(f"E_x = {props['elastic']['E_x']:.2e} Pa")
    print(f"nu = {props['elastic']['nu']:.3f}")

Unit Conversions
~~~~~~~~~~~~~~~~

.. code-block:: python

    from fibernet.units import convert_length, convert_pressure
    
    # Convert units
    length_mm = convert_length(1.0, 'm', 'mm')  # 1000.0
    pressure_gpa = convert_pressure(100.0, 'MPa', 'GPa')  # 0.1

Network Types
-------------

FiberNet supports many network types:

**Disordered:**
- Random straight (2D/3D)
- Random curved
- Poisson line process
- Voronoi tessellation

**Ordered:**
- Square lattice
- Triangular lattice
- Honeycomb
- Kagome
- Diamond
- Cubic

**Special:**
- Chiral metamaterials
- Woven/braided
- Hierarchical
- Biomimetic (collagen, fibrin, cellulose)
- CNT networks
- Fractal (Sierpinski, Koch, tree, Hilbert)
- Gradient (density, property, multi-zone)

Next Steps
----------

- Browse the `examples/` directory for more use cases
- Read the full `API documentation <api/index.html>`_
- Check out the `research case studies <examples/research_case_study.html>`_

Getting Help
------------

- Report issues on `GitHub <https://github.com/yourusername/fibernet/issues>`_
- Check the `FAQ <faq.html>`_
- Join the discussion on `GitHub Discussions <https://github.com/yourusername/fibernet/discussions>`_

