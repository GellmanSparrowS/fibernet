Examples
========

Real-world examples of FiberNet usage.

Basic Usage
-----------

.. code-block:: python

    import fibernet as fn
    from fibernet import gen

    # Generate a random 2D fiber network
    net = gen.random_straight_2d(
        num_fibers=100,
        fiber_length=10,
        box_size=(50, 50),
        seed=42
    )

    # Quick visualization
    net.plot()

    # Statistical summary
    print(net.describe())

    # Validate before simulation
    result = net.validate()
    if result['valid']:
        print("Network is ready for simulation!")

Parametric Study
----------------

.. code-block:: python

    from fibernet.utils.parametric import parametric_sweep
    from fibernet.analysis import MorphologyAnalyzer

    def analyze(net):
        morph = MorphologyAnalyzer(net)
        return {
            'nematic': morph.nematic_order_parameter(),
            'porosity': morph.porosity()
        }

    params, metrics = parametric_sweep(
        {'num_fibers': [50, 100, 200, 500]},
        lambda **kw: gen.random_straight_2d(**kw, fiber_length=10, box_size=(50, 50), seed=42),
        analyze
    )

Mechanical Simulation
---------------------

.. code-block:: python

    from fibernet.sim import FiberFEM

    # Create ordered lattice
    net = gen.square_lattice_2d(spacing=5, grid_size=(10, 10))

    # Run FEM simulation
    fem = FiberFEM(net, segments_per_fiber=5)
    result = fem.apply_uniaxial_strain(strain=0.01, axis=0)

    print(f"Energy: {result.energy:.4e} J")
    print(f"Max displacement: {result.max_displacement():.4e} m")

Export to VTK (Paraview)
-------------------------

.. code-block:: python

    from fibernet.io import to_vtk

    net = gen.random_straight_3d(num_fibers=50, fiber_length=8, box_size=(25, 25, 25))
    to_vtk(net, 'network.vtk')
    # Open in Paraview for 3D visualization

Pandas Integration
------------------

.. code-block:: python

    from fibernet.io import to_dataframe, network_summary

    net = gen.random_straight_2d(num_fibers=100, fiber_length=10, box_size=(30, 30), seed=42)

    # Convert to DataFrame for analysis
    df = to_dataframe(net)
    print(df.head())

    # Per-fiber summary
    summary = network_summary(net)
    print(summary.describe())
