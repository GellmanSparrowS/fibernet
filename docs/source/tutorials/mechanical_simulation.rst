Mechanical Simulation Tutorial
===============================

Learn how to perform linear elastic FEM analysis of fiber networks.

Setup
-----

.. code-block:: python

   from fibernet import gen
   from fibernet.sim.mechanical import FiberFEM

   # Generate network
   net = gen.random_straight_2d(
       num_fibers=50, fiber_length=12, box_size=(40, 40), seed=42
   )

   # Create FEM solver
   fem = FiberFEM(net, segments_per_fiber=5)
   print(f"Nodes: {fem.num_nodes}, Elements: {fem.num_elements}")

Uniaxial Tension
----------------

Apply uniaxial strain and compute response:

.. code-block:: python

   # Apply 1% strain in x-direction
   result = fem.apply_uniaxial_strain(strain=0.01, axis=0)

   print(f"Strain energy: {result.energy:.4e} J")
   print(f"Max displacement: {result.max_displacement():.4e} m")
   print(f"Max stress: {result.max_stress():.4e} Pa")

Effective Modulus
-----------------

Compute effective Young's modulus:

.. code-block:: python

   # Use small strain for linear regime
   E_x = fem.effective_modulus(strain=0.001, axis=0)
   E_y = fem.effective_modulus(strain=0.001, axis=1)

   print(f"E_x = {E_x:.4e} Pa")
   print(f"E_y = {E_y:.4e} Pa")
   print(f"Anisotropy ratio: {E_x/E_y:.3f}")

Shear Modulus
-------------

.. code-block:: python

   G = fem.shear_modulus(strain=0.001)
   print(f"Shear modulus: {G:.4e} Pa")

Stress-Strain Curve
-------------------

Generate a full stress-strain curve:

.. code-block:: python

   import numpy as np

   strains = np.linspace(0, 0.05, 10)
   stresses = []

   for eps in strains:
       result = fem.apply_uniaxial_strain(strain=eps, axis=0)
       stress = fem.compute_effective_stress(result, axis=0)
       stresses.append(stress)

   # Plot
   import matplotlib.pyplot as plt
   plt.plot(strains, stresses, 'b-o')
   plt.xlabel('Strain')
   plt.ylabel('Stress (Pa)')
   plt.title('Stress-Strain Curve')
   plt.grid(True)
   plt.show()
