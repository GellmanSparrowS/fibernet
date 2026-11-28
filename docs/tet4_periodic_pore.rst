Periodic pore flow with Tet4 geometry
=====================================

The optional pore-flow solver computes steady, incompressible Newtonian
Stokes flow through a bounded, connected pore. Supply the **fluid domain** as
a :class:`Tet4Bundle`, not the solid fiber geometry. The two end planes normal
to x must have matching triangulations under a translation; the other
boundary faces use no-slip conditions. The input bundle's material entries
keep geometric labels stable but do not set fluid properties. Supply dynamic
viscosity separately in Pa s.

On Python 3.10 or newer:

.. code-block:: console

   pip install "fibernet[pore-flow]"
   python -m fibernet.examples.tet4_periodic_pore
   python -m fibernet.examples.tet4_solid_to_pore_flow --matched-force

The first example checks flow through an empty duct and a three-path
scaffold. The second compares two stiffnesses under equal signed grip
forces by solving solid deformation, transferring the wall motion, and
recomputing the flow. Neither example needs external meshes.

.. code-block:: python

   from fibernet.sim.tet4_pore import solve_x_periodic_stokes

   flow = solve_x_periodic_stokes(
       pore_bundle,
       viscosity_Pa_s=1e-3,
       pressure_gradient_Pa_per_m=500.0,
   )
   print(flow.axial_flow_m3_s, flow.cap_flow_m3_s)

A positive driver is minus the axial pressure derivative, and gives positive
flow along +x. Both cap flow values use the same +x direction. The returned
bulk and cap values are separate finite-element readouts, not interchangeable
when assessing mesh convergence. The result includes a power check, pressure
residual, end balance and relative cap/bulk difference.

``solve_matched_pore_geometry`` applies the same viscosity and axial pressure
gradient to two supplied geometries when given ``pressure_gradient_Pa_per_m``.
To hold the end-to-end pressure difference instead, pass ``pressure_drop_Pa``:
each geometry uses that drop divided by its own axial period in metres.
These controls differ when loading changes the period. The matched solve
checks stable IDs and cell volumes; it does not compute wall motion or
apply fluid forces to the solid.

The reference scope is P2 velocity and P1 pressure on straight-sided
tetrahedra, small meshes, a periodically driven connected pore, and a
bounded sparse direct solve. The workload estimate is a pre-assembly
screening check, not a guaranteed ceiling on total process memory.
