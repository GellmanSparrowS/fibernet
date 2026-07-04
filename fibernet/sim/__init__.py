"""
Simulation engines for fiber networks.

Submodules:
- mechanical: Beam FEM for structural mechanics
- dynamics: MD/Brownian dynamics
- fracture: Progressive failure and damage
- thermal: Heat conduction
- electromagnetic: Electrical conductivity and percolation
- coupling: Multi-physics (thermo-mechanical, piezoresistive)
"""

from fibernet.sim.mechanical import FiberFEM, MechanicalResult, BeamElement, stress_strain_curve
from fibernet.sim.dynamics import FiberDynamics, DynamicsResult
from fibernet.sim.fracture import FiberFracture, FractureResult
from fibernet.sim.thermal import ThermalSolver, ThermalResult
from fibernet.sim.electromagnetic import EMSolver, EMResult
from fibernet.sim.coupling import ThermoMechanical, PiezoResistive, CoupledResult

__all__ = [
    "FiberFEM", "MechanicalResult", "BeamElement", "stress_strain_curve",
    "FiberDynamics", "DynamicsResult",
    "FiberFracture", "FractureResult",
    "ThermalSolver", "ThermalResult",
    "EMSolver", "EMResult",
    "ThermoMechanical", "PiezoResistive", "CoupledResult",
]
