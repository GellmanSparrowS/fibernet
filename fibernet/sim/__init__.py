"""
Simulation engines for fiber networks.

Submodules:
- mechanical: Beam FEM for structural mechanics
- dynamics: MD/Brownian dynamics
- fracture: Progressive failure and damage
- thermal: Heat conduction
- electromagnetic: Electrical conductivity and percolation
- coupling: Multi-physics (thermo-mechanical, piezoresistive)
- accelerated: Taichi-accelerated computations
- nonlinear: Hyperelastic, plasticity, viscoelasticity, large deformation
"""

from fibernet.sim.mechanical import FiberFEM, MechanicalResult, BeamElement, stress_strain_curve
from fibernet.sim.dynamics import FiberDynamics, DynamicsResult
from fibernet.sim.fracture import FiberFracture, FractureResult
from fibernet.sim.thermal import ThermalSolver, ThermalResult
from fibernet.sim.electromagnetic import EMSolver, EMResult
from fibernet.sim.coupling import ThermoMechanical, PiezoResistive, CoupledResult
from fibernet.sim.nonlinear import (
    NonlinearFEM, NonlinearResult,
    ConstitutiveModel, LinearElastic, BilinearPlasticity, PowerLawHardening,
    HyperelasticNeoHookean, HyperelasticMooneyRivlin, ArrudaBoyce,
    ViscoelasticModel, MaxwellModel, KelvinVoigtModel, StandardLinearSolid,
)

try:
    from fibernet.sim.accelerated import TaichiEngine, AcceleratedResult
except ImportError:
    pass

__all__ = [
    # Linear mechanics
    "FiberFEM", "MechanicalResult", "BeamElement", "stress_strain_curve",
    # Dynamics
    "FiberDynamics", "DynamicsResult",
    # Fracture
    "FiberFracture", "FractureResult",
    # Thermal
    "ThermalSolver", "ThermalResult",
    # Electromagnetic
    "EMSolver", "EMResult",
    # Coupling
    "ThermoMechanical", "PiezoResistive", "CoupledResult",
    # Nonlinear mechanics
    "NonlinearFEM", "NonlinearResult",
    "ConstitutiveModel", "LinearElastic", "BilinearPlasticity", "PowerLawHardening",
    "HyperelasticNeoHookean", "HyperelasticMooneyRivlin", "ArrudaBoyce",
    "ViscoelasticModel", "MaxwellModel", "KelvinVoigtModel", "StandardLinearSolid",
    # Acceleration
    "TaichiEngine", "AcceleratedResult",
]
