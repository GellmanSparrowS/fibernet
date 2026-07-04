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

# Fluid flow simulation
from fibernet.sim.fluid import DarcySolver, PoreNetworkModel, FluidResult

# Acoustic wave propagation
from fibernet.sim.acoustic import AcousticSolver, AcousticResult

# Update __all__
__all__ += [
    # Fluid
    "DarcySolver", "PoreNetworkModel", "FluidResult",
    # Acoustic
    "AcousticSolver", "AcousticResult",
]

# Multi-physics coupled simulations
from .coupled import (
    ThermoMechanicalSolver,
    ElectroMechanicalSolver,
    MultiPhysicsSolver,
    CoupledResult
)

# Viscoelastic models
from .viscoelastic import (
    ViscoelasticResult, MaxwellModel, KelvinVoigtModel,
    StandardLinearSolid, GeneralizedMaxwell
)
__all__.extend([
    "ViscoelasticResult", "MaxwellModel", "KelvinVoigtModel",
    "StandardLinearSolid", "GeneralizedMaxwell"
])

# Dynamic Mechanical Analysis
from .dma import (
    DMAResult, frequency_sweep, temperature_sweep, master_curve
)
__all__.extend([
    "DMAResult", "frequency_sweep", "temperature_sweep", "master_curve"
])

# Taichi-accelerated FEM solver
from .accelerated import TaichiFEMSolver
__all__.extend([
    "TaichiFEMSolver"
])
