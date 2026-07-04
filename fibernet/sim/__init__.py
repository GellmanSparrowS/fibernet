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

# Periodic boundary conditions
from .periodic import (
    PeriodicBoundary, create_periodic_network, 
    apply_periodic_strain, homogenize_properties
)
__all__.extend([
    "PeriodicBoundary", "create_periodic_network",
    "apply_periodic_strain", "homogenize_properties"
])

# Coupled multi-physics
from .coupled import (
    ThermoMechanicalSolver, ElectroMechanicalSolver,
    run_thermo_mechanical_analysis
)
__all__.extend([
    "ThermoMechanicalSolver", "ElectroMechanicalSolver",
    "run_thermo_mechanical_analysis"
])

# Fracture mechanics
from .fracture_mechanics import (
    CrackPropagationSolver, FractureResult, CrackTip,
    compute_energy_release_rate, compute_fracture_toughness
)
__all__.extend([
    "CrackPropagationSolver", "FractureResult", "CrackTip",
    "compute_energy_release_rate", "compute_fracture_toughness"
])

# Rheology
from .rheology import (
    FiberSuspensionRheology, RheologyResult, JefferyOrbit,
    compute_intrinsic_viscosity, compute_dilute_limit_viscosity
)
__all__.extend([
    "FiberSuspensionRheology", "RheologyResult", "JefferyOrbit",
    "compute_intrinsic_viscosity", "compute_dilute_limit_viscosity"
])

# Damage mechanics and fatigue
from .damage import (
    DamageMechanicsSolver, FatigueSolver,
    DamageState, FatigueResult, ProgressiveFailureResult,
    compute_damage_tolerance
)
__all__.extend([
    "DamageMechanicsSolver", "FatigueSolver",
    "DamageState", "FatigueResult", "ProgressiveFailureResult",
    "compute_damage_tolerance"
])

# Multi-scale modeling
from .multiscale import (
    HomogenizationSolver, RVEAnalyzer,
    HomogenizedProperties, RVEResult,
    compute_effective_properties, estimate_rve_size
)
__all__.extend([
    "HomogenizationSolver", "RVEAnalyzer",
    "HomogenizedProperties", "RVEResult",
    "compute_effective_properties", "estimate_rve_size"
])

# Incremental nonlinear FEM for stress-strain curves
from .incremental_fem import (
    IncrementalFEM, IncrementalResult,
    compute_stress_strain_curve
)
__all__.extend([
    "IncrementalFEM", "IncrementalResult",
    "compute_stress_strain_curve"
])

# Buckling analysis
from .buckling_analysis import (
    BucklingAnalyzer, FiberBucklingResult, NetworkBucklingResult,
    analyze_buckling
)
__all__.extend([
    "BucklingAnalyzer", "FiberBucklingResult", "NetworkBucklingResult",
    "analyze_buckling"
])

# Permeability and diffusion
from .permeability import (
    PermeabilitySolver, PermeabilityResult,
    DiffusionSolver, DiffusionResult,
    compute_permeability, compute_diffusion
)
__all__.extend([
    "PermeabilitySolver", "PermeabilityResult",
    "DiffusionSolver", "DiffusionResult",
    "compute_permeability", "compute_diffusion"
])

# Uncertainty quantification
from .uncertainty import (
    EnsembleResult, monte_carlo_ensemble,
    sensitivity_analysis, convergence_study
)
__all__.extend([
    "EnsembleResult", "monte_carlo_ensemble",
    "sensitivity_analysis", "convergence_study"
])

# Coefficient of Thermal Expansion
from .cte import CTEAnalyzer, CTEResult, compute_cte
__all__.extend(["CTEAnalyzer", "CTEResult", "compute_cte"])

# Molecular dynamics
from .molecular_dynamics import (
    FiberMDSolver, MDParameters, MDTrajectory,
    run_fiber_md
)
__all__.extend([
    "FiberMDSolver", "MDParameters", "MDTrajectory",
    "run_fiber_md"
])
