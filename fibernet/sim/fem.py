"""BeamFEM has been replaced by TaichiFEMSolver.

Use: from fibernet.sim.accelerated import TaichiFEMSolver
"""
from fibernet.sim.accelerated import TaichiFEMSolver, SimResult as FEMResult
BeamFEM = None  # removed
