"""
Multi-physics coupling module for fiber networks.

Implements:
- Thermo-mechanical coupling (thermal expansion, temperature-dependent properties)
- Electro-mechanical coupling (piezoresistive effects)
- Sequential coupling framework
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from fibernet.core.network import FiberNetwork
from fibernet.sim.mechanical import FiberFEM, MechanicalResult
from fibernet.sim.thermal import ThermalSolver, ThermalResult
from fibernet.sim.electromagnetic import EMSolver, EMResult


@dataclass
class CoupledResult:
    """Results from coupled simulation."""
    mechanical: MechanicalResult = None
    thermal: ThermalResult = None
    electromagnetic: EMResult = None
    coupling_history: List[Dict] = None


class ThermoMechanical:
    """Thermo-mechanical coupling: thermal expansion effects on mechanics."""
    
    def __init__(self, network: FiberNetwork):
        self.network = network
    
    def solve(
        self,
        delta_T: float = 100.0,
        mechanical_load: Optional[np.ndarray] = None,
        fixed_nodes: Optional[List[int]] = None,
    ) -> CoupledResult:
        """Solve coupled thermo-mechanical problem.
        
        Parameters
        ----------
        delta_T : float
            Temperature change from reference.
        """
        thermal_solver = ThermalSolver(self.network)
        thermal_result = thermal_solver.solve_steady_state(
            T_hot=delta_T, T_cold=0.0,
        )
        
        fem = FiberFEM(self.network)
        if fem.num_dof == 0:
            return CoupledResult(thermal=thermal_result)
        
        thermal_forces = np.zeros(fem.num_dof)
        
        if thermal_result.temperatures is not None:
            for e_idx, (elem, (ni, nj)) in enumerate(zip(fem.elements, fem.element_to_nodes)):
                T_i = thermal_result.temperatures[min(ni, len(thermal_result.temperatures) - 1)]
                T_j = thermal_result.temperatures[min(nj, len(thermal_result.temperatures) - 1)]
                T_avg = 0.5 * (T_i + T_j)
                
                alpha = self.network.fibers[min(e_idx // 5, self.network.num_fibers - 1)].material.thermal_expansion
                if alpha is None:
                    alpha = 1e-5
                
                strain_th = alpha * T_avg
                force_th = elem.E * elem.A * strain_th
                
                d = elem.direction
                for dim in range(3):
                    thermal_forces[ni * 6 + dim] -= force_th * d[dim]
                    thermal_forces[nj * 6 + dim] += force_th * d[dim]
        
        if mechanical_load is not None:
            thermal_forces += mechanical_load
        
        mech_result = fem.solve_static(
            forces=thermal_forces,
            fixed_nodes=fixed_nodes,
        )
        
        return CoupledResult(
            mechanical=mech_result,
            thermal=thermal_result,
        )


class PiezoResistive:
    """Electro-mechanical coupling: strain-dependent resistivity."""
    
    def __init__(self, network: FiberNetwork, gauge_factor: float = 2.0):
        self.network = network
        self.gauge_factor = gauge_factor
    
    def resistance_vs_strain(
        self,
        strains: np.ndarray = None,
        axis: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute relative resistance change under strain.
        
        Parameters
        ----------
        strains : np.ndarray
            Strain values to evaluate.
        axis : int
            Loading direction.
        
        Returns
        -------
        strains : np.ndarray
        dR_R0 : np.ndarray
            Relative resistance change.
        """
        if strains is None:
            strains = np.linspace(0, 0.05, 20)
        
        em_solver = EMSolver(self.network)
        base_result = em_solver.solve_conductivity(axis=axis)
        sigma_0 = base_result.effective_conductivity
        
        if sigma_0 < 1e-12:
            return strains, np.zeros_like(strains)
        
        dR_R0 = np.zeros(len(strains))
        
        for i, eps in enumerate(strains):
            if eps == 0:
                dR_R0[i] = 0.0
                continue
            
            dR_R0[i] = self.gauge_factor * eps
        
        return strains, dR_R0
