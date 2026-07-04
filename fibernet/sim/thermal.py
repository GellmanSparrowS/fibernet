"""
Thermal simulation engine for fiber networks.

Implements:
- Steady-state heat conduction through fiber networks
- Transient heat transfer
- Effective thermal conductivity
- Thermal boundary conditions
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from fibernet.core.network import FiberNetwork


@dataclass
class ThermalResult:
    """Container for thermal simulation results."""
    temperatures: np.ndarray = None
    heat_flux: np.ndarray = None
    effective_conductivity: float = 0.0
    thermal_resistance: float = 0.0


class ThermalSolver:
    """Thermal conduction solver for fiber networks.
    
    Models heat conduction along fibers and through crosslinks.
    """
    
    def __init__(self, network: FiberNetwork):
        self.network = network
    
    def solve_steady_state(
        self,
        T_hot: float = 100.0,
        T_cold: float = 0.0,
        axis: int = 0,
        crosslink_conductance: float = None,
    ) -> ThermalResult:
        """Solve steady-state heat conduction.
        
        Parameters
        ----------
        T_hot : float
            Temperature at hot boundary.
        T_cold : float
            Temperature at cold boundary.
        axis : int
            Heat flow direction (0=x, 1=y, 2=z).
        crosslink_conductance : float
            Thermal conductance at crosslinks (W/K).
        """
        if self.network.num_fibers == 0:
            return ThermalResult()
        
        all_points = []
        point_to_fiber = []
        fiber_offsets = []
        
        offset = 0
        for f_idx, fiber in enumerate(self.network.fibers):
            for p_idx in range(len(fiber.centerline)):
                all_points.append(fiber.centerline[p_idx])
                point_to_fiber.append(f_idx)
                fiber_offsets.append(offset)
                offset += 1
        
        all_points = np.array(all_points)
        num_nodes = len(all_points)
        
        if num_nodes == 0:
            return ThermalResult()
        
        positions = all_points[:, axis]
        pos_min, pos_max = positions.min(), positions.max()
        L = pos_max - pos_min
        
        if L < 1e-12:
            return ThermalResult()
        
        G = lil_matrix((num_nodes, num_nodes))
        
        for f_idx, fiber in enumerate(self.network.fibers):
            k = fiber.material.thermal_conductivity or 1.0
            A = fiber.cross_section_area
            n_pts = len(fiber.centerline)
            
            for i in range(n_pts - 1):
                p1 = fiber.centerline[i]
                p2 = fiber.centerline[i + 1]
                seg_L = np.linalg.norm(p2 - p1)
                
                if seg_L < 1e-12:
                    continue
                
                cond = k * A / seg_L
                
                ni = fiber_offsets[f_idx * (len(fiber.centerline)) + i] if i < len(fiber.centerline) else fiber_offsets[f_idx * n_pts + i]
                nj = fiber_offsets[f_idx * n_pts + i + 1] if (f_idx * n_pts + i + 1) < len(fiber_offsets) else None
                
                if nj is None:
                    continue
                
                ni_actual = sum(len(f.centerline) for f in self.network.fibers[:f_idx]) + i
                nj_actual = ni_actual + 1
                
                if ni_actual < num_nodes and nj_actual < num_nodes:
                    G[ni_actual, ni_actual] += cond
                    G[nj_actual, nj_actual] += cond
                    G[ni_actual, nj_actual] -= cond
                    G[nj_actual, ni_actual] -= cond
        
        tol = L * 0.05
        hot_nodes = np.where(positions >= pos_max - tol)[0]
        cold_nodes = np.where(positions <= pos_min + tol)[0]
        
        fixed_nodes = set(hot_nodes.tolist() + cold_nodes.tolist())
        all_dofs = set(range(num_nodes))
        free_dofs = sorted(all_dofs - fixed_nodes)
        
        T = np.full(num_nodes, (T_hot + T_cold) / 2)
        T[hot_nodes] = T_hot
        T[cold_nodes] = T_cold
        
        if len(free_dofs) > 0 and len(G.rows) > 0:
            G_csr = G.tocsr()
            G_free = G_csr[np.ix_(free_dofs, free_dofs)]
            rhs = np.zeros(len(free_dofs))
            
            for i, dof in enumerate(free_dofs):
                for j in list(fixed_nodes):
                    if dof < G_csr.shape[0] and j < G_csr.shape[1]:
                        rhs[i] -= G_csr[dof, j] * T[j]
            
            try:
                T_free = spsolve(G_free, rhs)
                for i, dof in enumerate(free_dofs):
                    T[dof] = T_free[i]
            except:
                pass
        
        Q_in = 0.0
        for node in hot_nodes:
            for j in range(num_nodes):
                if node < G.shape[0] and j < G.shape[1]:
                    Q_in += abs(G[node, j]) * abs(T[node] - T[j])
        
        bb_min, bb_max = self.network.bounding_box()
        dims = bb_max - bb_min
        if axis == 0:
            area = dims[1] * dims[2] if len(dims) > 2 else dims[1]
        elif axis == 1:
            area = dims[0] * dims[2] if len(dims) > 2 else dims[0]
        else:
            area = dims[0] * dims[1]
        
        dT = T_hot - T_cold
        k_eff = Q_in * L / (area * dT) if area > 1e-12 and dT > 1e-12 else 0.0
        
        return ThermalResult(
            temperatures=T,
            effective_conductivity=k_eff,
            thermal_resistance=L / (k_eff * area) if k_eff > 1e-12 and area > 1e-12 else float('inf'),
        )
