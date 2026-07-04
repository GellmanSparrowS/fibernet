"""
Thermal simulation for fiber networks.

Solves steady-state heat conduction through the fiber network.
Uses node-merged conductance matrix for proper connectivity.
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from typing import Optional, List, Dict, Tuple
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
    """Steady-state heat conduction solver for fiber networks.
    
    Parameters
    ----------
    network : FiberNetwork
        The fiber network to simulate.
    """
    
    def __init__(self, network: FiberNetwork):
        self.network = network
        self._build_thermal_mesh()
    
    def _build_thermal_mesh(self):
        """Build thermal mesh with merged nodes at fiber intersections."""
        self.node_positions = []
        self.node_map = {}
        self.elements = []
        
        nid = 0
        
        for f_idx, fiber in enumerate(self.network.fibers):
            pts = fiber.centerline
            fiber_nodes = []
            
            for p_idx, pt in enumerate(pts):
                key = tuple(np.round(pt, 8))
                if key not in self.node_map:
                    self.node_map[key] = nid
                    self.node_positions.append(pt)
                    nid += 1
                fiber_nodes.append(self.node_map[key])
            
            k = fiber.material.thermal_conductivity or 1.0
            A = fiber.cross_section_area
            
            for i in range(len(fiber_nodes) - 1):
                ni = fiber_nodes[i]
                nj = fiber_nodes[i + 1]
                p1 = self.node_positions[ni]
                p2 = self.node_positions[nj]
                L = np.linalg.norm(np.array(p2) - np.array(p1))
                
                if L > 1e-12 and ni != nj:
                    conductance = k * A / L
                    self.elements.append((ni, nj, conductance))
        
        self.num_nodes = nid
        self.num_elements = len(self.elements)
        
        if self.node_positions:
            self.node_positions = np.array(self.node_positions)
        else:
            self.node_positions = np.array([]).reshape(0, 3)
    
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
        crosslink_conductance : float, optional
            Additional thermal conductance at crosslinks.
        """
        if self.num_nodes == 0 or self.num_elements == 0:
            return ThermalResult()
        
        positions = self.node_positions[:, axis]
        pos_min = positions.min()
        pos_max = positions.max()
        L = pos_max - pos_min
        
        if L < 1e-12:
            return ThermalResult()
        
        # Build conductance matrix
        G = lil_matrix((self.num_nodes, self.num_nodes))
        
        for ni, nj, cond in self.elements:
            G[ni, ni] += cond
            G[nj, nj] += cond
            G[ni, nj] -= cond
            G[nj, ni] -= cond
        
        # Add crosslink conductance
        if crosslink_conductance:
            for cl in self.network.crosslinks:
                # Find nodes near crosslink position
                pos = cl.position
                key = tuple(np.round(pos, 8))
                if key in self.node_map:
                    ni = self.node_map[key]
                    # Connect to nearby nodes
                    for nj_key, nj in self.node_map.items():
                        if ni != nj:
                            dist = np.linalg.norm(np.array(nj_key) - np.array(key))
                            if dist < 1.0:
                                G[ni, ni] += crosslink_conductance
                                G[nj, nj] += crosslink_conductance
                                G[ni, nj] -= crosslink_conductance
                                G[nj, ni] -= crosslink_conductance
        
        # Identify boundary nodes
        tol = L * 0.05
        hot_nodes = np.where(positions >= pos_max - tol)[0]
        cold_nodes = np.where(positions <= pos_min + tol)[0]
        
        if len(hot_nodes) == 0 or len(cold_nodes) == 0:
            return ThermalResult()
        
        fixed_nodes = set(hot_nodes.tolist() + cold_nodes.tolist())
        all_dofs = set(range(self.num_nodes))
        free_dofs = sorted(all_dofs - fixed_nodes)
        
        # Set temperatures
        T = np.full(self.num_nodes, (T_hot + T_cold) / 2)
        T[hot_nodes] = T_hot
        T[cold_nodes] = T_cold
        
        # Solve for free nodes
        if len(free_dofs) > 0:
            G_csr = G.tocsr()
            G_free = G_csr[np.ix_(free_dofs, free_dofs)]
            
            # Add small regularization to avoid singular matrix
            G_free += 1e-12 * lil_matrix(np.eye(len(free_dofs))).tocsr()
            
            rhs = np.zeros(len(free_dofs))
            for i, dof in enumerate(free_dofs):
                for j in list(fixed_nodes):
                    if dof < G_csr.shape[0] and j < G_csr.shape[1]:
                        rhs[i] -= G_csr[dof, j] * T[j]
            
            try:
                T_free = spsolve(G_free, rhs)
                for i, dof in enumerate(free_dofs):
                    T[dof] = T_free[i]
            except Exception:
                pass
        
        # Compute heat flux
        Q_in = 0.0
        G_csr = G.tocsr()
        for node in hot_nodes:
            for j in range(self.num_nodes):
                if node < G_csr.shape[0] and j < G_csr.shape[1]:
                    Q_in += abs(G_csr[node, j]) * abs(T[node] - T[j])
        
        # Compute effective conductivity
        bb_min, bb_max = self.network.bounding_box()
        dims = bb_max - bb_min
        
        if self.network.dimension == 2:
            if axis == 0:
                area = dims[1] * 1.0
            elif axis == 1:
                area = dims[0] * 1.0
            else:
                area = dims[0] * dims[1]
        else:
            if axis == 0:
                area = dims[1] * dims[2]
            elif axis == 1:
                area = dims[0] * dims[2]
            else:
                area = dims[0] * dims[1]
        
        if area < 1e-12:
            return ThermalResult()
        
        delta_T = T_hot - T_cold
        if abs(delta_T) < 1e-12:
            return ThermalResult()
        
        k_eff = Q_in * L / (area * delta_T)
        
        return ThermalResult(
            temperatures=T,
            effective_conductivity=float(k_eff),
        )
