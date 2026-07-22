"""
Electromagnetic simulation engine for fiber networks.

Implements:
- Electrical conductivity/resistivity
- Effective permittivity/permeability
- Simple circuit model for conductive fiber networks
- Percolation analysis
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field
from fibernet.core.network import FiberNetwork


@dataclass
class EMResult:
    """Container for electromagnetic simulation results."""
    potentials: np.ndarray = None
    current_density: np.ndarray = None
    effective_conductivity: float = 0.0
    effective_permittivity: float = 0.0
    percolation_threshold: float = None
    is_percolating: bool = False


class EMSolver:
    """Electromagnetic solver for conductive/dielectric fiber networks."""
    
    def __init__(self, network: FiberNetwork):
        self.network = network
    
    def solve_conductivity(
        self,
        voltage: float = 1.0,
        axis: int = 0,
        contact_resistance: float = 0.0,
    ) -> EMResult:
        """Solve for effective electrical conductivity.
        
        Uses resistor network model where each fiber segment is a resistor.
        """
        if self.network.num_fibers == 0:
            return EMResult()
        
        num_nodes = sum(len(f.centerline) for f in self.network.fibers)
        if num_nodes == 0:
            return EMResult()
        
        all_points = np.vstack([f.centerline for f in self.network.fibers])
        positions = all_points[:, axis]
        pos_min, pos_max = positions.min(), positions.max()
        L = pos_max - pos_min
        
        if L < 1e-12:
            return EMResult()
        
        G = lil_matrix((num_nodes, num_nodes))
        
        node_offset = 0
        for f_idx, fiber in enumerate(self.network.fibers):
            sigma = fiber.material.electrical_conductivity or 0.0
            A = fiber.cross_section_area
            n_pts = len(fiber.centerline)
            
            for i in range(n_pts - 1):
                seg_L = np.linalg.norm(fiber.centerline[i + 1] - fiber.centerline[i])
                if seg_L < 1e-12:
                    continue
                
                cond = sigma * A / seg_L
                
                ni = node_offset + i
                nj = node_offset + i + 1
                
                G[ni, ni] += cond
                G[nj, nj] += cond
                G[ni, nj] -= cond
                G[nj, ni] -= cond
            
            node_offset += n_pts
        
        # Add conductance at crosslink points (connect different fibers)
        if hasattr(self.network, 'crosslinks') and self.network.crosslinks:
            fiber_node_offsets = []
            offset = 0
            for fiber in self.network.fibers:
                fiber_node_offsets.append(offset)
                offset += len(fiber.centerline)
            
            for cl in self.network.crosslinks:
                # Get the two fibers and find nearest nodes
                fi, fj = cl.fiber_i, cl.fiber_j
                if fi >= len(fiber_node_offsets) or fj >= len(fiber_node_offsets):
                    continue
                
                # Find nearest node on each fiber to the crosslink position
                cl_pos = cl.position
                fiber_i = self.network.fibers[fi]
                fiber_j = self.network.fibers[fj]
                
                # Nearest node on fiber i
                dists_i = np.linalg.norm(fiber_i.centerline - cl_pos, axis=1)
                node_i = fiber_node_offsets[fi] + np.argmin(dists_i)
                
                # Nearest node on fiber j
                dists_j = np.linalg.norm(fiber_j.centerline - cl_pos, axis=1)
                node_j = fiber_node_offsets[fj] + np.argmin(dists_j)
                
                if node_i != node_j:
                    # Connect with high conductance (low contact resistance)
                    contact_cond = 1e6  # High conductance at crosslinks
                    if contact_resistance > 0:
                        contact_cond = 1.0 / contact_resistance
                    
                    G[node_i, node_i] += contact_cond
                    G[node_j, node_j] += contact_cond
                    G[node_i, node_j] -= contact_cond
                    G[node_j, node_i] -= contact_cond
        
        tol = L * 0.05
        high_nodes = np.where(positions >= pos_max - tol)[0]
        low_nodes = np.where(positions <= pos_min + tol)[0]
        
        fixed = set(high_nodes.tolist() + low_nodes.tolist())
        free = sorted(set(range(num_nodes)) - fixed)
        
        V = np.zeros(num_nodes)
        V[high_nodes] = voltage
        V[low_nodes] = 0.0
        
        if len(free) > 0:
            G_csr = G.tocsr()
            G_free = G_csr[np.ix_(free, free)]
            rhs = np.zeros(len(free))
            
            for i, dof in enumerate(free):
                for j in list(fixed):
                    rhs[i] -= G_csr[dof, j] * V[j]
            
            # Add regularization for isolated nodes
            from scipy.sparse import diags
            diag = G_free.diagonal()
            isolated = np.abs(diag) < 1e-15
            if np.any(isolated):
                G_free = G_free + diags(np.where(isolated, 1.0, 0.0))
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    V_free = spsolve(G_free, rhs)
                if np.any(np.isnan(V_free)):
                    V_free = np.linalg.lstsq(G_free.toarray(), rhs, rcond=None)[0]
                for i, dof in enumerate(free):
                    V[dof] = V_free[i]
            except:
                pass
        
        I_total = 0.0
        G_csr = G.tocsr()
        for node in high_nodes:
            current = G_csr[node, :] @ V
            if hasattr(current, '__len__'):
                current = current.item() if current.size == 1 else current.sum()
            I_total += abs(float(current))
        
        bb_min, bb_max = self.network.bounding_box()
        dims = bb_max - bb_min
        
        # Handle 2D networks: use average fiber radius as thickness
        if np.any(dims < 1e-10):
            avg_radius = np.mean([f.radius for f in self.network.fibers])
            thickness = avg_radius * 10  # Use 10x average radius as thickness
            dims = np.where(dims < 1e-10, thickness, dims)
        
        if axis == 0:
            area = dims[1] * dims[2] if len(dims) > 2 else dims[1]
        elif axis == 1:
            area = dims[0] * dims[2] if len(dims) > 2 else dims[0]
        else:
            area = dims[0] * dims[1]
        
        sigma_eff = float(I_total * L / (area * voltage)) if area > 1e-12 and voltage > 1e-12 else 0.0
        
        is_perc = I_total > 1e-12
        
        return EMResult(
            potentials=V,
            effective_conductivity=sigma_eff,
            is_percolating=is_perc,
        )
    
    def percolation_analysis(
        self,
        fiber_volumes: np.ndarray = None,
        num_samples: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Analyze electrical percolation threshold.
        
        Returns probability of percolation vs. volume fraction.
        """
        if fiber_volumes is None:
            fiber_volumes = np.linspace(0.001, 0.1, num_samples)
        
        probabilities = np.zeros(len(fiber_volumes))
        
        for i, vol_frac in enumerate(fiber_volumes):
            n_active = max(1, int(self.network.num_fibers * vol_frac / max(self.network.density(), 1e-12)))
            n_active = min(n_active, self.network.num_fibers)
            
            rng = np.random.default_rng(42 + i)
            active = rng.choice(self.network.num_fibers, n_active, replace=False)
            
            sub_net = FiberNetwork(
                fibers=[self.network.fibers[j] for j in active],
                dimension=self.network.dimension,
                box_size=self.network.box_size,
            )
            sub_net.auto_crosslink()
            
            solver = EMSolver(sub_net)
            result = solver.solve_conductivity()
            if result.is_percolating:
                probabilities[i] = 1.0
        
        return fiber_volumes, probabilities
