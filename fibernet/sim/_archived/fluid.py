"""
Fluid Flow Simulation for Fiber Networks

Provides:
- Darcy's law permeability computation
- Pore network modeling for fluid transport
- Tortuosity and porosity calculations
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from fibernet.core.network import FiberNetwork


@dataclass
class FluidResult:
    """Container for fluid flow simulation results."""
    permeability: float = 0.0
    permeability_tensor: np.ndarray = None
    porosity: float = 0.0
    tortuosity: float = 1.0
    velocities: np.ndarray = None
    pressures: np.ndarray = None
    flow_rates: np.ndarray = None
    
    def hydraulic_conductivity(self, viscosity: float = 1e-3) -> float:
        """Compute hydraulic conductivity from permeability.
        
        Parameters
        ----------
        viscosity : float
            Dynamic viscosity (Pa·s). Default is water.
        """
        return self.permeability / viscosity


class DarcySolver:
    """Solver for Darcy's law permeability in fiber networks.
    
    Uses Kozeny-Carman relation and direct simulation to compute
    effective permeability of the fiber network.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    """
    
    def __init__(self, network: FiberNetwork):
        self.network = network
    
    def compute_porosity(self, box_size: np.ndarray = None) -> float:
        """Compute porosity of the fiber network.
        
        Parameters
        ----------
        box_size : np.ndarray, optional
            Box dimensions. Defaults to network bounding box.
        """
        if box_size is None:
            bb_min, bb_max = self.network.bounding_box()
            box_size = bb_max - bb_min
        
        # For 2D networks, z-dimension may be 0; use area instead
        if self.network.dimension == 2:
            box_volume = box_size[0] * box_size[1]
            if box_volume < 1e-15:
                box_volume = 1.0
        else:
            box_volume = np.prod(box_size)
            if box_volume < 1e-15:
                box_volume = 1.0
        
        # Compute fiber volume
        fiber_volume = 0.0
        for fiber in self.network.fibers:
            length = fiber.length
            radius = fiber.radius
            if self.network.dimension == 2:
                # 2D: area instead of volume
                fiber_volume += length * 2 * radius
            else:
                fiber_volume += np.pi * radius**2 * length
        
        porosity = 1.0 - fiber_volume / box_volume
        return max(0.0, min(1.0, porosity))
    
    def compute_tortuosity(self, axis: int = 0) -> float:
        """Compute tortuosity along an axis.
        
        Tortuosity = (actual path length) / (straight-line distance)
        
        Parameters
        ----------
        axis : int
            Direction axis (0=x, 1=y, 2=z).
        """
        tortuosities = []
        
        for fiber in self.network.fibers:
            pts = fiber.centerline
            if len(pts) < 2:
                continue
            
            # Actual path length
            path_length = fiber.length
            
            # Straight-line distance along axis
            straight_length = abs(pts[-1, axis] - pts[0, axis])
            
            if straight_length > 1e-10:
                tort = path_length / straight_length
                tortuosities.append(tort)
        
        if not tortuosities:
            return 1.0
        
        return np.mean(tortuosities)
    
    def kozeny_carman_permeability(
        self,
        porosity: float = None,
        k_carman: float = 5.0,
    ) -> float:
        """Compute permeability using Kozeny-Carman relation.
        
        K = (φ³ / (k_C * (1-φ)² * S²))
        
        where φ is porosity, k_C is Kozeny constant, and S is specific surface area.
        
        Parameters
        ----------
        porosity : float, optional
            Porosity. Computed if not provided.
        k_carman : float
            Kozeny constant (typically 4-5 for random packings).
        """
        if porosity is None:
            porosity = self.compute_porosity()
        
        if porosity >= 1.0:
            return 1e10  # Infinite permeability
        
        if porosity <= 0.0:
            return 0.0
        
        # Compute specific surface area (surface area per unit volume)
        total_surface = 0.0
        box_volume = 1.0
        
        bb_min, bb_max = self.network.bounding_box()
        box_size = bb_max - bb_min
        
        if self.network.dimension == 2:
            box_volume = box_size[0] * box_size[1]
            for fiber in self.network.fibers:
                # 2D: perimeter * length
                total_surface += 2 * fiber.length
        else:
            box_volume = np.prod(box_size)
            for fiber in self.network.fibers:
                # 3D: 2πrL
                total_surface += 2 * np.pi * fiber.radius * fiber.length
        
        S = total_surface / box_volume
        
        if S < 1e-15:
            return 1e10
        
        # Kozeny-Carman
        K = (porosity**3) / (k_carman * (1 - porosity)**2 * S**2)
        
        return float(K)
    
    def solve_flow(
        self,
        pressure_gradient: np.ndarray,
        viscosity: float = 1e-3,
        axis: int = 0,
    ) -> FluidResult:
        """Solve for flow through the fiber network.
        
        Uses simplified pore network model to compute flow.
        
        Parameters
        ----------
        pressure_gradient : np.ndarray
            Applied pressure gradient [Pa/m].
        viscosity : float
            Fluid viscosity [Pa·s].
        axis : int
            Flow direction.
        """
        porosity = self.compute_porosity()
        tortuosity = self.compute_tortuosity(axis=axis)
        
        # Compute permeability
        K = self.kozeny_carman_permeability(porosity)
        
        # Darcy's law: q = -K/μ * ∇P
        dp = np.linalg.norm(pressure_gradient)
        darcy_velocity = K * dp / viscosity
        
        # Compute permeability tensor (simplified)
        K_tensor = K * np.eye(3)
        
        return FluidResult(
            permeability=K,
            permeability_tensor=K_tensor,
            porosity=porosity,
            tortuosity=tortuosity,
            velocities=np.array([darcy_velocity, 0, 0]),
            pressures=None,
            flow_rates=None,
        )
    
    def compute_permeability_tensor(
        self,
        viscosity: float = 1e-3,
    ) -> np.ndarray:
        """Compute full permeability tensor.
        
        Parameters
        ----------
        viscosity : float
            Fluid viscosity.
        """
        K_tensor = np.zeros((3, 3))
        
        for i in range(3):
            grad = np.zeros(3)
            grad[i] = 1.0  # Unit gradient
            
            result = self.solve_flow(grad, viscosity, axis=i)
            
            for j in range(3):
                K_tensor[j, i] = result.permeability
        
        return K_tensor


class PoreNetworkModel:
    """Pore network model for fluid transport.
    
    Models the void space between fibers as a network of
    pores connected by throats.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    num_pores : int
        Number of pore nodes to generate.
    """
    
    def __init__(self, network: FiberNetwork, num_pores: int = 100):
        self.network = network
        self.num_pores = num_pores
        self._build_pore_network()
    
    def _build_pore_network(self):
        """Build pore network from fiber structure."""
        bb_min, bb_max = self.network.bounding_box()
        
        # Generate random pore positions in void space
        self.pore_positions = []
        self.pore_radii = []
        
        np.random.seed(42)
        
        attempts = 0
        max_attempts = self.num_pores * 10
        
        while len(self.pore_positions) < self.num_pores and attempts < max_attempts:
            # Random position in box
            pos = bb_min + np.random.rand(3) * (bb_max - bb_min)
            
            # Check if position is in void space (not inside fiber)
            in_void = True
            min_dist = 1e10
            
            for fiber in self.network.fibers:
                # Distance to fiber centerline
                pts = fiber.centerline
                for pt in pts:
                    dist = np.linalg.norm(pos - pt)
                    min_dist = min(min_dist, dist)
                    
                    if dist < fiber.radius:
                        in_void = False
                        break
                if not in_void:
                    break
            
            if in_void:
                self.pore_positions.append(pos)
                # Pore radius proportional to distance from nearest fiber
                self.pore_radii.append(max(min_dist * 0.5, 0.1))
            
            attempts += 1
        
        self.pore_positions = np.array(self.pore_positions) if self.pore_positions else np.array([]).reshape(0, 3)
        self.pore_radii = np.array(self.pore_radii) if self.pore_radii else np.array([])
        self.num_pores = len(self.pore_positions)
        
        # Build connectivity (Delaunay-like)
        self._build_connectivity()
    
    def _build_connectivity(self):
        """Build pore connectivity based on proximity."""
        from scipy.spatial import Delaunay
        
        if self.num_pores < 4:
            self.throats = []
            self.throat_radii = []
            return
        
        try:
            # Use Delaunay triangulation for connectivity
            tri = Delaunay(self.pore_positions)
            
            self.throats = []
            self.throat_radii = []
            
            for simplex in tri.simplices:
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        p1 = simplex[i]
                        p2 = simplex[j]
                        
                        if p1 < self.num_pores and p2 < self.num_pores:
                            self.throats.append((p1, p2))
                            
                            # Throat radius is minimum of connected pores
                            throat_r = min(self.pore_radii[p1], self.pore_radii[p2])
                            self.throat_radii.append(throat_r)
            
            # Remove duplicates
            self.throats = list(set([tuple(sorted(t)) for t in self.throats]))
            self.throat_radii = np.array(self.throat_radii[:len(self.throats)])
            
        except Exception:
            self.throats = []
            self.throat_radii = []
    
    def solve_pressure(
        self,
        inlet_pores: List[int],
        outlet_pores: List[int],
        inlet_pressure: float = 1.0,
        outlet_pressure: float = 0.0,
        viscosity: float = 1e-3,
    ) -> np.ndarray:
        """Solve pressure field in pore network.
        
        Parameters
        ----------
        inlet_pores : List[int]
            Indices of inlet pores.
        outlet_pores : List[int]
            Indices of outlet pores.
        inlet_pressure : float
            Pressure at inlet.
        outlet_pressure : float
            Pressure at outlet.
        viscosity : float
            Fluid viscosity.
        
        Returns
        -------
        np.ndarray
            Pressure at each pore.
        """
        if self.num_pores < 2 or len(self.throats) < 1:
            return np.zeros(self.num_pores)
        
        # Build conductance matrix
        G = lil_matrix((self.num_pores, self.num_pores))
        
        for idx, (p1, p2) in enumerate(self.throats):
            if idx >= len(self.throat_radii):
                break
            
            throat_r = self.throat_radii[idx]
            length = np.linalg.norm(self.pore_positions[p1] - self.pore_positions[p2])
            
            if length < 1e-10:
                continue
            
            # Hagen-Poiseuille conductance: g = πr⁴ / (8μL)
            conductance = np.pi * throat_r**4 / (8 * viscosity * length)
            
            G[p1, p2] = conductance
            G[p2, p1] = conductance
        
        G = G.tocsr()
        
        # Laplacian: L = D - G
        D = np.array(G.sum(axis=1)).flatten()
        L = csr_matrix((D, (range(self.num_pores), range(self.num_pores)))) - G
        
        # Boundary conditions
        fixed_pores = set(inlet_pores) | set(outlet_pores)
        free_pores = [i for i in range(self.num_pores) if i not in fixed_pores]
        
        if not free_pores:
            pressures = np.zeros(self.num_pores)
            for p in inlet_pores:
                if p < self.num_pores:
                    pressures[p] = inlet_pressure
            for p in outlet_pores:
                if p < self.num_pores:
                    pressures[p] = outlet_pressure
            return pressures
        
        # Solve
        L_free = L[np.ix_(free_pores, free_pores)]
        
        rhs = np.zeros(len(free_pores))
        
        # Contribution from fixed pores
        for i, pore in enumerate(free_pores):
            for fixed in inlet_pores:
                if fixed < self.num_pores:
                    rhs[i] += L[pore, fixed] * inlet_pressure
            for fixed in outlet_pores:
                if fixed < self.num_pores:
                    rhs[i] += L[pore, fixed] * outlet_pressure
        
        rhs = -rhs
        
        try:
            P_free = spsolve(L_free, rhs)
        except Exception:
            P_free = np.zeros(len(free_pores))
        
        # Assemble full pressure field
        pressures = np.zeros(self.num_pores)
        for i, pore in enumerate(free_pores):
            pressures[pore] = P_free[i]
        for p in inlet_pores:
            if p < self.num_pores:
                pressures[p] = inlet_pressure
        for p in outlet_pores:
            if p < self.num_pores:
                pressures[p] = outlet_pressure
        
        return pressures
    
    def compute_permeability(
        self,
        axis: int = 0,
        viscosity: float = 1e-3,
    ) -> float:
        """Compute effective permeability from pore network.
        
        Parameters
        ----------
        axis : int
            Flow direction.
        viscosity : float
            Fluid viscosity.
        """
        if self.num_pores < 2:
            return 0.0
        
        # Identify inlet and outlet pores
        coords = self.pore_positions[:, axis]
        threshold = 0.1 * (coords.max() - coords.min())
        
        inlet_pores = np.where(coords <= coords.min() + threshold)[0].tolist()
        outlet_pores = np.where(coords >= coords.max() - threshold)[0].tolist()
        
        if not inlet_pores or not outlet_pores:
            return 0.0
        
        # Solve pressure
        pressures = self.solve_pressure(
            inlet_pores, outlet_pores,
            inlet_pressure=1.0, outlet_pressure=0.0,
            viscosity=viscosity,
        )
        
        # Compute total flow
        total_flow = 0.0
        
        for idx, (p1, p2) in enumerate(self.throats):
            if idx >= len(self.throat_radii):
                break
            
            throat_r = self.throat_radii[idx]
            length = np.linalg.norm(self.pore_positions[p1] - self.pore_positions[p2])
            
            if length < 1e-10:
                continue
            
            dp = pressures[p1] - pressures[p2]
            
            # Flow along axis
            axis_vec = self.pore_positions[p2] - self.pore_positions[p1]
            axis_flow = np.pi * throat_r**4 * dp / (8 * viscosity * length)
            axis_flow *= abs(axis_vec[axis]) / (length + 1e-10)
            
            total_flow += abs(axis_flow)
        
        # Permeability: K = Q * μ * L / (A * ΔP)
        bb_min, bb_max = self.network.bounding_box()
        L = bb_max[axis] - bb_min[axis]
        
        box_size = bb_max - bb_min
        if self.network.dimension == 2:
            if axis == 0:
                A = box_size[1]
            else:
                A = box_size[0]
        else:
            if axis == 0:
                A = box_size[1] * box_size[2]
            elif axis == 1:
                A = box_size[0] * box_size[2]
            else:
                A = box_size[0] * box_size[1]
        
        if A < 1e-10 or L < 1e-10:
            return 0.0
        
        K = total_flow * viscosity * L / A
        
        return float(K)
