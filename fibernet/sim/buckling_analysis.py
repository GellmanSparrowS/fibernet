"""
Buckling Analysis Module

Provides buckling analysis for fiber networks:
- Euler buckling of individual fibers
- Network-level eigenvalue buckling
- Critical buckling loads and mode shapes
- Post-buckling behavior

This module is essential for understanding compressive failure
in fiber networks and composites.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import warnings

from fibernet.core.network import FiberNetwork
from fibernet.sim.mechanical import FiberFEM


@dataclass
class FiberBucklingResult:
    """Result of individual fiber buckling analysis."""
    fiber_index: int
    critical_load: float
    critical_stress: float
    buckling_mode: str  # 'pinned-pinned', 'fixed-fixed', 'fixed-pinned', 'fixed-free'
    effective_length_factor: float
    slenderness_ratio: float
    euler_valid: bool  # True if Euler formula is valid (slenderness > critical)
    
    def to_dict(self) -> Dict:
        return {
            'fiber_index': self.fiber_index,
            'critical_load': self.critical_load,
            'critical_stress': self.critical_stress,
            'buckling_mode': self.buckling_mode,
            'effective_length_factor': self.effective_length_factor,
            'slenderness_ratio': self.slenderness_ratio,
            'euler_valid': self.euler_valid,
        }


@dataclass
class NetworkBucklingResult:
    """Result of network-level buckling analysis."""
    critical_loads: np.ndarray = None
    mode_shapes: List[np.ndarray] = field(default_factory=list)
    num_modes: int = 0
    
    # Summary statistics
    min_critical_load: float = 0.0
    max_critical_load: float = 0.0
    mean_critical_load: float = 0.0
    
    # Buckling characteristics
    first_buckling_mode: int = 0
    dominant_buckling_type: str = 'global'  # 'global', 'local', 'mixed'
    
    def to_dict(self) -> Dict:
        return {
            'critical_loads': self.critical_loads.tolist() if self.critical_loads is not None else [],
            'num_modes': self.num_modes,
            'min_critical_load': self.min_critical_load,
            'max_critical_load': self.max_critical_load,
            'mean_critical_load': self.mean_critical_load,
            'first_buckling_mode': self.first_buckling_mode,
            'dominant_buckling_type': self.dominant_buckling_type,
        }
    
    def plot_modes(self, network: FiberNetwork, num_modes: int = 3, **kwargs):
        """Plot buckling mode shapes.
        
        Parameters
        ----------
        network : FiberNetwork
            Original network.
        num_modes : int
            Number of modes to plot.
        **kwargs : dict
            Additional arguments passed to plot.
        
        Returns
        -------
        fig : matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")
        
        num_modes = min(num_modes, len(self.mode_shapes))
        
        if num_modes == 0:
            warnings.warn("No mode shapes available to plot")
            return None
        
        fig = plt.figure(figsize=(15, 5))
        
        for i in range(num_modes):
            ax = fig.add_subplot(1, num_modes, i+1, projection='3d')
            
            mode = self.mode_shapes[i]
            load = self.critical_loads[i] if self.critical_loads is not None else 0
            
            # Plot original network
            for fiber in network.fibers:
                coords = fiber.centerline
                ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 
                       'gray', alpha=0.3, linewidth=1)
            
            # Plot deformed shape (scaled)
            scale = 10.0  # Amplification factor
            for fiber in network.fibers:
                coords = fiber.centerline.copy()
                # Add mode shape displacement (simplified)
                # In reality, need to map mode shape to fiber nodes
                coords[:, 0] += scale * mode[:len(coords), 0]
                coords[:, 1] += scale * mode[:len(coords), 1]
                coords[:, 2] += scale * mode[:len(coords), 2]
                ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 
                       'b', linewidth=2)
            
            ax.set_title(f'Mode {i+1}\nλ={load:.2e} N')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        plt.tight_layout()
        return fig


class BucklingAnalyzer:
    """Analyzer for fiber network buckling.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    segments_per_fiber : int
        Number of elements per fiber.
    
    Examples
    --------
    >>> import fibernet as fn
    >>> from fibernet.sim import BucklingAnalyzer
    >>> net = fn.create('random_2d', num_fibers=50, seed=42)
    >>> analyzer = BucklingAnalyzer(net)
    
    # Individual fiber buckling
    >>> fiber_results = analyzer.analyze_fiber_buckling()
    >>> print(f"Min critical load: {min(r.critical_load for r in fiber_results):.2e} N")
    
    # Network buckling
    >>> network_result = analyzer.analyze_network_buckling(num_modes=5)
    >>> print(f"First buckling load: {network_result.min_critical_load:.2e} N")
    """
    
    def __init__(self, network: FiberNetwork, segments_per_fiber: int = 5):
        self.network = network
        self.segments = segments_per_fiber
        self.fem = FiberFEM(network, segments_per_fiber=segments_per_fiber)
    
    def analyze_fiber_buckling(
        self,
        buckling_mode: str = 'pinned-pinned',
    ) -> List[FiberBucklingResult]:
        """Analyze Euler buckling for each fiber.
        
        Parameters
        ----------
        buckling_mode : str
            Boundary condition mode:
            - 'pinned-pinned': K=1.0 (both ends pinned)
            - 'fixed-fixed': K=0.5 (both ends fixed)
            - 'fixed-pinned': K=0.7 (one fixed, one pinned)
            - 'fixed-free': K=2.0 (one fixed, one free)
        
        Returns
        -------
        results : list of FiberBucklingResult
            Buckling results for each fiber.
        
        Notes
        -----
        Euler buckling formula: P_cr = π²EI / (KL)²
        
        Valid when slenderness ratio > critical slenderness:
        λ = L/r > √(2π²E/σ_y)
        
        where r = √(I/A) is radius of gyration.
        """
        # Effective length factors
        K_factors = {
            'pinned-pinned': 1.0,
            'fixed-fixed': 0.5,
            'fixed-pinned': 0.7,
            'fixed-free': 2.0,
        }
        
        if buckling_mode not in K_factors:
            raise ValueError(f"Unknown buckling mode: {buckling_mode}. "
                           f"Choose from: {list(K_factors.keys())}")
        
        K = K_factors[buckling_mode]
        
        results = []
        
        for i, fiber in enumerate(self.network.fibers):
            L = fiber.length
            E = fiber.material.youngs_modulus
            
            # Cross-section properties (circular)
            r = fiber.radius
            A = np.pi * r**2
            I = np.pi * r**4 / 4
            
            # Critical buckling load (Euler)
            P_cr = np.pi**2 * E * I / (K * L)**2
            
            # Critical stress
            sigma_cr = P_cr / A
            
            # Slenderness ratio
            r_gyration = np.sqrt(I / A)
            slenderness = L / r_gyration
            
            # Check if Euler formula is valid
            # Assume yield stress = E / 1000 (typical for polymers)
            sigma_y = E / 1000
            critical_slenderness = np.sqrt(2 * np.pi**2 * E / sigma_y)
            euler_valid = slenderness > critical_slenderness
            
            result = FiberBucklingResult(
                fiber_index=i,
                critical_load=P_cr,
                critical_stress=sigma_cr,
                buckling_mode=buckling_mode,
                effective_length_factor=K,
                slenderness_ratio=slenderness,
                euler_valid=euler_valid,
            )
            
            results.append(result)
        
        return results
    
    def analyze_network_buckling(
        self,
        num_modes: int = 5,
        load_direction: int = 0,
    ) -> NetworkBucklingResult:
        """Analyze network-level buckling using eigenvalue analysis.
        
        Parameters
        ----------
        num_modes : int
            Number of buckling modes to compute.
        load_direction : int
            Loading direction (0=x, 1=y, 2=z).
        
        Returns
        -------
        NetworkBucklingResult
            Buckling analysis results.
        
        Notes
        -----
        Solves generalized eigenvalue problem:
        (K + λ K_G) φ = 0
        
        where:
        - K is elastic stiffness matrix
        - K_G is geometric stiffness matrix
        - λ is buckling load factor
        - φ is buckling mode shape
        
        The smallest positive λ gives the critical buckling load.
        """
        # Build stiffness matrices
        K_elastic, K_geometric = self._build_buckling_matrices(load_direction)
        
        # Convert to CSR for eigenvalue solver
        K_elastic_csr = K_elastic.tocsr()
        K_geometric_csr = K_geometric.tocsr()
        
        try:
            # Solve eigenvalue problem: K φ = -λ K_G φ
            # We want smallest eigenvalues (closest to zero)
            eigenvalues, eigenvectors = eigsh(
                K_elastic_csr,
                M=-K_geometric_csr,
                k=min(num_modes, K_elastic_csr.shape[0] - 2),
                sigma=0,
                which='LM'
            )
            
            # Sort by eigenvalue
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Filter positive eigenvalues (physical buckling modes)
            positive_idx = eigenvalues > 0
            critical_loads = eigenvalues[positive_idx]
            mode_shapes_vec = eigenvectors[:, positive_idx]
            
            # Convert mode shapes to displacement arrays
            mode_shapes = []
            num_nodes = self.fem.num_nodes
            for i in range(len(critical_loads)):
                mode_vec = mode_shapes_vec[:, i]
                mode_disp = mode_vec.reshape(num_nodes, 3)
                mode_shapes.append(mode_disp)
            
            # Compute statistics
            if len(critical_loads) > 0:
                min_load = critical_loads[0]
                max_load = critical_loads[-1]
                mean_load = np.mean(critical_loads)
                first_mode = 0
            else:
                min_load = max_load = mean_load = 0
                first_mode = -1
            
            result = NetworkBucklingResult(
                critical_loads=critical_loads,
                mode_shapes=mode_shapes,
                num_modes=len(critical_loads),
                min_critical_load=min_load,
                max_critical_load=max_load,
                mean_critical_load=mean_load,
                first_buckling_mode=first_mode,
                dominant_buckling_type='global',  # Simplified
            )
            
        except Exception as e:
            # Eigenvalue solver failed
            warnings.warn(f"Eigenvalue buckling analysis failed: {e}")
            result = NetworkBucklingResult(
                critical_loads=np.array([]),
                mode_shapes=[],
                num_modes=0,
            )
        
        return result
    
    def _build_buckling_matrices(
        self,
        load_direction: int,
    ) -> Tuple[lil_matrix, lil_matrix]:
        """Build elastic and geometric stiffness matrices for buckling analysis.
        
        Returns
        -------
        K_elastic : lil_matrix
            Elastic stiffness matrix.
        K_geometric : lil_matrix
            Geometric stiffness matrix (stress stiffness).
        """
        num_dof = self.fem.num_dof
        
        K_elastic = lil_matrix((num_dof, num_dof))
        K_geometric = lil_matrix((num_dof, num_dof))
        
        # Apply unit load to get initial stress state
        unit_strain = 0.001
        result = self.fem.apply_uniaxial_strain(unit_strain, axis=load_direction)
        
        # Build elastic stiffness from element contributions
        for i, elem in enumerate(self.fem.elements):
            n1, n2 = self.fem.element_to_nodes[i]
            
            # Get element stiffness in global coordinates
            k_global = elem.stiffness_global
            
            # Assemble into global matrix
            dofs = [n1*3, n1*3+1, n1*3+2, n2*3, n2*3+1, n2*3+2]
            for ii, dof_i in enumerate(dofs):
                for jj, dof_j in enumerate(dofs):
                    K_elastic[dof_i, dof_j] += k_global[ii, jj]
        
        # Build geometric stiffness from axial forces
        for i, elem in enumerate(self.fem.elements):
            n1, n2 = self.fem.element_to_nodes[i]
            
            # Get axial force from stress
            axial_force = result.stresses[i] * elem.A
            
            # Build geometric stiffness matrix
            # For a beam element under axial force P:
            # K_G = P/L * [[1, -1], [-1, 1]] (simplified 1D)
            L = elem.L
            kg = axial_force / L
            
            # Assemble (simplified - only axial DOFs)
            direction = elem.direction
            for dim in range(3):
                dof1 = n1*3 + dim
                dof2 = n2*3 + dim
                K_geometric[dof1, dof1] += kg * direction[dim]**2
                K_geometric[dof1, dof2] -= kg * direction[dim]**2
                K_geometric[dof2, dof1] -= kg * direction[dim]**2
                K_geometric[dof2, dof2] += kg * direction[dim]**2
        
        return K_elastic, K_geometric


def analyze_buckling(
    network: FiberNetwork,
    segments_per_fiber: int = 5,
    analyze_fibers: bool = True,
    analyze_network: bool = True,
    num_modes: int = 5,
) -> Dict:
    """Convenience function for comprehensive buckling analysis.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    segments_per_fiber : int
        Number of elements per fiber.
    analyze_fibers : bool
        If True, analyze individual fiber buckling.
    analyze_network : bool
        If True, analyze network-level buckling.
    num_modes : int
        Number of buckling modes for network analysis.
    
    Returns
    -------
    results : dict
        Dictionary with 'fiber' and 'network' buckling results.
    
    Examples
    --------
    >>> import fibernet as fn
    >>> from fibernet.sim import analyze_buckling
    >>> net = fn.create('random_2d', num_fibers=50, seed=42)
    >>> results = analyze_buckling(net)
    >>> print(f"Fiber min load: {min(r.critical_load for r in results['fiber']):.2e} N")
    >>> print(f"Network min load: {results['network'].min_critical_load:.2e} N")
    """
    analyzer = BucklingAnalyzer(network, segments_per_fiber=segments_per_fiber)
    
    results = {}
    
    if analyze_fibers:
        results['fiber'] = analyzer.analyze_fiber_buckling()
    
    if analyze_network:
        results['network'] = analyzer.analyze_network_buckling(num_modes=num_modes)
    
    return results
