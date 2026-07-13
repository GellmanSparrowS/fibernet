"""
Acoustic Wave Propagation Simulation

Provides:
- Wave equation solving on fiber networks
- Sound velocity computation
- Frequency response analysis
- Acoustic band structure calculation
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve, eigsh
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from fibernet.core.network import FiberNetwork


@dataclass
class AcousticResult:
    """Container for acoustic simulation results."""
    frequencies: np.ndarray = None
    modes: np.ndarray = None
    sound_velocity: float = 0.0
    band_structure: Dict = None
    displacement_field: np.ndarray = None
    time_series: np.ndarray = None
    
    def fundamental_frequency(self) -> float:
        """Get the lowest non-zero frequency."""
        if self.frequencies is not None and len(self.frequencies) > 0:
            non_zero = self.frequencies[self.frequencies > 1e-6]
            if len(non_zero) > 0:
                return float(non_zero[0])
        return 0.0
    
    def density_of_states(self, num_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Compute density of states (DOS).
        
        Parameters
        ----------
        num_bins : int
            Number of frequency bins.
        """
        if self.frequencies is None or len(self.frequencies) == 0:
            return np.array([]), np.array([])
        
        freq_min = 0
        freq_max = np.max(self.frequencies) * 1.1
        
        bins = np.linspace(freq_min, freq_max, num_bins + 1)
        dos, bin_edges = np.histogram(self.frequencies, bins=bins)
        
        freq_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        df = bin_edges[1] - bin_edges[0]
        
        dos = dos.astype(float) / (df * len(self.frequencies) + 1e-15)
        
        return freq_centers, dos


class AcousticSolver:
    """Solver for acoustic wave propagation in fiber networks.
    
    Uses finite element method on beam elements to compute
    vibrational modes and acoustic properties.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    segments_per_fiber : int
        Number of elements per fiber.
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        segments_per_fiber: int = 5,
    ):
        self.network = network
        self.segments = segments_per_fiber
        self._build_mesh()
    
    def _build_mesh(self):
        """Build FEM mesh for acoustic analysis."""
        self.elements = []
        self.node_positions = []
        
        node_map = {}
        nid = 0
        
        for fiber in self.network.fibers:
            resampled = fiber.resample(self.segments + 1)
            pts = resampled.centerline
            
            fiber_nodes = []
            for pt in pts:
                key = tuple(np.round(pt, 8))
                if key not in node_map:
                    node_map[key] = nid
                    self.node_positions.append(pt.copy())
                    nid += 1
                fiber_nodes.append(node_map[key])
            
            E = fiber.material.youngs_modulus
            rho = fiber.material.density if fiber.material.density else 1000.0
            A = fiber.cross_section_area
            I = np.pi * fiber.radius**4 / 4
            
            for i in range(len(fiber_nodes) - 1):
                ni = fiber_nodes[i]
                nj = fiber_nodes[i + 1]
                if ni != nj:
                    p1 = np.array(self.node_positions[ni])
                    p2 = np.array(self.node_positions[nj])
                    L = np.linalg.norm(p2 - p1)
                    if L > 1e-12:
                        self.elements.append({
                            'n1': ni, 'n2': nj, 'L': L,
                            'E': E, 'rho': rho, 'A': A, 'I': I,
                        })
        
        self.num_nodes = nid
        self.num_elements = len(self.elements)
        self.num_dof = nid * 3
        
        if self.node_positions:
            self.node_positions = np.array(self.node_positions)
        else:
            self.node_positions = np.array([]).reshape(0, 3)
    
    def _assemble_mass_matrix(self) -> csr_matrix:
        """Assemble global mass matrix."""
        M = lil_matrix((self.num_dof, self.num_dof))
        
        for elem in self.elements:
            n1, n2 = elem['n1'], elem['n2']
            L = elem['L']
            rho = elem['rho']
            A = elem['A']
            
            # Lumped mass: m/2 at each node
            mass = rho * A * L
            mass_per_node = mass / 2.0
            
            for d in range(3):
                M[n1*3+d, n1*3+d] += mass_per_node
                M[n2*3+d, n2*3+d] += mass_per_node
        
        return M.tocsr()
    
    def _assemble_stiffness_matrix(self) -> csr_matrix:
        """Assemble global stiffness matrix."""
        K = lil_matrix((self.num_dof, self.num_dof))
        
        for elem in self.elements:
            n1, n2 = elem['n1'], elem['n2']
            L = elem['L']
            E = elem['E']
            A = elem['A']
            
            p1 = self.node_positions[n1]
            p2 = self.node_positions[n2]
            
            if L < 1e-15:
                continue
            
            # Direction vector
            n_hat = (p2 - p1) / L
            
            # Axial stiffness
            k_axial = E * A / L
            k_local = k_axial * np.outer(n_hat, n_hat)
            
            # Assemble
            dofs = [n1*3, n1*3+1, n1*3+2, n2*3, n2*3+1, n2*3+2]
            k_full = np.zeros((6, 6))
            k_full[0:3, 0:3] = k_local
            k_full[3:6, 3:6] = k_local
            k_full[0:3, 3:6] = -k_local
            k_full[3:6, 0:3] = -k_local
            
            for i in range(6):
                for j in range(6):
                    K[dofs[i], dofs[j]] += k_full[i, j]
        
        return K.tocsr()
    
    def compute_modes(
        self,
        num_modes: int = 20,
        fixed_nodes: Optional[List[int]] = None,
    ) -> AcousticResult:
        """Compute vibrational modes and natural frequencies.
        
        Parameters
        ----------
        num_modes : int
            Number of modes to compute.
        fixed_nodes : List[int], optional
            Fixed node indices (boundary conditions).
        """
        if self.num_nodes == 0:
            return AcousticResult()
        
        K = self._assemble_stiffness_matrix()
        M = self._assemble_mass_matrix()
        
        # Apply boundary conditions
        fixed_dofs = set()
        if fixed_nodes:
            for n in fixed_nodes:
                fixed_dofs.update([n*3+d for d in range(3)])
        
        # For 2D networks, fix z-direction
        if self.network.dimension == 2:
            for n in range(self.num_nodes):
                fixed_dofs.add(n*3+2)
        
        free_dofs = sorted(set(range(self.num_dof)) - fixed_dofs)
        
        if len(free_dofs) < 2:
            return AcousticResult()
        
        K_free = K[np.ix_(free_dofs, free_dofs)]
        M_free = M[np.ix_(free_dofs, free_dofs)]
        
        # Add small regularization
        K_free = K_free + 1e-10 * csr_matrix(np.eye(len(free_dofs)))
        
        # Solve eigenvalue problem: K*φ = ω²*M*φ
        num_modes = min(num_modes, len(free_dofs) - 1)
        
        try:
            # Shift-invert mode for smallest eigenvalues
            eigenvalues, eigenvectors = eigsh(
                K_free, M=M_free, k=num_modes, sigma=0.0,
                which='LM'
            )
            
            # Convert to frequencies: ω = √λ, f = ω/(2π)
            eigenvalues = np.maximum(eigenvalues, 0)
            omega = np.sqrt(eigenvalues)
            frequencies = omega / (2 * np.pi)
            
            # Sort by frequency
            idx = np.argsort(frequencies)
            frequencies = frequencies[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Reconstruct full mode shapes
            modes = np.zeros((self.num_dof, num_modes))
            for i, dof in enumerate(free_dofs):
                modes[dof, :] = eigenvectors[i, :]
            
            return AcousticResult(
                frequencies=frequencies,
                modes=modes,
            )
            
        except Exception as e:
            print(f"Warning: Eigenvalue computation failed: {e}")
            return AcousticResult()
    
    def compute_sound_velocity(
        self,
        mode: int = 0,
    ) -> float:
        """Compute sound velocity from dispersion relation.
        
        Parameters
        ----------
        mode : int
            Mode index (0 for fundamental).
        """
        result = self.compute_modes(num_modes=10)
        
        if result.frequencies is None or len(result.frequencies) == 0:
            return 0.0
        
        # Get fundamental frequency
        f0 = result.fundamental_frequency()
        
        # Estimate wavelength from network size
        bb_min, bb_max = self.network.bounding_box()
        L = np.max(bb_max - bb_min)
        
        # v = f * λ, where λ ≈ 2L for fundamental mode
        wavelength = 2 * L
        velocity = f0 * wavelength
        
        return float(velocity)
    
    def frequency_response(
        self,
        freq_range: Tuple[float, float] = (0, 1000),
        num_points: int = 100,
        damping: float = 0.01,
        force_node: int = 0,
        force_dof: int = 0,
        measure_node: int = -1,
        measure_dof: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute frequency response function (FRF).
        
        Parameters
        ----------
        freq_range : Tuple[float, float]
            Frequency range (f_min, f_max) in Hz.
        num_points : int
            Number of frequency points.
        damping : float
            Damping ratio.
        force_node : int
            Node where force is applied.
        force_dof : int
            DOF for force (0=x, 1=y, 2=z).
        measure_node : int
            Node where response is measured.
        measure_dof : int
            DOF for measurement.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (frequencies, response_magnitude)
        """
        if self.num_nodes == 0:
            return np.array([]), np.array([])
        
        K = self._assemble_stiffness_matrix()
        M = self._assemble_mass_matrix()
        
        # Apply boundary conditions (fix first node)
        fixed_dofs = set([0, 1, 2])
        
        # For 2D
        if self.network.dimension == 2:
            for n in range(self.num_nodes):
                fixed_dofs.add(n*3+2)
        
        free_dofs = sorted(set(range(self.num_dof)) - fixed_dofs)
        
        if len(free_dofs) < 2:
            return np.array([]), np.array([])
        
        K_free = K[np.ix_(free_dofs, free_dofs)].toarray()
        M_free = M[np.ix_(free_dofs, free_dofs)].toarray()
        
        # Add damping: C = 2*ζ*√(K*M) (proportional damping)
        C_free = damping * (K_free + M_free)
        
        frequencies = np.linspace(freq_range[0], freq_range[1], num_points)
        response = np.zeros(num_points)
        
        # Force vector
        F = np.zeros(self.num_dof)
        force_dof_idx = force_node * 3 + force_dof
        if force_dof_idx < self.num_dof:
            F[force_dof_idx] = 1.0
        F_free = F[free_dofs]
        
        # Measurement index
        if measure_node == -1:
            measure_node = self.num_nodes - 1
        measure_dof_idx = measure_node * 3 + measure_dof
        
        for i, freq in enumerate(frequencies):
            omega = 2 * np.pi * freq
            
            # Dynamic stiffness: H(ω) = -ω²M + iωC + K
            Z = -omega**2 * M_free + 1j * omega * C_free + K_free
            
            try:
                # Solve: Z * u = F
                u_free = np.linalg.solve(Z, F_free)
                
                # Extract response at measurement point
                if measure_dof_idx in free_dofs:
                    idx = free_dofs.index(measure_dof_idx)
                    response[i] = np.abs(u_free[idx])
                else:
                    response[i] = 0.0
                    
            except np.linalg.LinAlgError:
                response[i] = 0.0
        
        return frequencies, response
    
    def compute_band_structure(
        self,
        k_points: np.ndarray,
        num_bands: int = 10,
    ) -> AcousticResult:
        """Compute acoustic band structure (dispersion relation).
        
        Parameters
        ----------
        k_points : np.ndarray
            Wave vectors to sample, shape (N, 3).
        num_bands : int
            Number of bands to compute.
        """
        if self.num_nodes == 0:
            return AcousticResult()
        
        K_global = self._assemble_stiffness_matrix()
        M_global = self._assemble_mass_matrix()
        
        band_frequencies = []
        
        for k in k_points:
            # Bloch periodic boundary conditions
            # K(k) = sum over lattice vectors R: K_R * exp(ik·R)
            
            # Simplified: use Gamma point (k=0) for each k
            # Full implementation would require periodic cell
            
            K_k = K_global.copy()
            M_k = M_global.copy()
            
            # Apply phase factors for Bloch waves
            for elem in self.elements:
                n1, n2 = elem['n1'], elem['n2']
                r = self.node_positions[n2] - self.node_positions[n1]
                
                phase = np.exp(1j * np.dot(k, r))
                
                # Modify off-diagonal blocks
                for d1 in range(3):
                    for d2 in range(3):
                        dof1 = n1*3+d1
                        dof2 = n2*3+d2
                        
                        val = K_global[dof1, dof2]
                        K_k[dof1, dof2] = val * phase
                        K_k[dof2, dof1] = val * np.conj(phase)
            
            # Solve eigenvalue problem
            try:
                K_dense = K_k.toarray()
                M_dense = M_k.toarray()
                
                # Hermitian
                K_dense = 0.5 * (K_dense + K_dense.conj().T)
                M_dense = 0.5 * (M_dense + M_dense.conj().T)
                
                eigenvalues = np.linalg.eigvalsh(
                    np.linalg.solve(M_dense + 1e-10*np.eye(self.num_dof), K_dense)
                )
                
                eigenvalues = np.sort(np.real(eigenvalues))
                eigenvalues = np.maximum(eigenvalues, 0)
                
                omega = np.sqrt(eigenvalues[:num_bands])
                freq = omega / (2 * np.pi)
                
                band_frequencies.append(freq)
                
            except Exception:
                band_frequencies.append(np.zeros(num_bands))
        
        band_frequencies = np.array(band_frequencies)
        
        return AcousticResult(
            frequencies=band_frequencies.flatten(),
            band_structure={
                'k_points': k_points,
                'frequencies': band_frequencies,
            }
        )
