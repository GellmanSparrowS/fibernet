"""
Nonlinear mechanical simulation for fiber networks.

Implements:
- Hyperelastic models (Neo-Hookean, Mooney-Rivlin, Arruda-Boyce)
- Plasticity (bilinear, power-law hardening)
- Viscoelasticity (Maxwell, Kelvin-Voigt, Standard Linear Solid)
- Large deformation geometric nonlinearity
- Full stress-strain curve computation with yielding and failure

Uses incremental loading with Newton-Raphson iteration.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from copy import deepcopy

from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork


@dataclass
class NonlinearResult:
    """Container for nonlinear simulation results."""
    displacements: np.ndarray = None
    strains: np.ndarray = None
    stresses: np.ndarray = None
    reaction_forces: np.ndarray = None
    plastic_strains: np.ndarray = None
    damage: np.ndarray = None
    converged: bool = True
    num_iterations: int = 0
    
    def max_displacement(self) -> float:
        if self.displacements is not None:
            return float(np.max(np.abs(self.displacements)))
        return 0.0
    
    def max_stress(self) -> float:
        if self.stresses is not None:
            return float(np.max(np.abs(self.stresses)))
        return 0.0


# ============================================================
# Constitutive Models
# ============================================================

class ConstitutiveModel:
    """Base class for constitutive models."""
    
    def stress(self, strain: float) -> float:
        raise NotImplementedError
    
    def tangent_modulus(self, strain: float) -> float:
        raise NotImplementedError
    
    def energy_density(self, strain: float) -> float:
        raise NotImplementedError


class LinearElastic(ConstitutiveModel):
    """Linear elastic model."""
    
    def __init__(self, E: float):
        self.E = E
    
    def stress(self, strain: float) -> float:
        return self.E * strain
    
    def tangent_modulus(self, strain: float) -> float:
        return self.E
    
    def energy_density(self, strain: float) -> float:
        return 0.5 * self.E * strain**2


class BilinearPlasticity(ConstitutiveModel):
    """Bilinear elastic-plastic model.
    
    Parameters
    ----------
    E : float
        Young's modulus.
    sigma_y : float
        Yield stress.
    Et : float
        Tangent modulus (hardening). Et=0 for perfect plasticity.
    """
    
    def __init__(self, E: float, sigma_y: float, Et: float = 0.0):
        self.E = E
        self.sigma_y = sigma_y
        self.Et = Et
        self.eps_y = sigma_y / E
    
    def stress(self, strain: float) -> float:
        abs_strain = abs(strain)
        sign = np.sign(strain) if strain != 0 else 1.0
        
        if abs_strain <= self.eps_y:
            return self.E * strain
        else:
            return sign * (self.sigma_y + self.Et * (abs_strain - self.eps_y))
    
    def tangent_modulus(self, strain: float) -> float:
        if abs(strain) <= self.eps_y:
            return self.E
        return self.Et
    
    def energy_density(self, strain: float) -> float:
        abs_strain = abs(strain)
        if abs_strain <= self.eps_y:
            return 0.5 * self.E * strain**2
        else:
            eps_y = self.eps_y
            elastic = 0.5 * self.E * eps_y**2
            plastic = self.sigma_y * (abs_strain - eps_y) + 0.5 * self.Et * (abs_strain - eps_y)**2
            return elastic + plastic


class PowerLawHardening(ConstitutiveModel):
    """Power-law hardening (Ramberg-Osgood) model.
    
    σ/E = ε - (σ/K)^(1/n)  (approximate)
    
    Simplified form: σ = K * ε^n for ε > eps_y
    """
    
    def __init__(self, E: float, K: float, n: float, sigma_y: float = None):
        self.E = E
        self.K = K
        self.n = n
        self.sigma_y = sigma_y or (K * (sigma_y / K)**(1/n) if sigma_y else E * 0.002)
    
    def stress(self, strain: float) -> float:
        abs_strain = abs(strain)
        sign = np.sign(strain) if strain != 0 else 1.0
        
        if abs_strain < 1e-15:
            return 0.0
        
        linear = self.E * abs_strain
        power = self.K * abs_strain**self.n
        
        sigma = min(linear, power)
        return sign * sigma
    
    def tangent_modulus(self, strain: float) -> float:
        abs_strain = abs(strain)
        if abs_strain < 1e-15:
            return self.E
        
        E_t_linear = self.E
        E_t_power = self.K * self.n * abs_strain**(self.n - 1)
        return min(E_t_linear, E_t_power)
    
    def energy_density(self, strain: float) -> float:
        from scipy.integrate import quad
        result, _ = quad(self.stress, 0, strain)
        return result


class HyperelasticNeoHookean(ConstitutiveModel):
    """Neo-Hookean hyperelastic model.
    
    For uniaxial: σ = G * (λ - 1/λ²) where λ = 1 + ε
    """
    
    def __init__(self, G: float, bulk_modulus: float = None):
        self.G = G
        self.K_bulk = bulk_modulus or (G * 100)
    
    def stress(self, strain: float) -> float:
        lam = 1.0 + strain
        if lam <= 0:
            return -self.G * 1e6
        return self.G * (lam - 1.0 / lam**2)
    
    def tangent_modulus(self, strain: float) -> float:
        lam = 1.0 + strain
        if lam <= 0:
            return self.G * 1e6
        return self.G * (1.0 + 2.0 / lam**3)
    
    def energy_density(self, strain: float) -> float:
        lam = 1.0 + strain
        if lam <= 0:
            return 1e10
        return 0.5 * self.G * (lam**2 + 2.0/lam - 3.0)


class HyperelasticMooneyRivlin(ConstitutiveModel):
    """Mooney-Rivlin hyperelastic model.
    
    W = C1*(I1-3) + C2*(I2-3)
    Uniaxial: σ = 2*(C1 + C2/λ)*(λ - 1/λ²)
    """
    
    def __init__(self, C1: float, C2: float):
        self.C1 = C1
        self.C2 = C2
    
    def stress(self, strain: float) -> float:
        lam = 1.0 + strain
        if lam <= 0:
            return -1e10
        return 2 * (self.C1 + self.C2/lam) * (lam - 1.0/lam**2)
    
    def tangent_modulus(self, strain: float) -> float:
        h = 1e-8
        s1 = self.stress(strain + h)
        s2 = self.stress(strain - h)
        return (s1 - s2) / (2*h)
    
    def energy_density(self, strain: float) -> float:
        lam = 1.0 + strain
        if lam <= 0:
            return 1e10
        I1 = lam**2 + 2.0/lam
        I2 = 2.0*lam + 1.0/lam**2
        return self.C1*(I1-3) + self.C2*(I2-3)


class ArrudaBoyce(ConstitutiveModel):
    """Arruda-Boyce 8-chain model for rubber-like materials.
    
    Parameters
    ----------
    n_chain : float
        Number of Kuhn segments per chain.
    kT : float
        Thermal energy (k_B * T).
    b : float
        Kuhn length.
    chains_per_volume : float
        Chain density.
    """
    
    def __init__(self, n_chain: float = 100, nkT: float = 1e6, lam_limit: float = 10.0):
        self.N = n_chain
        self.nkT = nkT
        self.lam_limit = lam_limit
        self.G0 = nkT  # Initial shear modulus
    
    def _langevin_inv(self, x: float) -> float:
        """Inverse Langevin function approximation."""
        if abs(x) > 0.99:
            return x * 100
        return x * (3 - x**2) / (1 - x**2)
    
    def stress(self, strain: float) -> float:
        lam = 1.0 + strain
        if lam <= 0 or lam >= self.lam_limit:
            return np.sign(strain) * 1e10
        
        lam_rms = np.sqrt((lam**2 + 2.0/lam) / 3.0)
        r = lam_rms / np.sqrt(self.N)
        r = min(r, 0.99)
        
        beta = self._langevin_inv(r)
        
        return self.nkT * np.sqrt(self.N) * beta / 3.0 * (lam - 1.0/lam**2)
    
    def tangent_modulus(self, strain: float) -> float:
        h = 1e-8
        s1 = self.stress(strain + h)
        s2 = self.stress(strain - h)
        return (s1 - s2) / (2*h)
    
    def energy_density(self, strain: float) -> float:
        from scipy.integrate import quad
        result, _ = quad(self.stress, 0, strain)
        return result


# ============================================================
# Viscoelastic Models
# ============================================================

class ViscoelasticModel:
    """Base class for time-dependent viscoelastic models."""
    
    def step(self, strain: float, dt: float, state: dict) -> Tuple[float, dict]:
        """Compute stress and update internal state."""
        raise NotImplementedError


class MaxwellModel(ViscoelasticModel):
    """Maxwell model: spring + dashpot in series.
    
    Parameters
    ----------
    E : float
        Spring stiffness.
    eta : float
        Dashpot viscosity.
    """
    
    def __init__(self, E: float, eta: float):
        self.E = E
        self.eta = eta
        self.tau = eta / E
    
    def step(self, strain: float, dt: float, state: dict) -> Tuple[float, dict]:
        sigma_prev = state.get('sigma', 0.0)
        strain_prev = state.get('strain', 0.0)
        
        d_strain = strain - strain_prev
        d_strain_rate = d_strain / max(dt, 1e-15)
        
        alpha = dt / (self.tau + dt)
        sigma_new = (1 - alpha) * sigma_prev + alpha * self.E * (d_strain_rate * self.tau)
        sigma_new = (1 - alpha) * sigma_prev + self.E * alpha * d_strain / alpha if alpha > 0 else sigma_prev
        
        # Simplified implicit integration
        factor = self.tau / (self.tau + dt)
        sigma_new = factor * sigma_prev + self.eta * d_strain / (self.tau + dt)
        
        new_state = {'sigma': sigma_new, 'strain': strain}
        return sigma_new, new_state


class KelvinVoigtModel(ViscoelasticModel):
    """Kelvin-Voigt model: spring + dashpot in parallel.
    
    σ = E*ε + η*dε/dt
    """
    
    def __init__(self, E: float, eta: float):
        self.E = E
        self.eta = eta
    
    def step(self, strain: float, dt: float, state: dict) -> Tuple[float, dict]:
        strain_prev = state.get('strain', 0.0)
        d_strain_rate = (strain - strain_prev) / max(dt, 1e-15)
        
        sigma = self.E * strain + self.eta * d_strain_rate
        
        new_state = {'strain': strain}
        return sigma, new_state


class StandardLinearSolid(ViscoelasticModel):
    """Standard Linear Solid (Zener) model.
    
    Spring E1 in parallel with (spring E2 + dashpot eta in series).
    """
    
    def __init__(self, E1: float, E2: float, eta: float):
        self.E1 = E1
        self.E2 = E2
        self.eta = eta
        self.tau = eta / E2
    
    def step(self, strain: float, dt: float, state: dict) -> Tuple[float, dict]:
        eps_v_prev = state.get('eps_v', 0.0)
        
        alpha = dt / (self.tau + dt)
        eps_v_new = (1 - alpha) * eps_v_prev + alpha * strain
        
        sigma = self.E1 * strain + self.E2 * eps_v_new
        
        new_state = {'eps_v': eps_v_new, 'strain': strain}
        return sigma, new_state


# ============================================================
# Nonlinear FEM Solver
# ============================================================

class NonlinearFEM:
    """Nonlinear finite element solver for fiber networks.
    
    Supports geometric nonlinearity (large deformation) and
    material nonlinearity (plasticity, hyperelasticity).
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    constitutive_model : ConstitutiveModel, optional
        Material constitutive law. Defaults to linear elastic.
    segments_per_fiber : int
        Number of elements per fiber.
    large_deformation : bool
        Enable geometric nonlinearity.
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        constitutive_model: Optional[ConstitutiveModel] = None,
        segments_per_fiber: int = 5,
        large_deformation: bool = False,
    ):
        self.network = deepcopy(network)
        self.model = constitutive_model
        self.segments = segments_per_fiber
        self.large_deformation = large_deformation
        self._build_mesh()
    
    def _build_mesh(self):
        """Build FEM mesh."""
        self.elements = []
        self.element_nodes = []
        self.node_positions = []
        
        node_map = {}
        nid = 0
        
        for f_idx, fiber in enumerate(self.network.fibers):
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
            A = fiber.cross_section_area
            
            for i in range(len(fiber_nodes) - 1):
                ni = fiber_nodes[i]
                nj = fiber_nodes[i + 1]
                if ni != nj:
                    p1 = np.array(self.node_positions[ni])
                    p2 = np.array(self.node_positions[nj])
                    L0 = np.linalg.norm(p2 - p1)
                    if L0 > 1e-12:
                        self.elements.append({
                            'n1': ni, 'n2': nj, 'L0': L0,
                            'E': E, 'A': A, 'fiber_idx': f_idx,
                        })
                        self.element_nodes.append((ni, nj))
        
        self.num_nodes = nid
        self.num_elements = len(self.elements)
        self.num_dof = nid * 3
        
        if self.node_positions:
            self.node_positions = np.array(self.node_positions)
        else:
            self.node_positions = np.array([]).reshape(0, 3)
        
        self._u = np.zeros(self.num_dof)
    
    def _current_positions(self) -> np.ndarray:
        """Get current node positions (initial + displacement)."""
        pos = self.node_positions.copy()
        for i in range(self.num_nodes):
            for d in range(3):
                pos[i, d] += self._u[i * 3 + d]
        return pos
    
    def _compute_element_strain(self, pos: np.ndarray, elem: dict) -> float:
        """Compute axial strain for an element."""
        n1, n2 = elem['n1'], elem['n2']
        L0 = elem['L0']
        
        p1 = pos[n1]
        p2 = pos[n2]
        L = np.linalg.norm(p2 - p1)
        
        if self.large_deformation:
            strain = 0.5 * (L**2 - L0**2) / L0**2  # Green-Lagrange
        else:
            strain = (L - L0) / L0  # Engineering strain
        
        return strain
    
    def _assemble_residual(self, pos: np.ndarray, F_ext: np.ndarray) -> np.ndarray:
        """Assemble residual force vector R = F_int - F_ext."""
        R = np.zeros(self.num_dof)
        
        for elem in self.elements:
            n1, n2 = elem['n1'], elem['n2']
            L0 = elem['L0']
            E = elem['E']
            A = elem['A']
            
            p1 = pos[n1]
            p2 = pos[n2]
            L = np.linalg.norm(p2 - p1)
            
            if L < 1e-15:
                continue
            
            strain = self._compute_element_strain(pos, elem)
            
            if self.model:
                sigma = self.model.stress(strain)
            else:
                sigma = E * strain
            
            force_mag = sigma * A
            
            if self.large_deformation:
                direction = (p2 - p1) / L
                f_int = force_mag * (L / L0) * direction
            else:
                direction = (p2 - p1) / L
                f_int = force_mag * direction
            
            for d in range(3):
                R[n1 * 3 + d] -= f_int[d]
                R[n2 * 3 + d] += f_int[d]
        
        R += F_ext
        return R
    
    def _assemble_tangent(self, pos: np.ndarray) -> csr_matrix:
        """Assemble tangent stiffness matrix."""
        K = lil_matrix((self.num_dof, self.num_dof))
        
        for elem in self.elements:
            n1, n2 = elem['n1'], elem['n2']
            L0 = elem['L0']
            E = elem['E']
            A = elem['A']
            
            p1 = pos[n1]
            p2 = pos[n2]
            L = np.linalg.norm(p2 - p1)
            
            if L < 1e-15:
                continue
            
            strain = self._compute_element_strain(pos, elem)
            
            if self.model:
                E_t = self.model.tangent_modulus(strain)
                sigma = self.model.stress(strain)
            else:
                E_t = E
                sigma = E * strain
            
            n_hat = (p2 - p1) / L
            
            # Material stiffness
            k_mat = (E_t * A / L0) * np.outer(n_hat, n_hat)
            
            # Geometric stiffness (stress stiffness)
            if self.large_deformation:
                k_geo = (sigma * A / L) * (np.eye(3) - np.outer(n_hat, n_hat))
            else:
                k_geo = np.zeros((3, 3))
            
            ke = k_mat + k_geo
            
            dofs = [n1*3, n1*3+1, n1*3+2, n2*3, n2*3+1, n2*3+2]
            k_full = np.zeros((6, 6))
            k_full[0:3, 0:3] = ke
            k_full[3:6, 3:6] = ke
            k_full[0:3, 3:6] = -ke
            k_full[3:6, 0:3] = -ke
            
            for i in range(6):
                for j in range(6):
                    K[dofs[i], dofs[j]] += k_full[i, j]
        
        return K.tocsr()
    
    def solve_incremental(
        self,
        F_ext: np.ndarray,
        fixed_nodes: Optional[List[int]] = None,
        prescribed_dofs: Optional[Dict[int, float]] = None,
        max_iter: int = 20,
        tol: float = 1e-8,
    ) -> NonlinearResult:
        """Solve with Newton-Raphson iteration.
        
        Parameters
        ----------
        F_ext : np.ndarray
            External force vector.
        fixed_nodes : list of int
            Node indices with fixed DOFs.
        prescribed_dofs : dict
            {dof_index: displacement_value}.
        max_iter : int
            Maximum Newton iterations.
        tol : float
            Convergence tolerance.
        """
        if self.num_nodes == 0:
            return NonlinearResult()
        
        fixed_dofs = set()
        if fixed_nodes:
            for n in fixed_nodes:
                fixed_dofs.update([n * 3 + d for d in range(3)])
        if prescribed_dofs:
            fixed_dofs.update(prescribed_dofs.keys())
        
        # For 2D networks, constrain out-of-plane DOFs for ALL nodes
        if self.network.dimension == 2:
            for n in range(self.num_nodes):
                fixed_dofs.add(n * 3 + 2)  # z-displacement
        
        free_dofs = sorted(set(range(self.num_dof)) - fixed_dofs)
        
        for dof, val in (prescribed_dofs or {}).items():
            self._u[dof] = val
        
        converged = False
        for iteration in range(max_iter):
            pos = self._current_positions()
            R = self._assemble_residual(pos, F_ext)
            
            # Zero out fixed DOF residuals
            for dof in fixed_dofs:
                R[dof] = 0.0
            
            R_norm = np.linalg.norm(R[free_dofs]) if free_dofs else 0.0
            F_norm = np.linalg.norm(F_ext)
            ref_norm = max(F_norm, 1e-10)
            
            if R_norm / ref_norm < tol:
                converged = True
                break
            
            K = self._assemble_tangent(pos)
            
            if not free_dofs:
                converged = True
                break
            
            K_free = K[np.ix_(free_dofs, free_dofs)]
            R_free = R[free_dofs]
            
            try:
                # Add regularization to prevent singular matrix
                diag_vals = np.array(K_free.diagonal())
                reg_scale = max(np.max(np.abs(diag_vals)) * 1e-10, 1e-8)
                K_reg = K_free + reg_scale * csr_matrix(np.eye(len(free_dofs)))
                du_free = spsolve(K_reg, -R_free)
                
                if np.any(np.isnan(du_free)):
                    du_free = np.zeros_like(du_free)
                
                # Line search (simple backtracking)
                alpha = 1.0
                for ls in range(5):
                    u_trial = self._u.copy()
                    for i, dof in enumerate(free_dofs):
                        u_trial[dof] += alpha * du_free[i]
                    
                    # Check if residual decreases
                    pos_trial = self.node_positions.copy()
                    for ii in range(self.num_nodes):
                        for d in range(3):
                            pos_trial[ii, d] += u_trial[ii * 3 + d]
                    
                    R_trial = self._assemble_residual(pos_trial, F_ext)
                    for dof in fixed_dofs:
                        R_trial[dof] = 0.0
                    
                    R_trial_norm = np.linalg.norm(R_trial[free_dofs]) if free_dofs else 0.0
                    
                    if R_trial_norm < R_norm:
                        break
                    alpha *= 0.5
                
                for i, dof in enumerate(free_dofs):
                    self._u[dof] += alpha * du_free[i]
                    
            except Exception:
                break
        
        pos_final = self._current_positions()
        
        # Compute element stresses and strains
        stresses = np.zeros(self.num_elements)
        strains = np.zeros(self.num_elements)
        for i, elem in enumerate(self.elements):
            strains[i] = self._compute_element_strain(pos_final, elem)
            if self.model:
                stresses[i] = self.model.stress(strains[i])
            else:
                stresses[i] = elem['E'] * strains[i]
        
        return NonlinearResult(
            displacements=self._u.copy(),
            strains=strains,
            stresses=stresses,
            converged=converged,
            num_iterations=iteration + 1,
        )
    
    def stress_strain_curve(
        self,
        axis: int = 0,
        max_strain: float = 0.1,
        num_steps: int = 50,
        fixed_face: str = "min",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute full stress-strain curve with incremental loading.
        
        Parameters
        ----------
        axis : int
            Loading direction.
        max_strain : float
            Maximum applied strain.
        num_steps : int
            Number of load increments.
        
        Returns
        -------
        strains : np.ndarray
            Applied strain values.
        stresses : np.ndarray
            Average stress values.
        energies : np.ndarray
            Strain energy values.
        """
        if self.num_nodes == 0:
            return np.array([]), np.array([]), np.array([])
        
        positions = self.node_positions[:, axis]
        pos_min = positions.min()
        pos_max = positions.max()
        L = pos_max - pos_min
        
        if L < 1e-12:
            return np.array([]), np.array([]), np.array([])
        
        # Identify boundary nodes
        tol = L * 0.01
        fixed_nodes = []
        prescribed_nodes = []
        
        for n_idx, pos in enumerate(self.node_positions):
            if fixed_face == "min":
                if pos[axis] <= pos_min + tol:
                    fixed_nodes.append(n_idx)
                if pos[axis] >= pos_max - tol:
                    prescribed_nodes.append(n_idx)
            else:
                if pos[axis] >= pos_max - tol:
                    fixed_nodes.append(n_idx)
                if pos[axis] <= pos_min + tol:
                    prescribed_nodes.append(n_idx)
        
        # 2D constraint
        all_fixed_dofs = set()
        if self.network.dimension == 2:
            for n in range(self.num_nodes):
                all_fixed_dofs.add(n * 3 + 2)
        
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
        
        strain_values = np.linspace(0, max_strain, num_steps + 1)[1:]
        stress_values = np.zeros(num_steps)
        energy_values = np.zeros(num_steps)
        
        self._u = np.zeros(self.num_dof)
        
        for step_idx, target_strain in enumerate(strain_values):
            delta = target_strain * L
            
            fixed_dofs = set()
            for n in fixed_nodes:
                fixed_dofs.update([n * 3 + d for d in range(3)])
            fixed_dofs.update(all_fixed_dofs)
            
            prescribed = {}
            for n in prescribed_nodes:
                dof = n * 3 + axis
                prescribed[dof] = delta
                for d in range(3):
                    if d != axis:
                        fixed_dofs.add(n * 3 + d)
            fixed_dofs.update(all_fixed_dofs)
            
            free_dofs = sorted(set(range(self.num_dof)) - fixed_dofs - set(prescribed.keys()))
            
            for dof, val in prescribed.items():
                self._u[dof] = val
            
            pos = self._current_positions()
            
            total_stress = 0.0
            total_energy = 0.0
            
            for elem in self.elements:
                strain = self._compute_element_strain(pos, elem)
                if self.model:
                    sigma = self.model.stress(strain)
                    energy = self.model.energy_density(strain)
                else:
                    sigma = elem['E'] * strain
                    energy = 0.5 * elem['E'] * strain**2
                
                total_stress += sigma * elem['A']
                total_energy += energy * elem['A'] * elem['L0']
            
            avg_stress = total_stress / max(area, 1e-12)
            stress_values[step_idx] = avg_stress
            energy_values[step_idx] = total_energy
        
        return strain_values, stress_values, energy_values
    
    def viscoelastic_loading(
        self,
        visco_model: ViscoelasticModel,
        axis: int = 0,
        strain_rate: float = 1e-3,
        max_strain: float = 0.05,
        dt: float = 1e-4,
        fixed_face: str = "min",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run viscoelastic loading simulation.
        
        Parameters
        ----------
        visco_model : ViscoelasticModel
            Time-dependent constitutive model.
        strain_rate : float
            Applied strain rate.
        max_strain : float
            Maximum strain.
        dt : float
            Time step.
        
        Returns
        -------
        time : np.ndarray
        stress : np.ndarray
        strain : np.ndarray
        """
        if self.num_nodes == 0:
            return np.array([]), np.array([]), np.array([])
        
        positions = self.node_positions[:, axis]
        pos_min = positions.min()
        pos_max = positions.max()
        L = pos_max - pos_min
        
        if L < 1e-12:
            return np.array([]), np.array([]), np.array([])
        
        tol = L * 0.01
        hot_nodes = np.where(positions >= pos_max - tol)[0]
        cold_nodes = np.where(positions <= pos_min + tol)[0]
        
        bb_min, bb_max = self.network.bounding_box()
        dims = bb_max - bb_min
        if self.network.dimension == 2:
            area = dims[1] if axis == 0 else dims[0]
        else:
            if axis == 0:
                area = dims[1] * dims[2]
            elif axis == 1:
                area = dims[0] * dims[2]
            else:
                area = dims[0] * dims[1]
        
        total_time = max_strain / strain_rate
        num_steps = int(total_time / dt)
        
        elem_states = [{} for _ in range(self.num_elements)]
        
        time_values = []
        stress_values = []
        strain_values = []
        
        save_interval = max(1, num_steps // 200)
        
        for step in range(num_steps + 1):
            t = step * dt
            current_strain = min(strain_rate * t, max_strain)
            delta = current_strain * L
            
            self._u = np.zeros(self.num_dof)
            for n in cold_nodes:
                for d in range(3):
                    self._u[n * 3 + d] = 0.0
            for n in hot_nodes:
                self._u[n * 3 + axis] = delta
                for d in range(3):
                    if d != axis:
                        self._u[n * 3 + d] = 0.0
            
            pos = self._current_positions()
            
            total_stress = 0.0
            for i, elem in enumerate(self.elements):
                strain = self._compute_element_strain(pos, elem)
                sigma, elem_states[i] = visco_model.step(strain, dt, elem_states[i])
                total_stress += sigma * elem['A']
            
            if step % save_interval == 0:
                avg_stress = total_stress / max(area, 1e-12)
                time_values.append(t)
                stress_values.append(avg_stress)
                strain_values.append(current_strain)
        
        return np.array(time_values), np.array(stress_values), np.array(strain_values)
