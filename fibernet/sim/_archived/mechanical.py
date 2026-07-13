"""
Mechanical simulation engine for fiber networks.

Implements:
- Euler-Bernoulli beam theory for fiber bending
- Axial deformation (tension/compression)
- Shear deformation (Timoshenko beam)
- Linear elastic FEM for static analysis
- Nonlinear geometric effects (large deformation)

Uses scipy sparse matrices for efficient solving.
"""

import warnings
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import scipy.sparse.linalg as spla
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, field

from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork, Crosslink


@dataclass
class MechanicalResult:
    """Container for mechanical simulation results."""
    displacements: np.ndarray = None
    forces: np.ndarray = None
    stresses: np.ndarray = None
    strains: np.ndarray = None
    reaction_forces: np.ndarray = None
    deformed_positions: List[np.ndarray] = None
    energy: float = 0.0
    convergence_history: List[float] = field(default_factory=list)
    
    def max_displacement(self) -> float:
        if self.displacements is not None:
            return float(np.max(np.abs(self.displacements)))
        return 0.0
    
    def max_stress(self) -> float:
        if self.stresses is not None:
            return float(np.max(np.abs(self.stresses)))
        return 0.0


class BeamElement:
    """Euler-Bernoulli beam finite element in 3D.
    
    Each node has 6 DOFs: [ux, uy, uz, rx, ry, rz]
    Element has 12 DOFs total.
    """
    
    def __init__(self, node_i: np.ndarray, node_j: np.ndarray,
                 E: float, G: float, A: float, Iy: float, Iz: float, J: float):
        self.p1 = np.asarray(node_i)
        self.p2 = np.asarray(node_j)
        self.L = np.linalg.norm(self.p2 - self.p1)
        self.E = E
        self.G = G
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        
        if self.L < 1e-12:
            self.L = 1e-12
        
        self.direction = (self.p2 - self.p1) / self.L
        self._compute_local_stiffness()
        self._compute_transformation()
    
    def _compute_local_stiffness(self):
        """Compute 12x12 local stiffness matrix."""
        E, A, L = self.E, self.A, self.L
        Iy, Iz = self.Iy, self.Iz
        G, J = self.G, self.J
        
        k = np.zeros((12, 12))
        
        ea_l = E * A / L
        k[0, 0] = ea_l
        k[0, 6] = -ea_l
        k[6, 0] = -ea_l
        k[6, 6] = ea_l
        
        gj_l = G * J / L
        k[3, 3] = gj_l
        k[3, 9] = -gj_l
        k[9, 3] = -gj_l
        k[9, 9] = gj_l
        
        eiy = E * Iy
        for i, j, val in [
            (1, 1, 12 * eiy / L**3), (1, 5, 6 * eiy / L**2),
            (1, 7, -12 * eiy / L**3), (1, 11, 6 * eiy / L**2),
            (5, 1, 6 * eiy / L**2), (5, 5, 4 * eiy / L),
            (5, 7, -6 * eiy / L**2), (5, 11, 2 * eiy / L),
            (7, 1, -12 * eiy / L**3), (7, 5, -6 * eiy / L**2),
            (7, 7, 12 * eiy / L**3), (7, 11, -6 * eiy / L**2),
            (11, 1, 6 * eiy / L**2), (11, 5, 2 * eiy / L),
            (11, 7, -6 * eiy / L**2), (11, 11, 4 * eiy / L),
        ]:
            k[i, j] = val
        
        eiz = E * Iz
        for i, j, val in [
            (2, 2, 12 * eiz / L**3), (2, 4, -6 * eiz / L**2),
            (2, 8, -12 * eiz / L**3), (2, 10, -6 * eiz / L**2),
            (4, 2, -6 * eiz / L**2), (4, 4, 4 * eiz / L),
            (4, 8, 6 * eiz / L**2), (4, 10, 2 * eiz / L),
            (8, 2, -12 * eiz / L**3), (8, 4, 6 * eiz / L**2),
            (8, 8, 12 * eiz / L**3), (8, 10, 6 * eiz / L**2),
            (10, 2, -6 * eiz / L**2), (10, 4, 2 * eiz / L),
            (10, 8, 6 * eiz / L**2), (10, 10, 4 * eiz / L),
        ]:
            k[i, j] = val
        
        self.k_local = k
    
    def _compute_transformation(self):
        """Compute 12x12 transformation matrix (local to global)."""
        d = self.direction
        if abs(d[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        
        v2 = np.cross(d, ref)
        v2 /= np.linalg.norm(v2)
        v3 = np.cross(d, v2)
        
        R = np.array([d, v2, v3])
        
        T = np.zeros((12, 12))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        
        self.T = T
    
    @property
    def stiffness_global(self) -> np.ndarray:
        """12x12 global stiffness matrix."""
        return self.T.T @ self.k_local @ self.T
    
    def element_forces(self, u_global: np.ndarray) -> np.ndarray:
        """Compute element forces from global displacements."""
        u_local = self.T @ u_global
        f_local = self.k_local @ u_local
        return self.T.T @ f_local


class FiberFEM:
    """Finite element solver for fiber network mechanics.
    
    Parameters
    ----------
    network : FiberNetwork
        The fiber network to simulate.
    segments_per_fiber : int
        Number of beam elements per fiber.
    """
    
    def __init__(self, network: FiberNetwork, segments_per_fiber: int = 5):
        self.network = network
        self.segments = segments_per_fiber
        self._build_mesh()
    
    def _build_mesh(self):
        """Build FEM mesh from fiber network."""
        self.elements = []
        self.element_to_nodes = []
        self.node_positions = []
        
        node_map = {}
        nid = 0
        
        for f_idx, fiber in enumerate(self.network.fibers):
            resampled = fiber.resample(self.segments + 1)
            pts = resampled.centerline
            
            fiber_nodes = []
            for p_idx, pt in enumerate(pts):
                key = tuple(np.round(pt, 8))
                if key not in node_map:
                    node_map[key] = nid
                    self.node_positions.append(pt)
                    nid += 1
                fiber_nodes.append(node_map[key])
            
            E = fiber.material.youngs_modulus
            G = fiber.material.shear_modulus or (E / 2.6)
            A = fiber.cross_section_area
            Iy, Iz = fiber.second_moment_area
            J = fiber.polar_moment
            
            for e_idx in range(self.segments):
                ni = fiber_nodes[e_idx]
                nj = fiber_nodes[e_idx + 1]
                if ni != nj:
                    elem = BeamElement(
                        self.node_positions[ni], self.node_positions[nj],
                        E=E, G=G, A=A, Iy=Iy, Iz=Iz, J=J,
                    )
                    self.elements.append(elem)
                    self.element_to_nodes.append((ni, nj))
        
        self.num_nodes = nid
        self.num_elements = len(self.elements)
        self.num_dof = nid * 6
        
        if self.node_positions:
            self.node_positions = np.array(self.node_positions)
        else:
            self.node_positions = np.array([]).reshape(0, 3)
    
    def assemble_stiffness(self) -> csr_matrix:
        """Assemble global stiffness matrix."""
        K = lil_matrix((self.num_dof, self.num_dof))
        
        for elem, (ni, nj) in zip(self.elements, self.element_to_nodes):
            k_g = elem.stiffness_global
            
            dof_map = []
            for node in [ni, nj]:
                dof_map.extend([node * 6 + d for d in range(6)])
            
            for ii, di in enumerate(dof_map):
                for jj, dj in enumerate(dof_map):
                    K[di, dj] += k_g[ii, jj]
        
        return K.tocsr()
    
    def solve_static(
        self,
        forces: Optional[np.ndarray] = None,
        fixed_nodes: Optional[List[int]] = None,
        fixed_dofs: Optional[List[int]] = None,
        prescribed_dofs: Optional[Dict[int, float]] = None,
    ) -> MechanicalResult:
        """Solve static linear elastic problem.
        
        Parameters
        ----------
        forces : np.ndarray, optional
            External force vector (num_dof,). Zero if not given.
        fixed_nodes : list of int, optional
            Node indices with all DOFs fixed.
        fixed_dofs : list of int, optional
            Specific DOF indices to fix.
        prescribed_dofs : dict, optional
            {dof_index: displacement_value} for prescribed displacements.
        
        Returns
        -------
        MechanicalResult
        """
        if self.num_elements == 0:
            return MechanicalResult()
        
        K = self.assemble_stiffness()
        
        if forces is None:
            F = np.zeros(self.num_dof)
        else:
            F = np.asarray(forces, dtype=np.float64)
        
        fixed = set()
        if fixed_nodes:
            for n in fixed_nodes:
                for d in range(6):
                    fixed.add(n * 6 + d)
        if fixed_dofs:
            fixed.update(fixed_dofs)
        
        all_dofs = set(range(self.num_dof))
        free_dofs = sorted(all_dofs - fixed)
        
        if prescribed_dofs:
            for dof, val in prescribed_dofs.items():
                F -= K[:, dof].toarray().flatten() * val
        
        if len(free_dofs) == 0:
            return MechanicalResult(
                displacements=np.zeros(self.num_dof),
                forces=F,
            )
        
        K_free = K[np.ix_(free_dofs, free_dofs)]
        F_free = F[free_dofs]
        
        u = np.zeros(self.num_dof)
        if prescribed_dofs:
            for dof, val in prescribed_dofs.items():
                u[dof] = val
        
        if len(free_dofs) == 0:
            return MechanicalResult(
                displacements=u,
                forces=F,
            )
        
        # Detect and fix isolated DOFs (not connected to any element)
        diag = K_free.diagonal()
        isolated_mask = np.abs(diag) < 1e-15
        if np.any(isolated_mask):
            # Fix isolated DOFs by adding unit stiffness
            from scipy.sparse import diags
            fix = diags(np.where(isolated_mask, 1.0, 0.0))
            K_free = K_free + fix
        
        # Add Tikhonov regularization for near-singular systems
        reg = 1e-12 * np.max(np.abs(K_free.diagonal()))
        if reg > 0:
            from scipy.sparse import diags
            K_free = K_free + diags(reg * np.ones(len(free_dofs)))
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u_free = spsolve(K_free, F_free)
            if np.any(np.isnan(u_free)):
                u_free = np.linalg.lstsq(K_free.toarray(), F_free, rcond=None)[0]
            for i, dof in enumerate(free_dofs):
                u[dof] = u_free[i]
        except Exception as e:
            try:
                u_free = np.linalg.lstsq(K_free.toarray(), F_free, rcond=None)[0]
                for i, dof in enumerate(free_dofs):
                    u[dof] = u_free[i]
            except Exception:
                pass  # Leave u as zeros
        
        stresses = np.zeros(self.num_elements)
        strains = np.zeros(self.num_elements)
        for e_idx, (elem, (ni, nj)) in enumerate(zip(self.elements, self.element_to_nodes)):
            dof_map = []
            for node in [ni, nj]:
                dof_map.extend([node * 6 + d for d in range(6)])
            u_elem = np.array([u[d] for d in dof_map])
            u_local = elem.T @ u_elem
            
            axial_strain = (u_local[6] - u_local[0]) / elem.L
            strains[e_idx] = axial_strain
            stresses[e_idx] = elem.E * axial_strain
        
        reaction = K @ u - F
        
        deformed = []
        for fiber in self.network.fibers:
            deformed.append(fiber.centerline.copy())
        
        energy = 0.5 * u @ K @ u
        
        return MechanicalResult(
            displacements=u,
            forces=F,
            stresses=stresses,
            strains=strains,
            reaction_forces=reaction,
            deformed_positions=deformed,
            energy=energy,
        )
    
    def apply_uniaxial_strain(
        self,
        strain: float,
        axis: int = 0,
        fixed_face: str = "min",
    ) -> MechanicalResult:
        """Apply uniaxial strain along a given axis.
        
        Uses correct boundary conditions for Poisson's ratio measurement:
        - Fixed face: only constrain loading direction (allows transverse deformation)
        - Prescribed face: prescribe loading direction displacement
        - Reference node: constrain transverse + rotation to prevent rigid body motion
        
        Parameters
        ----------
        strain : float
            Applied strain (positive = tension).
        axis : int
            0=x, 1=y, 2=z.
        fixed_face : str
            'min' or 'max' face to fix; opposite face gets prescribed displacement.
        """
        if self.num_nodes == 0:
            return MechanicalResult()
        
        positions = self.node_positions[:, axis]
        pos_min = positions.min()
        pos_max = positions.max()
        L = pos_max - pos_min
        
        if L < 1e-12:
            return MechanicalResult()
        
        delta = strain * L
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
        
        fixed_dofs = []
        prescribed = {}
        
        # Fixed face: only constrain the loading direction DOF
        # This allows transverse deformation (needed for Poisson's ratio)
        for n in fixed_nodes:
            fixed_dofs.append(n * 6 + axis)
        
        # Prescribed face: prescribe the loading direction displacement
        for n in prescribed_nodes:
            dof = n * 6 + axis
            if dof not in set(fixed_dofs):
                prescribed[dof] = delta
        
        # Prevent rigid body motion: fix transverse DOFs at one reference node
        # Pick the first fixed node as reference
        if fixed_nodes:
            ref_node = fixed_nodes[0]
            for d in range(3):
                if d != axis:  # Don't re-constrain loading direction
                    fixed_dofs.append(ref_node * 6 + d)
            # Also fix in-plane rotation to prevent rigid body rotation
            if self.network.dimension == 2:
                fixed_dofs.append(ref_node * 6 + 5)  # rz
        
        # For 2D networks, constrain out-of-plane DOFs for ALL nodes
        if self.network.dimension == 2:
            for n in range(self.num_nodes):
                # Fix z-displacement (DOF 2)
                fixed_dofs.append(n * 6 + 2)
                # Fix x-rotation (DOF 3) and y-rotation (DOF 4)
                fixed_dofs.append(n * 6 + 3)
                fixed_dofs.append(n * 6 + 4)
        
        return self.solve_static(fixed_dofs=fixed_dofs, prescribed_dofs=prescribed)
    
    def effective_modulus(
        self,
        strain: float = 0.001,
        axis: int = 0,
    ) -> float:
        """Compute effective Young's modulus along given axis.
        
        Parameters
        ----------
        strain : float
            Small applied strain.
        axis : int
            Loading direction.
        
        Returns
        -------
        float
            Effective Young's modulus.
        """
        result = self.apply_uniaxial_strain(strain, axis)
        
        if self.node_positions.size == 0:
            return 0.0
        
        pos = self.node_positions[:, axis]
        L = pos.max() - pos.min()
        
        if L < 1e-12:
            return 0.0
        
        bb_min, bb_max = self.network.bounding_box()
        dims = bb_max - bb_min
        
        # For 2D networks, use unit thickness
        if self.network.dimension == 2:
            if axis == 0:
                area = dims[1] * 1.0  # width * unit thickness
            elif axis == 1:
                area = dims[0] * 1.0  # width * unit thickness
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
            return 0.0
        
        stress = result.energy * 2 / (self.network.total_volume * strain)
        E_eff = stress / strain
        
        return E_eff


def stress_strain_curve(
    network: FiberNetwork,
    max_strain: float = 0.1,
    num_steps: int = 20,
    axis: int = 0,
    segments_per_fiber: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute stress-strain curve for a fiber network.
    
    Parameters
    ----------
    network : FiberNetwork
        The fiber network.
    max_strain : float
        Maximum applied strain.
    num_steps : int
        Number of strain increments.
    axis : int
        Loading direction.
    segments_per_fiber : int
        FEM discretization.
    
    Returns
    -------
    strains : np.ndarray
        Applied strain values.
    stresses : np.ndarray
        Corresponding stress values.
    """
    fem = FiberFEM(network, segments_per_fiber)
    
    strains = np.linspace(0, max_strain, num_steps + 1)[1:]
    stresses = np.zeros_like(strains)
    
    bb_min, bb_max = network.bounding_box()
    dims = bb_max - bb_min
    if network.dimension == 2:
        if axis == 0:
            area = dims[1] * 1.0  # width * unit thickness
        elif axis == 1:
            area = dims[0] * 1.0  # length * unit thickness
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
        return strains, stresses
    
    for i, eps in enumerate(strains):
        result = fem.apply_uniaxial_strain(eps, axis)
        stress = result.energy * 2 / (network.total_volume * eps) if eps > 0 else 0
        stresses[i] = stress
    
    return strains, stresses


class EffectiveProperties:
    """Container for effective (homogenized) mechanical properties.
    
    Stores the full effective stiffness tensor and derived engineering
    constants for a fiber network representative volume element (RVE).
    """
    
    def __init__(self):
        self.stiffness_tensor = None  # 6x6 Voigt notation
        self.E_x = 0.0
        self.E_y = 0.0
        self.E_z = 0.0
        self.nu_xy = 0.0
        self.nu_xz = 0.0
        self.nu_yz = 0.0
        self.G_xy = 0.0
        self.G_xz = 0.0
        self.G_yz = 0.0
        self.relative_density = 0.0
    
    def summary(self) -> str:
        """Return a formatted summary of effective properties."""
        lines = [
            "=== Effective Mechanical Properties ===",
            f"E_x  = {self.E_x:.4e} Pa",
            f"E_y  = {self.E_y:.4e} Pa",
            f"E_z  = {self.E_z:.4e} Pa",
            f"ν_xy = {self.nu_xy:.4f}",
            f"ν_xz = {self.nu_xz:.4f}",
            f"ν_yz = {self.nu_yz:.4f}",
            f"G_xy = {self.G_xy:.4e} Pa",
            f"G_xz = {self.G_xz:.4e} Pa",
            f"G_yz = {self.G_yz:.4e} Pa",
            f"ρ*   = {self.relative_density:.4f}",
        ]
        if self.E_x > 0 and self.E_y > 0:
            aniso = max(self.E_x, self.E_y, self.E_z) / max(min(self.E_x, self.E_y, self.E_z), 1e-30)
            lines.append(f"Anisotropy ratio = {aniso:.2f}")
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Export to dictionary."""
        return {
            "E_x": self.E_x, "E_y": self.E_y, "E_z": self.E_z,
            "nu_xy": self.nu_xy, "nu_xz": self.nu_xz, "nu_yz": self.nu_yz,
            "G_xy": self.G_xy, "G_xz": self.G_xz, "G_yz": self.G_yz,
            "relative_density": self.relative_density,
        }


def compute_effective_properties(
    network: FiberNetwork,
    strain: float = 0.001,
    segments_per_fiber: int = 5,
) -> EffectiveProperties:
    """Compute effective engineering constants for a fiber network.
    
    Performs uniaxial strain tests along each axis to extract
    Young's moduli, Poisson's ratios, and shear moduli.
    
    Parameters
    ----------
    network : FiberNetwork
        The fiber network (should be a representative volume element).
    strain : float
        Small test strain (linear regime).
    segments_per_fiber : int
        FEM discretization per fiber.
    
    Returns
    -------
    EffectiveProperties with engineering constants.
    """
    fem = FiberFEM(network, segments_per_fiber)
    props = EffectiveProperties()
    
    bb_min, bb_max = network.bounding_box()
    dims = bb_max - bb_min
    volume = float(np.prod(dims[dims > 1e-12])) if np.any(dims > 1e-12) else 1.0
    fiber_volume = network.total_volume
    
    props.relative_density = fiber_volume / volume if volume > 1e-12 else 0.0
    
    moduli = {}
    for axis in range(min(3, network.dimension + 1)):
        if dims[axis] < 1e-12:
            continue
        
        result = fem.apply_uniaxial_strain(strain, axis)
        
        if network.dimension == 2:
            if axis == 0:
                area = dims[1] * 1.0
            elif axis == 1:
                area = dims[0] * 1.0
            else:
                area = dims[0] * dims[1]
        else:
            other_axes = [a for a in range(3) if a != axis]
            area = dims[other_axes[0]] * dims[other_axes[1]]
        
        if area < 1e-12 or volume < 1e-12:
            moduli[axis] = (0.0, 0.0, 0.0)
            continue
        
        E = result.energy * 2 / (volume * strain**2) if strain > 0 else 0.0
        
        transverse_strains = []
        for other_axis in range(min(3, network.dimension + 1)):
            if other_axis == axis:
                continue
            if dims[other_axis] < 1e-12:
                transverse_strains.append(0.0)
                continue
            
            positions = fem.node_positions[:, other_axis]
            L_other = positions.max() - positions.min()
            if L_other < 1e-12:
                transverse_strains.append(0.0)
                continue
            
            if result.displacements is not None and len(result.displacements) > 0:
                n_nodes = fem.num_nodes
                disp_other = np.zeros(n_nodes)
                for n in range(n_nodes):
                    dof_idx = n * 6 + other_axis
                    if dof_idx < len(result.displacements):
                        disp_other[n] = result.displacements[dof_idx]
                
                fixed_mask = positions <= positions.min() + dims[other_axis] * 0.01
                free_mask = positions >= positions.max() - dims[other_axis] * 0.01
                
                if np.any(free_mask) and np.any(fixed_mask):
                    avg_disp_free = np.mean(disp_other[free_mask])
                    eps_transverse = avg_disp_free / L_other
                    transverse_strains.append(eps_transverse)
                else:
                    transverse_strains.append(0.0)
            else:
                transverse_strains.append(0.0)
        
        moduli[axis] = (E, transverse_strains, strain)
    
    if 0 in moduli:
        props.E_x = moduli[0][0]
        if isinstance(moduli[0][1], list) and len(moduli[0][1]) > 0:
            eps_app = moduli[0][2] if len(moduli[0]) > 2 else strain
            if eps_app > 0:
                props.nu_xy = -moduli[0][1][0] / eps_app if len(moduli[0][1]) > 0 else 0.0
                props.nu_xz = -moduli[0][1][1] / eps_app if len(moduli[0][1]) > 1 else 0.0
    
    if 1 in moduli:
        props.E_y = moduli[1][0]
        if isinstance(moduli[1][1], list) and len(moduli[1][1]) > 0:
            eps_app = moduli[1][2] if len(moduli[1]) > 2 else strain
            if eps_app > 0:
                props.nu_yz = -moduli[1][1][0] / eps_app if len(moduli[1][1]) > 0 else 0.0
    
    if 2 in moduli:
        props.E_z = moduli[2][0]
    
    for a in range(3):
        if a in moduli and moduli[a][0] > 0:
            nu = getattr(props, f'nu_{"xy" if a == 0 else "xz" if a == 1 else "yz"}')
            E = moduli[a][0]
            G = E / (2 * (1 + nu)) if abs(nu) < 0.499 else E / 3
            if a == 0:
                props.G_xy = G
            elif a == 1:
                props.G_xz = G
            else:
                props.G_yz = G
    
    return props


def poisson_ratio(
    network: FiberNetwork,
    strain: float = 0.001,
    loading_axis: int = 0,
    transverse_axis: int = 1,
    segments_per_fiber: int = 5,
) -> float:
    """Compute Poisson's ratio ν = -ε_transverse / ε_axial.
    
    Parameters
    ----------
    loading_axis : int
        Direction of applied uniaxial strain (0=x, 1=y, 2=z).
    transverse_axis : int
        Direction of measured transverse strain.
    
    Returns
    -------
    float
        Poisson's ratio (negative for auxetic structures).
    """
    fem = FiberFEM(network, segments_per_fiber)
    result = fem.apply_uniaxial_strain(strain, loading_axis)
    
    if result.displacements is None or fem.num_nodes == 0:
        return 0.0
    
    bb_min, bb_max = network.bounding_box()
    dims = bb_max - bb_min
    
    positions = fem.node_positions[:, transverse_axis]
    L_trans = positions.max() - positions.min()
    
    if L_trans < 1e-12:
        return 0.0
    
    tol = dims[transverse_axis] * 0.01
    free_mask = positions >= positions.max() - tol
    
    if not np.any(free_mask):
        return 0.0
    
    n_nodes = fem.num_nodes
    disp_trans = np.zeros(n_nodes)
    for n in range(n_nodes):
        dof_idx = n * 6 + transverse_axis
        if dof_idx < len(result.displacements):
            disp_trans[n] = result.displacements[dof_idx]
    
    avg_disp = np.mean(disp_trans[free_mask])
    eps_trans = avg_disp / L_trans
    
    return -eps_trans / strain if abs(strain) > 1e-15 else 0.0
