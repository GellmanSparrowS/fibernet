"""
BeamFrameFEM — Beam/Frame Finite Element Method with Welded Joints.

A full beam/frame FEM solver treating fiber networks as structural frames
with welded (rigid, moment-resisting) joints. Supports both 2D and 3D.

Key Features
------------
- 2D: Euler-Bernoulli beam elements, 3 DOF/node (ux, uy, θz)
- 3D: Full 3D beam elements, 6 DOF/node (ux, uy, uz, θx, θy, θz)
- Welded joints: full moment transfer at connections
- Elastic material: linear elastic (E, G/ν, A, Iy, Iz, J)
- Circular cross-section support (A=πr², I=πr⁴/4, J=πr⁴/2)
- scipy.sparse backend — zero external dependencies beyond numpy/scipy
- Robust solving via SVD pseudoinverse for near-singular systems
- Integration with FiberNet graph structures

Comparison with DifferentiableSpringNetwork (truss model):
- Truss: pin-jointed, axial force only, 2/3 DOF/node (translation only)
- Beam: welded joints, axial + bending + torsion, 3/6 DOF/node

Usage
-----
>>> from fibernet.ml.beam_frame_fem import BeamFrameFEM
>>> solver = BeamFrameFEM(dim=2, E=1e9, nu=0.3)
>>> u, sigma, moments = solver.solve(edge_index, node_pos, radii, forces, fixed_nodes)
"""

import math
from typing import Optional, Tuple, Dict
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


class BeamFrameFEM:
    """Beam/Frame FEM solver with welded (rigid, moment-resisting) joints.

    Parameters
    ----------
    dim : int
        Spatial dimension (2 or 3).
    E : float
        Young's modulus (Pa).
    nu : float
        Poisson's ratio (used to compute G = E / (2(1+nu))).
    damping : float
        Regularization added to diagonal of K for numerical stability.
    """

    def __init__(self, dim: int = 2, E: float = 1e9, nu: float = 0.3,
                 damping: float = 1e-6):
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        self.dim = dim
        self.E = E
        self.nu = nu
        self.G = E / (2 * (1 + nu))  # Shear modulus
        self.damping = damping

        # DOF per node
        if dim == 2:
            self.dof_per_node = 3   # ux, uy, θz
            self.dof_per_element = 6
        else:
            self.dof_per_node = 6   # ux, uy, uz, θx, θy, θz
            self.dof_per_element = 12

    @staticmethod
    def circular_section_properties(radius: float) -> Dict[str, float]:
        """Compute cross-section properties for a circular cross-section.

        Parameters
        ----------
        radius : float
            Cross-section radius.

        Returns
        -------
        dict with keys: A, Iy, Iz, J
            A  = cross-section area = πr²
            Iy = second moment of area about y-axis = πr⁴/4
            Iz = second moment of area about z-axis = πr⁴/4
            J  = torsional constant = πr⁴/2
        """
        r = radius
        A = math.pi * r**2
        I_val = math.pi * r**4 / 4
        J_val = math.pi * r**4 / 2
        return {'A': A, 'Iy': I_val, 'Iz': I_val, 'J': J_val}

    def _beam_element_stiffness_2d(self, L: float, c: float, s: float,
                                    A: float, I: float) -> np.ndarray:
        """Compute 6×6 beam element stiffness matrix in global coords (2D).

        Parameters
        ----------
        L : element length
        c : cos(θ) — direction cosine
        s : sin(θ) — direction sine
        A : cross-section area
        I : second moment of area (Iz for 2D bending)

        Returns
        -------
        (6, 6) element stiffness matrix in global coordinates
        DOF order: [u1x, u1y, θ1z, u2x, u2y, θ2z]
        """
        E = self.E
        # Local stiffness matrix (Euler-Bernoulli beam)
        # DOF: [u1, v1, θ1, u2, v2, θ2]
        # Axial + bending combined

        # Axial part
        EA_L = E * A / L
        # Bending part
        EI_L3 = E * I / (L**3)
        EI_L2 = E * I / (L**2)
        EI_L = E * I / L

        # Local stiffness (6×6)
        k_local = np.array([
            [ EA_L,       0,          0,      -EA_L,       0,          0     ],
            [ 0,          12*EI_L3,   6*EI_L2, 0,         -12*EI_L3,   6*EI_L2],
            [ 0,          6*EI_L2,    4*EI_L,  0,         -6*EI_L2,    2*EI_L ],
            [-EA_L,       0,          0,       EA_L,       0,          0     ],
            [ 0,         -12*EI_L3,  -6*EI_L2, 0,          12*EI_L3,  -6*EI_L2],
            [ 0,          6*EI_L2,    2*EI_L,  0,         -6*EI_L2,    4*EI_L ],
        ])

        # Transformation matrix (local → global)
        # For 2D beam: T transforms [u_local, v_local, θ] for each node
        T = np.zeros((6, 6))
        T[0, 0] = c;  T[0, 1] = s   # u1_local = c*u1x + s*u1y
        T[1, 0] = -s; T[1, 1] = c   # v1_local = -s*u1x + c*u1y
        T[2, 2] = 1.0                # θ1z same
        T[3, 3] = c;  T[3, 4] = s
        T[4, 3] = -s; T[4, 4] = c
        T[5, 5] = 1.0

        # Global stiffness = T^T * k_local * T
        k_global = T.T @ k_local @ T
        return k_global

    def _beam_element_stiffness_3d(self, L: float, direction: np.ndarray,
                                    A: float, Iy: float, Iz: float,
                                    J: float) -> np.ndarray:
        """Compute 12×12 beam element stiffness matrix in global coords (3D).

        Parameters
        ----------
        L : element length
        direction : (3,) unit vector from node i to node j
        A : cross-section area
        Iy, Iz : second moments of area
        J : torsional constant

        Returns
        -------
        (12, 12) element stiffness matrix in global coordinates
        DOF order: [u1x, u1y, u1z, θ1x, θ1y, θ1z, u2x, u2y, u2z, θ2x, θ2y, θ2z]
        """
        E = self.E
        G = self.G

        # Build local coordinate system
        # Element axis: e1 = direction
        e1 = direction / np.linalg.norm(direction)

        # Choose a reference vector not parallel to e1
        if abs(e1[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])

        e2 = np.cross(e1, ref)
        e2 = e2 / np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        e3 = e3 / np.linalg.norm(e3)

        # Direction cosine matrix (3×3): rows are e1, e2, e3
        R = np.array([e1, e2, e3])  # (3, 3)

        # Local stiffness matrix (12×12)
        # DOF: [u1x, u1y, u1z, θ1x, θ1y, θ1z, u2x, u2y, u2z, θ2x, θ2y, θ2z]
        k = np.zeros((12, 12))

        # Axial stiffness (along local x)
        ea = E * A / L
        k[0, 0] = ea;   k[0, 6] = -ea
        k[6, 0] = -ea;  k[6, 6] = ea

        # Torsion (about local x)
        gj = G * J / L
        k[3, 3] = gj;   k[3, 9] = -gj
        k[9, 3] = -gj;  k[9, 9] = gj

        # Bending about local y (deflection in local z)
        eiz = E * Iz
        k[1, 1] = 12*eiz/L**3;   k[1, 5] = 6*eiz/L**2;    k[1, 7] = -12*eiz/L**3;  k[1, 11] = 6*eiz/L**2
        k[5, 1] = 6*eiz/L**2;    k[5, 5] = 4*eiz/L;       k[5, 7] = -6*eiz/L**2;   k[5, 11] = 2*eiz/L
        k[7, 1] = -12*eiz/L**3;  k[7, 5] = -6*eiz/L**2;   k[7, 7] = 12*eiz/L**3;   k[7, 11] = -6*eiz/L**2
        k[11, 1] = 6*eiz/L**2;   k[11, 5] = 2*eiz/L;      k[11, 7] = -6*eiz/L**2;  k[11, 11] = 4*eiz/L

        # Bending about local z (deflection in local y)
        eiy = E * Iy
        k[2, 2] = 12*eiy/L**3;   k[2, 4] = -6*eiy/L**2;   k[2, 8] = -12*eiy/L**3;  k[2, 10] = -6*eiy/L**2
        k[4, 2] = -6*eiy/L**2;   k[4, 4] = 4*eiy/L;       k[4, 8] = 6*eiy/L**2;    k[4, 10] = 2*eiy/L
        k[8, 2] = -12*eiy/L**3;  k[8, 4] = 6*eiy/L**2;    k[8, 8] = 12*eiy/L**3;   k[8, 10] = 6*eiy/L**2
        k[10, 2] = -6*eiy/L**2;  k[10, 4] = 2*eiy/L;      k[10, 8] = 6*eiy/L**2;   k[10, 10] = 4*eiy/L

        # Transformation to global coordinates
        # Build 12×12 transformation matrix
        T = np.zeros((12, 12))
        for block in range(4):  # 4 blocks: u1, θ1, u2, θ2
            row = block * 3
            col = block * 3
            T[row:row+3, col:col+3] = R

        # Global stiffness = T^T * k * T
        k_global = T.T @ k @ T
        return k_global

    def assemble_global_stiffness(self, edge_index: np.ndarray,
                                   node_pos: np.ndarray,
                                   radii: np.ndarray) -> np.ndarray:
        """Assemble global stiffness matrix K.

        Parameters
        ----------
        edge_index : (2, n_edges) directed edge connectivity (bidirectional)
        node_pos : (n_nodes, dim) node positions
        radii : (n_edges,) element radii

        Returns
        -------
        K : (n_dof, n_dof) global stiffness matrix (dense)
        """
        n_nodes = node_pos.shape[0]
        n_dof = n_nodes * self.dof_per_node
        K = np.zeros((n_dof, n_dof))

        for e in range(edge_index.shape[1]):
            i = int(edge_index[0, e])
            j = int(edge_index[1, e])

            # Element geometry
            d = node_pos[j] - node_pos[i]
            L = np.linalg.norm(d)
            if L < 1e-12:
                continue

            # Section properties
            sec = self.circular_section_properties(float(radii[e]))

            if self.dim == 2:
                c = d[0] / L
                s = d[1] / L
                ke = self._beam_element_stiffness_2d(L, c, s, sec['A'], sec['Iz'])

                # Global DOF mapping
                dofs_i = [i * 3, i * 3 + 1, i * 3 + 2]
                dofs_j = [j * 3, j * 3 + 1, j * 3 + 2]
                dofs = dofs_i + dofs_j
            else:
                direction = d / L
                ke = self._beam_element_stiffness_3d(
                    L, direction, sec['A'], sec['Iy'], sec['Iz'], sec['J'])

                # Global DOF mapping
                dofs_i = list(range(i * 6, (i + 1) * 6))
                dofs_j = list(range(j * 6, (j + 1) * 6))
                dofs = dofs_i + dofs_j

            # Scatter into global matrix
            for a in range(self.dof_per_element):
                for b in range(self.dof_per_element):
                    K[dofs[a], dofs[b]] += ke[a, b]

        return K

    def solve(self, edge_index, node_pos, radii, forces,
              fixed_nodes, moments=None) -> Tuple:
        """Solve Ku = f with boundary conditions.

        Parameters
        ----------
        edge_index : (2, n_edges) torch.Tensor or numpy array
        node_pos : (n_nodes, dim) torch.Tensor or numpy array
        radii : (n_edges,) torch.Tensor or numpy array
        forces : (n_nodes, dim) external forces (torch.Tensor or numpy array)
        fixed_nodes : list/tensor of fixed node indices
        moments : (n_nodes, n_mom) optional nodal moments
            2D: (n_nodes, 1) — moment about z
            3D: (n_nodes, 3) — moments about x, y, z

        Returns
        -------
        2D: (displacements, axial_stresses, bending_moments)
            displacements : (n_nodes, 3) — [ux, uy, θz]
            axial_stresses : (n_edges,) — σ = E * ε_axial
            bending_moments : (n_edges, 2) — M at node i and j end
        3D: (displacements, axial_stresses, bending_moments)
            displacements : (n_nodes, 6) — [ux, uy, uz, θx, θy, θz]
            axial_stresses : (n_edges,)
            bending_moments : (n_edges, 4) — [My_i, My_j, Mz_i, Mz_j]
        """
        # Convert tensors to numpy
        import torch
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.numpy()
        if isinstance(node_pos, torch.Tensor):
            node_pos = node_pos.numpy()
        if isinstance(radii, torch.Tensor):
            radii = radii.numpy()
        if isinstance(forces, torch.Tensor):
            forces = forces.numpy()
        if isinstance(fixed_nodes, torch.Tensor):
            fixed_nodes = fixed_nodes.numpy().tolist()

        edge_index = np.asarray(edge_index)
        node_pos = np.asarray(node_pos)
        radii = np.asarray(radii)
        forces = np.asarray(forces)

        # Deduplicate bidirectional edges (fiber network graphs use bidirectional edges)
        edge_index, radii, _ = deduplicate_edges(edge_index, radii)

        n_nodes = node_pos.shape[0]
        n_dof = n_nodes * self.dof_per_node

        # Assemble
        K = self.assemble_global_stiffness(edge_index, node_pos, radii)

        # Add damping
        K += self.damping * np.eye(n_dof)

        # Build force vector (including moments)
        f = np.zeros(n_dof)
        for nn in range(n_nodes):
            for d in range(self.dim):
                f[nn * self.dof_per_node + d] = forces[nn, d]

        if moments is not None:
            if isinstance(moments, torch.Tensor):
                moments = moments.numpy()
            moments = np.asarray(moments)
            for nn in range(n_nodes):
                if self.dim == 2:
                    f[nn * 3 + 2] += moments[nn] if moments.ndim == 1 else moments[nn, 0]
                else:
                    for d in range(3):
                        f[nn * 6 + 3 + d] += moments[nn, d] if moments.ndim > 1 else moments[nn]

        # Apply boundary conditions
        fixed_dofs = set()
        for fn in fixed_nodes:
            fn_val = int(fn)
            for d in range(self.dof_per_node):
                fixed_dofs.add(fn_val * self.dof_per_node + d)

        free_dofs = np.array(sorted(set(range(n_dof)) - fixed_dofs))

        if len(free_dofs) == 0:
            # All DOFs fixed
            u = np.zeros((n_nodes, self.dof_per_node))
            sigma = np.zeros(edge_index.shape[1])
            if self.dim == 2:
                return u, sigma, np.zeros((edge_index.shape[1], 2))
            else:
                return u, sigma, np.zeros((edge_index.shape[1], 4))

        K_ff = K[free_dofs][:, free_dofs]
        f_f = f[free_dofs]

        # Robust solve
        u_f = self._robust_solve(K_ff, f_f)

        u_full = np.zeros(n_dof)
        u_full[free_dofs] = u_f
        u = u_full.reshape(n_nodes, self.dof_per_node)

        # Compute element forces
        sigma, moments_out = self._compute_element_forces(
            u, edge_index, node_pos, radii)

        return u, sigma, moments_out

    def _robust_solve(self, K: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Robust linear solve using pseudoinverse for near-singular systems."""
        n = K.shape[0]
        # Try direct solve first (faster for well-conditioned systems)
        try:
            K_sparse = sparse.csr_matrix(K)
            u = spsolve(K_sparse, f)
            if np.all(np.isfinite(u)):
                return u
        except Exception:
            pass

        # Fall back to pseudoinverse
        rcond = n * np.finfo(np.float32).eps * 10
        K_pinv = np.linalg.pinv(K, rcond=rcond)
        return K_pinv @ f

    def _compute_element_forces(self, u: np.ndarray, edge_index: np.ndarray,
                                 node_pos: np.ndarray,
                                 radii: np.ndarray) -> Tuple:
        """Compute axial stresses and bending moments for each element.

        Returns
        -------
        sigma : (n_edges,) axial stress
        moments : (n_edges, 2) for 2D or (n_edges, 4) for 3D
        """
        n_edges = edge_index.shape[1]
        sigma = np.zeros(n_edges)

        if self.dim == 2:
            moments = np.zeros((n_edges, 2))  # Mi, Mj (bending moment at each end)
        else:
            moments = np.zeros((n_edges, 4))  # My_i, My_j, Mz_i, Mz_j

        for e in range(n_edges):
            i = int(edge_index[0, e])
            j = int(edge_index[1, e])

            d = node_pos[j] - node_pos[i]
            L = np.linalg.norm(d)
            if L < 1e-12:
                continue

            sec = self.circular_section_properties(float(radii[e]))

            if self.dim == 2:
                c = d[0] / L
                s = d[1] / L

                # Extract nodal displacements in local coords
                ui = u[i]  # [ux, uy, θz]
                uj = u[j]

                # Transform to local
                u1_local = c * ui[0] + s * ui[1]
                v1_local = -s * ui[0] + c * ui[1]
                t1_local = ui[2]
                u2_local = c * uj[0] + s * uj[1]
                v2_local = -s * uj[0] + c * uj[1]
                t2_local = uj[2]

                # Axial strain
                eps_axial = (u2_local - u1_local) / L
                sigma[e] = self.E * eps_axial

                # Bending moments (Euler-Bernoulli)
                EI = self.E * sec['Iz']
                # M_i = EI/L² * (6/L*(v1-v2) + 4*θ1 + 2*θ2)
                # M_j = EI/L² * (6/L*(v1-v2) - 2*θ1 - 4*θ2)
                # Using standard beam formula:
                M_i = EI / L**2 * (6/L * (v1_local - v2_local) + 4*t1_local + 2*t2_local) * L
                M_j = EI / L**2 * (6/L * (v1_local - v2_local) - 2*t1_local - 4*t2_local) * L
                # Simplify:
                M_i = EI * (6 * (v1_local - v2_local) / L**2 + 4 * t1_local / L + 2 * t2_local / L)
                M_j = EI * (6 * (v1_local - v2_local) / L**2 - 2 * t1_local / L - 4 * t2_local / L)
                moments[e, 0] = M_i
                moments[e, 1] = M_j

            else:
                direction = d / L
                # Local coordinate system
                e1 = direction
                if abs(e1[0]) < 0.9:
                    ref = np.array([1.0, 0.0, 0.0])
                else:
                    ref = np.array([0.0, 1.0, 0.0])
                e2 = np.cross(e1, ref)
                e2 /= np.linalg.norm(e2)
                e3 = np.cross(e1, e2)
                e3 /= np.linalg.norm(e3)
                R = np.array([e1, e2, e3])

                # Extract displacements and rotations
                di = u[i, :3]  # translation
                ri = u[i, 3:]  # rotation
                dj = u[j, :3]
                rj = u[j, 3:]

                # Transform to local
                di_local = R @ di
                dj_local = R @ dj
                ri_local = R @ ri
                rj_local = R @ rj

                # Axial strain
                eps_axial = (dj_local[0] - di_local[0]) / L
                sigma[e] = self.E * eps_axial

                # Bending moments
                EIy = self.E * sec['Iy']
                EIz = self.E * sec['Iz']

                # Bending about local y (deflection in z)
                My_i = EIz * (6 * (di_local[2] - dj_local[2]) / L**2
                             + 4 * ri_local[1] / L + 2 * rj_local[1] / L)
                My_j = EIz * (6 * (di_local[2] - dj_local[2]) / L**2
                             - 2 * ri_local[1] / L - 4 * rj_local[1] / L)

                # Bending about local z (deflection in y)
                Mz_i = EIy * (6 * (di_local[1] - dj_local[1]) / L**2
                             - 4 * ri_local[2] / L - 2 * rj_local[2] / L)
                Mz_j = EIy * (6 * (di_local[1] - dj_local[1]) / L**2
                             + 2 * ri_local[2] / L + 4 * rj_local[2] / L)

                moments[e] = [My_i, My_j, Mz_i, Mz_j]

        return sigma, moments

    def get_element_forces(self, u: np.ndarray, edge_index: np.ndarray,
                           node_pos: np.ndarray, radii: np.ndarray) -> Dict:
        """Get comprehensive element force data.

        Returns dict with:
            axial_force : (n_edges,) N = σ * A
            axial_stress : (n_edges,) σ
            shear_force : (n_edges,) V = (Mi + Mj) / L  (2D) or per plane (3D)
            bending_moment : (n_edges, 2) or (n_edges, 4)
            torsion : (n_edges,)  (3D only, zero for 2D)
            max_combined_stress : (n_edges,) σ_max = |N/A| + |M*c/I|
        """
        n_edges = edge_index.shape[1]
        sigma, moments = self._compute_element_forces(u, edge_index, node_pos, radii)

        axial_force = np.zeros(n_edges)
        shear_force = np.zeros(n_edges)
        torsion = np.zeros(n_edges)
        max_stress = np.zeros(n_edges)

        for e in range(n_edges):
            sec = self.circular_section_properties(float(radii[e]))
            c = float(radii[e])  # outer radius = extreme fiber distance
            axial_force[e] = sigma[e] * sec['A']

            if self.dim == 2:
                # Shear from bending moments
                i = int(edge_index[0, e])
                j = int(edge_index[1, e])
                L = np.linalg.norm(node_pos[j] - node_pos[i])
                if L > 1e-12:
                    shear_force[e] = abs(moments[e, 0] + moments[e, 1]) / L
                # Combined stress: axial + bending
                max_moment = max(abs(moments[e, 0]), abs(moments[e, 1]))
                max_stress[e] = abs(sigma[e]) + abs(max_moment * c / sec['Iz'])
            else:
                i = int(edge_index[0, e])
                j = int(edge_index[1, e])
                L = np.linalg.norm(node_pos[j] - node_pos[i])
                if L > 1e-12:
                    Vy = abs(moments[e, 2] + moments[e, 3]) / L
                    Vz = abs(moments[e, 0] + moments[e, 1]) / L
                    shear_force[e] = np.sqrt(Vy**2 + Vz**2)
                max_M = max(abs(moments[e, 0]), abs(moments[e, 1]),
                           abs(moments[e, 2]), abs(moments[e, 3]))
                max_stress[e] = abs(sigma[e]) + abs(max_M * c / sec['Iy'])

        return {
            'axial_force': axial_force,
            'axial_stress': sigma,
            'shear_force': shear_force,
            'bending_moment': moments,
            'torsion': torsion,
            'max_combined_stress': max_stress,
        }


def deduplicate_edges(edge_index, radii=None):
    """Remove duplicate bidirectional edges, keeping only one direction.

    Parameters
    ----------
    edge_index : (2, n_edges) array
    radii : (n_edges,) array, optional

    Returns
    -------
    unique_edge_index : (2, n_unique) array
    unique_radii : (n_unique,) array (if radii provided)
    unique_indices : list of indices into original edge_index
    """
    import numpy as np
    edge_index = np.asarray(edge_index)
    
    seen = set()
    unique_edges = []
    unique_idx = []
    
    for e in range(edge_index.shape[1]):
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            unique_edges.append([i, j])
            unique_idx.append(e)
    
    unique_edge_index = np.array(unique_edges).T
    
    if radii is not None:
        radii = np.asarray(radii)
        unique_radii = radii[unique_idx]
        return unique_edge_index, unique_radii, unique_idx
    else:
        return unique_edge_index, unique_idx
