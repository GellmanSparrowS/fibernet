"""Sparse Beam Frame FEM for large-scale fiber networks"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Dict, List, Tuple
import torch

class SparseBeamFrameFEM:
    """Sparse matrix implementation for beam frame FEM (scalable to large structures)"""
    
    def __init__(self, E=1e9, nu=0.3):
        """
        Args:
            E: Young's modulus (Pa)
            nu: Poisson's ratio
        """
        self.E = E
        self.nu = nu
        self.G = E / (2 * (1 + nu))  # Shear modulus
    
    def section_properties(self, r: float) -> Dict[str, float]:
        """Compute circular cross-section properties
        
        Args:
            r: fiber radius
            
        Returns:
            A: cross-sectional area
            I: second moment of area (bending stiffness)
            J: torsional constant
        """
        A = np.pi * r**2
        I = np.pi * r**4 / 4
        J = np.pi * r**4 / 2
        return {'A': A, 'I': I, 'J': J}
    
    def build_sparse_stiffness_2d(self, edge_index: np.ndarray, node_pos: np.ndarray, 
                                    radii: np.ndarray, deduplicate: bool = True) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """Build sparse global stiffness matrix for 2D beam frame
        
        Args:
            edge_index: (2, n_edges) edge connectivity
            node_pos: (n_nodes, 2) node positions
            radii: (n_edges,) fiber radii
            deduplicate: whether to remove duplicate edges
            
        Returns:
            K_global: (3*n_nodes, 3*n_nodes) sparse stiffness matrix
            edge_list: deduplicated edge indices
        """
        if deduplicate:
            # Remove duplicate edges
            seen = set()
            unique_edges = []
            for e in range(edge_index.shape[1]):
                i, j = edge_index[0, e], edge_index[1, e]
                key = (min(i, j), max(i, j))
                if key not in seen:
                    seen.add(key)
                    unique_edges.append(e)
            edge_list = np.array(unique_edges)
        else:
            edge_list = np.arange(edge_index.shape[1])
        
        n_nodes = node_pos.shape[0]
        n_dof = 3 * n_nodes
        
        # COO format for efficient assembly
        rows = []
        cols = []
        vals = []
        
        for e in edge_list:
            i, j = edge_index[0, e], edge_index[1, e]
            r = radii[e]
            
            # Element geometry
            dx = node_pos[j, 0] - node_pos[i, 0]
            dy = node_pos[j, 1] - node_pos[i, 1]
            L = np.sqrt(dx**2 + dy**2)
            
            if L < 1e-12:
                continue
            
            # Direction cosines
            cx, cy = dx / L, dy / L
            
            # Section properties
            props = self.section_properties(r)
            A, I = props['A'], props['I']
            
            # Local stiffness matrix (6x6 for 2D beam)
            # DOF order: [u1, v1, θ1, u2, v2, θ2]
            E, L2, L3 = self.E, L**2, L**3
            
            k_local = np.array([
                [E*A/L, 0, 0, -E*A/L, 0, 0],
                [0, 12*E*I/L3, 6*E*I/L2, 0, -12*E*I/L3, 6*E*I/L2],
                [0, 6*E*I/L2, 4*E*I/L, 0, -6*E*I/L2, 2*E*I/L],
                [-E*A/L, 0, 0, E*A/L, 0, 0],
                [0, -12*E*I/L3, -6*E*I/L2, 0, 12*E*I/L3, -6*E*I/L2],
                [0, 6*E*I/L2, 2*E*I/L, 0, -6*E*I/L2, 4*E*I/L]
            ])
            
            # Transformation matrix (local → global)
            T = np.array([
                [cx, cy, 0, 0, 0, 0],
                [-cy, cx, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, cx, cy, 0],
                [0, 0, 0, -cy, cx, 0],
                [0, 0, 0, 0, 0, 1]
            ])
            
            # Global element stiffness
            k_global = T.T @ k_local @ T
            
            # DOF indices
            dofs = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
            
            # Assemble into sparse matrix
            for a in range(6):
                for b in range(6):
                    rows.append(dofs[a])
                    cols.append(dofs[b])
                    vals.append(k_global[a, b])
        
        K_global = sparse.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
        return K_global.tocsr(), edge_list
    
    def solve_2d(self, edge_index, node_pos, radii, forces, fixed_nodes, 
                 damping=1e-6, deduplicate=True):
        """Solve 2D beam frame problem
        
        Args:
            edge_index: (2, n_edges) edge connectivity
            node_pos: (n_nodes, 2) node positions
            radii: (n_edges,) fiber radii
            forces: (n_nodes, 2) applied forces
            fixed_nodes: list of fixed node indices
            damping: regularization parameter
            deduplicate: whether to remove duplicate edges
            
        Returns:
            u: (n_nodes, 3) displacements [ux, uy, θ]
            sigma: (n_unique_edges,) axial stresses
            moments: (n_unique_edges, 2) bending moments at each end
        """
        # Convert to numpy if needed
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
        
        # Build stiffness matrix
        K_global, edge_list = self.build_sparse_stiffness_2d(
            edge_index, node_pos, radii, deduplicate=deduplicate
        )
        
        n_nodes = node_pos.shape[0]
        n_dof = 3 * n_nodes
        
        # Add damping (regularization)
        K_global = K_global + damping * sparse.eye(n_dof, format='csr')
        
        # Build force vector
        f_global = np.zeros(n_dof)
        f_global[0::3] = forces[:, 0]  # fx
        f_global[1::3] = forces[:, 1]  # fy
        # No moments applied (f_global[2::3] = 0)
        
        # Apply boundary conditions
        all_dofs = np.arange(n_dof)
        fixed_dofs = []
        for node in fixed_nodes:
            fixed_dofs.extend([3*node, 3*node+1, 3*node+2])
        fixed_dofs = np.array(fixed_dofs)
        
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
        
        # Reduced system
        K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
        f_reduced = f_global[free_dofs]
        
        # Solve
        u_reduced = spsolve(K_reduced, f_reduced)
        
        # Full solution
        u_full = np.zeros(n_dof)
        u_full[free_dofs] = u_reduced
        
        # Reshape to (n_nodes, 3)
        u = u_full.reshape(n_nodes, 3)
        
        # Compute stresses and moments
        sigma = np.zeros(len(edge_list))
        moments = np.zeros((len(edge_list), 2))
        
        for idx, e in enumerate(edge_list):
            i, j = edge_index[0, e], edge_index[1, e]
            r = radii[e]
            
            # Element geometry
            dx = node_pos[j, 0] - node_pos[i, 0]
            dy = node_pos[j, 1] - node_pos[i, 1]
            L = np.sqrt(dx**2 + dy**2)
            
            if L < 1e-12:
                continue
            
            # Direction cosines
            cx, cy = dx / L, dy / L
            
            # Section properties
            props = self.section_properties(r)
            A, I = props['A'], props['I']
            
            # Local displacements
            ui_local = np.array([cx*u[i,0] + cy*u[i,1], -cy*u[i,0] + cx*u[i,1], u[i,2]])
            uj_local = np.array([cx*u[j,0] + cy*u[j,1], -cy*u[j,0] + cx*u[j,1], u[j,2]])
            
            # Axial strain
            eps = (uj_local[0] - ui_local[0]) / L
            sigma[idx] = self.E * eps
            
            # Bending moments (from local displacements)
            E, L2, L3 = self.E, L**2, L**3
            M_i = E*I/L2 * (6*(ui_local[1] - uj_local[1])/L + 4*ui_local[2] + 2*uj_local[2])
            M_j = E*I/L2 * (6*(ui_local[1] - uj_local[1])/L + 2*ui_local[2] + 4*uj_local[2])
            moments[idx] = [M_i, M_j]
        
        return u, sigma, moments, edge_list
    
    def solve_3d(self, edge_index, node_pos, radii, forces, fixed_nodes, 
                 damping=1e-6, deduplicate=True):
        """Solve 3D beam frame problem (6 DOF per node)"""
        # Convert to numpy if needed
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
        
        # Deduplicate edges
        if deduplicate:
            seen = set()
            unique_edges = []
            for e in range(edge_index.shape[1]):
                i, j = edge_index[0, e], edge_index[1, e]
                key = (min(i, j), max(i, j))
                if key not in seen:
                    seen.add(key)
                    unique_edges.append(e)
            edge_list = np.array(unique_edges)
        else:
            edge_list = np.arange(edge_index.shape[1])
        
        n_nodes = node_pos.shape[0]
        n_dof = 6 * n_nodes
        
        # COO format for assembly
        rows = []
        cols = []
        vals = []
        
        for e in edge_list:
            i, j = edge_index[0, e], edge_index[1, e]
            r = radii[e]
            
            # Element geometry
            dx = node_pos[j] - node_pos[i]
            L = np.linalg.norm(dx)
            
            if L < 1e-12:
                continue
            
            # Local coordinate system
            e1 = dx / L
            if abs(e1[0]) < 0.9:
                ref = np.array([1.0, 0, 0])
            else:
                ref = np.array([0, 1.0, 0])
            e2 = np.cross(e1, ref)
            e2 /= np.linalg.norm(e2)
            e3 = np.cross(e1, e2)
            
            # Section properties
            props = self.section_properties(r)
            A, I, J = props['A'], props['I'], props['J']
            
            # Local stiffness (12x12 for 3D beam)
            E, G = self.E, self.G
            L2, L3 = L**2, L**3
            
            # Build local stiffness matrix (simplified diagonal-dominant form)
            k_local = np.zeros((12, 12))
            
            # Axial
            k_local[0, 0] = k_local[6, 6] = E*A/L
            k_local[0, 6] = k_local[6, 0] = -E*A/L
            
            # Torsion
            k_local[3, 3] = k_local[9, 9] = G*J/L
            k_local[3, 9] = k_local[9, 3] = -G*J/L
            
            # Bending about local y-axis (deflection in z)
            k_local[1, 1] = k_local[7, 7] = 12*E*I/L3
            k_local[1, 7] = k_local[7, 1] = -12*E*I/L3
            k_local[1, 5] = k_local[5, 1] = 6*E*I/L2
            k_local[1, 11] = k_local[11, 1] = 6*E*I/L2
            k_local[7, 5] = k_local[5, 7] = -6*E*I/L2
            k_local[7, 11] = k_local[11, 7] = -6*E*I/L2
            k_local[5, 5] = k_local[11, 11] = 4*E*I/L
            k_local[5, 11] = k_local[11, 5] = 2*E*I/L
            
            # Bending about local z-axis (deflection in y)
            k_local[2, 2] = k_local[8, 8] = 12*E*I/L3
            k_local[2, 8] = k_local[8, 2] = -12*E*I/L3
            k_local[2, 4] = k_local[4, 2] = -6*E*I/L2
            k_local[2, 10] = k_local[10, 2] = -6*E*I/L2
            k_local[8, 4] = k_local[4, 8] = 6*E*I/L2
            k_local[8, 10] = k_local[10, 8] = 6*E*I/L2
            k_local[4, 4] = k_local[10, 10] = 4*E*I/L
            k_local[4, 10] = k_local[10, 4] = 2*E*I/L
            
            # Transformation matrix (12x12)
            T = np.zeros((12, 12))
            for k in range(4):  # 4 nodes (2 endpoints × 2 sets of DOFs)
                T[3*k:3*k+3, 3*k:3*k+3] = np.array([e1, e2, e3])
            
            # Global element stiffness
            k_global = T.T @ k_local @ T
            
            # DOF indices
            dofs = [6*i + k for k in range(6)] + [6*j + k for k in range(6)]
            
            # Assemble
            for a in range(12):
                for b in range(12):
                    rows.append(dofs[a])
                    cols.append(dofs[b])
                    vals.append(k_global[a, b])
        
        # Build sparse matrix
        K_global = sparse.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
        K_global = K_global.tocsr()
        
        # Add damping
        K_global = K_global + damping * sparse.eye(n_dof, format='csr')
        
        # Build force vector
        f_global = np.zeros(n_dof)
        f_global[0::6] = forces[:, 0]  # fx
        f_global[1::6] = forces[:, 1]  # fy
        f_global[2::6] = forces[:, 2]  # fz
        # No moments applied
        
        # Apply boundary conditions
        all_dofs = np.arange(n_dof)
        fixed_dofs = []
        for node in fixed_nodes:
            fixed_dofs.extend([6*node + k for k in range(6)])
        fixed_dofs = np.array(fixed_dofs)
        
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
        
        # Reduced system
        K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
        f_reduced = f_global[free_dofs]
        
        # Solve
        u_reduced = spsolve(K_reduced, f_reduced)
        
        # Full solution
        u_full = np.zeros(n_dof)
        u_full[free_dofs] = u_reduced
        
        # Reshape to (n_nodes, 6)
        u = u_full.reshape(n_nodes, 6)
        
        # Compute axial stresses
        sigma = np.zeros(len(edge_list))
        for idx, e in enumerate(edge_list):
            i, j = edge_index[0, e], edge_index[1, e]
            r = radii[e]
            
            dx = node_pos[j] - node_pos[i]
            L = np.linalg.norm(dx)
            
            if L < 1e-12:
                continue
            
            e1 = dx / L
            props = self.section_properties(r)
            A = props['A']
            
            # Local displacements
            ui_local = e1 @ u[i, :3]
            uj_local = e1 @ u[j, :3]
            
            # Axial strain
            eps = (uj_local - ui_local) / L
            sigma[idx] = self.E * eps
        
        # No moments computed for 3D yet (TODO)
        moments = None
        
        return u, sigma, moments, edge_list
