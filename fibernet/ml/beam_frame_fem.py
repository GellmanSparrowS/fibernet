"""
Beam Frame FEM - Corrected & Enhanced
==========================================
Fixes from v4/v5:
  1. Corrected moment formula (shape function 2nd derivatives, not stiffness eq)
  2. Added bending stress sigma = M*c/I (was missing)
  3. Added displacement BC support (prescribed non-zero displacements)
  4. Fixed nonlinear solver stress computation
  5. Per-node stress/strain for graph-level analysis
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Dict, List, Tuple, Optional
import torch


class BeamFrameFEM:
    """Corrected beam frame FEM with welded joints, displacement BCs, nonlinear solver."""
    
    def __init__(self, E: float = 1e9, nu: float = 0.3):
        self.E = E
        self.nu = nu
        self.G = E / (2 * (1 + nu))
    
    @staticmethod
    def section_properties(r: float) -> Dict[str, float]:
        A = np.pi * r**2
        I = np.pi * r**4 / 4
        J = np.pi * r**4 / 2
        return {'A': A, 'I': I, 'J': J, 'r': r}
    
    @staticmethod
    def _validate_nodes(n_nodes, fixed_nodes, prescribed_disp):
        """Validate node indices against n_nodes. Raises ValueError for out-of-range."""
        bad_fixed = [n for n in fixed_nodes if int(n) < 0 or int(n) >= n_nodes]
        if bad_fixed:
            raise ValueError(
                f"fixed_nodes contains invalid indices {bad_fixed} "
                f"(n_nodes={n_nodes}, valid range: 0..{n_nodes-1})"
            )
        bad_prescribed = [n for n in prescribed_disp if int(n) < 0 or int(n) >= n_nodes]
        if bad_prescribed:
            raise ValueError(
                f"prescribed_disp contains invalid node indices {bad_prescribed} "
                f"(n_nodes={n_nodes}, valid range: 0..{n_nodes-1})"
            )

    def _deduplicate_edges(self, edge_index: np.ndarray) -> np.ndarray:
        seen = set()
        unique = []
        for e in range(edge_index.shape[1]):
            i, j = int(edge_index[0, e]), int(edge_index[1, e])
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return np.array(unique)
    
    def build_stiffness_2d(self, edge_index, node_pos, radii, deduplicate=True):
        if deduplicate:
            edge_list = self._deduplicate_edges(edge_index)
        else:
            edge_list = np.arange(edge_index.shape[1])
        
        n_nodes = node_pos.shape[0]
        n_dof = 3 * n_nodes
        rows, cols, vals = [], [], []
        
        for e in edge_list:
            i, j = int(edge_index[0, e]), int(edge_index[1, e])
            r = radii[e]
            dx = node_pos[j, 0] - node_pos[i, 0]
            dy = node_pos[j, 1] - node_pos[i, 1]
            L = np.sqrt(dx**2 + dy**2)
            if L < 1e-12:
                continue
            cx, cy = dx / L, dy / L
            props = self.section_properties(r)
            A, I = props['A'], props['I']
            E = self.E
            L2, L3 = L**2, L**3
            
            k_local = np.array([
                [E*A/L, 0, 0, -E*A/L, 0, 0],
                [0, 12*E*I/L3, 6*E*I/L2, 0, -12*E*I/L3, 6*E*I/L2],
                [0, 6*E*I/L2, 4*E*I/L, 0, -6*E*I/L2, 2*E*I/L],
                [-E*A/L, 0, 0, E*A/L, 0, 0],
                [0, -12*E*I/L3, -6*E*I/L2, 0, 12*E*I/L3, -6*E*I/L2],
                [0, 6*E*I/L2, 2*E*I/L, 0, -6*E*I/L2, 4*E*I/L]
            ])
            
            T = np.array([
                [cx, cy, 0, 0, 0, 0],
                [-cy, cx, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, cx, cy, 0],
                [0, 0, 0, -cy, cx, 0],
                [0, 0, 0, 0, 0, 1]
            ])
            
            k_global = T.T @ k_local @ T
            dofs = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
            
            for a in range(6):
                for b in range(6):
                    if abs(k_global[a, b]) > 0:
                        rows.append(dofs[a])
                        cols.append(dofs[b])
                        vals.append(k_global[a, b])
        
        K = sparse.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
        return K, edge_list
    
    def _compute_element_stress_2d(self, node_pos, u, edge_index, edge_list, radii):
        """Compute stresses using shape function 2nd derivatives (correct formula).
        
        M(ξ) = EI * N''(ξ) * d_local
        where N'' are second derivatives of Hermite shape functions.
        """
        n_edges = len(edge_list)
        sigma_axial = np.zeros(n_edges)
        sigma_bending = np.zeros(n_edges)
        moments = np.zeros((n_edges, 2))
        edge_forces = np.zeros((n_edges, 3))  # [axial_N, shear_V, moment_avg]
        
        for idx, e in enumerate(edge_list):
            i, j = int(edge_index[0, e]), int(edge_index[1, e])
            r = radii[e]
            
            dx = node_pos[j, 0] - node_pos[i, 0]
            dy = node_pos[j, 1] - node_pos[i, 1]
            L = np.sqrt(dx**2 + dy**2)
            if L < 1e-12:
                continue
            
            cx, cy = dx / L, dy / L
            props = self.section_properties(r)
            A, I_val = props['A'], props['I']
            
            # Transform to local frame
            ui_local = np.array([cx*u[i,0]+cy*u[i,1], -cy*u[i,0]+cx*u[i,1], u[i,2]])
            uj_local = np.array([cx*u[j,0]+cy*u[j,1], -cy*u[j,0]+cx*u[j,1], u[j,2]])
            
            v_i, th_i = ui_local[1], ui_local[2]
            v_j, th_j = uj_local[1], uj_local[2]
            
            # Axial strain and stress
            eps_axial = (uj_local[0] - ui_local[0]) / L
            sigma_axial[idx] = self.E * eps_axial
            
            # Bending moment from shape function 2nd derivatives
            # M(ξ=0) = EI * N''(0) * d_local
            # N''(0) = [12/L², 6/L, -12/L², 6/L]
            # M(ξ=1) = EI * N''(1) * d_local
            # N''(1) = [12/L², -6/L, -12/L², -6/L]
            d_trans = np.array([v_i, th_i, v_j, th_j])
            
            Npp_left = np.array([12/L**2, 6/L, -12/L**2, 6/L])
            Npp_right = np.array([12/L**2, -6/L, -12/L**2, -6/L])
            
            # Sign convention: M = -EI * v'' for standard beam
            # But for internal moment, use: M = EI * N'' * d (positive = sagging)
            M_i = -self.E * I_val * np.dot(Npp_left, d_trans)
            M_j = -self.E * I_val * np.dot(Npp_right, d_trans)
            
            moments[idx] = [M_i, M_j]
            
            # Bending stress at outer fiber
            c = r
            sigma_b_i = abs(M_i) * c / I_val
            sigma_b_j = abs(M_j) * c / I_val
            sigma_bending[idx] = max(sigma_b_i, sigma_b_j)
            
            # Axial force and shear
            N = self.E * A * eps_axial
            # Shear from V = dM/dx ≈ (M_j - M_i) / L
            V = (M_j - M_i) / L
            edge_forces[idx] = [N, V, (M_i + M_j) / 2]
        
        sigma_total = np.abs(sigma_axial) + sigma_bending
        return sigma_axial, sigma_bending, sigma_total, moments, edge_forces
    
    def solve_2d(self, edge_index, node_pos, radii,
                 forces=None, fixed_nodes=None,
                 prescribed_disp=None,
                 damping=1e-6, deduplicate=True):
        """Solve 2D beam frame with force and/or displacement BCs.
        
        Returns dict with:
            u: (n_nodes, 3) [ux, uy, theta]
            sigma_axial, sigma_bending, sigma_total: per-edge stresses
            moments: (n_edges, 2) bending moments at ends
            node_stress: (n_nodes,) max stress at each node
            reactions: (n_nodes, 3) reaction forces/moments
        """
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.numpy()
        if isinstance(node_pos, torch.Tensor):
            node_pos = node_pos.numpy()
        if isinstance(radii, torch.Tensor):
            radii = radii.numpy()
        if forces is not None and isinstance(forces, torch.Tensor):
            forces = forces.numpy()
        
        if fixed_nodes is None:
            fixed_nodes = []
        if prescribed_disp is None:
            prescribed_disp = {}
        if forces is None:
            forces = np.zeros((node_pos.shape[0], 2))
        
        self._validate_nodes(node_pos.shape[0], fixed_nodes, prescribed_disp)
        
        K, edge_list = self.build_stiffness_2d(edge_index, node_pos, radii, deduplicate)
        n_nodes = node_pos.shape[0]
        n_dof = 3 * n_nodes
        
        f_global = np.zeros(n_dof)
        f_global[0::3] = forces[:, 0]
        f_global[1::3] = forces[:, 1]
        
        fixed_dofs = set()
        prescribed_dof_values = {}
        
        for node in fixed_nodes:
            for k in range(3):
                dof = 3 * int(node) + k
                fixed_dofs.add(dof)
                prescribed_dof_values[dof] = 0.0
        
        for node, (ux, uy) in prescribed_disp.items():
            dof_x = 3 * int(node)
            dof_y = 3 * int(node) + 1
            fixed_dofs.add(dof_x)
            fixed_dofs.add(dof_y)
            prescribed_dof_values[dof_x] = ux
            prescribed_dof_values[dof_y] = uy
        
        fixed_dofs = np.array(sorted(fixed_dofs))
        all_dofs = np.arange(n_dof)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
        
        f_modified = f_global.copy()
        K_damped = K + damping * sparse.eye(n_dof, format='csr')
        
        for dof, val in prescribed_dof_values.items():
            if val != 0.0:
                col = K_damped[:, dof].toarray().flatten()
                f_modified -= col * val
        
        K_reduced = K_damped[np.ix_(free_dofs, free_dofs)]
        f_reduced = f_modified[free_dofs]
        u_reduced = spsolve(K_reduced, f_reduced)
        
        u_full = np.zeros(n_dof)
        u_full[free_dofs] = u_reduced
        for dof, val in prescribed_dof_values.items():
            u_full[dof] = val
        
        u = u_full.reshape(n_nodes, 3)
        
        # Compute stresses with corrected formula
        sigma_axial, sigma_bending, sigma_total, moments, edge_forces = \
            self._compute_element_stress_2d(node_pos, u, edge_index, edge_list, radii)
        
        # Per-node max stress
        node_stress = np.zeros(n_nodes)
        node_moment = np.zeros(n_nodes)
        for idx, e in enumerate(edge_list):
            i, j = int(edge_index[0, e]), int(edge_index[1, e])
            node_stress[i] = max(node_stress[i], sigma_total[idx])
            node_stress[j] = max(node_stress[j], sigma_total[idx])
            node_moment[i] = max(node_moment[i], abs(moments[idx, 0]))
            node_moment[j] = max(node_moment[j], abs(moments[idx, 1]))
        
        # Reactions
        reactions = (K @ u_full - f_global).reshape(n_nodes, 3)
        
        return {
            'u': u, 'sigma_axial': sigma_axial, 'sigma_bending': sigma_bending,
            'sigma_total': sigma_total, 'moments': moments,
            'edge_forces': edge_forces, 'node_stress': node_stress,
            'node_moment': node_moment, 'reactions': reactions,
            'edge_list': edge_list, 'K': K,
            'n_nodes': n_nodes, 'n_edges': len(edge_list)
        }
    
    def solve_2d_nonlinear(self, edge_index, node_pos, radii,
                           prescribed_disp, fixed_nodes=None,
                           forces=None, n_steps=10, tol=1e-6, max_iter=20,
                           damping=1e-6, deduplicate=True):
        """Geometrically nonlinear solver using incremental co-rotational approach.
        
        For each increment:
          1. Apply fraction of total prescribed displacement
          2. Solve linear system at current configuration
          3. Update node positions
          4. Repeat
        """
        if isinstance(node_pos, torch.Tensor):
            node_pos = node_pos.numpy().copy()
        else:
            node_pos = node_pos.copy()
        if isinstance(edge_index, torch.Tensor):
            edge_index_np = edge_index.numpy()
        else:
            edge_index_np = edge_index
        
        if fixed_nodes is None:
            fixed_nodes = []
        
        n_nodes = node_pos.shape[0]
        self._validate_nodes(n_nodes, fixed_nodes, prescribed_disp)
        current_pos = node_pos.copy()
        total_u = np.zeros((n_nodes, 3))
        
        step_disp = {}
        for node, (ux, uy) in prescribed_disp.items():
            step_disp[node] = (ux / n_steps, uy / n_steps)
        
        history = []
        
        for step in range(n_steps):
            step_prescribed = {}
            for node, (dux, duy) in step_disp.items():
                step_prescribed[node] = (dux, duy)
            
            for iteration in range(max_iter):
                result = self.solve_2d(
                    edge_index_np, current_pos, radii,
                    forces=forces, fixed_nodes=fixed_nodes,
                    prescribed_disp=step_prescribed,
                    damping=damping, deduplicate=deduplicate
                )
                
                u_step = result['u']
                du_norm = np.linalg.norm(u_step[:, :2])
                
                # Update positions
                current_pos[:, 0] += u_step[:, 0]
                current_pos[:, 1] += u_step[:, 1]
                total_u[:, 0] += u_step[:, 0]
                total_u[:, 1] += u_step[:, 1]
                total_u[:, 2] += u_step[:, 2]
                
                if du_norm < tol * (1.0 + np.linalg.norm(total_u[:, :2])):
                    break
                
                step_prescribed = {}
            
            max_disp = np.max(np.linalg.norm(total_u[:, :2], axis=1))
            max_stress = np.max(result['sigma_total']) if result['sigma_total'] is not None else 0
            
            history.append({
                'step': step, 'max_disp': float(max_disp),
                'max_stress': float(max_stress), 'iterations': iteration + 1
            })
        
        # Final stress computation from total displacement on original geometry
        sigma_axial, sigma_bending, sigma_total, moments, edge_forces = \
            self._compute_element_stress_2d(
                node_pos, total_u, edge_index_np,
                result['edge_list'], radii
            )
        
        node_stress = np.zeros(n_nodes)
        for idx, e in enumerate(result['edge_list']):
            i, j = int(edge_index_np[0, e]), int(edge_index_np[1, e])
            node_stress[i] = max(node_stress[i], sigma_total[idx])
            node_stress[j] = max(node_stress[j], sigma_total[idx])
        
        result['u'] = total_u
        result['u_total'] = total_u
        result['deformed_pos'] = current_pos
        result['sigma_axial'] = sigma_axial
        result['sigma_bending'] = sigma_bending
        result['sigma_total'] = sigma_total
        result['moments'] = moments
        result['edge_forces'] = edge_forces
        result['node_stress'] = node_stress
        result['history'] = history
        
        return result
    
    def solve_3d(self, edge_index, node_pos, radii,
                 forces=None, fixed_nodes=None,
                 prescribed_disp=None,
                 damping=1e-6, deduplicate=True):
        """Solve 3D beam frame with corrected stress computation."""
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.numpy()
        if isinstance(node_pos, torch.Tensor):
            node_pos = node_pos.numpy()
        if isinstance(radii, torch.Tensor):
            radii = radii.numpy()
        if forces is not None and isinstance(forces, torch.Tensor):
            forces = forces.numpy()
        
        if fixed_nodes is None:
            fixed_nodes = []
        if prescribed_disp is None:
            prescribed_disp = {}
        if forces is None:
            forces = np.zeros((node_pos.shape[0], 3))
        
        self._validate_nodes(node_pos.shape[0], fixed_nodes, prescribed_disp)
        
        edge_list = self._deduplicate_edges(edge_index) if deduplicate else np.arange(edge_index.shape[1])
        n_nodes = node_pos.shape[0]
        n_dof = 6 * n_nodes
        rows, cols, vals = [], [], []
        
        for e in edge_list:
            i, j = int(edge_index[0, e]), int(edge_index[1, e])
            r = radii[e]
            dx = node_pos[j] - node_pos[i]
            L = np.linalg.norm(dx)
            if L < 1e-12:
                continue
            e1 = dx / L
            ref = np.array([0, 1.0, 0]) if abs(e1[0]) > 0.9 else np.array([1.0, 0, 0])
            e2 = np.cross(e1, ref); e2 /= np.linalg.norm(e2)
            e3 = np.cross(e1, e2)
            
            props = self.section_properties(r)
            A, I, J = props['A'], props['I'], props['J']
            E, G = self.E, self.G
            L2, L3 = L**2, L**3
            
            k_local = np.zeros((12, 12))
            k_local[0,0]=k_local[6,6]=E*A/L; k_local[0,6]=k_local[6,0]=-E*A/L
            k_local[3,3]=k_local[9,9]=G*J/L; k_local[3,9]=k_local[9,3]=-G*J/L
            k_local[1,1]=k_local[7,7]=12*E*I/L3; k_local[1,7]=k_local[7,1]=-12*E*I/L3
            k_local[1,5]=k_local[5,1]=6*E*I/L2; k_local[1,11]=k_local[11,1]=6*E*I/L2
            k_local[7,5]=k_local[5,7]=-6*E*I/L2; k_local[7,11]=k_local[11,7]=-6*E*I/L2
            k_local[5,5]=k_local[11,11]=4*E*I/L; k_local[5,11]=k_local[11,5]=2*E*I/L
            k_local[2,2]=k_local[8,8]=12*E*I/L3; k_local[2,8]=k_local[8,2]=-12*E*I/L3
            k_local[2,4]=k_local[4,2]=-6*E*I/L2; k_local[2,10]=k_local[10,2]=-6*E*I/L2
            k_local[8,4]=k_local[4,8]=6*E*I/L2; k_local[8,10]=k_local[10,8]=6*E*I/L2
            k_local[4,4]=k_local[10,10]=4*E*I/L; k_local[4,10]=k_local[10,4]=2*E*I/L
            
            R_block = np.array([e1, e2, e3])
            T = np.zeros((12, 12))
            for k in range(4): T[3*k:3*k+3, 3*k:3*k+3] = R_block
            k_global = T.T @ k_local @ T
            
            dofs = list(range(6*i, 6*i+6)) + list(range(6*j, 6*j+6))
            for a in range(12):
                for b in range(12):
                    if abs(k_global[a,b]) > 0:
                        rows.append(dofs[a]); cols.append(dofs[b]); vals.append(k_global[a,b])
        
        K = sparse.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
        K_damped = K + damping * sparse.eye(n_dof, format='csr')
        
        f_global = np.zeros(n_dof)
        f_global[0::6] = forces[:, 0]
        f_global[1::6] = forces[:, 1]
        f_global[2::6] = forces[:, 2]
        
        fixed_dofs = set()
        prescribed_dof_values = {}
        for node in fixed_nodes:
            for k in range(6):
                dof = 6*int(node)+k; fixed_dofs.add(dof); prescribed_dof_values[dof] = 0.0
        for node, disp in prescribed_disp.items():
            for k, val in enumerate(disp):
                dof = 6*int(node)+k; fixed_dofs.add(dof); prescribed_dof_values[dof] = val
        
        fixed_dofs = np.array(sorted(fixed_dofs))
        free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)
        
        f_modified = f_global.copy()
        for dof, val in prescribed_dof_values.items():
            if val != 0.0:
                f_modified -= K_damped[:, dof].toarray().flatten() * val
        
        u_reduced = spsolve(K_damped[np.ix_(free_dofs, free_dofs)], f_modified[free_dofs])
        u_full = np.zeros(n_dof); u_full[free_dofs] = u_reduced
        for dof, val in prescribed_dof_values.items(): u_full[dof] = val
        u = u_full.reshape(n_nodes, 6)
        
        # Stresses
        n_edges = len(edge_list)
        sigma_axial = np.zeros(n_edges)
        sigma_bending = np.zeros(n_edges)
        
        for idx, e in enumerate(edge_list):
            i, j = int(edge_index[0, e]), int(edge_index[1, e])
            r = radii[e]
            dx = node_pos[j] - node_pos[i]
            L = np.linalg.norm(dx)
            if L < 1e-12: continue
            e1 = dx / L
            props = self.section_properties(r)
            A, I_val = props['A'], props['I']
            
            ui_axial = np.dot(e1, u[i, :3])
            uj_axial = np.dot(e1, u[j, :3])
            eps = (uj_axial - ui_axial) / L
            sigma_axial[idx] = self.E * eps
            
            ref = np.array([0, 1.0, 0]) if abs(e1[0]) > 0.9 else np.array([1.0, 0, 0])
            e2 = np.cross(e1, ref); e2 /= np.linalg.norm(e2)
            e3 = np.cross(e1, e2)
            
            vi = np.dot(e2, u[i,:3]); vj = np.dot(e2, u[j,:3])
            wi = np.dot(e3, u[i,:3]); wj = np.dot(e3, u[j,:3])
            th_yi = np.dot(e2, u[i,3:6]); th_yj = np.dot(e2, u[j,3:6])
            th_zi = np.dot(e3, u[i,3:6]); th_zj = np.dot(e3, u[j,3:6])
            
            d_v = np.array([vi, th_yi, vj, th_yj])
            d_w = np.array([wi, th_zi, wj, th_zj])
            Npp_left = np.array([12/L**2, 6/L, -12/L**2, 6/L])
            
            M_z = abs(self.E * I_val * np.dot(Npp_left, d_v))
            M_y = abs(self.E * I_val * np.dot(Npp_left, d_w))
            sigma_bending[idx] = (M_z + M_y) * r / I_val
        
        sigma_total = np.abs(sigma_axial) + sigma_bending
        node_stress = np.zeros(n_nodes)
        for idx, e in enumerate(edge_list):
            i, j = int(edge_index[0, e]), int(edge_index[1, e])
            node_stress[i] = max(node_stress[i], sigma_total[idx])
            node_stress[j] = max(node_stress[j], sigma_total[idx])
        
        # Reactions: R = K*u - f
        reactions = (K_damped @ u_full - f_global).reshape(n_nodes, 6)
        
        # Edge forces [axial_N, shear_V_y, shear_V_z]
        edge_forces = np.zeros((len(edge_list), 3))
        for idx, e in enumerate(edge_list):
            i, j = int(edge_index[0, e]), int(edge_index[1, e])
            r = radii[e]
            dx = node_pos[j] - node_pos[i]
            L = np.linalg.norm(dx)
            if L < 1e-12:
                continue
            e1_dir = dx / L
            props = self.section_properties(r)
            A = props['A']
            ui_axial = np.dot(e1_dir, u[i, :3])
            uj_axial = np.dot(e1_dir, u[j, :3])
            N = self.E * A * (uj_axial - ui_axial) / L
            edge_forces[idx, 0] = N
        
        return {
            'u': u, 'sigma_axial': sigma_axial, 'sigma_bending': sigma_bending,
            'sigma_total': sigma_total, 'node_stress': node_stress,
            'reactions': reactions, 'edge_forces': edge_forces,
            'edge_list': edge_list, 'K': K,
            'n_nodes': n_nodes, 'n_edges': len(edge_list)
        }

    # ═══════════════════════════════════════════════════════
    # CONVENIENCE METHODS (v4.1.1 additions)
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def graph_to_fem_input(graph, dim=2, pct=0.10):
        """Convert StructureGraph to BeamFrameFEM input arrays.
        
        Parameters
        ----------
        graph : StructureGraph
        dim : int
            2 for 2D, 3 for 3D analysis.
        pct : float
            Boundary detection percentage (0.10 = 10% each side).
        
        Returns
        -------
        dict with keys:
            edge_index: (2, n_edges) int64
            node_pos: (n_nodes, dim) float64
            radii: (n_edges,) float64
            boundaries: dict with 'left', 'right' (and 'top', 'bottom', etc.)
            x_range: float (for displacement calculation)
        """
        from fibernet.sim.accelerated import _graph_to_arrays, _get_boundary_indices
        pos, elements, _, _ = _graph_to_arrays(graph)
        edge_index = elements.T.astype(np.int64)
        node_pos = pos[:, :dim] if dim == 2 else pos
        # Extract per-edge radius from StructureGraph.edges (Dict[int, SEdge])
        if hasattr(graph, 'edges') and graph.edges:
            radii = np.array([e.radius for e in graph.edges.values()])
            if len(radii) != edge_index.shape[1]:
                radii = np.full(edge_index.shape[1], radii[0] if len(radii) > 0 else 0.05)
        else:
            radii = np.full(edge_index.shape[1], 0.05)
        
        boundaries = _get_boundary_indices(pos, pct=pct)
        x_range = pos[:, 0].max() - pos[:, 0].min()
        
        return {
            'edge_index': edge_index,
            'node_pos': node_pos,
            'radii': radii,
            'boundaries': boundaries,
            'x_range': x_range,
        }

    def stretch_test(self, graph, target_stretch=2.0, dim=2, pct=0.10,
                     nonlinear=None):
        """One-liner uniaxial stretch test using beam frame FEM.
        
        Analogous to TaichiEngine.stretch_test() but uses FEM.
        
        Parameters
        ----------
        graph : StructureGraph
        target_stretch : float
            Target stretch ratio (2.0 = stretch to 2x original length).
        dim : int
            2 for 2D, 3 for 3D.
        pct : float
            Boundary percentage.
        nonlinear : bool or None
            If None, auto-selects nonlinear for |stretch-1| > 0.3.
            If True, always use nonlinear solver.
        
        Returns
        -------
        dict : FEM result with u, stresses, reactions, etc.
        """
        inp = self.graph_to_fem_input(graph, dim=dim, pct=pct)
        
        left = inp['boundaries'].get('left', [])
        right = inp['boundaries'].get('right', [])
        target_disp = inp['x_range'] * (target_stretch - 1.0)
        
        if dim == 2:
            prescribed = {ni: (target_disp, 0.0) for ni in right}
            if nonlinear is None:
                nonlinear = abs(target_stretch - 1.0) > 0.3
            if nonlinear:
                return self.solve_2d_nonlinear(
                    inp['edge_index'], inp['node_pos'], inp['radii'],
                    prescribed_disp=prescribed, fixed_nodes=left, n_steps=10
                )
            return self.solve_2d(
                inp['edge_index'], inp['node_pos'], inp['radii'],
                fixed_nodes=left, prescribed_disp=prescribed
            )
        else:
            prescribed = {ni: (target_disp, 0.0, 0.0) for ni in right}
            return self.solve_3d(
                inp['edge_index'], inp['node_pos'], inp['radii'],
                fixed_nodes=left, prescribed_disp=prescribed
            )

    def to_sim_result(self, fem_result, graph=None):
        """Convert FEM dict result to SimResult for backend compatibility.
        
        Parameters
        ----------
        fem_result : dict
            Result from solve_2d, solve_3d, or stretch_test.
        graph : StructureGraph, optional
            Original graph for metadata.
        
        Returns
        -------
        SimResult
        """
        from fibernet.sim.accelerated import SimResult, _graph_to_arrays
        
        u = fem_result['u']
        n_nodes = fem_result['n_nodes']
        
        # Displacement magnitude
        if u.shape[1] == 3 and fem_result.get('edge_list') is not None:
            # 2D: u is (n, 3) = [ux, uy, theta]
            disp_3d = np.column_stack([u[:, :2], np.zeros(n_nodes)])
        elif u.shape[1] == 6:
            # 3D: u is (n, 6) = [ux, uy, uz, thx, thy, thz]
            disp_3d = u[:, :3]
        else:
            disp_3d = u
        
        max_disp = float(np.max(np.linalg.norm(disp_3d, axis=1)))
        
        # Deformed positions
        if graph is not None:
            pos, _, _, _ = _graph_to_arrays(graph)
            deformed = pos + np.column_stack([disp_3d, np.zeros(pos.shape[1] - disp_3d.shape[1])]) if pos.shape[1] > disp_3d.shape[1] else pos + disp_3d
        else:
            deformed = disp_3d  # Just the displacement
        
        # Energy (elastic strain energy)
        sigma_axial = fem_result.get('sigma_axial', np.array([]))
        sigma_bending = fem_result.get('sigma_bending', np.array([]))
        edge_list = fem_result.get('edge_list', np.array([]))
        
        # Approximate energy from stresses
        energy = 0.0
        if graph is not None and len(sigma_axial) > 0:
            pos, elements, _, _ = _graph_to_arrays(graph)
            for idx, e in enumerate(edge_list):
                i, j = int(elements[e, 0]), int(elements[e, 1])
                L = np.linalg.norm(pos[i] - pos[j])
                r = 0.05
                try:
                    edge_vals = list(graph.edges.values())
                    if 0 <= e < len(edge_vals):
                        r = edge_vals[e].radius
                except Exception:
                    pass
                A = np.pi * r**2
                strain = sigma_axial[idx] / self.E if self.E > 0 else 0
                energy += 0.5 * self.E * strain**2 * A * L
        
        return SimResult(
            displacements=disp_3d,
            deformed_positions=deformed,
            energy=energy,
            max_displacement=max_disp,
            max_force=float(np.max(np.abs(sigma_axial))) * np.pi * 0.05**2 if len(sigma_axial) > 0 else 0.0,
            max_stretch=float(np.max(np.abs(sigma_axial) / self.E + 1.0)) if len(sigma_axial) > 0 else 1.0,
            mean_stretch=float(np.mean(np.abs(sigma_axial) / self.E + 1.0)) if len(sigma_axial) > 0 else 1.0,
            n_nodes=n_nodes,
            n_edges=fem_result.get('n_edges', 0),
            mode='fem_beam_frame',
            metadata={
                'sigma_axial': sigma_axial,
                'sigma_bending': sigma_bending,
                'sigma_total': fem_result.get('sigma_total', np.array([])),
                'reactions': fem_result.get('reactions'),
                'edge_forces': fem_result.get('edge_forces'),
            },
        )

