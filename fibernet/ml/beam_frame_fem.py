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
import sys
import warnings


def _beam_resultant_contract(dimension, status='available'):
    """Return fresh component metadata for the known beam producer layout.

    Availability is not an accuracy, equilibrium or unit-system assertion.
    The caller supplies the producer dimension; array width is not evidence.
    """
    if dimension not in (2, 3):
        raise ValueError('beam resultant dimension must be 2 or 3')
    components = [{'name': 'axial_N', 'quantity_kind': 'force', 'status': status}]
    if dimension == 2:
        components.extend([
            {'name': 'shear_V', 'quantity_kind': 'force', 'status': status},
            {'name': 'mean_bending_moment', 'quantity_kind': 'moment',
             'status': status, 'definition': '(M_i + M_j) / 2'},
        ])
    else:
        components.extend([
            {'name': name, 'quantity_kind': 'force', 'status': 'unavailable',
             'reason': 'not_recovered_by_3d_solver'}
            for name in ('shear_V_y', 'shear_V_z')
        ])
    return {
        'schema_version': 1, 'dimension': dimension, 'components': components,
        'unit_system': 'not_declared',
    }


class BeamFrameFEM:
    """Beam frames with welded joints: 2-D incremental geometry, 3-D linear only."""
    
    def __init__(self, E: float = 1e9, nu: float = 0.3):
        self.E = E
        self.nu = nu
        self.G = E / (2 * (1 + nu))

    def history(self, graph, *, material, times, constraints, loads=None,
                length_unit, initial_state=None):
        """Quasistatic 2-D viscoelastic response through the existing frame.

        Explicit uniform LinearViscoelasticMaterial overrides this instance's
        static E and the graph's static material moduli. times[0] is the
        unloaded initial state, or the supplied committed checkpoint time.
        Histories map stable node IDs to (len(times),3) endpoint samples.
        Loads are Fx/Fy/Mz in N/N/N m. Constraint samples are ux/uy/rz in
        m/m/rad; None means free and zero means constrained to zero. The mask
        is constant in time. Graph positions and radii use length_unit.

        Returns ViscoelasticResult with separate physical/numerical loss and
        a detached JSON-restartable final_state. No input is mutated. Curved
        members, 3-D, mechanisms and nonfinite results are refused.
        """
        from fibernet.sim._viscoelastic_fem import solve_history
        return solve_history(self, graph, material=material, times=times,
                             constraints=constraints, loads=loads,
                             length_unit=length_unit, initial_state=initial_state)
    
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

    @staticmethod
    def _validate_force_shape(forces, n_nodes, dim):
        """Reject omitted or extra load components before stiffness assembly."""
        expected_shape = (n_nodes, dim)
        actual_shape = getattr(forces, 'shape', None)
        if actual_shape != expected_shape:
            raise ValueError(
                f"forces must have shape {expected_shape}: one row per node "
                f"and {dim} translational force components; got {actual_shape}. "
                "Applied nodal moments/rotational columns are not supported. "
                "Use forces=None for zero loads."
            )

    @staticmethod
    def _linear_numerical_diagnostics(reactions, free_dofs, damping):
        """Copy physical free-DOF residuals; this is not a stability verdict."""
        if np.ndim(reactions) != 2 or reactions.shape[1] not in (3, 6):
            raise ValueError('numerical_diagnostics requires 3 or 6 DOFs per node')
        components = ('ux', 'uy', 'rz') if reactions.shape[1] == 3 else (
            'ux', 'uy', 'uz', 'rx', 'ry', 'rz')
        ndof = len(components)
        coefficient = float(damping)
        regularization = {
            'status': 'available', 'method': 'diagonal_identity',
            'coefficient': coefficient,
            'interpretation': 'numerical_not_physical_support',
        }
        if not np.isfinite(coefficient):
            regularization.update(
                status='unavailable', coefficient=None,
                reason='nonfinite_regularization_coefficient')
        values = np.asarray(reactions).ravel()[free_dofs]
        residual = {
            'status': 'available',
            'definition': 'K_physical_u_minus_applied_load',
            'index_kind': 'input_node_row',
            'node_rows': [int(dof // ndof) for dof in free_dofs],
            'dof_components': [components[int(dof % ndof)] for dof in free_dofs],
            'values': [float(value) for value in values],
        }
        if not np.isfinite(values).all():
            residual.update(status='unavailable', values=None,
                            reason='nonfinite_recovered_residual')
        return {
            'schema_version': 1, 'model': 'linear_beam_frame',
            'regularization': regularization,
            'physical_free_residual': residual,
            'physical_equilibrium': 'not_evaluated',
            'load_compatibility': 'not_evaluated',
            'solution_uniqueness': 'not_evaluated',
        }

    @staticmethod
    def _copy_numerical_diagnostics(diagnostics, n_nodes, ndof, node_ids=None):
        """Validate v1 metadata and, when mapped, replace row indices with IDs."""
        from copy import deepcopy
        import json

        def invalid(message):
            raise ValueError(f"numerical_diagnostics {message}")

        def finite_number(value):
            return (not isinstance(value, (bool, np.bool_))
                    and isinstance(value, (int, float, np.integer, np.floating))
                    and np.isfinite(value))

        if ndof not in (3, 6):
            invalid('requires 3 or 6 DOFs per node')
        diagnostic = deepcopy(diagnostics)
        if (not isinstance(diagnostic, dict)
                or type(diagnostic.get('schema_version')) is not int
                or diagnostic['schema_version'] != 1):
            invalid('requires schema_version 1')
        for name in ('physical_equilibrium', 'load_compatibility', 'solution_uniqueness'):
            if diagnostic.get(name) != 'not_evaluated':
                invalid(f'{name} must remain not_evaluated')
        regularization = diagnostic.get('regularization')
        if (not isinstance(regularization, dict)
                or regularization.get('method') != 'diagonal_identity'
                or regularization.get('interpretation') != 'numerical_not_physical_support'):
            invalid('requires diagonal_identity regularization metadata')
        if regularization.get('status') == 'available':
            if not finite_number(regularization.get('coefficient')) or 'reason' in regularization:
                invalid('available regularization requires a finite coefficient and no reason')
        elif regularization.get('status') == 'unavailable':
            if (regularization.get('coefficient', object()) is not None
                    or regularization.get('reason') != 'nonfinite_regularization_coefficient'):
                invalid('unavailable regularization requires null coefficient and its reason')
        else:
            invalid('has an invalid regularization status')
        residual = diagnostic.get('physical_free_residual')
        if (not isinstance(residual, dict)
                or residual.get('definition') != 'K_physical_u_minus_applied_load'):
            invalid('requires the physical free residual definition')
        nonlinear = diagnostic.get('model') == 'incremental_updated_geometry_linear_solves'
        if nonlinear:
            if (residual.get('status') != 'unavailable'
                    or residual.get('reason') != 'final_nonlinear_equilibrium_residual_not_recovered'
                    or residual.get('values', object()) is not None
                    or any(key in residual for key in ('index_kind', 'node_rows', 'node_ids', 'dof_components'))
                    or regularization.get('scope') != 'inner_linear_solves'):
                invalid('final nonlinear residual must be unavailable without increment indices')
        else:
            if (diagnostic.get('model') != 'linear_beam_frame'
                    or residual.get('index_kind') != 'input_node_row'
                    or 'node_ids' in residual):
                invalid('linear raw residuals require input_node_row indices')
            rows = residual.get('node_rows')
            components = residual.get('dof_components')
            allowed = ('ux', 'uy', 'rz') if ndof == 3 else ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')
            if (not isinstance(rows, list) or not isinstance(components, list)
                    or len(rows) != len(components)
                    or any(type(row) is not int or row < 0 or row >= n_nodes for row in rows)
                    or any(type(component) is not str or component not in allowed
                           for component in components)):
                invalid('row/component arrays must have equal lengths and valid input DOFs')
            if len(set(zip(rows, components))) != len(rows):
                invalid('free DOF records must not contain duplicate row/component pairs')
            if residual.get('status') == 'available':
                values = residual.get('values')
                if (not isinstance(values, list) or len(values) != len(rows)
                        or not all(finite_number(value) for value in values)
                        or 'reason' in residual):
                    invalid('available residual requires equally sized finite values and no reason')
            elif residual.get('status') == 'unavailable':
                if (residual.get('values', object()) is not None
                        or residual.get('reason') != 'nonfinite_recovered_residual'):
                    invalid('unavailable residual requires null values and its reason')
            else:
                invalid('has an invalid physical residual status')
            if node_ids is not None:
                if (len(node_ids) != n_nodes or len(set(node_ids)) != n_nodes
                        or any(isinstance(nid, (bool, np.bool_))
                               or not isinstance(nid, (int, np.integer)) or nid < 0
                               for nid in node_ids)):
                    invalid('resolved node map must cover all input node rows')
                residual['node_ids'] = [int(node_ids[row]) for row in rows]
                del residual['node_rows']
                residual['index_kind'] = 'stable_node_id'
        try:
            json.dumps(diagnostic, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError('numerical_diagnostics must contain JSON-native finite values or null') from exc
        return diagnostic

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
    
    def build_stiffness_2d(self, edge_index, node_pos, radii, deduplicate=False):
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
    
    @staticmethod
    def _hermite_endpoint_curvatures(length, transverse_dofs):
        """Return v'' at x=0,L for Hermite DOFs [v_i, v'_i, v_j, v'_j]."""
        L = length
        second_derivatives = np.array([
            [-6/L**2, -4/L, 6/L**2, -2/L],
            [6/L**2, 2/L, -6/L**2, 4/L],
        ])
        return second_derivatives @ transverse_dofs

    def _compute_element_stress_2d(self, node_pos, u, edge_index, edge_list, radii,
                                 *, axial_eigenstrain=None):
        """Recover linear stresses from cubic Hermite curvature.

        The scalar convention M=-EI*v'' gives M_i=q_local[2],
        M_j=-q_local[5], and V=(M_j-M_i)/L=-q_local[1]=q_local[4].
        Thus the endpoint energy-gradient vectors are (-N,-V,M_i) and
        (N,V,-M_j); reported endpoint moments are not both nodal moments.
        """
        if axial_eigenstrain is not None:
            axial_eigenstrain = np.asarray(axial_eigenstrain, dtype=float)
            if (axial_eigenstrain.shape != (edge_index.shape[1],)
                    or not np.isfinite(axial_eigenstrain).all()):
                raise ValueError('axial_eigenstrain requires finite original input edge rows')
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
            if axial_eigenstrain is not None:
                eps_axial = eps_axial - axial_eigenstrain[e]
            sigma_axial[idx] = self.E * eps_axial
            
            # Rotational DOFs are the transverse slopes in the local plane.
            d_trans = np.array([v_i, th_i, v_j, th_j])
            curvature = self._hermite_endpoint_curvatures(L, d_trans)
            M_i, M_j = -self.E * I_val * curvature

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
                 damping=1e-6, deduplicate=False):
        """Solve 2D beam frame with force and/or displacement BCs.

        edge_forces columns are [axial_N, shear_V, mean_bending_moment];
        the third is (M_i + M_j) / 2, not a force or endpoint moment.
        beam_resultant_contract describes their quantity kinds and availability.

        forces: None (zero loads), or an exact (n_nodes, 2) array [Fx, Fy].
        Applied nodal moments are unsupported; extra columns raise ValueError.
        
        Returns dict with:
            u: (n_nodes, 3) [ux, uy, theta]
            sigma_axial, sigma_bending, sigma_total: per-edge stresses
            moments: (n_edges, 2) bending moments at ends
            node_stress: (n_nodes,) max stress at each node
            reactions: (n_nodes, 3) reaction forces/moments
        """
        # A tensor can only exist after its owner has imported Torch.
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.numpy()
        if torch is not None and isinstance(node_pos, torch.Tensor):
            node_pos = node_pos.numpy()
        if torch is not None and isinstance(radii, torch.Tensor):
            radii = radii.numpy()
        if torch is not None and isinstance(forces, torch.Tensor):
            forces = forces.numpy()
        
        if fixed_nodes is None:
            fixed_nodes = []
        if prescribed_disp is None:
            prescribed_disp = {}
        if forces is None:
            forces = np.zeros((node_pos.shape[0], 2))

        self._validate_force_shape(forces, node_pos.shape[0], 2)
        
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
            'beam_resultant_contract': _beam_resultant_contract(2),
            'node_moment': node_moment, 'reactions': reactions,
            'edge_list': edge_list, 'K': K,
            'n_nodes': n_nodes, 'n_edges': len(edge_list),
            'numerical_diagnostics': self._linear_numerical_diagnostics(
                reactions, free_dofs, damping),
        }
    
    def solve_2d_nonlinear(self, edge_index, node_pos, radii,
                           prescribed_disp, fixed_nodes=None,
                           forces=None, n_steps=10, tol=1e-6, max_iter=20,
                           damping=1e-6, deduplicate=False):
        """Incremental updated-geometry linear solves (equilibrium unverified).

        Final stress/resultant recovery is a linear-reference approximation.
        Final K, reactions and node_moment are unavailable, not physical zeros.
        
        For each increment:
          1. Apply fraction of total prescribed displacement
          2. Solve linear system at current configuration
          3. Update node positions
          4. Repeat
        """
        for name, value in (('n_steps', n_steps), ('max_iter', max_iter)):
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value <= 0):
                raise ValueError(f"{name} must be a positive integer (not bool)")
        if (isinstance(tol, (bool, np.bool_))
                or not isinstance(tol, (int, float, np.integer, np.floating))
                or not np.isfinite(tol) or tol < 0):
            raise ValueError("tol must be a finite nonnegative real number (not bool)")

        torch = sys.modules.get("torch")
        if torch is not None and isinstance(node_pos, torch.Tensor):
            node_pos = node_pos.numpy().copy()
        else:
            node_pos = node_pos.copy()
        if torch is not None and isinstance(edge_index, torch.Tensor):
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
                
                update_threshold = tol * (1.0 + np.linalg.norm(total_u[:, :2]))
                update_criterion_met = bool(du_norm < update_threshold)
                if update_criterion_met:
                    break
                
                step_prescribed = {}
            
            max_disp = np.max(np.linalg.norm(total_u[:, :2], axis=1))
            max_stress = np.max(result['sigma_total']) if result['sigma_total'] is not None else 0
            
            history.append({
                'step': step, 'max_disp': float(max_disp),
                'max_stress': float(max_stress), 'iterations': iteration + 1,
                'update_norm': float(du_norm),
                'update_threshold': float(update_threshold),
                'update_criterion_met': update_criterion_met,
                'iteration_limit_reached': bool(
                    not update_criterion_met and iteration + 1 == max_iter),
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

        # The last inner solve describes an increment on updated geometry, not
        # the total state above. Do not expose its diagnostics as final fields.
        model = 'incremental_updated_geometry_linear_solves'
        result['numerical_diagnostics'] = {
            'schema_version': 1, 'model': model,
            'regularization': {
                **result['numerical_diagnostics']['regularization'],
                'scope': 'inner_linear_solves',
            },
            'physical_free_residual': {
                'status': 'unavailable',
                'definition': 'K_physical_u_minus_applied_load',
                'reason': 'final_nonlinear_equilibrium_residual_not_recovered',
                'values': None,
            },
            'physical_equilibrium': 'not_evaluated',
            'load_compatibility': 'not_evaluated',
            'solution_uniqueness': 'not_evaluated',
        }
        result['field_status'] = {}
        for field, reason in (
            ('K', 'final_nonlinear_stiffness_not_recovered'),
            ('reactions', 'final_nonlinear_equilibrium_reactions_not_recovered'),
            ('node_moment', 'final_total_nodal_moments_not_recovered'),
        ):
            result[field] = None
            result['field_status'][field] = {
                'status': 'not_supported', 'reason': reason, 'model': model,
            }
        for field in ('sigma_axial', 'sigma_bending', 'sigma_total',
                      'moments', 'edge_forces', 'node_stress'):
            result['field_status'][field] = {
                'status': 'approximate',
                'approximation': 'original_geometry_total_displacement_linear_reference',
                'model': model,
            }
        result['beam_resultant_contract'] = _beam_resultant_contract(
            2, result['field_status']['edge_forces']['status'])
        result['solver_status'] = {
            'model': model,
            'equilibrium': 'unverified',
            'history_field_scope': {
                'max_disp': 'accumulated_total_translation',
                'max_stress': 'last_inner_linear_increment',
            },
            'update_criterion': 'translation_increment_norm',
            'update_criterion_is_equilibrium_check': False,
            'max_iter': max_iter,
            'steps': [{key: entry[key] for key in (
                'step', 'iterations', 'update_norm', 'update_threshold',
                'update_criterion_met', 'iteration_limit_reached')}
                for entry in history],
        }

        return result
    
    def solve_3d(self, edge_index, node_pos, radii,
                 forces=None, fixed_nodes=None,
                 prescribed_disp=None,
                 damping=1e-6, deduplicate=False):
        """Solve 3D beam frame with corrected stress computation.

        edge_forces retains [axial_N, shear_V_y, shear_V_z] for compatibility.
        Only axial_N is recovered; both shear columns are uncomputed zero
        placeholders, not measured zero shear. beam_resultant_contract marks
        these components unavailable. No unit system is inferred.

        forces: None (zero loads), or an exact (n_nodes, 3) array [Fx, Fy, Fz].
        Applied nodal moments are unsupported; extra columns raise ValueError.
        """
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.numpy()
        if torch is not None and isinstance(node_pos, torch.Tensor):
            node_pos = node_pos.numpy()
        if torch is not None and isinstance(radii, torch.Tensor):
            radii = radii.numpy()
        if torch is not None and isinstance(forces, torch.Tensor):
            forces = forces.numpy()
        
        if fixed_nodes is None:
            fixed_nodes = []
        if prescribed_disp is None:
            prescribed_disp = {}
        if forces is None:
            forces = np.zeros((node_pos.shape[0], 3))

        self._validate_force_shape(forces, node_pos.shape[0], 3)
        
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
            
            # In the right-handed local frame, theta x e1 = theta_z*e2
            # - theta_y*e3, so v'=theta_z and w'=-theta_y.
            d_v = np.array([vi, th_zi, vj, th_zj])
            d_w = np.array([wi, -th_yi, wj, -th_yj])
            curvature_v = self._hermite_endpoint_curvatures(L, d_v)
            curvature_w = self._hermite_endpoint_curvatures(L, d_w)

            # Circular-section peak normal stress uses the moment norm at
            # each endpoint, then the maximum; the curvature is linear in x.
            moment_norms = self.E * I_val * np.hypot(curvature_v, curvature_w)
            sigma_bending[idx] = np.max(moment_norms) * r / I_val

        sigma_total = np.abs(sigma_axial) + sigma_bending
        node_stress = np.zeros(n_nodes)
        for idx, e in enumerate(edge_list):
            i, j = int(edge_index[0, e]), int(edge_index[1, e])
            node_stress[i] = max(node_stress[i], sigma_total[idx])
            node_stress[j] = max(node_stress[j], sigma_total[idx])
        
        # Reactions: R = K*u - f
        reactions = (K @ u_full - f_global).reshape(n_nodes, 6)
        
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
            'beam_resultant_contract': _beam_resultant_contract(3),
            'edge_list': edge_list, 'K': K,
            'n_nodes': n_nodes, 'n_edges': len(edge_list),
            'numerical_diagnostics': self._linear_numerical_diagnostics(
                reactions, free_dofs, damping),
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
            Straight members between graph nodes. Nonempty edge-owned
            internal_points are not a curved-beam discretization and reject.
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
        curved_edges = [
            edge_id for edge_id, edge in graph.edges.items()
            if edge.internal_points is not None and len(edge.internal_points)
        ]
        if curved_edges:
            raise ValueError(
                'BeamFrameFEM supports straight graph members; nonempty '
                'internal_points would be ignored on edge IDs '
                + repr(tuple(curved_edges))
                + '. Represent the fiber with actual intermediate graph nodes '
                'and straight edges, with separately specified mechanics.'
            )
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
                     nonlinear=None, deduplicate=False):
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
            In 2-D, None auto-selects incremental updated-geometry solves for
            |stretch-1| > 0.3; True selects this experimental path explicitly.
            This path warns at any stretch: final nonlinear equilibrium is
            unverified, and final stiffness/reactions/nodal moments are unavailable.
            False retains the small-deformation linear solve and warns when
            |stretch-1| > 0.3. This legacy switch is not a validity bound.
            In 3-D, only the linear solver exists: True is unsupported, while
            None and False retain the linear solve. Large 3-D stretches warn
            because this small-deformation model is not a nonlinear solution.
        deduplicate : bool
            False preserves independent parallel fibers. True is available
            only for reproducing legacy simple-graph calculations.
        
        Returns
        -------
        dict : FEM result with u, stresses, reactions, etc.
        """
        if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer)) or dim not in (2, 3):
            raise ValueError('dim must be 2 or 3')
        if nonlinear is not None and not isinstance(nonlinear, (bool, np.bool_)):
            raise TypeError('nonlinear must be bool or None')
        if dim == 3 and nonlinear is not None and bool(nonlinear):
            raise NotImplementedError('3-D beam-frame stretch is linear-only; nonlinear=True is unsupported')
        if (isinstance(target_stretch, (bool, np.bool_))
                or not isinstance(target_stretch, (int, float, np.integer, np.floating))
                or not np.isfinite(target_stretch)):
            raise ValueError('target_stretch must be a finite real stretch ratio')
        if dim == 3 and abs(target_stretch - 1.0) > 0.3:
            warnings.warn('3-D beam-frame stretch uses a small-deformation linear solver even at large stretch; '
                          'no 3-D nonlinear solver is available', UserWarning, stacklevel=2)
        inp = self.graph_to_fem_input(graph, dim=dim, pct=pct)
        
        left = inp['boundaries'].get('left', [])
        right = inp['boundaries'].get('right', [])
        target_disp = inp['x_range'] * (target_stretch - 1.0)
        
        if dim == 2:
            prescribed = {ni: (target_disp, 0.0) for ni in right}
            if nonlinear is None:
                nonlinear = abs(target_stretch - 1.0) > 0.3
            if nonlinear:
                warnings.warn(
                    '2-D beam-frame stretch uses experimental incremental updated-geometry linear solves; '
                    'final nonlinear equilibrium is unverified, and final stiffness, reactions and nodal moments '
                    'are unavailable. For small-deformation analysis, use nonlinear=False with a suitably small '
                    'stretch (for example, target_stretch=1.01).', UserWarning, stacklevel=2)
                result = self.solve_2d_nonlinear(
                    inp['edge_index'], inp['node_pos'], inp['radii'],
                    prescribed_disp=prescribed, fixed_nodes=left, n_steps=10,
                    deduplicate=deduplicate
                )
            else:
                if abs(target_stretch - 1.0) > 0.3:
                    warnings.warn(
                        '2-D beam-frame stretch uses a small-deformation linear solver even at large stretch; '
                        'nonlinear=False does not provide a large-deformation solution.', UserWarning, stacklevel=2)
                result = self.solve_2d(
                    inp['edge_index'], inp['node_pos'], inp['radii'],
                    fixed_nodes=left, prescribed_disp=prescribed,
                    deduplicate=deduplicate
                )
        else:
            prescribed = {ni: (target_disp, 0.0, 0.0) for ni in right}
            result = self.solve_3d(
                inp['edge_index'], inp['node_pos'], inp['radii'],
                fixed_nodes=left, prescribed_disp=prescribed,
                deduplicate=deduplicate
            )
        displacement = np.asarray(result['u'])[:, :dim]
        result['max_displacement'] = float(np.linalg.norm(displacement, axis=1).max())
        return result

    def to_sim_result(self, fem_result, graph=None, *, node_ids=None, edge_ids=None):
        """Convert FEM dict result to SimResult for backend compatibility.

        ``edge_forces`` contains signed axial forces. The solver's full
        three-column force/resultant array is retained as the distinct
        ``metadata['beam_resultants']`` value, including in summary exports.
        ``beam_resultant_contract`` describes dimension, quantity kinds and
        availability. N denotes the axial-force symbol, not an asserted newton
        unit. In 3-D only axial N is recovered; the retained shear slots are
        legacy placeholders, not measured zero forces. No unit system is inferred.
        Present numerical diagnostics are independently copied; input-row
        residual records become stable-node-ID records only with a graph map.
        
        Parameters
        ----------
        fem_result : dict
            Result from solve_2d, solve_3d, or stretch_test.
        graph : StructureGraph, optional
            Original graph for geometry and metadata. Without explicit maps,
            raw rows retain the legacy graph_to_fem_input / _graph_to_arrays
            insertion-order assumption. Without graph, ID maps are omitted.
        node_ids, edge_ids : sequence of int, optional, keyword-only
            IDs of the INPUT node and element rows supplied to the raw solve,
            before deduplication. Supply both maps and graph together; each
            map must cover its graph entities exactly. Output edge IDs are
            edge_ids[fem_result['edge_list']]. For graph.to_numpy() inputs use
            sorted(graph.nodes) and list(graph.edges), respectively (and apply
            any input-row permutations to the maps too).
            These maps assert caller provenance; they cannot verify that
            arbitrary raw geometry, connectivity or materials match graph.
        
        Returns
        -------
        SimResult
        """
        from fibernet.sim.accelerated import SimResult, _graph_to_arrays
        
        u = fem_result['u']
        n_nodes = fem_result['n_nodes']
        explicit_ids = node_ids is not None or edge_ids is not None
        if explicit_ids and (graph is None or node_ids is None or edge_ids is None):
            raise ValueError("Explicit node_ids and edge_ids require both maps and graph")

        def validate_ids(values, expected, name):
            try:
                values = list(values)
            except TypeError as exc:
                raise ValueError(f"{name} must be a sequence of integer IDs") from exc
            if any(isinstance(v, (bool, np.bool_)) or
                   not isinstance(v, (int, np.integer)) or v < 0 for v in values):
                raise ValueError(f"{name} must contain nonnegative integer IDs, not bools")
            values = [int(v) for v in values]
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique IDs")
            if set(values) != set(expected):
                raise ValueError(f"{name} must have exact graph coverage")
            return values

        edge_list = fem_result.get('edge_list', np.array([]))
        identity = {}
        if graph is not None:
            if explicit_ids:
                node_ids = validate_ids(node_ids, graph.nodes, 'node_ids')
                graph_edge_ids = validate_ids(edge_ids, graph.edges, 'edge_ids')
                if (np.ndim(u) != 2 or len(node_ids) != u.shape[0]
                        or len(node_ids) != n_nodes):
                    raise ValueError("node_ids length must match raw u rows and n_nodes")
            else:
                node_ids = list(graph.nodes)
                graph_edge_ids = list(graph.edges)
            # Validate before any retained-row indexing or geometric calculation.
            # Check Python values first so mixed bool/int lists cannot coerce bools.
            try:
                selected = np.asarray(edge_list)
            except (TypeError, ValueError) as exc:
                raise ValueError("FEM edge_list must contain unique valid input edge row indices") from exc
            if (selected.ndim != 1 or selected.dtype.kind not in 'iu'
                    or any(isinstance(v, (bool, np.bool_)) for v in edge_list)
                    or np.any(selected < 0) or np.any(selected >= len(graph_edge_ids))
                    or len(set(selected.tolist())) != len(selected)):
                raise ValueError("FEM edge_list must contain unique valid input edge row indices")
            if explicit_ids:
                pos = np.array([graph.nodes[nid].position for nid in node_ids])
                node_rows = {nid: i for i, nid in enumerate(node_ids)}
                edge_vals = [graph.edges[eid] for eid in graph_edge_ids]
                elements = np.array([[node_rows[e.node_i], node_rows[e.node_j]]
                                     for e in edge_vals], dtype=int).reshape(-1, 2)
            else:
                pos, elements, _, _ = _graph_to_arrays(graph)
                edge_vals = list(graph.edges.values())
            identity = {'node_ids': list(node_ids),
                        'edge_ids': [graph_edge_ids[int(i)] for i in selected]}

        numerical = {}
        if 'numerical_diagnostics' in fem_result:
            if np.ndim(u) != 2 or u.shape[0] != n_nodes or u.shape[1] not in (3, 6):
                raise ValueError('numerical_diagnostics requires matching raw u rows and n_nodes')
            numerical['numerical_diagnostics'] = self._copy_numerical_diagnostics(
                fem_result['numerical_diagnostics'], n_nodes, u.shape[1],
                node_ids if graph is not None else None)

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
            deformed = pos + np.column_stack([disp_3d, np.zeros(pos.shape[1] - disp_3d.shape[1])]) if pos.shape[1] > disp_3d.shape[1] else pos + disp_3d
        else:
            deformed = disp_3d  # Just the displacement
        
        # Energy (axial-stress proxy over the assembled beam elements)
        sigma_axial = fem_result.get('sigma_axial', np.array([]))
        sigma_bending = fem_result.get('sigma_bending', np.array([]))
        beam_forces = np.asarray(fem_result.get('edge_forces', np.empty((0, 3))))
        axial_forces = (beam_forces[:, 0].copy()
                        if beam_forces.ndim == 2 and beam_forces.shape[1] >= 1
                        else np.zeros(0))
        resultant_contract = {}
        if (np.ndim(u) == 2 and u.shape[1] in (3, 6) and
                beam_forces.ndim == 2 and beam_forces.shape[1] == 3):
            dimension = 2 if u.shape[1] == 3 else 3
            # Preserve producer qualifications, especially nonlinear approximations.
            status = fem_result.get('field_status', {}).get('edge_forces', {}).get('status', 'available')
            resultant_contract['beam_resultant_contract'] = _beam_resultant_contract(dimension, status)
        edge_stretches = np.zeros(0)
        
        # Approximate energy from stresses
        energy = 0.0
        if graph is not None and len(sigma_axial) > 0:
            selected = elements[np.asarray(edge_list, dtype=int)]
            lengths = np.linalg.norm(pos[selected[:, 1]] - pos[selected[:, 0]], axis=1)
            final_lengths = np.linalg.norm(deformed[selected[:, 1]] -
                                           deformed[selected[:, 0]], axis=1)
            edge_stretches = final_lengths / np.maximum(lengths, 1e-12)
            for idx, e in enumerate(edge_list):
                i, j = int(elements[e, 0]), int(elements[e, 1])
                L = np.linalg.norm(pos[i] - pos[j])
                r = 0.05
                try:
                    if 0 <= e < len(edge_vals):
                        r = edge_vals[e].radius
                except Exception:
                    pass
                A = np.pi * r**2
                strain = sigma_axial[idx] / self.E if self.E > 0 else 0
                energy += 0.5 * self.E * strain**2 * A * L
        
        # Do not invent validity for legacy/linear dictionaries. Only nonlinear
        # producers currently emit this availability contract.
        availability = {key: fem_result[key] for key in
                        ('field_status', 'solver_status') if key in fem_result}
        if 'field_status' in fem_result:
            availability['field_status'] = dict(fem_result['field_status'])
            if 'edge_forces' in fem_result['field_status']:
                availability['field_status']['beam_resultants'] = dict(
                    fem_result['field_status']['edge_forces'])
            for key in ('K', 'node_moment'):
                if key in fem_result['field_status'] and fem_result.get(key) is None:
                    availability[key] = None

        return SimResult(
            displacements=disp_3d,
            deformed_positions=deformed,
            energy=energy,
            edge_forces=axial_forces,
            edge_stretches=edge_stretches,
            max_displacement=max_disp,
            max_force=float(np.max(np.abs(axial_forces))) if axial_forces.size else 0.0,
            max_stretch=float(edge_stretches.max()) if edge_stretches.size else 1.0,
            mean_stretch=float(edge_stretches.mean()) if edge_stretches.size else 1.0,
            std_stretch=float(edge_stretches.std()) if edge_stretches.size else 0.0,
            n_nodes=n_nodes,
            n_edges=fem_result.get('n_edges', 0),
            mode='fem_beam_frame',
            metadata={
                **identity,
                **availability,
                **numerical,
                **resultant_contract,
                'sigma_axial': sigma_axial,
                'sigma_bending': sigma_bending,
                'sigma_total': fem_result.get('sigma_total', np.array([])),
                'reactions': fem_result.get('reactions'),
                'beam_resultants': fem_result.get('edge_forces'),
                'energy_kind': 'axial_stress_proxy',
            },
        )

