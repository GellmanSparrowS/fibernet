"""
Differentiable Physics Simulation for FiberNet.

Transforms fiber network mechanics into differentiable computations,
enabling end-to-end gradient-based optimization from structure parameters
to mechanical properties.

Implements:
- DifferentiableSpringNetwork: Differentiable spring/beam network solver
- DifferentiableFEA: Lightweight differentiable finite element analysis
- PhysicsOptimizer: Gradient-based structure optimization
- DifferentiableMaterialModel: Learnable constitutive law

Features
--------
- Backpropagation through simulation (structure → mechanics → loss)
- Automatic differentiation via PyTorch autograd
- Linear and nonlinear (geometric) mechanics
- Supports truss/beam elements with axial + bending stiffness
- Gradient-based topology and sizing optimization
- Compatible with StructureGraph input

References
----------
- Article section 5: "Differentiable physics simulations transform
  traditional FEA into differentiable computational graphs"
- Degrave et al., "A Differentiable Physics Engine for Deep Learning
  in Robotics" (Frontiers in Neurorobotics, 2019)
- Werling et al., "Fast and Feature-Complete Differentiable Physics
  Engine for Articulated Rigid-Body Dynamics" (RSS 2021)

Examples
--------
>>> from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
>>> sim = DifferentiableSpringNetwork(n_nodes=10, youngs_modulus=1e9)
>>> # Set design variables (cross-section radii)
>>> radii = torch.ones(n_edges, requires_grad=True) * 0.01
>>> displacement, stress = sim.solve(edge_index, node_pos, radii, forces)
>>> loss = (displacement[target_nodes] - target_disp).pow(2).sum()
>>> loss.backward()  # Gradients flow through simulation!
>>> # Optimize radii to minimize displacement
>>> optimizer = torch.optim.Adam([radii], lr=0.01)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch required: pip install torch")


if HAS_TORCH:

    # ==================================================================
    # Differentiable Spring/Truss Network
    # ==================================================================

    class DifferentiableSpringNetwork(nn.Module):
        """Differentiable spring/truss network solver.

        Solves static equilibrium for a network of spring/truss elements
        using the direct stiffness method, fully differentiable.

        Parameters
        ----------
        youngs_modulus : float
            Material Young's modulus E (Pa).
        dim : int
            Spatial dimension (2 or 3).
        solver : str
            "direct" (solve Ku=f) or "iterative" (conjugate gradient).
        damping : float
            Small regularization for numerical stability.

        Examples
        --------
        >>> sim = DifferentiableSpringNetwork(youngs_modulus=1e9)
        >>> # Define 2D truss: 4 nodes, 4 edges
        >>> node_pos = torch.tensor([[0,0],[1,0],[1,1],[0,1]], dtype=torch.float32)
        >>> edge_index = torch.tensor([[0,1],[1,2],[2,3],[3,0]], dtype=torch.long)
        >>> radii = torch.ones(4, requires_grad=True) * 0.01
        >>> forces = torch.zeros(4, 2)
        >>> forces[1] = torch.tensor([100.0, 0.0])  # apply force at node 1
        >>> fixed_nodes = torch.tensor([0, 3])
        >>> u, sigma = sim.solve(edge_index, node_pos, radii, forces, fixed_nodes)
        """

        def __init__(
            self,
            youngs_modulus: float = 1e9,
            dim: int = 2,
            solver: str = "direct",
            damping: float = 1e-5,
        ):
            super().__init__()
            self.E = youngs_modulus
            self.dim = dim
            self.solver = solver
            self.damping = damping

        def compute_element_stiffness(
            self,
            pos_i: torch.Tensor,
            pos_j: torch.Tensor,
            area: torch.Tensor,
        ) -> torch.Tensor:
            """Compute 2D truss element stiffness matrix (4x4).

            Parameters
            ----------
            pos_i, pos_j : (dim,) node positions
            area : scalar, cross-section area

            Returns
            -------
            (2*dim, 2*dim) element stiffness matrix
            """
            d = pos_j - pos_i
            L = torch.sqrt((d ** 2).sum() + 1e-12)
            c = d / L  # direction cosines

            # Axial stiffness k = EA/L
            k = self.E * area / L

            # Outer product for direction coupling
            cc = torch.outer(c, c)  # (dim, dim)

            # Element stiffness in global coords (block form)
            # K_e = k * [cc, -cc; -cc, cc]
            K = torch.zeros(2 * self.dim, 2 * self.dim, device=pos_i.device)
            K[:self.dim, :self.dim] = k * cc
            K[:self.dim, self.dim:] = -k * cc
            K[self.dim:, :self.dim] = -k * cc
            K[self.dim:, self.dim:] = k * cc

            return K

        def assemble_global_stiffness(
            self,
            edge_index: torch.Tensor,
            node_pos: torch.Tensor,
            radii: torch.Tensor,
        ) -> torch.Tensor:
            """Assemble global stiffness matrix K.

            Parameters
            ----------
            edge_index : (2, n_edges) edge connectivity
            node_pos : (n_nodes, dim) node positions
            radii : (n_edges,) element radii (for area = πr²)

            Returns
            -------
            (n_nodes*dim, n_nodes*dim) global stiffness matrix
            """
            n_nodes = node_pos.shape[0]
            n_dof = n_nodes * self.dim
            K = torch.zeros(n_dof, n_dof, device=node_pos.device)

            areas = math.pi * radii ** 2

            for e in range(edge_index.shape[1]):
                i = edge_index[0, e].item()
                j = edge_index[1, e].item()

                K_e = self.compute_element_stiffness(
                    node_pos[i], node_pos[j], areas[e]
                )

                # Scatter into global matrix
                dof_i = list(range(i * self.dim, (i + 1) * self.dim))
                dof_j = list(range(j * self.dim, (j + 1) * self.dim))
                dofs = dof_i + dof_j

                for a in range(2 * self.dim):
                    for b in range(2 * self.dim):
                        K[dofs[a], dofs[b]] = K[dofs[a], dofs[b]] + K_e[a, b]

            return K

        def solve(
            self,
            edge_index: torch.Tensor,
            node_pos: torch.Tensor,
            radii: torch.Tensor,
            forces: torch.Tensor,
            fixed_nodes: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Solve Ku = f with boundary conditions.

            Parameters
            ----------
            edge_index : (2, n_edges)
            node_pos : (n_nodes, dim)
            radii : (n_edges,) element radii
            forces : (n_nodes, dim) external forces
            fixed_nodes : (n_fixed,) indices of fixed nodes

            Returns
            -------
            displacements : (n_nodes, dim) nodal displacements
            stresses : (n_edges,) axial stress in each element
            """
            n_nodes = node_pos.shape[0]
            n_dof = n_nodes * self.dim

            K = self.assemble_global_stiffness(edge_index, node_pos, radii)

            # Apply damping for numerical stability
            K = K + self.damping * torch.eye(n_dof, device=K.device)

            # Flatten forces
            f = forces.flatten()

            # Apply boundary conditions (fix specified nodes)
            if fixed_nodes is not None and len(fixed_nodes) > 0:
                fixed_dofs = []
                for fn in fixed_nodes:
                    fn_val = fn.item() if hasattr(fn, 'item') else fn
                    for d in range(self.dim):
                        fixed_dofs.append(fn_val * self.dim + d)

                free_dofs = sorted(set(range(n_dof)) - set(fixed_dofs))
                K_free = K[free_dofs][:, free_dofs]
                f_free = f[free_dofs]

                # Solve reduced system
                u_free = torch.linalg.solve(K_free, f_free)

                u = torch.zeros(n_dof, device=node_pos.device)
                for idx, dof in enumerate(free_dofs):
                    u[dof] = u_free[idx]
            else:
                u = torch.linalg.solve(K, f)

            displacements = u.reshape(n_nodes, self.dim)

            # Compute element stresses
            areas = math.pi * radii ** 2
            stresses = []
            for e in range(edge_index.shape[1]):
                i = edge_index[0, e].item()
                j = edge_index[1, e].item()
                d = node_pos[j] - node_pos[i]
                L = torch.sqrt((d ** 2).sum() + 1e-12)
                direction = d / L
                # Axial strain = (u_j - u_i) · direction / L
                du = displacements[j] - displacements[i]
                strain = (du * direction).sum() / L
                stress = self.E * strain
                stresses.append(stress)

            stresses = torch.stack(stresses)

            return displacements, stresses

        def compliance(self, displacements: torch.Tensor,
                       forces: torch.Tensor) -> torch.Tensor:
            """Compute structural compliance C = f^T u (to minimize)."""
            return (forces.flatten() * displacements.flatten()).sum()


    class DifferentiableBeamNetwork(nn.Module):
        """Differentiable Euler-Bernoulli beam network solver.

        Extends spring/truss with bending stiffness for more realistic
        fiber network mechanics.

        Parameters
        ----------
        youngs_modulus : float
            Young's modulus E.
        dim : int
            Must be 2 for beam elements.
        include_bending : bool
            If True, includes bending stiffness (EI terms).
        """

        def __init__(
            self,
            youngs_modulus: float = 1e9,
            dim: int = 2,
            include_bending: bool = True,
            damping: float = 1e-5,
        ):
            super().__init__()
            self.E = youngs_modulus
            self.dim = dim
            self.include_bending = include_bending
            self.damping = damping

        def solve(
            self,
            edge_index: torch.Tensor,
            node_pos: torch.Tensor,
            radii: torch.Tensor,
            forces: torch.Tensor,
            fixed_nodes: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Solve beam network with axial + bending DOFs.

            Parameters
            ----------
            edge_index : (2, n_edges)
            node_pos : (n_nodes, 2)
            radii : (n_edges,) element radii
            forces : (n_nodes, 2) or (n_nodes, 3) if including moments
            fixed_nodes : fixed node indices

            Returns
            -------
            displacements : (n_nodes, 3) [ux, uy, theta]
            stresses : (n_edges,) max axial stress
            """
            n_nodes = node_pos.shape[0]
            # 3 DOFs per node: ux, uy, theta (rotation)
            n_dof = n_nodes * 3

            K = torch.zeros(n_dof, n_dof, device=node_pos.device)

            areas = math.pi * radii ** 2
            I_vals = math.pi * radii ** 4 / 4  # second moment of area

            for e in range(edge_index.shape[1]):
                i = edge_index[0, e].item()
                j = edge_index[1, e].item()
                d = node_pos[j] - node_pos[i]
                L = torch.sqrt((d ** 2).sum() + 1e-12)
                c = d / L
                cx, cy = c[0], c[1]

                A = areas[e]
                I_val = I_vals[e]

                # Axial stiffness terms
                EA_L = self.E * A / L
                EI_L = self.E * I_val / L
                EI_L2 = self.E * I_val / (L ** 2)
                EI_L3 = self.E * I_val / (L ** 3)

                # Simplified 6x6 beam element stiffness (local → global)
                # For 2D: DOFs are [ux_i, uy_i, theta_i, ux_j, uy_j, theta_j]
                k_local = torch.zeros(6, 6, device=node_pos.device)

                if self.include_bending:
                    # Axial terms
                    k_local[0, 0] = EA_L
                    k_local[0, 3] = -EA_L
                    k_local[3, 0] = -EA_L
                    k_local[3, 3] = EA_L

                    # Bending terms (Euler-Bernoulli)
                    k_local[1, 1] = 12 * EI_L3
                    k_local[1, 2] = 6 * EI_L2
                    k_local[1, 4] = -12 * EI_L3
                    k_local[1, 5] = 6 * EI_L2

                    k_local[2, 1] = 6 * EI_L2
                    k_local[2, 2] = 4 * EI_L
                    k_local[2, 4] = -6 * EI_L2
                    k_local[2, 5] = 2 * EI_L

                    k_local[4, 1] = -12 * EI_L3
                    k_local[4, 2] = -6 * EI_L2
                    k_local[4, 4] = 12 * EI_L3
                    k_local[4, 5] = -6 * EI_L2

                    k_local[5, 1] = 6 * EI_L2
                    k_local[5, 2] = 2 * EI_L
                    k_local[5, 4] = -6 * EI_L2
                    k_local[5, 5] = 4 * EI_L
                else:
                    # Pure truss (only axial)
                    k_local[0, 0] = EA_L
                    k_local[0, 3] = -EA_L
                    k_local[3, 0] = -EA_L
                    k_local[3, 3] = EA_L
                    # Small bending regularization
                    k_local[1, 1] = 1e-3 * EI_L3
                    k_local[2, 2] = 1e-3 * EI_L
                    k_local[4, 4] = 1e-3 * EI_L3
                    k_local[5, 5] = 1e-3 * EI_L

                # Rotation matrix (local to global)
                T = torch.zeros(6, 6, device=node_pos.device)
                T[0, 0] = cx; T[0, 1] = cy
                T[1, 0] = -cy; T[1, 1] = cx
                T[2, 2] = 1.0
                T[3, 3] = cx; T[3, 4] = cy
                T[4, 3] = -cy; T[4, 4] = cx
                T[5, 5] = 1.0

                K_e = T.t() @ k_local @ T

                # Assemble
                dofs_i = [i * 3, i * 3 + 1, i * 3 + 2]
                dofs_j = [j * 3, j * 3 + 1, j * 3 + 2]
                dofs = dofs_i + dofs_j
                for a in range(6):
                    for b in range(6):
                        K[dofs[a], dofs[b]] = K[dofs[a], dofs[b]] + K_e[a, b]

            K = K + self.damping * torch.eye(n_dof, device=K.device)

            # Flatten forces (pad to 3 DOFs if needed)
            if forces.shape[1] == 2:
                f = torch.zeros(n_dof, device=forces.device)
                for n in range(n_nodes):
                    f[n * 3] = forces[n, 0]
                    f[n * 3 + 1] = forces[n, 1]
            else:
                f = forces.flatten()

            # Boundary conditions
            if fixed_nodes is not None and len(fixed_nodes) > 0:
                fixed_dofs = []
                for fn in fixed_nodes:
                    fn_val = fn.item() if hasattr(fn, 'item') else fn
                    fixed_dofs.extend([fn_val * 3, fn_val * 3 + 1, fn_val * 3 + 2])
                free_dofs = sorted(set(range(n_dof)) - set(fixed_dofs))
                K_free = K[free_dofs][:, free_dofs]
                f_free = f[free_dofs]
                u_free = torch.linalg.solve(K_free, f_free)
                u = torch.zeros(n_dof, device=node_pos.device)
                for idx, dof in enumerate(free_dofs):
                    u[dof] = u_free[idx]
            else:
                u = torch.linalg.solve(K, f)

            displacements = u.reshape(n_nodes, 3)

            # Element stresses
            stresses = []
            for e in range(edge_index.shape[1]):
                i_node = edge_index[0, e].item()
                j_node = edge_index[1, e].item()
                d = node_pos[j_node] - node_pos[i_node]
                L = torch.sqrt((d ** 2).sum() + 1e-12)
                direction = d / L
                du = displacements[j_node, :2] - displacements[i_node, :2]
                axial_strain = (du * direction).sum() / L
                stress = self.E * axial_strain
                stresses.append(stress)

            return displacements, torch.stack(stresses)


    # ==================================================================
    # Differentiable FEA Wrapper
    # ==================================================================

    class DifferentiableFEA(nn.Module):
        """High-level differentiable FEA wrapper for structure optimization.

        Wraps DifferentiableSpringNetwork or DifferentiableBeamNetwork
        with convenient interface for topology/sizing optimization.

        Parameters
        ----------
        youngs_modulus : float
            Material Young's modulus.
        element_type : str
            "spring" or "beam".
        include_bending : bool
            Include bending stiffness (beam only).
        """

        def __init__(
            self,
            youngs_modulus: float = 1e9,
            element_type: str = "beam",
            include_bending: bool = True,
        ):
            super().__init__()
            self.youngs_modulus = youngs_modulus
            self.element_type = element_type

            if element_type == "beam":
                self.solver = DifferentiableBeamNetwork(
                    youngs_modulus=youngs_modulus,
                    include_bending=include_bending,
                )
            else:
                self.solver = DifferentiableSpringNetwork(
                    youngs_modulus=youngs_modulus,
                )

        def forward(
            self,
            edge_index: torch.Tensor,
            node_pos: torch.Tensor,
            radii: torch.Tensor,
            forces: torch.Tensor,
            fixed_nodes: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Run differentiable FEA and return results dict.

            Returns
            -------
            dict with keys:
                - displacements: (n_nodes, dim)
                - stresses: (n_edges,)
                - compliance: scalar
                - max_stress: scalar
                - volume: scalar
            """
            u, sigma = self.solver.solve(edge_index, node_pos, radii, forces, fixed_nodes)
            compliance = self.solver.compliance(u, forces) if hasattr(self.solver, 'compliance') else (forces.flatten() * u[:, :forces.shape[1]].flatten()).sum()

            volumes = math.pi * radii ** 2 * self._edge_lengths(edge_index, node_pos)

            return {
                "displacements": u,
                "stresses": sigma,
                "compliance": compliance,
                "max_stress": sigma.abs().max(),
                "volume": volumes.sum(),
            }

        def _edge_lengths(self, edge_index: torch.Tensor, node_pos: torch.Tensor) -> torch.Tensor:
            lengths = []
            for e in range(edge_index.shape[1]):
                i = edge_index[0, e].item()
                j = edge_index[1, e].item()
                d = node_pos[j] - node_pos[i]
                lengths.append(torch.sqrt((d ** 2).sum() + 1e-12))
            return torch.stack(lengths)

        @classmethod
        def from_structure_graph(cls, g, youngs_modulus=1e9, element_type="beam"):
            """Create DifferentiableFEA from a StructureGraph."""
            node_ids = sorted(g.nodes.keys())
            node_map = {nid: i for i, nid in enumerate(node_ids)}
            n_nodes = len(node_ids)

            positions = []
            for nid in node_ids:
                pos = g.nodes[nid].position[:2]
                positions.append(pos.tolist())
            node_pos = torch.tensor(positions, dtype=torch.float32)

            src, dst = [], []
            for edge in g.edges.values():
                src.append(node_map[edge.node_i])
                dst.append(node_map[edge.node_j])
            edge_index = torch.tensor([src, dst], dtype=torch.long)

            return cls(youngs_modulus=youngs_modulus, element_type=element_type), node_pos, edge_index


    # ==================================================================
    # Physics-Based Structure Optimizer
    # ==================================================================

    class PhysicsOptimizer:
        """Gradient-based optimizer for structure design using differentiable physics.

        Optimizes cross-section radii (and optionally node positions) to minimize
        compliance or stress subject to volume constraints.

        Parameters
        ----------
        fea : DifferentiableFEA
            Differentiable FEA solver.
        lr : float
            Learning rate.
        volume_constraint : float or None
            Maximum total volume (None = unconstrained).
        penalty_weight : float
            Lagrange penalty for volume constraint.
        """

        def __init__(
            self,
            fea: DifferentiableFEA,
            lr: float = 0.01,
            volume_constraint: Optional[float] = None,
            penalty_weight: float = 1.0,
        ):
            self.fea = fea
            self.lr = lr
            self.volume_constraint = volume_constraint
            self.penalty_weight = penalty_weight

        def optimize(
            self,
            edge_index: torch.Tensor,
            node_pos: torch.Tensor,
            initial_radii: torch.Tensor,
            forces: torch.Tensor,
            fixed_nodes: Optional[torch.Tensor] = None,
            n_iterations: int = 100,
            min_radius: float = 0.001,
            max_radius: float = 0.1,
            objective: str = "compliance",
            verbose: bool = True,
        ) -> Dict[str, Any]:
            """Run topology/sizing optimization.

            Parameters
            ----------
            objective : str
                "compliance" (minimize compliance) or
                "stress" (minimize max stress).

            Returns
            -------
            dict with optimized radii, history, etc.
            """
            radii = initial_radii.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([radii], lr=self.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_iterations, eta_min=1e-4
            )

            history = {"objective": [], "volume": [], "max_stress": []}

            for it in range(n_iterations):
                optimizer.zero_grad()
                result = self.fea(edge_index, node_pos, radii, forces, fixed_nodes)

                if objective == "compliance":
                    obj = result["compliance"]
                else:
                    obj = result["max_stress"]

                loss = obj
                if self.volume_constraint is not None:
                    vol_penalty = F.relu(result["volume"] - self.volume_constraint)
                    loss = loss + self.penalty_weight * vol_penalty

                loss.backward()
                optimizer.step()
                scheduler.step()

                # Clamp radii
                with torch.no_grad():
                    radii.clamp_(min_radius, max_radius)

                history["objective"].append(obj.item())
                history["volume"].append(result["volume"].item())
                history["max_stress"].append(result["max_stress"].item())

                if verbose and (it % max(n_iterations // 10, 1) == 0 or it == n_iterations - 1):
                    print(f"Iter {it:3d} | obj={obj.item():.6f} | "
                          f"vol={result['volume'].item():.4f} | "
                          f"max_stress={result['max_stress'].item():.2f}")

            return {
                "optimized_radii": radii.detach(),
                "history": history,
                "final_result": self.fea(edge_index, node_pos, radii.detach(), forces, fixed_nodes),
            }


    # ==================================================================
    # Differentiable Material Model
    # ==================================================================

    class DifferentiableMaterialModel(nn.Module):
        """Learnable differentiable constitutive law.

        Maps strain to stress using a neural network, trained to satisfy
        physical constraints (monotonicity, positive tangent modulus).

        Parameters
        ----------
        hidden : list of int
            Hidden layer sizes.
        ensure_monotonic : bool
            Enforce monotonic stress-strain via positive weights.
        """

        def __init__(self, hidden: Optional[List[int]] = None, ensure_monotonic: bool = False):
            super().__init__()
            if hidden is None:
                hidden = [32, 16]
            self.ensure_monotonic = ensure_monotonic

            layers = []
            prev = 1  # strain input
            for h in hidden:
                layers.append(nn.Linear(prev, h))
                if ensure_monotonic:
                    layers.append(nn.Softplus())
                else:
                    layers.append(nn.GELU())
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

            if ensure_monotonic:
                self._enforce_positive_weights()

        def _enforce_positive_weights(self):
            """Ensure all linear layer weights are positive for monotonicity."""
            with torch.no_grad():
                for layer in self.net:
                    if isinstance(layer, nn.Linear):
                        layer.weight.abs_()
                        layer.bias.zero_()

        def forward(self, strain: torch.Tensor) -> torch.Tensor:
            """Map strain → stress.

            Parameters
            ----------
            strain : (N,) or (N, 1) strain values

            Returns
            -------
            (N,) stress values
            """
            if strain.dim() == 1:
                strain = strain.unsqueeze(-1)
            stress = self.net(strain).squeeze(-1)
            return stress

        def tangent_modulus(self, strain: torch.Tensor) -> torch.Tensor:
            """Compute dσ/dε (tangent modulus) via autograd.

            Parameters
            ----------
            strain : (N,) strain values

            Returns
            -------
            (N,) tangent modulus values
            """
            s = strain.detach().requires_grad_(True)
            sigma = self.forward(s)
            dsigma = torch.autograd.grad(
                sigma.sum(), s, create_graph=True
            )[0]
            return dsigma

        def fit(
            self,
            strain_data: np.ndarray,
            stress_data: np.ndarray,
            epochs: int = 200,
            lr: float = 1e-3,
            verbose: bool = True,
        ) -> Dict[str, Any]:
            """Train material model on experimental data.

            Parameters
            ----------
            strain_data, stress_data : 1D arrays
            epochs : int
            lr : float

            Returns
            -------
            dict with training history
            """
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            strain_t = torch.tensor(strain_data, dtype=torch.float32)
            stress_t = torch.tensor(stress_data, dtype=torch.float32)

            history = {"train_loss": []}
            for epoch in range(epochs):
                optimizer.zero_grad()
                pred = self.forward(strain_t)
                loss = F.mse_loss(pred, stress_t)

                # Optional: monotonicity penalty
                if self.ensure_monotonic:
                    tangent = self.tangent_modulus(strain_t)
                    mono_penalty = F.relu(-tangent).mean()
                    loss = loss + 10.0 * mono_penalty

                loss.backward()
                optimizer.step()

                if self.ensure_monotonic:
                    self._enforce_positive_weights()

                history["train_loss"].append(loss.item())

                if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch:3d} | loss={loss.item():.6f}")

            return history

else:
    class DifferentiableSpringNetwork:
        def __init__(self, *a, **kw):
            _require_torch()

    class DifferentiableBeamNetwork:
        def __init__(self, *a, **kw):
            _require_torch()

    class DifferentiableFEA:
        def __init__(self, *a, **kw):
            _require_torch()

    class PhysicsOptimizer:
        def __init__(self, *a, **kw):
            _require_torch()

    class DifferentiableMaterialModel:
        def __init__(self, *a, **kw):
            _require_torch()
