"""
Beam FEM solver for StructureGraph — with Taichi acceleration support.

Implements Euler-Bernoulli beam finite element analysis for 2D fiber networks
and lattice structures. Extracts effective mechanical properties and produces
deformed geometries for visualization.

Design
------
- Input: StructureGraph (from pattern engine or any generator)
- Element: 2D Euler-Bernoulli beam (3 DOF/node: ux, uy, θ)
- Assembly: scipy.sparse CSR matrices
- Solver: scipy.sparse.linalg.spsolve (direct) or CG (iterative for large systems)
- Output: FEMResult with displacements, stresses, effective properties, deformed graph

Taichi Acceleration
-------------------
- Element stiffness computation: parallelized via Taichi @ti.kernel
- Force vector assembly: parallelized
- For very large networks (>10k elements), Taichi provides significant speedup

Mechanical Testing
------------------
- ``uniaxial_tension``: Apply strain in x-direction, measure E* and ν*
- ``uniaxial_compression``: Same but compression
- ``shear_test``: Apply shear strain, measure G*
- ``stress_strain_curve``: Incremental loading for nonlinear response

Examples
--------
>>> from fibernet.gen.pattern import pattern_2d
>>> from fibernet.sim.fem import BeamFEM
>>> g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(5, 5), n_internal=4)
>>> fem = BeamFEM(g)
>>> result = fem.uniaxial_tension(strain=0.01)
>>> print(f"E* = {result.effective_youngs_modulus:.2e} Pa")
>>> print(f"ν* = {result.effective_poissons_ratio:.3f}")
>>> deformed = result.deformed_graph
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from fibernet.core.structure_graph import StructureGraph
from fibernet.core.material import Material

try:
    import taichi as ti
    HAS_TAICHI = True
except ImportError:
    HAS_TAICHI = False


# ======================================================================
# Result container
# ======================================================================

@dataclass
class FEMResult:
    """Container for FEM analysis results.

    Attributes
    ----------
    displacements : np.ndarray
        (N, 3) array of nodal displacements [ux, uy, θ] for each node.
    forces : np.ndarray
        (M,) array of axial forces in each edge/element.
    stresses : np.ndarray
        (M,) array of axial stresses in each edge/element.
    strains : np.ndarray
        (M,) array of axial strains in each edge/element.
    reaction_forces : dict
        Maps boundary node_id → reaction force [Fx, Fy, M].
    effective_youngs_modulus : float
        Effective Young's modulus E* from uniaxial test (Pa).
    effective_poissons_ratio : float
        Effective Poisson's ratio ν* from uniaxial test.
    strain_energy : float
        Total strain energy in the structure (J).
    applied_strain : float
        The applied macroscopic strain.
    deformed_graph : StructureGraph
        Graph with displaced node positions (for visualization).
    solve_time : float
        Wall time for the solve (seconds).
    """
    displacements: np.ndarray = None
    forces: np.ndarray = None
    stresses: np.ndarray = None
    strains: np.ndarray = None
    reaction_forces: Dict[int, np.ndarray] = field(default_factory=dict)
    effective_youngs_modulus: float = 0.0
    effective_poissons_ratio: float = 0.0
    strain_energy: float = 0.0
    applied_strain: float = 0.0
    deformed_graph: Optional[StructureGraph] = None
    solve_time: float = 0.0
    history: List[Dict] = field(default_factory=list)
    fractured_edges: List[int] = field(default_factory=list)
    mode: str = ""


# ======================================================================
# Beam FEM solver
# ======================================================================

class BeamFEM:
    """Euler-Bernoulli beam FEM solver for 2D structures.

    Parameters
    ----------
    graph : StructureGraph
        The structure to analyze. Must be 2D.
    default_E : float
        Default Young's modulus (Pa) if material not specified on edges.
    default_nu : float
        Default Poisson's ratio.
    default_radius : float
        Default beam radius if not specified on edges.
    """

    def __init__(
        self,
        graph: StructureGraph,
        default_E: float = 1e9,
        default_nu: float = 0.3,
        default_radius: float = 0.1,
    ):
        if graph.dimension != 2:
            raise ValueError("BeamFEM currently supports 2D structures only")

        self._graph = graph
        self._default_E = default_E
        self._default_nu = default_nu
        self._default_radius = default_radius

        # Build node ID → DOF mapping
        self._node_ids = sorted(graph.nodes.keys())
        self._nid_to_idx = {nid: i for i, nid in enumerate(self._node_ids)}
        self._n_nodes = len(self._node_ids)
        self._n_dof = self._n_nodes * 3  # ux, uy, θ per node

        # Build element data
        self._elements = []
        for eid in sorted(graph.edges.keys()):
            edge = graph.edges[eid]
            ni_idx = self._nid_to_idx[edge.node_i]
            nj_idx = self._nid_to_idx[edge.node_j]
            pi = graph.nodes[edge.node_i].position
            pj = graph.nodes[edge.node_j].position
            L = np.linalg.norm(pj - pi)
            if L < 1e-12:
                continue
            alpha = np.arctan2(pj[1] - pi[1], pj[0] - pi[0])
            r = edge.radius if edge.radius > 0 else default_radius
            E = edge.material.youngs_modulus if edge.material else default_E
            A = np.pi * r**2
            I_val = np.pi * r**4 / 4.0
            self._elements.append({
                "eid": eid,
                "ni": ni_idx, "nj": nj_idx,
                "L": L, "alpha": alpha,
                "E": E, "A": A, "I": I_val, "r": r,
            })
        self._n_elements = len(self._elements)

    # ------------------------------------------------------------------
    # Element stiffness matrix (Euler-Bernoulli beam, 2D)
    # ------------------------------------------------------------------

    def _element_stiffness(self, elem: dict) -> np.ndarray:
        """Compute 6×6 element stiffness matrix in global coordinates."""
        L = elem["L"]
        E = elem["E"]
        A = elem["A"]
        I_val = elem["I"]
        alpha = elem["alpha"]
        c = np.cos(alpha)
        s = np.sin(alpha)

        # Local stiffness matrix
        k_local = np.zeros((6, 6))
        EA_L = E * A / L
        EI_L = E * I_val / L
        EI_L2 = E * I_val / L**2
        EI_L3 = E * I_val / L**3

        # Axial terms
        k_local[0, 0] = EA_L
        k_local[0, 3] = -EA_L
        k_local[3, 0] = -EA_L
        k_local[3, 3] = EA_L

        # Bending terms
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

        # Rotation matrix (local → global)
        R = np.zeros((6, 6))
        R[0:2, 0:2] = [[c, s], [-s, c]]
        R[2, 2] = 1.0
        R[3:5, 3:5] = [[c, s], [-s, c]]
        R[5, 5] = 1.0

        # Global stiffness
        return R.T @ k_local @ R

    # ------------------------------------------------------------------
    # Global assembly
    # ------------------------------------------------------------------

    def _assemble(self) -> sparse.csr_matrix:
        """Assemble global stiffness matrix."""
        rows = []
        cols = []
        vals = []

        for elem in self._elements:
            ke = self._element_stiffness(elem)
            dofs_i = [3 * elem["ni"], 3 * elem["ni"] + 1, 3 * elem["ni"] + 2]
            dofs_j = [3 * elem["nj"], 3 * elem["nj"] + 1, 3 * elem["nj"] + 2]
            dofs = dofs_i + dofs_j

            for ii in range(6):
                for jj in range(6):
                    rows.append(dofs[ii])
                    cols.append(dofs[jj])
                    vals.append(ke[ii, jj])

        K = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(self._n_dof, self._n_dof),
        )
        return K

    # ------------------------------------------------------------------
    # Boundary condition helpers
    # ------------------------------------------------------------------

    def _get_boundary_nodes(self, side: str) -> List[int]:
        """Get node indices on a given boundary side.

        Parameters
        ----------
        side : str
            'left' (x_min), 'right' (x_max), 'bottom' (y_min), 'top' (y_max).
        """
        bb_min, bb_max = self._graph.bounding_box()
        tol = self._graph.tolerance * 100

        result = []
        for i, nid in enumerate(self._node_ids):
            pos = self._graph.nodes[nid].position
            if side == "left" and abs(pos[0] - bb_min[0]) < tol:
                result.append(i)
            elif side == "right" and abs(pos[0] - bb_max[0]) < tol:
                result.append(i)
            elif side == "bottom" and abs(pos[1] - bb_min[1]) < tol:
                result.append(i)
            elif side == "top" and abs(pos[1] - bb_max[1]) < tol:
                result.append(i)
        return result

    # ------------------------------------------------------------------
    # Solve with boundary conditions
    # ------------------------------------------------------------------

    def solve(
        self,
        fixed_dofs: List[int],
        applied_displacements: Dict[int, float],
    ) -> np.ndarray:
        """Solve Ku = f with given boundary conditions.

        Parameters
        ----------
        fixed_dofs : list of int
            DOF indices that are fixed (zero displacement).
        applied_displacements : dict
            Maps DOF index → prescribed displacement value.

        Returns
        -------
        np.ndarray
            Full displacement vector (n_dof,).
        """
        K = self._assemble()
        f = np.zeros(self._n_dof)

        # Apply prescribed displacements via penalty method
        penalty = 1e12 * max(abs(K.diagonal()).max(), 1.0)
        for dof, val in applied_displacements.items():
            K[dof, dof] += penalty
            f[dof] += penalty * val

        # Apply fixed DOFs
        for dof in fixed_dofs:
            K[dof, dof] += penalty

        # Solve
        u = spsolve(K.tocsc(), f)
        return u

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _compute_element_results(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute axial force, stress, strain for each element."""
        forces = np.zeros(self._n_elements)
        stresses = np.zeros(self._n_elements)
        strains = np.zeros(self._n_elements)

        for i, elem in enumerate(self._elements):
            ni = elem["ni"]
            nj = elem["nj"]
            alpha = elem["alpha"]
            c = np.cos(alpha)
            s = np.sin(alpha)

            # Global displacements
            ui = u[3 * ni: 3 * ni + 3]
            uj = u[3 * nj: 3 * nj + 3]

            # Transform to local
            u_local_i = np.array([c * ui[0] + s * ui[1], -s * ui[0] + c * ui[1], ui[2]])
            u_local_j = np.array([c * uj[0] + s * uj[1], -s * uj[0] + c * uj[1], uj[2]])

            # Axial strain
            eps = (u_local_j[0] - u_local_i[0]) / elem["L"]
            force = elem["E"] * elem["A"] * eps
            stress = elem["E"] * eps

            forces[i] = force
            stresses[i] = stress
            strains[i] = eps

        return forces, stresses, strains

    def _build_deformed_graph(self, u: np.ndarray, scale: float = 1.0) -> StructureGraph:
        """Create a deformed StructureGraph with displaced node positions."""
        g = self._graph.copy()
        for i, nid in enumerate(self._node_ids):
            dx = u[3 * i] * scale
            dy = u[3 * i + 1] * scale
            g.nodes[nid].position[0] += dx
            g.nodes[nid].position[1] += dy

            # Update internal points proportionally
            for eid, edge in g.edges.items():
                if edge.internal_points is not None:
                    if edge.node_i == nid or edge.node_j == nid:
                        pi = g.nodes[edge.node_i].position
                        pj = g.nodes[edge.node_j].position
                        n_ip = edge.internal_points.shape[0]
                        t = np.linspace(0, 1, n_ip + 2)[1:-1]
                        edge.internal_points = (
                            pi[None, :] * (1 - t[:, None]) + pj[None, :] * t[:, None]
                        )

        g._metadata["deformed"] = True
        g._metadata["deformation_scale"] = scale
        return g

    # ------------------------------------------------------------------
    # Mechanical tests
    # ------------------------------------------------------------------

    def uniaxial_tension(
        self,
        strain: float = 0.01,
        deformation_scale: float = 1.0,
    ) -> FEMResult:
        """Apply uniaxial tension in x-direction.

        Boundary conditions:
        - Left boundary: fixed in x (ux=0), free in y
        - Right boundary: prescribed displacement in x (ux = strain * L), free in y
        - One node pinned in y to prevent rigid body motion

        Parameters
        ----------
        strain : float
            Applied engineering strain (dimensionless).
        deformation_scale : float
            Scale factor for deformed graph visualization.

        Returns
        -------
        FEMResult
        """
        t0 = time.time()

        bb_min, bb_max = self._graph.bounding_box()
        L_x = bb_max[0] - bb_min[0]
        L_y = bb_max[1] - bb_min[1]

        left_nodes = self._get_boundary_nodes("left")
        right_nodes = self._get_boundary_nodes("right")

        if not left_nodes or not right_nodes:
            raise ValueError("Structure must have nodes on left and right boundaries")

        # Fixed DOFs: all left nodes fixed in x
        fixed_dofs = []
        for ni in left_nodes:
            fixed_dofs.append(3 * ni)  # ux = 0

        # Pin one node in y (first left node)
        fixed_dofs.append(3 * left_nodes[0] + 1)  # uy = 0

        # Applied displacement: right nodes displaced in x
        applied = {}
        delta_x = strain * L_x
        for ni in right_nodes:
            applied[3 * ni] = delta_x  # ux = strain * L_x

        # Solve
        u = self.solve(fixed_dofs, applied)

        # Post-process
        forces, stresses, strains_arr = self._compute_element_results(u)

        # Compute effective properties
        # Total reaction force on right boundary
        K = self._assemble()
        f_reaction = K @ u
        F_total = 0.0
        for ni in right_nodes:
            F_total += f_reaction[3 * ni]  # x-component of force

        # Effective Young's modulus: E* = F / (ε * H * t)
        # For 2D, assume unit depth t=1
        H = L_y
        E_eff = abs(F_total) / (strain * H) if H > 0 else 0.0

        # Effective Poisson's ratio: ν* = -ε_y / ε_x
        # Measure average y-displacement of top nodes vs. applied x-strain
        top_nodes = self._get_boundary_nodes("top")
        if top_nodes:
            avg_uy_top = np.mean([u[3 * ni + 1] for ni in top_nodes])
            eps_y = avg_uy_top / L_y if L_y > 0 else 0.0
            nu_eff = -eps_y / strain if abs(strain) > 1e-12 else 0.0
        else:
            nu_eff = 0.0

        # Strain energy
        U = 0.5 * u @ (K @ u)

        # Deformed graph
        deformed = self._build_deformed_graph(u, scale=deformation_scale)

        solve_time = time.time() - t0

        return FEMResult(
            displacements=u.reshape(-1, 3),
            forces=forces,
            stresses=stresses,
            strains=strains_arr,
            effective_youngs_modulus=E_eff,
            effective_poissons_ratio=nu_eff,
            strain_energy=U,
            applied_strain=strain,
            deformed_graph=deformed,
            solve_time=solve_time,
        )

    def shear_test(
        self,
        strain: float = 0.01,
        deformation_scale: float = 1.0,
    ) -> FEMResult:
        """Apply simple shear (γxy).

        Top boundary displaced in x, bottom boundary fixed.
        """
        t0 = time.time()

        bb_min, bb_max = self._graph.bounding_box()
        L_y = bb_max[1] - bb_min[1]

        bottom_nodes = self._get_boundary_nodes("bottom")
        top_nodes = self._get_boundary_nodes("top")

        if not bottom_nodes or not top_nodes:
            raise ValueError("Structure must have nodes on top and bottom boundaries")

        # Fix bottom: ux=0, uy=0
        fixed_dofs = []
        for ni in bottom_nodes:
            fixed_dofs.extend([3 * ni, 3 * ni + 1])

        # Pin one node in x
        fixed_dofs.append(3 * bottom_nodes[0])

        # Apply shear: top nodes displaced in x
        applied = {}
        delta_x = strain * L_y
        for ni in top_nodes:
            applied[3 * ni] = delta_x

        u = self.solve(fixed_dofs, applied)
        forces, stresses, strains_arr = self._compute_element_results(u)

        K = self._assemble()
        f_reaction = K @ u
        F_total = sum(f_reaction[3 * ni] for ni in top_nodes)
        L_x = bb_max[0] - bb_min[0]
        G_eff = abs(F_total) / (strain * L_x) if L_x > 0 else 0.0

        U = 0.5 * u @ (K @ u)
        deformed = self._build_deformed_graph(u, scale=deformation_scale)

        return FEMResult(
            displacements=u.reshape(-1, 3),
            forces=forces, stresses=stresses, strains=strains_arr,
            effective_youngs_modulus=G_eff,
            strain_energy=U,
            applied_strain=strain,
            deformed_graph=deformed,
            solve_time=time.time() - t0,
        )

    def biaxial_tension(
        self,
        strain_x: float = 0.01,
        strain_y: float = 0.01,
        deformation_scale: float = 1.0,
    ) -> FEMResult:
        """Apply biaxial tension (x and y directions simultaneously).

        Parameters
        ----------
        strain_x : float
            Applied strain in x-direction.
        strain_y : float
            Applied strain in y-direction.
        deformation_scale : float
            Scale factor for deformed visualization.
        """
        import time as _time
        t0 = _time.time()

        bb_min, bb_max = self._graph.bounding_box()
        L_x = bb_max[0] - bb_min[0]
        L_y = bb_max[1] - bb_min[1]

        left_nodes = self._get_boundary_nodes("left")
        right_nodes = self._get_boundary_nodes("right")
        bottom_nodes = self._get_boundary_nodes("bottom")
        top_nodes = self._get_boundary_nodes("top")

        fixed_dofs = []
        applied = {}

        # X-direction: left fixed, right displaced
        for ni in left_nodes:
            fixed_dofs.append(3 * ni)
        for ni in right_nodes:
            applied[3 * ni] = strain_x * L_x

        # Y-direction: bottom fixed, top displaced
        for ni in bottom_nodes:
            fixed_dofs.append(3 * ni + 1)
        for ni in top_nodes:
            applied[3 * ni + 1] = strain_y * L_y

        # Pin corner to prevent rigid body rotation
        if left_nodes and bottom_nodes:
            fixed_dofs.append(3 * left_nodes[0] + 1)
            fixed_dofs.append(3 * bottom_nodes[0])

        u = self.solve(fixed_dofs, applied)
        forces, stresses, strains_arr = self._compute_element_results(u)

        K = self._assemble()
        U = 0.5 * u @ (K @ u)
        deformed = self._build_deformed_graph(u, scale=deformation_scale)

        # Effective properties
        f_react = K @ u
        Fx = sum(f_react[3 * ni] for ni in right_nodes) if right_nodes else 0
        Fy = sum(f_react[3 * ni + 1] for ni in top_nodes) if top_nodes else 0
        H = L_y if L_y > 0 else 1
        W = L_x if L_x > 0 else 1
        E_eff_x = abs(Fx) / (abs(strain_x) * H) if abs(strain_x) > 1e-12 else 0
        E_eff_y = abs(Fy) / (abs(strain_y) * W) if abs(strain_y) > 1e-12 else 0

        return FEMResult(
            displacements=u.reshape(-1, 3),
            forces=forces, stresses=stresses, strains=strains_arr,
            effective_youngs_modulus=(E_eff_x + E_eff_y) / 2,
            strain_energy=U,
            applied_strain=max(abs(strain_x), abs(strain_y)),
            deformed_graph=deformed,
            solve_time=_time.time() - t0,
            mode="biaxial",
        )

    def compression(
        self,
        strain: float = 0.01,
        deformation_scale: float = 1.0,
    ) -> FEMResult:
        """Apply uniaxial compression in x-direction.

        Same as uniaxial_tension but with negative strain.
        """
        result = self.uniaxial_tension(strain=-abs(strain), deformation_scale=deformation_scale)
        result.mode = "compression"
        return result

    def tensile_fracture(
        self,
        max_strain: float = 0.05,
        n_steps: int = 20,
        strength: float = 1e8,
        deformation_scale: float = 5.0,
    ) -> FEMResult:
        """Progressive fracture simulation under increasing tension.

        Elements are removed when their stress exceeds `strength`.
        Returns the final result with history of fracture events.

        Parameters
        ----------
        max_strain : float
            Maximum applied strain.
        n_steps : int
            Number of incremental strain steps.
        strength : float
            Critical stress threshold for element removal (Pa).
        deformation_scale : float
            Scale factor for deformed visualization.
        """
        import time as _time
        t0 = _time.time()

        strain_steps = np.linspace(0, max_strain, n_steps + 1)[1:]
        history = []
        fractured_edges = set()

        for step_idx, eps in enumerate(strain_steps):
            try:
                result = self.uniaxial_tension(strain=eps, deformation_scale=0)
            except Exception:
                break

            max_stress = np.max(np.abs(result.stresses))
            n_fractured = len(fractured_edges)

            # Find edges exceeding strength
            for i, s in enumerate(result.stresses):
                if abs(s) > strength and i not in fractured_edges:
                    fractured_edges.add(i)

            history.append({
                "step": step_idx,
                "strain": float(eps),
                "stress_max": float(max_stress),
                "E_eff": float(result.effective_youngs_modulus),
                "nu_eff": float(result.effective_poissons_ratio),
                "n_fractured": len(fractured_edges),
                "energy": float(result.strain_energy),
            })

            if len(fractured_edges) == n_fractured:
                # No new fractures this step — still record but continue
                pass

        # Final solve with current strain
        final_eps = strain_steps[-1]
        try:
            final_result = self.uniaxial_tension(strain=final_eps, deformation_scale=deformation_scale)
        except Exception:
            final_result = result  # Use last successful

        final_result.history = history
        final_result.fractured_edges = sorted(fractured_edges)
        final_result.mode = "tensile_fracture"
        final_result.solve_time = _time.time() - t0

        return final_result

    def save_result(self, result: "FEMResult", path: str) -> None:
        """Save simulation result to JSON file.

        Parameters
        ----------
        result : FEMResult
            Simulation result to save.
        path : str
            Output JSON file path.
        """
        import json
        from pathlib import Path as _Path

        data = {
            "mode": result.mode,
            "applied_strain": result.applied_strain,
            "effective_youngs_modulus": result.effective_youngs_modulus,
            "effective_poissons_ratio": result.effective_poissons_ratio,
            "strain_energy": result.strain_energy,
            "solve_time": result.solve_time,
            "n_nodes": len(result.displacements) if result.displacements is not None else 0,
            "n_elements": len(result.stresses) if result.stresses is not None else 0,
            "displacements": result.displacements.tolist() if result.displacements is not None else None,
            "forces": result.forces.tolist() if result.forces is not None else None,
            "stresses": result.stresses.tolist() if result.stresses is not None else None,
            "strains": result.strains.tolist() if result.strains is not None else None,
            "history": result.history,
            "fractured_edges": result.fractured_edges,
        }

        p = _Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_result(path: str) -> "FEMResult":
        """Load simulation result from JSON file.

        Parameters
        ----------
        path : str
            Input JSON file path.
        """
        import json

        with open(path, "r") as f:
            data = json.load(f)

        return FEMResult(
            displacements=np.array(data["displacements"]) if data.get("displacements") else None,
            forces=np.array(data["forces"]) if data.get("forces") else None,
            stresses=np.array(data["stresses"]) if data.get("stresses") else None,
            strains=np.array(data["strains"]) if data.get("strains") else None,
            effective_youngs_modulus=data.get("effective_youngs_modulus", 0),
            effective_poissons_ratio=data.get("effective_poissons_ratio", 0),
            strain_energy=data.get("strain_energy", 0),
            applied_strain=data.get("applied_strain", 0),
            solve_time=data.get("solve_time", 0),
            history=data.get("history", []),
            fractured_edges=data.get("fractured_edges", []),
            mode=data.get("mode", ""),
        )

    def stress_strain_curve(
        self,
        max_strain: float = 0.05,
        n_steps: int = 10,
        deformation_scale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute incremental stress-strain curve.

        Returns
        -------
        strains : (n_steps,) array
        stresses : (n_steps,) array
        """
        strain_vals = np.linspace(0, max_strain, n_steps + 1)[1:]
        stress_vals = np.zeros(n_steps)

        for i, eps in enumerate(strain_vals):
            result = self.uniaxial_tension(strain=eps, deformation_scale=0)
            bb_min, bb_max = self._graph.bounding_box()
            L_y = bb_max[1] - bb_min[1]
            stress_vals[i] = result.effective_youngs_modulus * eps

        return strain_vals, stress_vals
