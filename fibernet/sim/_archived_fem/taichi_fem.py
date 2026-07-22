"""
Archived: TaichiFEMSolver (static truss FEM)

Status: 雪藏 (2026-07-13)
Reason: 杆单元 FEM 效果不佳，E* 偏低，非三角化结构靠正则化
Replaced by: TaichiEngine (mass-spring dynamics) — all-in 质点弹簧

Usage (if needed in future):
    from fibernet.sim._archived_fem.taichi_fem import TaichiFEMSolver
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np

try:
    import taichi as ti
    HAS_TAICHI = True
except ImportError:
    HAS_TAICHI = False

import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class SimResult:
    """Result container (shared with accelerated.py)."""
    displacements: np.ndarray = None
    forces: np.ndarray = None
    stresses: np.ndarray = None
    strains: np.ndarray = None
    energy: float = 0.0
    effective_youngs_modulus: float = 0.0
    effective_poissons_ratio: float = 0.0
    effective_shear_modulus: float = 0.0
    time_seconds: float = 0.0
    mode: str = ""
    deformed_positions: np.ndarray = None
    history: List[Dict] = field(default_factory=list)
    positions_trajectory: List[np.ndarray] = field(default_factory=list)


def _ensure_taichi(arch="cpu", num_threads=4):
    if not HAS_TAICHI:
        raise ImportError("Taichi required")
    try:
        if not ti.is_initialized(): pass
    except: pass
    try:
        arch_map = {"cpu": ti.cpu, "gpu": ti.gpu}
        ti.init(arch=arch_map.get(arch, ti.cpu), cpu_max_num_threads=num_threads)
    except: pass


def _graph_to_arrays(graph):
    node_ids = list(graph.nodes.keys())
    nid_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    pos = np.array([graph.nodes[nid].position for nid in node_ids])
    elements = np.array([[nid_to_idx[e.node_i], nid_to_idx[e.node_j]] for e in graph.edges.values()])
    return pos, elements, node_ids, nid_to_idx


def _element_data(pos, elements):
    diff = pos[elements[:, 1]] - pos[elements[:, 0]]
    lengths = np.linalg.norm(diff, axis=1)
    lengths = np.maximum(lengths, 1e-12)
    directions = diff / lengths[:, None]
    return lengths, directions


def _get_boundary_indices(pos, tol=None):
    if tol is None:
        span = pos.max(0) - pos.min(0)
        tol = max(span.min() * 0.05, 0.1)
    bb_min, bb_max = pos.min(0), pos.max(0)
    return {
        "left": list(np.where(pos[:, 0] < bb_min[0] + tol)[0]),
        "right": list(np.where(pos[:, 0] > bb_max[0] - tol)[0]),
        "bottom": list(np.where(pos[:, 1] < bb_min[1] + tol)[0]),
        "top": list(np.where(pos[:, 1] > bb_max[1] - tol)[0]),
    }


class TaichiFEMSolver:
    """Static truss FEM with Taichi parallel assembly.

    Truss/bar elements: axial force only (no bending).
    Uses 2 DOF/node for 2D, 3 DOF/node for 3D.
    """

    def __init__(self, arch: str = "cpu", num_threads: int = 4):
        _ensure_taichi(arch, num_threads)

    def _solve_truss(
        self,
        pos: np.ndarray,
        elements: np.ndarray,
        youngs_modulus: float,
        radii: np.ndarray,
        fixed_dofs: List[int],
        applied_displacements: Dict[int, float] = None,
        applied_forces: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Core truss solver with displacement or force BCs."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        num_nodes = pos.shape[0]
        num_elements = elements.shape[0]
        dim = 2 if np.allclose(pos[:, 2], pos[0, 2]) else 3
        ndof = num_nodes * dim

        lengths, directions = _element_data(pos, elements)
        areas = np.pi * radii[:num_elements] ** 2
        axial_k = youngs_modulus * areas / lengths

        # Taichi parallel element stiffness
        ti_ke = ti.field(dtype=ti.f64, shape=(num_elements, dim, dim))
        ti_dir = ti.Vector.field(dim, dtype=ti.f64, shape=num_elements)
        ti_k = ti.field(dtype=ti.f64, shape=num_elements)

        ti_dir.from_numpy(directions[:, :dim].astype(np.float64))
        ti_k.from_numpy(axial_k.astype(np.float64))

        @ti.kernel
        def assemble_ke():
            for e in range(num_elements):
                k = ti_k[e]
                d = ti_dir[e]
                for i in ti.static(range(dim)):
                    for j in ti.static(range(dim)):
                        ti_ke[e, i, j] = k * d[i] * d[j]

        assemble_ke()
        ke_all = ti_ke.to_numpy()

        # Global assembly
        rows, cols, vals = [], [], []
        for e in range(num_elements):
            ni, nj = elements[e]
            for a in range(dim):
                for b in range(dim):
                    v = ke_all[e, a, b]
                    if abs(v) > 1e-15:
                        rows.extend([ni * dim + a, ni * dim + a, nj * dim + a, nj * dim + a])
                        cols.extend([ni * dim + b, nj * dim + b, ni * dim + b, nj * dim + b])
                        vals.extend([v, -v, -v, v])

        K = sp.csr_matrix((vals, (rows, cols)), shape=(ndof, ndof))

        # Boundary conditions
        all_dofs = set(range(ndof))
        fixed_set = set(fixed_dofs)
        # Also fix applied displacement DOFs
        if applied_displacements:
            fixed_set.update(applied_displacements.keys())
        free_dofs = sorted(all_dofs - fixed_set)

        K_ff = K[free_dofs, :][:, free_dofs]

        # Regularization: small ground springs to stabilize mechanisms
        # Scale relative to matrix diagonal for robustness
        diag_max = abs(K_ff.diagonal()).max() if len(free_dofs) > 0 else 1.0
        reg = max(diag_max * 1e-8, 1e-6)
        K_ff = K_ff + sp.eye(len(free_dofs), format="csr") * reg

        # Right-hand side
        F = np.zeros(ndof)
        if applied_forces is not None:
            F = applied_forces.flatten()[:ndof]

        # Handle prescribed displacements
        u = np.zeros(ndof)
        if applied_displacements:
            for dof, val in applied_displacements.items():
                u[dof] = val
            # Modify RHS for prescribed displacements
            u_prescribed = np.zeros(ndof)
            for dof, val in applied_displacements.items():
                u_prescribed[dof] = val
            F_mod = F - K @ u_prescribed
            F_free = F_mod[free_dofs]
        else:
            F_free = F[free_dofs]

        u_free = spla.spsolve(K_ff, F_free)
        u[free_dofs] = u_free

        displacements = u.reshape(num_nodes, dim)
        if dim == 2:
            displacements = np.hstack([displacements, np.zeros((num_nodes, 1))])

        # Element results
        strains = np.zeros(num_elements)
        stresses = np.zeros(num_elements)
        for e in range(num_elements):
            ni, nj = elements[e]
            delta = displacements[nj, :dim] - displacements[ni, :dim]
            strains[e] = np.dot(delta, directions[e, :dim]) / lengths[e]
            stresses[e] = youngs_modulus * strains[e]

        return displacements, strains, stresses

    def uniaxial_tension(
        self,
        graph: StructureGraph,
        strain: float = 0.01,
        youngs_modulus: float = 1e9,
        radius: float = 0.05,
    ) -> SimResult:
        """Uniaxial tension in x-direction."""
        t0 = time.time()
        pos, elements, _, _ = _graph_to_arrays(graph)
        radii = np.full(len(elements), radius)
        bnd = _get_boundary_indices(pos)

        dim = 2 if np.allclose(pos[:, 2], pos[0, 2]) else 3
        L_x = pos[:, 0].max() - pos[:, 0].min()
        L_y = pos[:, 1].max() - pos[:, 1].min()
        delta_x = strain * L_x

        # Fix left boundary (x), pin one y
        fixed_dofs = []
        for ni in bnd["left"]:
            fixed_dofs.append(ni * dim)
        if bnd["bottom"]:
            fixed_dofs.append(bnd["bottom"][0] * dim + 1)

        # Prescribe right boundary displacement
        applied = {}
        for ni in bnd["right"]:
            applied[ni * dim] = delta_x

        displacements, strains, stresses = self._solve_truss(
            pos, elements, youngs_modulus, radii, fixed_dofs,
            applied_displacements=applied,
        )

        # Effective properties
        K = self._build_stiffness(pos, elements, youngs_modulus, radii, dim)
        u_flat = displacements[:, :dim].flatten()
        f_react = K @ u_flat

        F_total = sum(f_react[ni * dim] for ni in bnd["right"])
        H = L_y if dim >= 2 else 1
        E_eff = abs(F_total) / (abs(strain) * H) if abs(strain) > 1e-12 else 0

        # Poisson's ratio
        top_nodes = bnd.get("top", [])
        if top_nodes and abs(strain) > 1e-12:
            avg_uy = np.mean([displacements[ni, 1] for ni in top_nodes])
            eps_y = avg_uy / L_y
            nu_eff = -eps_y / strain
        else:
            nu_eff = 0.0

        energy = 0.5 * u_flat @ (K @ u_flat)
        deformed_pos = pos + displacements

        forces = np.array([
            stresses[e] * np.pi * radii[e] ** 2 for e in range(len(elements))
        ])

        return SimResult(
            displacements=displacements,
            forces=forces,
            stresses=stresses,
            strains=strains,
            energy=energy,
            effective_youngs_modulus=E_eff,
            effective_poissons_ratio=nu_eff,
            time_seconds=time.time() - t0,
            mode="uniaxial_tension",
            deformed_positions=deformed_pos,
        )

    def biaxial_tension(
        self,
        graph: StructureGraph,
        strain_x: float = 0.01,
        strain_y: float = 0.01,
        youngs_modulus: float = 1e9,
        radius: float = 0.05,
    ) -> SimResult:
        """Biaxial tension in x and y."""
        t0 = time.time()
        pos, elements, _, _ = _graph_to_arrays(graph)
        radii = np.full(len(elements), radius)
        bnd = _get_boundary_indices(pos)
        dim = 2 if np.allclose(pos[:, 2], pos[0, 2]) else 3
        L_x = pos[:, 0].max() - pos[:, 0].min()
        L_y = pos[:, 1].max() - pos[:, 1].min()

        fixed_dofs = []
        applied = {}

        for ni in bnd["left"]:
            fixed_dofs.append(ni * dim)
        for ni in bnd["right"]:
            applied[ni * dim] = strain_x * L_x
        for ni in bnd["bottom"]:
            fixed_dofs.append(ni * dim + 1)
        for ni in bnd["top"]:
            applied[ni * dim + 1] = strain_y * L_y

        displacements, strains, stresses = self._solve_truss(
            pos, elements, youngs_modulus, radii, fixed_dofs,
            applied_displacements=applied,
        )

        K = self._build_stiffness(pos, elements, youngs_modulus, radii, dim)
        u_flat = displacements[:, :dim].flatten()
        energy = 0.5 * u_flat @ (K @ u_flat)

        return SimResult(
            displacements=displacements, forces=None,
            stresses=stresses, strains=strains,
            energy=energy,
            effective_youngs_modulus=(abs(strain_x) + abs(strain_y)) / 2 * youngs_modulus,
            time_seconds=time.time() - t0,
            mode="biaxial_tension",
            deformed_positions=pos + displacements,
        )

    def compression(
        self,
        graph: StructureGraph,
        strain: float = 0.01,
        youngs_modulus: float = 1e9,
        radius: float = 0.05,
    ) -> SimResult:
        """Uniaxial compression."""
        result = self.uniaxial_tension(graph, strain=-abs(strain),
                                       youngs_modulus=youngs_modulus, radius=radius)
        result.mode = "compression"
        return result

    def shear_test(
        self,
        graph: StructureGraph,
        strain: float = 0.01,
        youngs_modulus: float = 1e9,
        radius: float = 0.05,
    ) -> SimResult:
        """Simple shear test."""
        t0 = time.time()
        pos, elements, _, _ = _graph_to_arrays(graph)
        radii = np.full(len(elements), radius)
        bnd = _get_boundary_indices(pos)
        dim = 2
        L_y = pos[:, 1].max() - pos[:, 1].min()

        fixed_dofs = []
        applied = {}
        for ni in bnd["bottom"]:
            fixed_dofs.extend([ni * dim, ni * dim + 1])
        for ni in bnd["top"]:
            applied[ni * dim] = strain * L_y
            fixed_dofs.append(ni * dim + 1)

        displacements, strains, stresses = self._solve_truss(
            pos, elements, youngs_modulus, radii, fixed_dofs,
            applied_displacements=applied,
        )

        K = self._build_stiffness(pos, elements, youngs_modulus, radii, dim)
        u_flat = displacements[:, :dim].flatten()
        f_react = K @ u_flat
        F_total = sum(f_react[ni * dim] for ni in bnd["top"])
        L_x = pos[:, 0].max() - pos[:, 0].min()
        G_eff = abs(F_total) / (abs(strain) * L_x) if abs(strain) > 1e-12 else 0

        energy = 0.5 * u_flat @ (K @ u_flat)

        return SimResult(
            displacements=displacements, forces=None,
            stresses=stresses, strains=strains,
            energy=energy,
            effective_shear_modulus=G_eff,
            time_seconds=time.time() - t0,
            mode="shear",
            deformed_positions=pos + displacements,
        )

    def _build_stiffness(self, pos, elements, E, radii, dim):
        import scipy.sparse as sp
        num_nodes, num_elements = pos.shape[0], elements.shape[0]
        ndof = num_nodes * dim
        lengths, directions = _element_data(pos, elements)
        areas = np.pi * radii[:num_elements] ** 2
        axial_k = E * areas / lengths
        rows, cols, vals = [], [], []
        for e in range(num_elements):
            ni, nj = elements[e]
            for a in range(dim):
                for b in range(dim):
                    v = axial_k[e] * directions[e, a] * directions[e, b]
                    if abs(v) > 1e-15:
                        rows.extend([ni*dim+a, ni*dim+a, nj*dim+a, nj*dim+a])
                        cols.extend([ni*dim+b, nj*dim+b, ni*dim+b, nj*dim+b])
                        vals.extend([v, -v, -v, v])
        return sp.csr_matrix((vals, (rows, cols)), shape=(ndof, ndof))


# ======================================================================
# Backend 2: TaichiEngine (Mass-Spring Dynamics)
# ======================================================================
