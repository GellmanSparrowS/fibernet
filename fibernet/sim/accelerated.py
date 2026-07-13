"""
Taichi-accelerated simulations for fiber networks.

Two backends:
TaichiEngine is the primary simulation backend (mass-spring dynamics).
2. **TaichiEngine** — Mass-spring dynamics (explicit Verlet integration, Taichi parallel forces)

Both accept StructureGraph directly via high-level test methods.
"""

from __future__ import annotations

import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Union

import numpy as np

try:
    import taichi as ti
    HAS_TAICHI = True
except ImportError:
    HAS_TAICHI = False

from fibernet.core.structure_graph import StructureGraph


@dataclass
class SimResult:
    """Unified result container for both backends."""
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

    def save(self, path: str):
        data = {
            "mode": self.mode,
            "energy": self.energy,
            "effective_youngs_modulus": self.effective_youngs_modulus,
            "effective_poissons_ratio": self.effective_poissons_ratio,
            "effective_shear_modulus": self.effective_shear_modulus,
            "time_seconds": self.time_seconds,
            "displacements": self.displacements.tolist() if self.displacements is not None else None,
            "forces": self.forces.tolist() if self.forces is not None else None,
            "stresses": self.stresses.tolist() if self.stresses is not None else None,
            "strains": self.strains.tolist() if self.strains is not None else None,
            "history": self.history,
        }
        if self.deformed_positions is not None:
            data["deformed_positions"] = self.deformed_positions.tolist()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str) -> "SimResult":
        with open(path) as f:
            data = json.load(f)
        r = SimResult(
            mode=data.get("mode", ""),
            energy=data.get("energy", 0),
            effective_youngs_modulus=data.get("effective_youngs_modulus", 0),
            effective_poissons_ratio=data.get("effective_poissons_ratio", 0),
            effective_shear_modulus=data.get("effective_shear_modulus", 0),
            time_seconds=data.get("time_seconds", 0),
            history=data.get("history", []),
        )
        for key in ("displacements", "forces", "stresses", "strains", "deformed_positions"):
            if data.get(key) is not None:
                setattr(r, key, np.array(data[key]))
        return r


def _ensure_taichi(arch: str = "cpu", num_threads: int = 4):
    if not HAS_TAICHI:
        raise ImportError("Taichi required: pip install taichi")
    try:
        if not ti.is_initialized():
            pass
    except AttributeError:
        pass
    try:
        arch_map = {"cpu": ti.cpu, "gpu": ti.gpu}
        ti.init(arch=arch_map.get(arch, ti.cpu), cpu_max_num_threads=num_threads)
    except RuntimeError:
        pass


def _graph_to_arrays(graph: StructureGraph):
    """Extract positions, elements, node_id mapping from StructureGraph."""
    node_ids = list(graph.nodes.keys())
    nid_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    pos = np.array([graph.nodes[nid].position for nid in node_ids])
    elements = np.array([[nid_to_idx[e.node_i], nid_to_idx[e.node_j]] for e in graph.edges.values()])
    return pos, elements, node_ids, nid_to_idx


def _get_boundary_indices(pos: np.ndarray, tol: float = None) -> Dict[str, List[int]]:
    """Find boundary node indices for each side."""
    if tol is None:
        span = pos.max(0) - pos.min(0)
        tol = max(span.min() * 0.05, 0.1)
    bb_min, bb_max = pos.min(0), pos.max(0)
    result = {}
    result["left"] = list(np.where(pos[:, 0] < bb_min[0] + tol)[0])
    result["right"] = list(np.where(pos[:, 0] > bb_max[0] - tol)[0])
    result["bottom"] = list(np.where(pos[:, 1] < bb_min[1] + tol)[0])
    result["top"] = list(np.where(pos[:, 1] > bb_max[1] - tol)[0])
    if pos.shape[1] >= 3:
        result["back"] = list(np.where(pos[:, 2] < bb_min[2] + tol)[0])
        result["front"] = list(np.where(pos[:, 2] > bb_max[2] - tol)[0])
    return result


def _element_data(pos, elements):
    """Compute element lengths, directions, areas."""
    diff = pos[elements[:, 1]] - pos[elements[:, 0]]
    lengths = np.linalg.norm(diff, axis=1)
    lengths = np.maximum(lengths, 1e-12)
    directions = diff / lengths[:, None]
    return lengths, directions


# ======================================================================
# ======================================================================


class TaichiEngine:
    """Mass-spring dynamics with Taichi parallel force computation.

    Axial spring model: F = k * (L - L0) / L0 * direction
    Explicit Verlet integration with damping.
    """

    def __init__(self, arch: str = "cpu", num_threads: int = 4):
        _ensure_taichi(arch, num_threads)

    def compute_forces(
        self,
        positions: np.ndarray,
        rest_lengths: np.ndarray,
        stiffness: np.ndarray,
        edges: np.ndarray,
    ) -> np.ndarray:
        """Compute spring forces in parallel."""
        num_nodes = positions.shape[0]
        num_edges = edges.shape[0]
        dim = positions.shape[1]

        pos = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
        forces = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
        edge_arr = ti.Vector.field(2, dtype=ti.i32, shape=num_edges)
        L0 = ti.field(dtype=ti.f64, shape=num_edges)
        k_field = ti.field(dtype=ti.f64, shape=num_edges)
        f_temp = ti.Vector.field(dim, dtype=ti.f64, shape=num_edges)

        pos.from_numpy(positions.astype(np.float64))
        edge_arr.from_numpy(edges.astype(np.int32))
        L0.from_numpy(rest_lengths.astype(np.float64))
        k_field.from_numpy(stiffness.astype(np.float64))

        @ti.kernel
        def compute():
            for e in range(num_edges):
                i = edge_arr[e][0]
                j = edge_arr[e][1]
                diff = pos[j] - pos[i]
                L = diff.norm()
                if L > 1e-12:
                    strain_val = (L - L0[e]) / L0[e]
                    force_mag = k_field[e] * strain_val
                    f_temp[e] = force_mag * (diff / L)
                else:
                    for d in ti.static(range(dim)):
                        f_temp[e][d] = 0.0

        @ti.kernel
        def zero():
            for n in range(num_nodes):
                for d in ti.static(range(dim)):
                    forces[n][d] = 0.0

        @ti.kernel
        def accumulate():
            for e in range(num_edges):
                i = edge_arr[e][0]
                j = edge_arr[e][1]
                for d in ti.static(range(dim)):
                    forces[i][d] += f_temp[e][d]
                    forces[j][d] -= f_temp[e][d]

        zero()
        compute()
        accumulate()
        return forces.to_numpy()

    def dynamics(
        self,
        graph: StructureGraph,
        fixed_nodes: List[int] = None,
        displacement_schedule: Dict[int, List[Tuple[float, np.ndarray]]] = None,
        external_force: np.ndarray = None,
        stiffness: float = 1e5,
        damping: float = 0.3,
        dt: float = 1e-4,
        num_steps: int = 5000,
        save_interval: int = 500,
        spring_k: float = None,
        dashpot: float = 10.0,
        drag: float = 1.0,
    ) -> SimResult:
        """Run mass-spring dynamics with dashpot damping and air drag.

        Based on reference implementation with proper physics:
        - Spring force: F = -k * dir * (dist/rest - 1)
        - Dashpot damping: F_damp = -damp * (vi-vj).dot(dir) * dir * rest_len
        - Air drag: v *= exp(-drag * dt)
        - Constraints by position clamping + velocity zeroing

        Parameters
        ----------
        graph : StructureGraph
        fixed_nodes : list of node indices to fix
        displacement_schedule : dict mapping node_idx → [(step, displacement), ...]
        external_force : (N, 3) array
        stiffness : global spring stiffness (overridden by spring_k if given)
        damping : dashpot damping coefficient
        dt : time step
        num_steps : total integration steps
        save_interval : save trajectory every N steps
        spring_k : per-edge spring stiffness array (overrides stiffness)
        dashpot : dashpot damping coefficient
        drag : air drag coefficient
        """
        t0 = time.time()
        pos_orig, elements, node_ids, _ = _graph_to_arrays(graph)
        dim = pos_orig.shape[1]
        num_nodes = len(node_ids)
        num_edges = len(elements)

        lengths, _ = _element_data(pos_orig, elements)
        rest_lengths_np = lengths.copy()

        # Spring stiffness
        if spring_k is not None:
            stiff_np = np.asarray(spring_k, dtype=np.float64)
        else:
            stiff_np = np.full(num_edges, stiffness)

        # Taichi fields (allocated once)
        ti_pos = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
        ti_vel = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
        ti_edges = ti.Vector.field(2, dtype=ti.i32, shape=num_edges)
        ti_L0 = ti.field(dtype=ti.f64, shape=num_edges)
        ti_k = ti.field(dtype=ti.f64, shape=num_edges)
        ti_fixed = ti.field(dtype=ti.i32, shape=num_nodes)
        ti_ext = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
        ti_pos0 = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)

        ti_pos0.from_numpy(pos_orig.astype(np.float64))
        ti_pos.from_numpy(pos_orig.astype(np.float64))
        ti_vel.from_numpy(np.zeros_like(pos_orig))
        ti_edges.from_numpy(elements.astype(np.int32))
        ti_L0.from_numpy(rest_lengths_np.astype(np.float64))
        ti_k.from_numpy(stiff_np.astype(np.float64))

        fixed_arr = np.zeros(num_nodes, dtype=np.int32)
        for ni in (fixed_nodes or []):
            fixed_arr[ni] = 1
        ti_fixed.from_numpy(fixed_arr)

        ext_f = external_force if external_force is not None else np.zeros((num_nodes, dim))
        ti_ext.from_numpy(ext_f.astype(np.float64)[:, :dim])

        # Schedule fields
        schedule_nodes = []
        schedule_targets = np.zeros((num_nodes, dim))
        if displacement_schedule:
            for ni, sched in displacement_schedule.items():
                schedule_nodes.append(ni)
                schedule_targets[ni] = np.array(sched[-1][1])[:dim]
        sched_mask = np.zeros(num_nodes, dtype=np.int32)
        for ni in schedule_nodes:
            sched_mask[ni] = 1
        ti_sched = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
        ti_sched.from_numpy(schedule_targets.astype(np.float64))
        ti_sched_mask = ti.field(dtype=ti.i32, shape=num_nodes)
        ti_sched_mask.from_numpy(sched_mask)

        # Parameter fields
        ti_dt = ti.field(dtype=ti.f64, shape=())
        ti_dashpot = ti.field(dtype=ti.f64, shape=())
        ti_drag = ti.field(dtype=ti.f64, shape=())
        ti_ramp = ti.field(dtype=ti.f64, shape=())

        @ti.kernel
        def substep():
            _dt = ti_dt[None]
            _dash = ti_dashpot[None]
            _drag = ti_drag[None]
            _ramp = ti_ramp[None]

            # Spring + dashpot forces
            for e in range(num_edges):
                ia = ti_edges[e][0]
                ib = ti_edges[e][1]
                d = ti_pos[ia] - ti_pos[ib]
                dist = d.norm() + 1e-6
                dir_ = d / dist

                # Spring force: F = -k * dir * (dist/rest - 1)
                f_spring = -ti_k[e] * dir_ * (dist / ti_L0[e] - 1.0)

                # Dashpot: F_damp = -damp * (vi-vj).dot(dir) * dir * rest
                rel_v = ti_vel[ia] - ti_vel[ib]
                f_damp = -_dash * rel_v.dot(dir_) * dir_ * ti_L0[e]

                f_total = f_spring + f_damp

                for k in ti.static(range(dim)):
                    ti.atomic_add(ti_vel[ia][k], f_total[k] * _dt)
                    ti.atomic_add(ti_vel[ib][k], -f_total[k] * _dt)

            # Euler step + air drag + external force
            for i in range(num_nodes):
                # External force
                for k in ti.static(range(dim)):
                    ti_vel[i][k] += ti_ext[i][k] * _dt

                # Air drag
                ti_vel[i] *= ti.exp(-_drag * _dt)

                # Position update
                for k in ti.static(range(dim)):
                    ti_pos[i][k] += ti_vel[i][k] * _dt

            # Constraints: fixed nodes
            for i in range(num_nodes):
                if ti_fixed[i] == 1:
                    for k in ti.static(range(dim)):
                        ti_pos[i][k] = ti_pos0[i][k]
                        ti_vel[i][k] = 0.0

            # Constraints: schedule nodes (ramped displacement)
            for i in range(num_nodes):
                if ti_sched_mask[i] == 1:
                    for k in ti.static(range(dim)):
                        ti_pos[i][k] = ti_pos0[i][k] + ti_sched[i][k] * _ramp
                        ti_vel[i][k] = 0.0

        # Run loop
        ti_dt[None] = dt
        ti_dashpot[None] = dashpot
        ti_drag[None] = drag

        trajectory = [pos_orig.copy()]
        max_stretch_history = []

        for step in range(num_steps):
            ramp = min(1.0, (step + 1) / num_steps)
            ti_ramp[None] = ramp
            substep()

            if (step + 1) % save_interval == 0:
                cur_pos = ti_pos.to_numpy()
                trajectory.append(cur_pos.copy())
                new_len = np.array([
                    np.linalg.norm(cur_pos[elements[e, 1]] - cur_pos[elements[e, 0]])
                    for e in range(num_edges)
                ])
                max_stretch_history.append(float(np.max(new_len / rest_lengths_np)))

        pos_final = ti_pos.to_numpy()
        displacements = pos_final - pos_orig
        if dim < 3:
            displacements = np.hstack([displacements, np.zeros((num_nodes, 3 - dim))])

        return SimResult(
            displacements=displacements,
            time_seconds=time.time() - t0,
            mode="dynamics",
            deformed_positions=pos_final,
            positions_trajectory=trajectory,
            history=[{"step": (i+1)*save_interval, "max_stretch": ms}
                     for i, ms in enumerate(max_stretch_history)],
        )
    def stretch_test(
        self,
        graph: StructureGraph,
        target_stretch: float = 2.0,
        stiffness: float = 1e5,
        damping: float = 0.3,
        num_steps: int = None,
        save_interval: int = 1000,
        ramp_fraction: float = 0.2,
        auto_steps: bool = True,
    ) -> SimResult:
        """Displacement-controlled uniaxial stretch with automatic step calculation.

        Two-phase simulation:
        1. Ramp phase: gradually move right boundary to target_stretch
        2. Relax phase: hold displacement, let waves propagate and settle

        Parameters
        ----------
        target_stretch : float
            Target stretch ratio (e.g., 2.0 = double the length)
        stiffness : float
            Spring stiffness (higher = stiffer springs)
        damping : float
            Velocity damping coefficient (0-1)
        num_steps : int, optional
            Total simulation steps. If None and auto_steps=True, calculated from graph diameter
        save_interval : int
            Save trajectory every N steps
        ramp_fraction : float
            Fraction of total steps for ramp phase (default 0.2 = 20%)
        auto_steps : bool
            Auto-calculate steps based on graph diameter if num_steps is None

        Returns
        -------
        SimResult
            Result with deformed_positions and trajectory
        """
        import networkx as nx

        pos, elements, _, _ = _graph_to_arrays(graph)
        bnd = _get_boundary_indices(pos)
        L_x = pos[:, 0].max() - pos[:, 0].min()
        target_disp = L_x * (target_stretch - 1)

        # Auto-calculate steps based on graph diameter
        if num_steps is None and auto_steps:
            # Build graph to find diameter
            G = nx.Graph()
            for e in elements:
                G.add_edge(int(e[0]), int(e[1]))
            
            # Find max hops from right to left
            right_set = set(bnd['right'])
            left_set = set(bnd['left'])
            max_hops = 0
            for rid in right_set:
                for lid in left_set:
                    try:
                        hops = nx.shortest_path_length(G, rid, lid)
                        max_hops = max(max_hops, hops)
                    except:
                        pass
            
            # Estimate steps needed
            # Wave propagation: ~100 steps per hop
            wave_steps = max_hops * 100
            # Relaxation: 30000 steps for drag=1.0
            relax_steps = 30000
            # Total with safety margin
            num_steps = max(wave_steps + relax_steps, 20000) * 2
            print(f"Auto-calculated {num_steps} steps (diameter={max_hops} hops)")
        elif num_steps is None:
            num_steps = 20000

        # Displacement schedule: ramp phase + hold phase
        ramp_steps = int(num_steps * ramp_fraction)
        schedule = {}
        for ni in bnd["right"]:
            schedule[ni] = [
                (0, np.array([0.0, 0.0, 0.0])),
                (ramp_steps, np.array([target_disp, 0.0, 0.0])),
                (num_steps, np.array([target_disp, 0.0, 0.0]))  # hold
            ]

        fixed = bnd["left"] + (bnd.get("bottom", [])[:1] if bnd.get("bottom") else [])

        return self.dynamics(
            graph,
            fixed_nodes=fixed,
            displacement_schedule=schedule,
            stiffness=stiffness,
            damping=damping,
            dt=1e-5,
            num_steps=num_steps,
            save_interval=save_interval,
        )

    @staticmethod
    def _interpolate_schedule(schedule, step, total_steps):
        """Linearly interpolate displacement from schedule."""
        if not schedule:
            return np.zeros(3)
        schedule = sorted(schedule, key=lambda x: x[0])
        if step <= schedule[0][0]:
            return np.array(schedule[0][1])
        if step >= schedule[-1][0]:
            return np.array(schedule[-1][1])
        for i in range(len(schedule) - 1):
            s0, d0 = schedule[i]
            s1, d1 = schedule[i + 1]
            if s0 <= step <= s1:
                t = (step - s0) / max(1, s1 - s0)
                return np.array(d0) * (1 - t) + np.array(d1) * t
        return np.array(schedule[-1][1])


