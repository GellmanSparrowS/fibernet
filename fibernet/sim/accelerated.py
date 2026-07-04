"""
Taichi-accelerated simulations for large fiber networks.

Provides GPU/CPU-parallel implementations of:
- Beam FEM assembly (parallel element processing)
- Mass-spring dynamics (parallel force computation)
- Random network generation (parallel fiber deposition)

Uses Taichi's CPU backend for parallelism without GPU requirement.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

try:
    import taichi as ti
    HAS_TAICHI = True
except ImportError:
    HAS_TAICHI = False


@dataclass
class AcceleratedResult:
    """Container for accelerated simulation results."""
    displacements: np.ndarray = None
    forces: np.ndarray = None
    positions: List[np.ndarray] = None
    energy: float = 0.0
    time_seconds: float = 0.0


class TaichiEngine:
    """Taichi-accelerated computation engine.
    
    Parameters
    ----------
    arch : str
        'cpu' or 'gpu'. Defaults to 'cpu'.
    num_threads : int
        Number of CPU threads.
    """
    
    def __init__(self, arch: str = "cpu", num_threads: int = 4):
        if not HAS_TAICHI:
            raise ImportError("Taichi required. Install with: pip install taichi")
        
        try:
            initialized = ti.is_initialized()
        except AttributeError:
            initialized = False
        
        if not initialized:
            try:
                if arch == "cpu":
                    ti.init(arch=ti.cpu, cpu_max_num_threads=num_threads)
                elif arch == "gpu":
                    ti.init(arch=ti.gpu)
                else:
                    ti.init(arch=ti.cpu)
            except RuntimeError:
                pass
    
    def parallel_force_computation(
        self,
        positions: np.ndarray,
        rest_lengths: np.ndarray,
        stiffness: np.ndarray,
        edges: np.ndarray,
    ) -> np.ndarray:
        """Compute spring forces in parallel using Taichi.
        
        Parameters
        ----------
        positions : np.ndarray
            Node positions (N, 3).
        rest_lengths : np.ndarray
            Rest lengths for each edge.
        stiffness : np.ndarray
            Spring stiffness for each edge.
        edges : np.ndarray
            Edge connectivity (M, 2) - node indices.
        """
        num_nodes = positions.shape[0]
        num_edges = edges.shape[0]
        
        pos = ti.Vector.field(3, dtype=ti.f64, shape=num_nodes)
        forces = ti.Vector.field(3, dtype=ti.f64, shape=num_nodes)
        edge_arr = ti.Vector.field(2, dtype=ti.i32, shape=num_edges)
        L0 = ti.field(dtype=ti.f64, shape=num_edges)
        k = ti.field(dtype=ti.f64, shape=num_edges)
        f_temp = ti.Vector.field(3, dtype=ti.f64, shape=num_edges)
        
        pos.from_numpy(positions.astype(np.float64))
        edge_arr.from_numpy(edges.astype(np.int32))
        L0.from_numpy(rest_lengths.astype(np.float64))
        k.from_numpy(stiffness.astype(np.float64))
        
        @ti.kernel
        def compute_edge_forces():
            for e in range(num_edges):
                i = edge_arr[e][0]
                j = edge_arr[e][1]
                diff = pos[j] - pos[i]
                L = ti.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
                if L > 1e-12:
                    strain = (L - L0[e]) / L0[e]
                    force_mag = k[e] * strain
                    direction = diff / L
                    f_temp[e] = force_mag * direction
                else:
                    f_temp[e] = ti.Vector([0.0, 0.0, 0.0])
        
        @ti.kernel
        def zero_forces():
            for n in range(num_nodes):
                forces[n] = ti.Vector([0.0, 0.0, 0.0])
        
        @ti.kernel
        def accumulate_forces():
            for e in range(num_edges):
                i = edge_arr[e][0]
                j = edge_arr[e][1]
                ti.atomic_add(forces[i], f_temp[e])
                ti.atomic_add(forces[j], -f_temp[e])
        
        zero_forces()
        compute_edge_forces()
        accumulate_forces()
        
        return forces.to_numpy()
    
    def parallel_dynamics(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        rest_lengths: np.ndarray,
        stiffness: np.ndarray,
        edges: np.ndarray,
        dt: float = 1e-6,
        damping: float = 0.01,
        num_steps: int = 1000,
        save_interval: int = 100,
        fixed_nodes: Optional[List[int]] = None,
    ) -> AcceleratedResult:
        """Run parallel mass-spring dynamics using Taichi.
        
        Parameters
        ----------
        positions : np.ndarray
            Initial node positions (N, 3).
        velocities : np.ndarray
            Initial node velocities (N, 3).
        masses : np.ndarray
            Node masses (N,).
        edges : np.ndarray
            Edge connectivity (M, 2).
        dt : float
            Time step.
        num_steps : int
            Number of steps.
        """
        import time
        start_time = time.time()
        
        num_nodes = positions.shape[0]
        num_edges = edges.shape[0]
        fixed_set = set(fixed_nodes) if fixed_nodes else set()
        
        pos = ti.Vector.field(3, dtype=ti.f64, shape=num_nodes)
        vel = ti.Vector.field(3, dtype=ti.f64, shape=num_nodes)
        mass = ti.field(dtype=ti.f64, shape=num_nodes)
        frc = ti.Vector.field(3, dtype=ti.f64, shape=num_nodes)
        edge_arr = ti.Vector.field(2, dtype=ti.i32, shape=num_edges)
        L0 = ti.field(dtype=ti.f64, shape=num_edges)
        k_spring = ti.field(dtype=ti.f64, shape=num_edges)
        f_temp = ti.Vector.field(3, dtype=ti.f64, shape=num_edges)
        is_fixed = ti.field(dtype=ti.i32, shape=num_nodes)
        
        pos.from_numpy(positions.astype(np.float64))
        vel.from_numpy(velocities.astype(np.float64))
        mass.from_numpy(masses.astype(np.float64))
        edge_arr.from_numpy(edges.astype(np.int32))
        L0.from_numpy(rest_lengths.astype(np.float64))
        k_spring.from_numpy(stiffness.astype(np.float64))
        
        fixed_arr = np.zeros(num_nodes, dtype=np.int32)
        for n in fixed_set:
            fixed_arr[n] = 1
        is_fixed.from_numpy(fixed_arr)
        
        @ti.kernel
        def compute_forces():
            for n in range(num_nodes):
                frc[n] = ti.Vector([0.0, 0.0, 0.0])
            
            for e in range(num_edges):
                i = edge_arr[e][0]
                j = edge_arr[e][1]
                diff = pos[j] - pos[i]
                L = ti.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
                if L > 1e-12:
                    strain = (L - L0[e]) / L0[e]
                    force_mag = k_spring[e] * strain
                    direction = diff / L
                    f = force_mag * direction
                    ti.atomic_add(frc[i], f)
                    ti.atomic_add(frc[j], -f)
        
        @ti.kernel
        def integrate(damp: ti.f64, step_dt: ti.f64):
            for n in range(num_nodes):
                if is_fixed[n] == 0:
                    vel[n] = vel[n] + step_dt * (frc[n] - damp * vel[n]) / mass[n]
                    pos[n] = pos[n] + step_dt * vel[n]
        
        saved_positions = []
        for step in range(num_steps):
            compute_forces()
            integrate(damping, dt)
            
            if step % save_interval == 0:
                saved_positions.append(pos.to_numpy().copy())
        
        elapsed = time.time() - start_time
        
        return AcceleratedResult(
            positions=saved_positions,
            time_seconds=elapsed,
        )
    
    def parallel_generate_random_3d(
        self,
        num_fibers: int,
        fiber_length: float,
        box_size: Tuple[float, float, float],
        radius: float = 0.1,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate random fiber endpoints in parallel.
        
        Returns array of shape (num_fibers, 2, 3) for start/end points.
        """
        rng = np.random.default_rng(seed)
        Lx, Ly, Lz = box_size
        
        centers = rng.uniform(0, 1, (num_fibers, 3)) * np.array([Lx, Ly, Lz])
        
        theta = rng.uniform(0, 2 * np.pi, num_fibers)
        cos_phi = rng.uniform(-1, 1, num_fibers)
        sin_phi = np.sqrt(1 - cos_phi**2)
        
        directions = np.column_stack([
            sin_phi * np.cos(theta),
            sin_phi * np.sin(theta),
            cos_phi,
        ])
        
        lengths = fiber_length * (0.5 + rng.uniform(size=num_fibers))
        
        starts = centers - 0.5 * lengths[:, None] * directions
        ends = centers + 0.5 * lengths[:, None] * directions
        
        endpoints = np.stack([starts, ends], axis=1)
        return endpoints
